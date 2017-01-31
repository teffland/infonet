import numpy as np
import numpy.random as npr
import chainer as ch
import chainer.functions as F
import chainer.links as L

from tagger import extract_all_mentions, TaggerLoss
from special_functions import batch_weighted_softmax_cross_entropy
from gru import GRU, BidirectionalGRU
from masked_softmax import masked_softmax

class Extractor(ch.Chain):
    def __init__(self,
                 tagger,
                 n_mention_class,
                 n_relation_class,
                 null_idx, coref_idx,
                 lstm_size=50,
                 bidirectional=False,
                 n_layers=1,
                 use_mlp=False,
                 shortcut_embeds=False,
                 dropout=.25,
                 start_tags=(2,),
                 in_tags=(1,2),
                 out_tags=(0,),
                 tag2mtype=None,
                 mtype2msubtype=None,
                 msubtype2rtype=None,
                 max_rel_dist=10000):
        # setup rnn layer
        if shortcut_embeds:
            tagger_feature_size = tagger.feature_size + tagger.embedding_size
        else:
            tagger_feature_size = tagger.feature_size
        if bidirectional:
            feature_size = 2*lstm_size
            lstms = [BidirectionalGRU(lstm_size, n_inputs=tagger_feature_size)]
            for i in range(1,n_layers):
                lstms.append(BidirectionalGRU(lstm_size, n_inputs=feature_size))
        else:
            feature_size = lstm_size
            lstms = [GRU(lstm_size, n_inputs=tagger_feature_size)]
            for i in range(1,n_layers):
                lstms.append(GRU(lstm_size, n_inputs=feature_size))
        # setup other links
        super(Extractor, self).__init__(
            tagger=tagger,
            mlp = L.Linear(feature_size, feature_size),
            out=L.Linear(feature_size, feature_size),
            f_m=L.Linear(feature_size+1, n_mention_class),
            f_r=L.Linear(2*feature_size+3, n_relation_class)
        )
        self.n_mention_class = n_mention_class
        self.n_relation_class = n_relation_class
        self.lstms = lstms
        for i, lstm in enumerate(self.lstms):
            self.add_link('lstm_{}'.format(i), lstm)
        self.lstm_size = lstm_size
        self.feature_size = feature_size
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.use_mlp = use_mlp
        self.shortcut_embeds = shortcut_embeds
        self.n_layers = n_layers
        self.start_tags = start_tags
        self.in_tags = in_tags
        self.out_tags = out_tags
        self.tag2mtype = tag2mtype
        self.max_rel_dist = max_rel_dist
        self.null_idx = null_idx
        self.coref_idx = coref_idx

        # convert the typemaps to indicator array masks
        # mention type -> subtype uses the string label, so keep it as a dict
        # for mtypes ('entity' and 'event-anchor') the indices
        # are kept as the raw tokens.  theyre assembled
        for k,v in mtype2msubtype.items():
            mask = np.zeros(n_mention_class).astype(np.float32)
            mask[np.array(v)] = 1.
            mtype2msubtype[k] = mask
        self.mtype2msubtype = mtype2msubtype
        # for mention subtype -> relation type
        # we will use the predictions for the mentions
        # which means its easiest to use the indices of the labels
        # so we instead create a a mask matrix and look them up with EmbedID
        left_masks = np.zeros((n_mention_class, n_relation_class)).astype(np.float32)
        right_masks = np.zeros((n_mention_class, n_relation_class)).astype(np.float32)
        for k,v in msubtype2rtype['left'].items():
            left_masks[k, np.array(v)] = 1.
        for k,v in msubtype2rtype['right'].items():
            right_masks[k, np.array(v)] = 1.
        self.left_masks = left_masks
        self.right_masks = right_masks

        # print self.mtype2msubtype
        # print
        # print self.msubtype2rtype

    def reset_state(self):
        self.tagger.reset_state()
        for lstm in self.lstms:
            lstm.reset_state()

    def _extract_graph(self, tagger_preds, features):
        """ Subroutine responsible for extracting the graph and graph features
        from the tagger predictions using `extract_all_mentions`.

        Note: This function can be slow for large documents.
          This is unavoidable for documents with large mention counts (m)
          because the number of relations r is naively (m choose 2).
          This graph can be pruned by setting `max_rel_dist`,
          which will omit all relations for mentions `max_rel_dist` apart
          (as measured from the left edge of the consituent mention spans.)
        """
        # TODO: Possible extension: using separate features for mentions and relations

        # convert from time-major to batch-major
        tagger_preds = F.transpose_sequence(tagger_preds)
        # for p in tagger_preds:
            # print p.shape, p.data
        features = F.transpose_sequence(features)

        # extract the mentions and relations for each doc
        all_boundaries = extract_all_mentions(tagger_preds,
                                              start_tags=self.start_tags,
                                              in_tags=self.in_tags,
                                              out_tags=self.out_tags,
                                              tag2mtype=self.tag2mtype)

        all_mentions = [] # features for mentions
        all_mention_spans = [] # spans in doc for each mention
        all_mention_masks = [] # bool masks for constraining predictions
        all_left_mention_idxs = [] # idxs for left mention of a relation in mention table
        all_right_mention_idxs = [] # idxs for right mention of a relation in mention table
        all_relation_spans = [] # spans in doc for left and right mentions
        all_null_rel_spans = [] # predict null for mention pairs > max_rel_dist
        zipped = zip(all_boundaries, tagger_preds, features)

        # extract graph and features for each doc
        for s, (boundaries, seq, features) in enumerate(zipped):
            mentions = []
            mention_spans = []
            mention_masks = []
            left_mention_idxs = []
            right_mention_idxs = []
            relation_spans = []
            null_rel_spans = []
            # left_mention_masks = []
            # right_mention_masks = []
            moving_rel_idx = 0
            # print '{} mentions'.format(len(boundaries))
            for i, b in enumerate(boundaries):
                # mention feature is average of its span features
                mention = F.sum(features[b[0]:b[1]], axis=0)
                mention /= F.broadcast_to(ch.Variable(np.array(b[1]-b[0],
                                                             dtype=np.float32)),
                                                 mention.shape)
                mentions.append(mention)
                mention_spans.append((b[0], b[1]))
                mention_masks.append(self.mtype2msubtype[b[2]])
                # make a relation to all previous mentions (M choose 2)
                # except those that are further than max_rel_dist away
                # (prune for speed, accuracy)
                for j in range(i):
                # for j in range(moving_rel_idx, i):
                    # print j,i
                    bj = boundaries[j]
                    if abs(bj[0] - b[0]) < self.max_rel_dist:
                        relation_spans.append((bj[0], bj[1], b[0], b[1]))
                        left_mention_idxs.append(np.array(j).astype(np.int32))
                        right_mention_idxs.append(np.array(i).astype(np.int32))
                    # if that's too far, then it'll def be too far for the next
                    else:
                        null_rel_spans.append((bj[0], bj[1], b[0], b[1]))
                        # moving_rel_idx = j

            # convert list of mentions to matrix and append
            mentions = F.vstack(mentions)
            # add in span widths as features
            m_spans = np.array(mention_spans).astype(np.float32)
            m_wids = m_spans[:,1]-m_spans[:,0]
            m_wids  = m_wids.reshape((-1,1))
            # m_dists = np.array([ s[1]-s[0] for s in mention_spans ] ).astype(np.float32).reshape((-1,1))
            mentions = F.hstack([mentions, ch.Variable(m_wids)])
            mention_masks = F.vstack(mention_masks)
            all_mentions.append(mentions)
            all_mention_masks.append(mention_masks)
            all_mention_spans.append(m_spans)

            # same for relations
            left_mention_idxs = F.vstack(left_mention_idxs)
            all_left_mention_idxs.append(left_mention_idxs)
            right_mention_idxs = F.vstack(right_mention_idxs)
            all_right_mention_idxs.append(right_mention_idxs)
            all_relation_spans.append(relation_spans)
            all_null_rel_spans.append(null_rel_spans)

        return (all_mentions, all_mention_masks,
                all_left_mention_idxs, all_right_mention_idxs,
                all_mention_spans, all_relation_spans, all_null_rel_spans)

    def __call__(self, x_list, train=True, backprop_to_tagger=False):
        drop = F.dropout
        # first tag the doc
        tagger_preds, tagger_features = self.tagger.predict(x_list,
                                                            return_features=True)

        if not backprop_to_tagger:
            # allow backprop through tagger_features but not features
            features = [ F.identity(f) for f in tagger_features ]
            for f in features:
                f.unchain_backward()
        else:
            features = tagger_features

        if self.shortcut_embeds:
            embeds = self.tagger.embeds
            features = [ F.hstack([f,e])
                         for f,e in zip(features, embeds)]

        # do another layer of features on the tagger layer
        if self.bidirectional:
            # helper function
            def bilstm(inputs, lstm):
                f_lstms, b_lstms = [], []
                for x_f, x_b in zip(inputs, inputs[::-1]):
                    h_f, h_b = lstm(x_f, x_b)
                    f_lstms.append(h_f)
                    b_lstms.append(h_b)
                b_lstms = b_lstms[::-1]
                return [ F.hstack([f,b]) for f,b in zip(f_lstms, b_lstms)]
            # run the layers of bilstms
            lstms = [ drop(h, self.dropout, train) for h in bilstm(features, self.lstms[0]) ]
            for lstm in self.lstms[1:]:
                lstms = [ drop(h, self.dropout, train) for h in bilstm(lstms, lstm) ]
        else:
            lstms = [ drop(self.lstms[0](x), self.dropout, train) for x in features ]
            for lstm in self.lstms[1:]:
                lstms = [ drop(lstm(h), self.dropout, train) for h in lstms ]

        f = F.leaky_relu
        # rnn output layer
        lstms = [ drop(f(self.out(h)) , self.dropout, train) for h in lstms ]

        # hidden layer
        if self.use_mlp:
            lstms = [ drop(f(self.mlp(h)) , self.dropout, train) for h in lstms ]

        # extract the information graph from the tagger
        (mentions, mention_masks,
        left_idxs, right_idxs,
         m_spans, r_spans, null_r_spans) = self._extract_graph(tagger_preds,lstms)
        # print [m.shape for m in mentions]

        # score mentions and take predictions for relations
        m_logits = [ self.f_m(m) for m in mentions ]

        m_preds = [ F.argmax(masked_softmax(m, mask), axis=1).data.astype('float32').reshape((-1,1))
                    for m, mask in zip(m_logits, mention_masks) ]
        # print 'm pad count', [ np.sum(m == 0.) for m in m_preds]

        # get features and type constraints for left and right mentions
        embed_id = F.embed_id
        left_mentions = [ F.squeeze(embed_id(idxs, ms))
                          for idxs, ms in zip(left_idxs, mentions) ]
        right_mentions = [ F.squeeze(embed_id(idxs, ms))
                          for idxs, ms in zip(right_idxs, mentions) ]

        mention_dists = [ F.reshape(F.squeeze(embed_id(r_idxs, mspan)
                          - embed_id(l_idxs, mspan), axis=1)[:,0], (-1,1))
                          for l_idxs, r_idxs, mspan in zip(left_idxs, right_idxs, m_spans)]
        # print embed_id(right_idxs[0], m_spans[0]).shape, mention_dists[0].shape, left_mentions[0].shape, right_mentions[0].shape
        left_masks = [ F.squeeze(embed_id(F.cast(embed_id(idxs, preds), 'int32'),
                                self.left_masks))
                       for idxs, preds in zip(left_idxs, m_preds) ]
        right_masks = [ F.squeeze(embed_id(F.cast(embed_id(idxs, preds), 'int32'),
                                 self.right_masks))
                       for idxs, preds in zip(right_idxs, m_preds) ]
        rel_masks = [ l_mask * r_mask for l_mask, r_mask in zip(left_masks, right_masks) ]

        # concat left and right mentions into one vector per relation
        relation_features = [ F.hstack(ms)
                              for ms in zip(left_mentions, right_mentions, mention_dists) ]

        # score relations
        # print [ (np.sum(mask.data[:,0] != 0.), len(mask.data)) for mask in rel_masks ]
        r_logits = [ self.f_r(r) for r in relation_features ]
        # print 'shapes',[(mask.shape, r.shape) for mask, r in zip(rel_masks, r_logits)]
        # print 'pad count', [(np.sum(mask.data[:,0] != 0), np.sum(r.data[:,0] != 0.))
        #                      for mask, r in zip(rel_masks, r_logits)]
        # r_logits = [ mask * self.f_r(r)
        #              for r, mask in zip(relation_features, rel_masks) ]
        # print 'r shapes', [(r.shape) for r in r_logits]

        # convert mention spans to lists
        m_spans = [tuple([tuple(mspan) for mspan in mspans.tolist() ]) for mspans in m_spans]
        return (tagger_features, tagger_preds,
                m_logits, r_logits,
                mention_masks, rel_masks,
                m_spans, r_spans, null_r_spans)

    def predict(self, x_list, reset_state=True):
        if reset_state:
            self.reset_state()

        (b_features, b_preds,
        m_logits, r_logits,
        m_masks, r_masks,
        m_spans, r_spans, null_r_spans) = self(x_list)

        m_preds = [ F.argmax(masked_softmax(m, mask), axis=1).data
                    for m, mask in zip(m_logits, m_masks) ]
        r_preds = [ F.argmax(masked_softmax(r, mask), axis=1).data
                    for r, mask in zip(r_logits, r_masks) ]
        # r_dists = [ F.softmax(r).data for r in r_logits]
        # predict null for all mention pairs > max rel dist from eachother
        # print 'masked pad count', [np.sum(r.data[:,0] != 0.) for r in r_logits]
        # print 'zero r count', [np.sum(np.all(r.data == 0., axis=1)) for r in r_logits]
        # pad_counts = [ (np.sum(r == 0), len(r)) for r in r_preds ]
        # for c, p, d in zip(pad_counts, r_preds, r_dists):
        #     if c:
        #         for r, l in zip(p.tolist(), d.tolist()):
        #             if r == 0:
        #                 print 'label dist', l
        r_preds = [ np.hstack([r, self.null_idx*np.ones(len(null_rs), dtype=np.int32)])
                    for r, null_rs in zip(r_preds, null_r_spans)]
        # print [ (np.sum(r == 0), len(r)) for r in r_preds ]
        # print 'rpred shapes', [r.shape for r in r_preds]
        r_spans = [ r+null_rs for r, null_rs in zip(r_spans, null_r_spans) ]
        return b_preds, m_preds, r_preds, m_spans, r_spans

    def report(self):
        summary = {}
        for link in self.children(): #links(skipself=True):
            if link is not self.tagger:
                for param in link.params():
                    d = '{}/{}/{}'.format(link.name, param.name, 'data')
                    summary[d] = param.data
                    g = '{}/{}/{}'.format(link.name, param.name, 'grad')
                    summary[g] = param.grad
                    # print d,g
            else:
                for sublink in link.children():
                    for param in sublink.params():
                        d = 'tagger/{}/{}/{}'.format(sublink.name, param.name, 'data')
                        summary[d] = param.data
                        g = 'tagger/{}/{}/{}'.format(sublink.name, param.name, 'grad')
                        summary[g] = param.grad
                        # print d,g
        return summary

class ExtractorLoss(ch.Chain):
    def __init__(self, extractor):
        super(ExtractorLoss, self).__init__(
            extractor=extractor,
            tagger_loss=TaggerLoss(extractor.tagger.copy())
        )

    def __call__(self, x_list, gold_b_list, gold_m_list, gold_r_list,
                 b_loss=True, m_loss=True, r_loss=True,
                 downsample=False, reweight=False, boundary_reweighting=False,
                 **kwds):
        assert not (downsample and reweight), "cannot downsample and reweight"
        # extract the graph
        (b_features, b_preds,
        men_logits, rel_logits,
        men_masks, rel_masks,
        men_spans, rel_spans, null_rspans) = self.extractor(x_list, **kwds)
        # print zip([len(m) for m in men_logits], [len(r) for r in rel_logits])
        # compute loss per sequence
        # print b_features
        if b_loss:
            boundary_loss = self.tagger_loss(x_list, gold_b_list,
                                             features=b_features)
        mention_loss = relation_loss = 0
        batch_size = float(len(men_logits))
        zipped = zip(men_logits, rel_logits,
                     men_spans, gold_m_list,
                     rel_spans, gold_r_list)
        for (m_logits, r_logits, m_spans, gold_m, r_spans, gold_r) in zipped:
            # using gold mentions, construct a matching label and truth array
            # that indicates if a mention boundary prediction is correct
            # and if so the index of the correct mention type (or 0 if not)
            # So the type loss is only calculated for correctly detected mentions.
            #
            # To prevent degenerate solutions that force the tagger to not output
            # as many correct mentions (resulting in trivially lower loss),
            # we rescale the loss by (# true mentions / # correct mentions).
            # Intuitively this creates higher losses for less correct mentions
            gold_spans = {m[:2] for m in gold_m}
            span2label = {m[:2]:m[2] for m in gold_m}
            weights = []
            labels = []
            for m in m_spans:
                if m in gold_spans:
                    weights.append(1.0)
                    labels.append(span2label[m])
                else:
                    weights.append(0.0)
                    labels.append(0)
            weights = np.array(weights, dtype=np.float32)
            # print '-'*80
            # print "{} true, {} pred, {} correct mention spans".format(
            #   len(gold_spans), len(m_spans), np.sum(weights))
            # print len(gold_m), gold_m
            # print len(m_spans), m_spans
            labels = np.array(labels, dtype=np.int32)
            # print type(m_logits), len(m_logits)
            doc_mention_loss = batch_weighted_softmax_cross_entropy(m_logits, labels,
                                                                    instance_weight=weights)
            if boundary_reweighting:
                doc_mention_loss *= len(weights) / (np.sum(weights) + 1e-15)
            mention_loss += doc_mention_loss

            # do the same for relations
            # but only if BOTH mention boundaries are correct
            # gold_rel_spans = set([r[:4] for r in gold_r])

            rel2label = {r[:4]:r[4] for r in gold_r}
            weights = []
            labels = []
            for r in r_spans:
                if (r[:2] in gold_spans) and (r[2:4] in gold_spans):
                    weights.append(1.0)
                    labels.append(rel2label[r[:4]])
                else:
                    weights.append(0.0)
                    labels.append(0)
            weights = np.array(weights, dtype=np.float32)
            # print "{} true, {} pred, {} correct relation spans".format(
            #   len(gold_r), len(r_spans), np.sum(weights))
            labels = np.array(labels, dtype=np.int32)
            # we downsample coref and null entries to the average number of examples
            # for all of the other correct labels
            # that way learning isn't overtaken
            # by 90% NULL and 5% coref and 5% everything else
            if downsample:
                i_n, i_c = self.extractor.null_idx, self.extractor.coref_idx
                # get max count of non coref/null examples
                good_labels = labels[weights==1.]
                # print "Label counts before down sample:"
                # print '\t', np.vstack(np.unique(good_labels, return_counts=True))
                unique, counts = np.unique(good_labels[np.all([good_labels!=i_n,
                                                               good_labels!=i_c], axis=0)],
                                           return_counts=True)

                max_count = np.max(counts) if counts.size > 0 else 1
                # now get all possible down-sample-able indices of NULLs
                # and set all but max_count of them to have 0 weight
                possible_labels = np.argwhere(np.all([weights==1.,labels==i_n],axis=0)).reshape(-1)
                if possible_labels.size > 0:
                    down_idxs = npr.choice(possible_labels, size=len(possible_labels)-max_count, replace=False)
                    weights[down_idxs] = 0.
                # do the same for coref
                possible_labels = np.argwhere(np.all([weights==1.,labels==i_c],axis=0)).reshape(-1)
                if possible_labels.size > 0:
                    down_idxs = npr.choice(possible_labels, size=len(possible_labels)-max_count, replace=False)
                    weights[down_idxs] = 0.
                doc_relation_loss = batch_weighted_softmax_cross_entropy(r_logits, labels,
                                                                     instance_weight=weights)
            elif reweight:
                unique, counts = np.unique(labels[weights==1.], return_counts=True)
                # print np.vstack([unique, counts])
                counts = np.sum(counts)/counts
                # print np.vstack([unique, counts])
                class_weights = np.ones(self.extractor.n_relation_class, dtype=np.float32)
                class_weights[unique] = counts
                # print class_weights
                # class_weights =
                # good_labels = labels[weights==1.]
                # print "Label counts after down sample:"
                # print '\t', np.vstack(np.unique(good_labels, return_counts=True))
                # print np.sum(weights), len(weights)
                doc_relation_loss = batch_weighted_softmax_cross_entropy(r_logits, labels,
                                                                     class_weight=class_weights,
                                                                     instance_weight=weights)
            else:
                doc_relation_loss = batch_weighted_softmax_cross_entropy(r_logits, labels,
                                                                     instance_weight=weights)

            if boundary_reweighting:
                doc_relation_loss *= len(weights) / (np.sum(weights) + 1e-15)
            relation_loss += doc_relation_loss
        mention_loss /= batch_size
        relation_loss /= batch_size
        # print "Extract Loss: B:{0:2.4f}, M:{1:2.4f}, R:{2:2.4f}".format(
        #     np.asscalar(boundary_loss.data),
        #     np.asscalar(mention_loss.data),
        #     np.asscalar(relation_loss.data))

        loss = 0
        if b_loss:
            loss += boundary_loss
        if m_loss:
            loss += mention_loss
        if r_loss:
            loss += relation_loss
        return loss

    def report(self):
        return self.extractor.report()
