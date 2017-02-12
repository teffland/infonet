import numpy as np
import numpy.random as npr
import chainer as ch
import chainer.functions as F
import chainer.links as L

from tagger import TaggerLoss
from special_functions import batch_weighted_softmax_cross_entropy
from gru import StackedGRU
from masked_softmax import masked_softmax
from simple_attention import SimpleAttention
from report import ReporterMixin
from evaluation import mention_boundary_stats, mention_relation_stats

class Extractor(ch.Chain, ReporterMixin):
    def __init__(self,
                 tagger,
                 build_on_tagger_features,
                 backprop_to_tagger,
                 n_mention_class, n_relation_class,
                 null_idx, coref_idx,
                 shared_options,
                 mention_options,
                 relation_options,
                 classification_options,
                 mtype2msubtype,
                 msubtype2rtype,
                 **kwds):
        ch.Chain.__init__(self)
        # tagger composition options
        self.add_link('tagger', tagger.copy())
        self.build_on_tagger_features = build_on_tagger_features
        self.backprop_to_tagger = backprop_to_tagger

        # shared representation layers
        self.shared_opt = opt = shared_options
        if self.build_on_tagger_features:
            self.shared_feature_size = self.tagger.feature_size
        else:
            self.shared_feature_size = self.tagger.embed_size
        ## gru
        if opt['gru_state_sizes']:
            gru = StackedGRU(self.shared_feature_size, opt['gru_state_sizes'],
                             dropouts=opt['gru_dropouts'],
                             hdropouts=opt['gru_hdropouts'])
            self.add_link('shared_gru', gru)
            self.shared_feature_size = opt['gru_state_sizes'][-1]
        ## mlp
        self.shared_mlp_dropouts = opt['mlp_dropouts']
        self.shared_activations = [ getattr(F, f) for f in opt['mlp_activations'] ]
        self.shared_mlps = []
        for i, hidden_dim in enumerate(opt['mlp_sizes']):
            mlp = L.Linear(self.shared_feature_size, hidden_dim)
            self.shared_mlps.append(mlp)
            self.add_link('shared_mlp_{}'.format(i), mlp)
            self.shared_feature_size = hidden_dim

        # mention representation layers
        self.mention_opt = opt = mention_options
        self.mention_feature_size = self.shared_feature_size
        ## gru
        if opt['gru_state_sizes']:
            gru = StackedGRU(self.mention_feature_size, opt['gru_state_sizes'],
                             dropouts=opt['gru_dropouts'],
                             hdropouts=opt['gru_hdropouts'])
            self.add_link('mention_gru', gru)
            self.mention_feature_size = opt['gru_state_sizes'][-1]
        ## mlp
        self.mention_mlp_dropouts = opt['mlp_dropouts']
        self.mention_activations = [ getattr(F, f) for f in opt['mlp_activations'] ]
        self.mention_mlps = []
        for i, hidden_dim in enumerate(opt['mlp_sizes']):
            mlp = L.Linear(self.mention_feature_size, hidden_dim)
            self.shared_mlps.append(mlp)
            self.add_link('mention_mlp_{}'.format(i), mlp)
            self.mention_feature_size = hidden_dim
        ## feature pooling
        self.mention_pooling = opt['pooling']
        self.mention_include_width = opt['include_width']
        if self.mention_pooling == 'attention':
            self.add_link('mention_attn',
                          SimpleAttention(self.mention_feature_size))

        # relation representation layers
        self.relation_opt = opt = relation_options
        self.relation_feature_size = self.shared_feature_size
        ## gru
        if opt['gru_state_sizes']:
            gru = StackedGRU(self.relation_feature_size, opt['gru_state_sizes'],
                             dropouts=opt['gru_dropouts'],
                             hdropouts=opt['gru_hdropouts'])
            self.add_link('relation_gru', gru)
            self.relation_feature_size = opt['gru_state_sizes'][-1]
        ## mlp
        self.relation_mlp_dropouts = opt['mlp_dropouts']
        self.relation_activations = [ getattr(F, f) for f in opt['mlp_activations'] ]
        self.relation_mlps = []
        for i, hidden_dim in enumerate(opt['mlp_sizes']):
            mlp = L.Linear(self.relation_feature_size, hidden_dim)
            self.shared_mlps.append(mlp)
            self.add_link('relation_mlp_{}'.format(i), mlp)
            self.relation_feature_size = hidden_dim
        ## feature pooling
        self.relation_pooling = opt['pooling']
        self.relation_outer_window = opt['outer_window_size']
        self.relation_include_width = opt['include_width']
        if self.relation_pooling == 'attention':
            self.add_link('relation_attn',
                          SimpleAttention(self.relation_feature_size))
        self.max_r_dist = opt['max_r_dist']

        # classification configuration
        ## how to do classifications
        self.prediction_method = classification_options['method']
        if self.prediction_method == 'staged':
            mention_logit = L.Linear(self.mention_feature_size, n_mention_class)
            self.add_link('mention_logit', mention_logit)
            relation_logit = L.Linear(self.relation_feature_size, n_relation_class)
            self.add_link('relation_logit', relation_logit)
        else:
            raise NotImplementedError, "Joint prediction not yet implemented"
        ## configure class compatibility masking
        do_mask = classification_options['constraint_mask']
        # convert the typemaps to indicator array masks
        # mention type -> subtype uses the string label, so keep it as a dict
        # for mtypes ('entity' and 'event-anchor') the indices
        # are kept as the raw tokens.
        for k,v in mtype2msubtype.items():
            if do_mask:
                mask = np.zeros(n_mention_class).astype(np.float32)
                mask[np.array(v)] = 1.
            else:
                mask = np.ones(n_mention_class).astype(np.float32)
            mtype2msubtype[k] = mask
        self.mtype2msubtype = mtype2msubtype
        # for mention subtype -> relation type
        # we will use the predictions for the mentions
        # which means its easiest to use the indices of the labels
        # so we instead create a a mask matrix and look them up with EmbedID
        if do_mask:
            left_masks = np.zeros((n_mention_class, n_relation_class)).astype(np.float32)
            right_masks = np.zeros((n_mention_class, n_relation_class)).astype(np.float32)
        else:
            left_masks = np.ones((n_mention_class, n_relation_class)).astype(np.float32)
            right_masks = np.ones((n_mention_class, n_relation_class)).astype(np.float32)
        for k,v in msubtype2rtype['left'].items():
            left_masks[k, np.array(v)] = 1.
        for k,v in msubtype2rtype['right'].items():
            right_masks[k, np.array(v)] = 1.
        self.left_masks = left_masks
        self.right_masks = right_masks

        ReporterMixin.__init__(self)

    def reset_state(self):
        self.tagger.reset_state()
        if hasattr(self, 'shared_gru'):
            self.shared_gru.reset_state()
        if hasattr(self, 'mention_gru'):
            self.mention_gru.reset_state()
        if hasattr(self, 'relation_gru'):
            self.relation_gru.reset_state()

    def rescale_Us(self):
        if self.backprop_to_tagger:
            self.tagger.rescale_Us()
        if hasattr(self, 'shared_gru'):
            self.shared_gru.rescale_Us()
        if hasattr(self, 'mention_gru'):
            self.mention_gru.rescale_Us()
        if hasattr(self, 'relation_gru'):
            self.relation_gru.rescale_Us()

    def _mention_feature_agg(self, features, span):
        if self.mention_pooling == 'sum':
            mention = F.sum(features[span[0]:span[1]], axis=0)
        elif self.mention_pooling == 'avg':
            mention = F.sum(features[span[0]:span[1]], axis=0)
            mention /= F.broadcast_to(ch.Variable(np.array(span[1]-span[0],
                                                         dtype=np.float32)),
                                             mention.shape)
        elif self.mention_pooling == 'max':
            mention = F.max(features[span[0]:span[1]], axis=0)
        elif self.mention_pooling == 'logsumexp':
            mention = F.logsumexp(features[span[0]:span[1]], axis=0)
        elif self.mention_pooling == 'attention':
            mention = self.mention_attn(features[span[0]:span[1]])
        else:
            raise ValueError, "Unknown mention pooling function"

        if self.mention_include_width:
            w = ch.Variable(np.array(span[1]-span[0]).astype(np.float32).reshape((1,)))
            mention = F.hstack([mention, w])
        return mention

    def _relation_feature_agg(self, features, span):
        left = max(span[0]-self.relation_outer_window, 0)
        right = min(span[1]+self.relation_outer_window, features.shape[0])
        if self.relation_pooling == 'sum':
            relation = F.sum(features[left:right], axis=0)
        elif self.relation_pooling == 'avg':
            relation = F.sum(features[left:right], axis=0)
            relation /= F.broadcast_to(ch.Variable(np.array(right-left,
                                                         dtype=np.float32)),
                                             relation.shape)
        elif self.relation_pooling == 'max':
            relation = F.max(features[left:right], axis=0)
        elif self.relation_pooling == 'logsumexp':
            relation = F.logsumexp(features[left:right], axis=0)
        elif self.relation_pooling == 'attention':
            relation = self.relation_attn(features[left:right])
        else:
            raise ValueError, "Unknown relation pooling function"
        if self.mention_include_width:
            w = ch.Variable(np.array(span[1]-span[0]).astype(np.float32).reshape((1,)))
            relation = F.hstack([relation, w])
        return relation

    def _extract_graph(self, sequence_tags, mention_features, relation_features):
        """ Subroutine responsible for extracting the graph and graph features
        from the tagger using `extract_all_mentions`.

        Note: This function can be slow for large documents.
          This is unavoidable for documents with large mention counts (m)
          because the number of relations r is naively (m choose 2).
          This graph can be pruned by setting `max_r_dist`,
          which will omit all relations for mentions `max_r_dist` apart.
        """
        # convert from time-major to batch-major
        # aka we switch from a per-timestep representation to a per-doc one
        # this is for two reasons: (1) ease of implementation
        # and (2) typically there are more mentions and relations in a doc
        # than there are docs in a batch, so the corresponding matrices are larger
        # and there is higher variance in these quantities than seq lengths in the batch
        # so this way is actually more efficient
        sequence_tags = F.transpose_sequence(sequence_tags)
        mention_features = F.transpose_sequence(mention_features)
        relation_features = F.transpose_sequence(relation_features)

        # extract the mentions and relations for each doc
        all_boundaries = self.tagger.extract_all_mentions(sequence_tags)

        all_mentions = [] # features for mentions
        all_mention_spans = [] # spans in doc for each mention
        all_mention_masks = [] # bool masks for constraining predictions
        all_relations = [] # features for relations
        all_relation_left_nbrs = [] # idxs for left mention of a relation in mention table
        all_relation_right_nbrs = [] # idxs for right mention of a relation in mention table
        all_mention_nbrs = [] # idxs for relations connected to a mention, and whether its on left or right
        all_relation_spans = [] # spans in doc for left and right mentions
        all_null_rel_spans = [] # predict null for mention pairs > max_rel_dist

        # extract graph and features for each doc
        zipped = zip(all_boundaries, mention_features, relation_features)
        for s, (boundaries, men_features, rel_features) in enumerate(zipped):
            mentions = []
            mention_spans = []
            mention_masks = []
            mention_nbrs = []
            relations = []
            relation_left_nbrs = []
            relation_right_nbrs = []
            relation_spans = []
            null_rel_spans = []

            # print '{} mentions'.format(len(boundaries))
            for i, b in enumerate(boundaries):
                mention = self._mention_feature_agg(men_features, b)
                mentions.append(mention)
                mention_spans.append((b[0], b[1]))
                mention_masks.append(self.mtype2msubtype[b[2]])
                # make a relation to all previous mentions (M choose 2)
                # except those that are further than max_r_dist away
                mention_nbrs.append([])
                for j in range(i):
                    bj = boundaries[j]
                    if bj[1] - b[0] < self.max_r_dist:
                        relation_spans.append((bj[0], bj[1], b[0], b[1]))
                        relation = self._relation_feature_agg(rel_features, (bj[0], b[1]))
                        relations.append(relation)
                        # for each relation, keep track of its neighboring mentions
                        relation_left_nbrs.append(j)
                        relation_right_nbrs.append(i)

                        # for each mention, keep track of its neighboring relations
                        # and also, whether the mention is the left or right constituent
                        r = len(relation_spans)
                        mention_nbrs[i].append((r,0)) # 0 means relation is on left
                        mention_nbrs[j].append((r,1)) # 1 means relation is on right
                    else:
                        null_rel_spans.append((bj[0], bj[1], b[0], b[1]))

            # rearrange mentions in order from most to least neighboring relations
            # needed for efficient inference in bipartite crf
            sort_idxs = [x[0] for x in sorted(zip(range(len(mention_nbrs)), mention_nbrs),
                                              key=lambda x:len(x[1]), reverse=True)]
            mentions = [ mentions[i] for i in sort_idxs]
            mention_nbrs = [ mention_nbrs[i] for i in sort_idxs ]
            mention_masks = [ mention_masks[i] for i in sort_idxs ]
            mention_spans = [ mention_spans[i] for i in sort_idxs ]
            relation_left_nbrs = [ np.array(sort_idxs[lm_i]).astype(np.int32)
                                  for lm_i in relation_left_nbrs ]
            relation_right_nbrs = [ np.array(sort_idxs[rm_i]).astype(np.int32)
                                  for rm_i in relation_right_nbrs ]

            # convert list of mentions to matrix and append
            mentions = F.vstack(mentions)
            mention_masks = F.vstack(mention_masks)
            all_mentions.append(mentions)
            all_mention_masks.append(mention_masks)
            all_mention_spans.append(mention_spans)
            all_mention_nbrs.append(mention_nbrs)

            # same for relations
            relations = F.vstack(relations)
            relation_left_nbrs = F.vstack(relation_left_nbrs)
            relation_right_nbrs = F.vstack(relation_right_nbrs)
            all_relations.append(relations)
            all_relation_left_nbrs.append(relation_left_nbrs)
            all_relation_right_nbrs.append(relation_right_nbrs)
            all_relation_spans.append(relation_spans)
            all_null_rel_spans.append(null_rel_spans)

        return (all_mentions, all_mention_masks, all_mention_nbrs,
                all_relations, all_relation_left_nbrs, all_relation_right_nbrs,
                all_mention_spans, all_relation_spans, all_null_rel_spans)

    def __call__(self, x_list, p_list, *_, **kwds):
        train = kwds.pop('train', True)
        gold_boundaries = kwds.pop('gold_boundaries', None)

        # get features and prediction from tagger, depending on configuration
        if not gold_boundaries:
            sequence_tags = self.tagger.predict(x_list, p_list,
                                                train=train,
                                                unfold_preds=False)
            if self.build_on_tagger_features:
                features = self.tagger.features
                if not self.backprop_to_tagger:
                    features = [ F.identity(features) for feature in features ]
                    for feature in features:
                        feature.unchain_backward()
            else:
                features = self.tagger.embeds
        else:
            sequence_tags = gold_boundaries
            if self.build_on_tagger_features:
                features = self.tagger(x_list, p_list, train=train)
                if not self.backprop_to_tagger:
                    features = [ F.identity(features) for feature in features ]
                    for feature in features:
                        feature.unchain_backward()
            else:
                features = self.tagger.embed(x_list, p_list, train=train)

        # run shared feature layers
        if hasattr(self, 'shared_gru'):
            features = self.shared_gru(features, train=train)
        zipped_mlp = zip(self.shared_mlp_dropouts, self.shared_activations, self.shared_mlps)
        for dropout, activation, mlp in zipped_mlp:
            for i in range(len(features)):
                features[i] = F.dropout(activation(mlp(features[i])), dropout, train)

        # run mention feature layers
        mention_features = features
        if hasattr(self, 'mention_gru'):
            mention_features = self.mention_gru(mention_features, train=train)
        zipped_mlp = zip(self.mention_mlp_dropouts, self.mention_activations, self.mention_mlps)
        for dropout, activation, mlp in zipped_mlp:
            for i in range(len(mention_features)):
                mention_features[i] = F.dropout(activation(mlp(mention_features[i])), dropout, train)

        # run relation feature layers
        relation_features = features
        if hasattr(self, 'relation_gru'):
            relation_features = self.relation_gru(relation_features, train=train)
        zipped_mlp = zip(self.relation_mlp_dropouts, self.relation_activations, self.relation_mlps)
        for dropout, activation, mlp in zipped_mlp:
            for i in range(len(relation_features)):
                relation_features[i] = F.dropout(activation(mlp(relation_features[i])), dropout, train)

        # extract the information graph and its features
        (mentions, mention_masks, mention_nbrs,
         relations, relation_left_nbrs, relation_right_nbrs,
         m_spans, r_spans, null_r_spans) = self._extract_graph(sequence_tags,
                                                               mention_features,
                                                               relation_features)

        # now do classification scoring
        if self.prediction_method == 'staged':
            # score mentions and take predictions for relations
            m_logits = [ self.mention_logit(m) for m in mentions ]

            m_preds = [ F.argmax(masked_softmax(m, mask), axis=1).data.astype('float32').reshape((-1,1))
                        for m, mask in zip(m_logits, mention_masks) ]

            # get type constraints for left and right mentions
            left_masks = [ F.squeeze(F.embed_id(F.cast(F.embed_id(idxs, preds), 'int32'),
                                    self.left_masks))
                           for idxs, preds in zip(relation_left_nbrs, m_preds) ]
            right_masks = [ F.squeeze(F.embed_id(F.cast(F.embed_id(idxs, preds), 'int32'),
                                     self.right_masks))
                           for idxs, preds in zip(relation_right_nbrs, m_preds) ]
            relation_masks = [ l_mask * r_mask for l_mask, r_mask in zip(left_masks, right_masks) ]

            # score relations
            r_logits = [ self.relation_logit(r) for r in relations ]
        else:
            raise NotImplementedError, "Joint predictions not yet implemented"

        return (sequence_tags,
                m_logits, r_logits,
                mention_masks, relation_masks,
                m_spans, r_spans, null_r_spans)

    def predict(self, x_list, p_list, *_, **kwds):
        reset_state = kwds.pop('reset_state', True)
        if reset_state:
            self.reset_state()

        (b_preds,
         m_logits, r_logits,
         m_masks, r_masks,
         m_spans, r_spans, null_r_spans) = self(x_list, p_list)

        # unfold the tagger preds
        b_preds = [ pred.data for pred in F.transpose_sequence(b_preds) ]

        # get predictions
        m_preds = [ F.argmax(masked_softmax(m, mask), axis=1).data
                    for m, mask in zip(m_logits, m_masks) ]
        r_preds = [ F.argmax(masked_softmax(r, mask), axis=1).data
                    for r, mask in zip(r_logits, r_masks) ]
        # automatically predict null for pruned relations
        r_preds = [ np.hstack([r, self.null_idx*np.ones(len(null_rs), dtype=np.int32)])
                    for r, null_rs in zip(r_preds, null_r_spans)]
        r_spans = [ r+null_rs for r, null_rs in zip(r_spans, null_r_spans) ]

        # convert to back to gold formats
        m_preds = [ [ (s[0],s[1], p) for p,s in zip(preds, spans)]
                    for preds,spans in zip(m_preds, m_spans)]
        r_preds = [ [ (s[0],s[1],s[2],s[3], p) for p,s in zip(preds, spans)]
                    for preds,spans in zip(r_preds, r_spans)]
        return b_preds, m_preds, r_preds

class ExtractorLoss(ch.Chain):
    def __init__(self, extractor, use_gold_boundaries=False):
        super(ExtractorLoss, self).__init__(
            extractor=extractor,
            tagger_loss=TaggerLoss(extractor.tagger.copy())
        )
        self.use_gold_boundaries = use_gold_boundaries
    def __call__(self, x_list, p_list, gold_b_list, gold_m_list, gold_r_list, *_,
                 **kwds):
        b_loss = kwds.pop('b_loss', True)
        m_loss = kwds.pop('m_loss', True)
        r_loss = kwds.pop('r_loss', True)
        reweight_relations = kwds.pop('reweight_relations', False)
        boundary_reweighting = kwds.pop('boundary_reweighting', False)

        # extract the graph
        if self.use_gold_boundaries:
            stuff = self.extractor(x_list, p_list, gold_boundaries=gold_b_list, **kwds)
        else:
            stuff = self.extractor(x_list, p_list, **kwds)
        (b_preds,
        men_logits, rel_logits,
        men_masks, rel_masks,
        men_spans, rel_spans, null_rspans) = stuff

        # compute loss per sequence
        if b_loss:
            boundary_loss = self.tagger_loss(x_list, p_list, gold_b_list,
                                             features=self.extractor.tagger.features)

        if m_loss:
            mention_loss = 0
            batch_size = float(len(men_logits))
            zipped = zip(men_logits, men_spans, gold_m_list)
            for (m_logits, m_spans, gold_m) in zipped:
                # using gold mentions, construct a matching label and truth array
                # that indicates if a mention boundary prediction is correct
                # and if so the index of the correct mention type (or 0 if not)
                # So the type loss is only calculated for correctly detected mentions.
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
                labels = np.array(labels, dtype=np.int32)
                doc_mention_loss = batch_weighted_softmax_cross_entropy(m_logits, labels,
                                                                        instance_weight=weights)
                # To prevent degenerate solutions that force the tagger to not output
                # as many correct mentions (resulting in trivially lower loss),
                # we rescale the loss by (# true mentions / # correct mentions).
                # Intuitively this creates higher losses for less correct mentions
                if boundary_reweighting:
                    doc_mention_loss *= len(weights) / (np.sum(weights) + 1e-15)
                mention_loss += doc_mention_loss
            mention_loss /= batch_size

        if r_loss:
            relation_loss = 0
            batch_size = float(len(rel_logits))
            zipped = zip(rel_logits, rel_spans, gold_r_list)
            for (r_logits, r_spans, gold_r) in zipped:
                # do the same for relations
                # but only if BOTH mention boundaries are correct
                gold_rel_spans = set([r[:4] for r in gold_r])
                rel2label = {r[:4]:r[4] for r in gold_r}
                weights = []
                labels = []
                for r in r_spans:
                    if (r[:4] in gold_rel_spans):
                        weights.append(1.0)
                        labels.append(rel2label[r[:4]])
                    else:
                        weights.append(0.0)
                        labels.append(0)
                weights = np.array(weights, dtype=np.float32)
                labels = np.array(labels, dtype=np.int32)
                # the relation labels can be heavily biased.
                # so we reweight the class losses proportional
                # to the inverse frequency of the class label counts
                # rescaled to sum to the number of unique class labels
                # eg, where normal class weights would be [1,1,...,1] (sum=N)
                # they would look something like [.5, 1.5, ..., 1] (sum=N)
                # NOTE: unseen labels get a count of one, for smoothing
                if reweight_relations:
                    unique, counts = np.unique(labels[weights==1.], return_counts=True)
                    class_weights = np.ones(self.extractor.n_relation_class, dtype=np.float32)
                    class_weights[unique] = counts
                    total = counts.sum().astype(np.float32)
                    class_weights = self.extractor.n_relation_class/(total/counts)
                    assert class_weights.sum() == self.extractor.n_relation_class
                    doc_relation_loss = batch_weighted_softmax_cross_entropy(r_logits, labels,
                                                                         class_weight=class_weights,
                                                                         instance_weight=weights)
                else:
                    doc_relation_loss = batch_weighted_softmax_cross_entropy(r_logits, labels,
                                                                         instance_weight=weights)

                if boundary_reweighting:
                    doc_relation_loss *= len(weights) / (np.sum(weights) + 1e-15)
                relation_loss += doc_relation_loss
            relation_loss /= batch_size

        print "Extract Loss: B:{0:2.4f}, M:{1:2.4f}, R:{2:2.4f}".format(
            np.asscalar(boundary_loss.data),
            np.asscalar(mention_loss.data),
            np.asscalar(relation_loss.data))

        loss = 0
        if b_loss:
            loss += boundary_loss
        if m_loss:
            loss += mention_loss
        if r_loss:
            loss += relation_loss
        return loss

    def reset_state(self):
        self.extractor.reset_state()

    def report(self):
        return self.extractor.report()

    def save_model(self, save_prefix):
        ch.serializers.save_npz(save_prefix+'extractor.model', self.extractor)

class ExtractorEvaluator():
    def __init__(self,
                 extractor,
                 token_vocab,
                 pos_vocab,
                 boundary_vocab,
                 mention_vocab,
                 relation_vocab,
                 tag_map):
        self.extractor = extractor
        self.token_vocab = token_vocab
        self.pos_vocab = pos_vocab
        self.boundary_vocab = boundary_vocab
        self.mention_vocab = mention_vocab
        self.relation_vocab = relation_vocab
        self.tag_map = tag_map

    def evaluate(self, batch_iter, save_prefix=None):
        all_xs, all_truexs = [], []
        all_bpreds, all_bs = [], []
        all_mpreds, all_ms = []
        all_rpreds, all_rs = []
        all_ps = []
        all_fs = []

        assert batch_iter.is_new_epoch
        for batch in batch_iter:
            ix, ip, ib, im, ir, f, truex = zip(*batch)
            ib_preds, im_preds, ir_preds = self.extractor.predict(ix, ip)

            all_bpreds.extend(convert_sequences(ib_preds, self.boundary_vocab.token))
            all_bs.extend(convert_sequences(ib, self.boundary_vocab.token))
            all_xs.extend(convert_sequences(ix, self.token_vocab.token))
            all_truexs.extend(truex) # never passed through vocab. no UNKS
            all_ps.extend(convert_sequences(ip, self.pos_vocab.token))
            all_fs.extend(f)
            convert_mention = lambda x: x[:-1]+(self.mention_vocab.token(x[-1]),) # type is las
            all_mpreds.extend(convert_sequences(im_preds, convert_mention))
            all_ms.extend(convert_sequences(im, convert_mention))
            convert_relation = lambda x: x[:-1]+(self.relation_vocab.token(x[-1]),) # type is last
            all_rpreds.extend(convert_sequences(ir_preds, convert_relation))
            all_rs.extend(convert_sequences(ir, convert_relation))
            if batch_iter.is_new_epoch:
                break

        if save_prefix:
            print "Saving predictions to {} ...".format(save_prefix),
            # save each true and predicted doc to separate yaat file
            trues = zip(all_fs, all_truexs, all_ps, all_bs, all_ms, all_rs)
            for f, xs, ps, bs, ms, rs in trues:
                fname = save_prefix+f+'_true'
                self.save_doc(fname, xs, ps, bs, ms, rs)

            preds = zip(all_fs, all_xs, all_ps, all_bpreds, all_mpreds, all_rpreds)
            for f, xs, ps, bs, ms, rs in preds:
                fname = save_prefix+f+'_pred'
                self.save_doc(fname, xs, ps, bs, ms, rs)
            print "Done"

        stats = mention_relation_stats(all_ms, all_mpreds, all_res, all_rpreds)
        stats.update({'boundary-'+k:v for k,v in
                         mention_boundary_stats(all_bs, all_bpreds,
                                                self.extractor.tagger, **self.tag_map).items()})
        stats['score'] = stats['f1']
        return stats

    def save_doc(self, fname, xs, ps, bs, ms, rs):
        """ Save predictions to a yaat file """
        yaat_ps = []
        for i, p in enumerate(ps):
            yaat_ps.append({'ann-type':'node',
                            'ann-uid':'p_'+str(i),
                            'ann-span':(i, i+1),
                            'node-type':'pos',
                            'type':p})
        yaat_bs = []
        for i, b in enumerate(bs):
            yaat_bs.append({'ann-type':'node',
                            'ann-uid':'b_'+str(i),
                            'ann-span':(i, i+1),
                            'node-type':'boundary',
                            'type':b})
        yaat_ms = []
        mspan2id = {}
        for i, m in enumerate(ms):
            muid = 'm_'+str(i)
            mspan2id[m[:2]] = muid
            yaat_ms.append({'ann-type':'node',
                            'ann-uid':muid,
                            'ann-span':m[:2],
                            'node-type':m[2].split(':')[0],
                            'type':':'.join(m[2].split(':')[1:])})

        yaat_rs = []
        for i, r in enumerate(rs):
            yaat_rs.append({'ann-type':'edge',
                            'ann-uid':'r_'+str(i),
                            'ann-left':mspan2id[r[:2]],
                            'ann-right':mspan2id[r[2:4]],
                            'edge-type':r[4].split(':')[0],
                            'type':':'.join(r[4].split(':')[1:])})
        doc = {
            'tokens':xs,
            'annotations':yaat_ps+yaat_bs+yaat_ms+yaat_rs
        }
        with open(fname+'.yaat', 'w', encoding='utf8') as f:
            f.write(unicode(json.dumps(doc, ensure_ascii=False, indent=2)))
