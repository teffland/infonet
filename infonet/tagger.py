import json
from io import open

import numpy as np
import chainer as ch
import chainer.functions as F
import chainer.links as L

from util import SequenceIterator, sequences2arrays, convert_sequences, mode
from crf_linear import LinearChainCRF
from gru import StackedGRU
from report import ReporterMixin
from evaluation import mention_boundary_stats

class Tagger(ch.Chain, ReporterMixin):
    def __init__(self,
                 word_embeddings,
                 n_label,
                 backprop_to_embeds=False,
                 word_dropout=.5,
                 pos_vocab_size=0,
                 pos_vector_size=25,
                 pos_dropout=.5,
                 gru_state_sizes=[100],
                 bidirectional=False,
                 gru_dropouts=[.5],
                 gru_hdropouts=[.0],
                 mlp_sizes=[],
                 mlp_activations=[],
                 mlp_dropouts=[],
                 crf_type=None,
                 start_tags=[],
                 in_tags=[],
                 out_tags=[],
                 tag2mtype=[],
                 **kwds):
        ch.Chain.__init__(self)
        # embeddings
        self.word_size = word_embeddings.shape[1]
        word_embed = L.EmbedID(word_embeddings.shape[0], word_embeddings.shape[1],
                          word_embeddings)
        self.add_link('word_embed', word_embed)
        self.word_dropout = word_dropout
        self.backprop_to_embeds = backprop_to_embeds
        if pos_vocab_size:
            self.use_pos = True
            self.pos_size = pos_vector_size
            pos_embed = L.EmbedID(pos_vocab_size, pos_vector_size)
            self.add_link('pos_embed', pos_embed)
            self.pos_dropout = pos_dropout
        else:
            self.use_pos = False

        # gru layers
        in_size = self.word_size + self.pos_size if self.use_pos else self.word_size
        gru = StackedGRU(in_size, gru_state_sizes,
                         dropouts=gru_dropouts,
                         hdropouts=gru_hdropouts)
        self.add_link('gru', gru)
        out_size = gru_state_sizes[-1]

        # mlp layers
        self.mlp_dropouts = mlp_dropouts
        self.activations = [ getattr(F, f) for f in mlp_activations ]
        self.mlps = []
        for i, hidden_dim in enumerate(mlp_sizes):
            mlp = L.Linear(out_size, hidden_dim)
            self.mlps.append(mlp)
            self.add_link('mlp_'+i, mlp)
            out_size = hidden_dim

        # crf layer
        self.crf_type = crf_type if crf_type.lower() != 'none' else None
        if self.crf_type:
            crf = LinearChainCRF(out_size, n_label, self.crf_type)
            self.add_link('crf', crf)
        else:
            logit = L.Linear(out_size, n_label)
            self.add_link('logit', logit)

        # for decoding sequences to mention boundaries
        self.start_tags = start_tags
        self.in_tags = in_tags
        self.out_tags = out_tags
        self.tag2mtype = tag2mtype

        ReporterMixin.__init__(self)

    def reset_state(self):
        self.gru.reset_state()

    def rescale_Us(self):
        self.gru.rescale_Us

    def __call__(self, x_list, p_list, *_, **kwds):
        train = kwds.pop('train', True)
        # convert lists of sequences to lists of timesteps
        x_list = sequences2arrays(x_list)
        p_list = sequences2arrays(p_list)

        # embed the tokens and pos
        self.embeds = [ F.dropout(self.word_embed(x), self.word_dropout, train)
                        for x in x_list ]
        if self.backprop_to_embeds:
            for embed in self.embeds:
                embed.unchain_backward()
        if self.use_pos:
            pos_embeds = [ F.dropout(self.pos_embed(p), self.pos_dropout, train)
                           for p in p_list ]
            self.embeds = [ F.hstack([x,p]) for x,p in zip(self.embeds, pos_embeds)]

        # run gru over them
        features = self.gru(self.embeds, train=train)

        # run mlp over features
        for dropout, activation, mlp in zip(self.mlp_dropouts, self.activations, self.mlps):
            for i in range(len(features)):
                features[i] = F.dropout(activation(mlp(features[i])), dropout, train)

        return features

    def predict(self, x_list, p_list, *_, **kwds):
        reset = kwds.pop('reset', True)
        train = kwds.pop('train', False)
        unfold_preds = kwds.pop('unfold_preds', True)

        if reset:
            self.reset_state()

        self.features = self(x_list, p_list, train=train)
        if self.crf_type:
            _, preds = self.crf.argmax(self.features)
        else:
            preds = [ ch.functions.argmax(self.logit(feature), axis=1)
                      for feature in self.features ]
        if unfold_preds:
            return ch.functions.transpose_sequence(preds)
        return preds

    def extract_mentions(self, seq,
                         start_tags=None,
                         in_tags=None,
                         out_tags=None,
                         tag2mtype=None):
        """ We extract mentions according to the BIO or BILOU schemes

        We scan across the sequence, encountering 3 cases:
        1. We are not in a mention, but encounter an in_tag, and start a mention
            eg, ... O <B|I|L|U>
        2. We are in a mention, but encounter an out_tag, and end the current mention
            eg, ... <B|I|L|U> O
        3. We are in a mention, but encounter a start tag,
           and end the current mention and start a new mention
            eg, ... <B|I|L|U> <B|U>


        When computing the type of the mention
        we simply take the mode of the types of it's constituent tokens.
        """
        start_tags = start_tags if start_tags else self.start_tags
        in_tags = in_tags if in_tags else self.in_tags
        out_tags = out_tags if out_tags else self.out_tags
        tag2mtype = tag2mtype if tag2mtype else self.tag2mtype

        if type(seq[0]) == ch.Variable:
            seq = [ np.asscalar(s.data) for s in seq ]
        mentions = []
        in_mention = False
        mention_start = mention_end = 0
        for i, s in enumerate(seq):
            if not in_mention and s in in_tags: # case 1
                mention_start = i
                in_mention = True
            elif in_mention and s in out_tags: # case 2
                if tag2mtype:
                    mention_type = mode([ tag2mtype[s] for s in seq[mention_start:i] ])
                else:
                    mention_type = None
                mentions.append((mention_start, i, mention_type))
                in_mention=False
            elif in_mention and s in start_tags: # case 3
                if tag2mtype:
                    mention_type = mode([ tag2mtype[s] for s in seq[mention_start:i] ])
                else:
                    mention_type = None
                mentions.append((mention_start, i, mention_type))
                mention_start = i

        if in_mention: # we end on a mention
            if tag2mtype:
                mention_type = mode([ tag2mtype[s] for s in seq[mention_start:i+1] ])
            else:
                mention_type = None
            mentions.append((mention_start, i+1, mention_type))
        return mentions

    def extract_all_mentions(self, seqs, **kwds):
        return [ self.extract_mentions(seq, **kwds) for seq in seqs ]

class TaggerLoss(ch.Chain):
    def __init__(self, tagger):
        super(TaggerLoss, self).__init__(tagger=tagger)

    def __call__(self, x_list, p_list, b_list, *_, **kwds):
        features = kwds.pop('features', None)

        # convert lists of sequences to lists of timesteps
        b_list = sequences2arrays(b_list)
        # inputing features skips evaluation of the network
        if self.tagger.crf_type:
            if features is None:
                features = self.tagger(x_list, p_list, train=True)
            return self.tagger.crf(features, b_list)

        else:
            loss = 0
            if features is None:
                features = self.tagger(x_list, p_list, train=True)
            for feature, b in zip(features, b_list):
                loss += F.softmax_cross_entropy(self.tagger.logit(feature), b)
            return loss / float(len(b_list))

    def reset_state(self):
        self.tagger.reset_state()

    def report(self):
        return self.tagger.report()

    def save_model(self, save_prefix):
        ch.serializers.save_npz(save_prefix+'tagger.model', self.tagger)

class TaggerEvaluator():
    def __init__(self,
                 tagger,
                 token_vocab,
                 pos_vocab,
                 boundary_vocab,
                 mention_vocab,
                 tag_map):
        self.tagger = tagger
        self.token_vocab = token_vocab
        self.pos_vocab = pos_vocab
        self.boundary_vocab = boundary_vocab
        self.mention_vocab = mention_vocab
        self.tag_map = tag_map

    def evaluate(self, batch_iter, save_prefix=None):
        all_bpreds, all_bs = [], []
        all_xs, all_truexs = [], []
        all_ps = []
        all_fs = []
        all_ms = []
        assert batch_iter.is_new_epoch
        for batch in batch_iter:
            ix, ip, ib, f, truex, m = zip(*batch)
            ib_preds = [ pred.data for pred in self.tagger.predict(ix, ip) ]

            all_bpreds.extend(convert_sequences(ib_preds, self.boundary_vocab.token))
            all_bs.extend(convert_sequences(ib, self.boundary_vocab.token))
            all_xs.extend(convert_sequences(ix, self.token_vocab.token))
            all_truexs.extend(truex) # never passed through vocab. no UNKS
            all_ps.extend(convert_sequences(ip, self.pos_vocab.token))
            all_fs.extend(f)
            all_ms.extend(m)
            if batch_iter.is_new_epoch:
                break

        if save_prefix:
            print "Saving predictions to {} ...".format(save_prefix),
            # extract the true and predicted mention boundaries
            convert_mention = lambda x: x[:-1]+(self.mention_vocab.token(x[-1]),)
            all_ms = convert_sequences(all_ms, convert_mention)
            ib_preds = convert_sequences(all_bpreds, self.boundary_vocab.idx)
            all_mpreds = self.tagger.extract_all_mentions(ib_preds)
            all_mpreds = convert_sequences(all_mpreds, convert_mention)

            # save each true and predicted doc to separate yaat file
            trues = zip(all_fs, all_truexs, all_ps, all_bs, all_ms)
            for f, xs, ps, bs, ms in trues:
                fname = save_prefix+f+'_true'
                self.save_doc(fname, xs, ps, bs, ms)

            preds = zip(all_fs, all_xs, all_ps, all_bpreds, all_mpreds)
            for f, xs, ps, bs, ms in preds:
                fname = save_prefix+f+'_pred'
                self.save_doc(fname, xs, ps, bs, ms)
            print "Done"

        stats = mention_boundary_stats(all_bs, all_bpreds, self.tagger, **self.tag_map)
        stats['score'] = stats['f1']
        return stats

    def save_doc(self, fname, xs, ps, bs, ms):
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
        for i, m in enumerate(ms):
            yaat_ms.append({'ann-type':'node',
                            'ann-uid':'m_'+str(i),
                            'ann-span':m[:2],
                            'node-type':m[2],
                            'type':m[2]})
        doc = {
            'tokens':xs,
            'annotations':yaat_ps+yaat_bs+yaat_ms
        }
        with open(fname+'.yaat', 'w', encoding='utf8') as f:
            f.write(unicode(json.dumps(doc, ensure_ascii=False, indent=2)))
