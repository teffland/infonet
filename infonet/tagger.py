import numpy as np
import chainer as ch
import chainer.functions as F
import chainer.links as L

from util import SequenceIterator, sequences2arrays
from crf_linear import LinearChainCRF
from gru import StackedGRU

class Tagger(ch.Chain):
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
                 **kwds):
                # lstm_size, out_size,
                #  pos_d, pos_v,
                #  bidirectional=False,
                #  use_mlp=False,
                #  dropout=.25,
                #  use_hdropout=False,
                #  n_layers=1,101693

                #  crf_type='none'):
        super(Tagger, self).__init__()

        # embeddings
        self.word_size = word_embeddings.shape[1]
        word_embed = L.EmbedID(word_embeddings.shape[0], word_embeddings.shape[1],
                          word_embeddings)
        self.add_link('word_embed', word_embed)
        self.word_dropout = word_dropout
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

        # # setup rnn layer
        # hdropout = dropout if use_hdropout else 0.0
        # if bidirectional:
        #     feature_size = 2*lstm_size
        #     lstms = [BidirectionalGRU(lstm_size, n_inputs=embeddings.shape[1]+pos_d,
        #                               dropout=hdropout)]
        #     for i in range(1,n_layers):
        #         lstms.append(BidirectionalGRU(lstm_size, n_inputs=feature_size, dropout=hdropout))
        # else:
        #     feature_size = lstm_size
        #     lstms = [GRU(lstm_size, n_inputs=embeddings.shape[1]+pos_d, dropout=hdropout)]
        #     for i in range(1,n_layers):
        #         lstms.append(GRU(lstm_size, n_inputs=feature_size, dropout=hdropout))

        # setup crf layer
        # if crf_type in 'none':
        #     self.crf_type = None
        #     crf_type = 'simple' # it won't actually be used
        # else:
        #     self.crf_type = crf_type
        #
        # super(Tagger, self).__init__(
        #     embed = L.EmbedID(embeddings.shape[0], embeddings.shape[1],
        #                       embeddings),
        #     pos_embed = L.EmbedID(pos_v, pos_d),
        #     mlp = L.Linear(feature_size, feature_size),
        #     out = L.Linear(feature_size, feature_size),
        #     logit = L.Linear(feature_size, out_size),
        #     crf = LinearChainCRF(out_size, feature_size, crf_type)
        # )
        # self.lstms = lstms
        # for i, lstm in enumerate(self.lstms):
        #     self.add_link('lstm_{}'.format(i), lstm)
        # self.embedding_size = embeddings.shape[1]
        # self.pos_d = pos_d
        # self.lstm_size = lstm_size
        # self.feature_size = feature_size
        # self.out_size = out_size
        # self.dropout = dropout
        # self.bidirectional = bidirectional
        # self.use_mlp = use_mlp
        # self.use_hdropout = use_hdropout

    def reset_state(self):
        self.gru.reset_state()

    def rescale_Us(self):
        self.gru.rescale_Us

    def __call__(self, x_list, p_list, train=True):#, return_logits=False):
        # embed the tokens and pos
        self.embeds = [ F.dropout(self.word_embed(x), self.word_dropout, train)
                   for x in x_list ]
        if self.use_pos:
            pos_embeds = [ F.dropout(self.pos_embed(p), self.pos_dropout, train)
                           for p in p_list ]
            self.embeds = [ F.hstack([x,p]) for x,p in zip(self.embeds, pos_embeds)]

        # run gru over them
        features = self.gru(self.embeds, train=train)

        # run mlp over features
        for dropout, activation, mlp in zip(self.mlp_dropouts, self.activations, self.mlps):
            for i in range(len(features)):
                features[i] = F.dropout(activation(mlp(features[i])), dropout, train=train)

        return features



        # drop = F.dropout
        # embeds = [ drop(self.embed(x), self.dropout, train) for x in x_list ]
        # if self.pos_d > 0:
        #     pos_embeds = [ drop(self.pos_embed(p), self.dropout, train) for p in p_list ]
        #     self.embeds = [F.hstack([x,p]) for x,p in zip(embeds, pos_embeds)]
        # else:
        #     self.embeds = embeds
        #
        # # run lstm layer over embeddings
        # if self.bidirectional:
        #     # helper function
        #     def bilstm(inputs, lstm):
        #         f_lstms, b_lstms = [], []
        #         for x_f, x_b in zip(inputs, inputs[::-1]):
        #             h_f, h_b = lstm(x_f, x_b, train=train)
        #             f_lstms.append(h_f)
        #             b_lstms.append(h_b)
        #         b_lstms = b_lstms[::-1]
        #         return [ F.hstack([f,b]) for f,b in zip(f_lstms, b_lstms) ]
        #
        #     # run the layers of bilstms
        #     # don't dropout h twice if using horizontal dropout
        #     if self.use_hdropout:
        #         lstms = [ h for h in bilstm(self.embeds, self.lstms[0]) ]
        #         for lstm in self.lstms[1:]:
        #             lstms = [ h for h in bilstm(lstms, lstm) ]
        #     else:
        #         lstms = [ drop(h, self.dropout, train) for h in bilstm(self.embeds, self.lstms[0]) ]
        #         for lstm in self.lstms[1:]:
        #             lstms = [ drop(h, self.dropout, train) for h in bilstm(lstms, lstm) ]
        # else:
        #     if self.use_hdropout:
        #         # lstms = []
        #         # for i, x in enumerate(self.embeds):
        #         #     print i
        #         #     lstms.append(drop(self.lstms[0](x), self.dropout, train))
        #         # print
        #         lstms = [ drop(self.lstms[0](x), self.dropout, train) for x in self.embeds ]
        #         for lstm in self.lstms[1:]:
        #             lstms = [ lstm(h, train=train) for h in lstms ]
        #     else:
        #         lstms = [ drop(self.lstms[0](x, train=train), self.dropout, train) for x in self.embeds ]
        #         for lstm in self.lstms[1:]:
        #             lstms = [ drop(lstm(h, train=train), self.dropout, train) for h in lstms ]
        #
        #
        # f = F.leaky_relu
        # # rnn output layer
        # lstms = [ drop(f(self.out(h)), self.dropout, train) for h in lstms]
        #
        # # hidden layer
        # if self.use_mlp:
        #     lstms = [ drop(f(self.mlp(h)) , self.dropout, train) for h in lstms ]
        #
        # if return_logits: # no crf layer, so do simple logit layer
        #     return [ self.logit(h) for h in lstms ]
        # else:
        #     return lstms

    def predict(self, x_list, p_list,
                reset=True, train=False, **kwds):
        if reset:
            self.reset_state()

        self.features = self(x_list, p_list, train=train)
        if self.crf_type:
            _, preds = self.crf.argmax(self.features)
        else:
            preds = [ ch.functions.argmax(self.logit(feature), axis=1)
                      for feature in self.features ]
        return preds

    def report(self):
        summary = {}
        for link in self.links(skipself=True):
            for param in link.params():
                d = '{}/{}/{}'.format(link.name, param.name, 'data')
                summary[d] = param.data
                g = '{}/{}/{}'.format(link.name, param.name, 'grad')
                summary[g] = param.grad
        return summary

class TaggerLoss(ch.Chain):
    def __init__(self, tagger):
        super(TaggerLoss, self).__init__(
            tagger=tagger
        )

    def __call__(self, x_list, p_list, b_list, features=None):
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

    def report(self):
        return self.tagger.report()

def mode(L):
    """ Compute the mode of a list """
    types = {}
    for e in L:
        if e in types:
            types[e] += 1
        else:
            types[e] = 1
    return sorted(types.items(), reverse=True, key=lambda x:x[1])[0][0]

def extract_mentions(seq,
                     start_tags,#=('B', 'U'),
                     in_tags,#=('B', 'I', 'L', 'U'),
                     out_tags,#=('O'),
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

def extract_all_mentions(seqs, **kwds):
    return [extract_mentions(seq, **kwds) for seq in seqs]
