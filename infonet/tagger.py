import numpy as np
import chainer as ch
import chainer.functions as F
import chainer.links as L
from util import SequenceIterator, sequences2arrays
from crf_linear import LinearChainCRF
from gru import GRU, BidirectionalGRU

# import monitor

class Tagger(ch.Chain):
    def __init__(self, embeddings, lstm_size, out_size,
                 bidirectional=False,
                 use_mlp=False,
                 dropout=.25,
                 use_hdropout=False,
                 n_layers=1,
                 crf_type='none'):
        # setup rnn layer
        hdropout = dropout if use_hdropout else 0.0
        if bidirectional:
            feature_size = 2*lstm_size
            lstms = [BidirectionalGRU(lstm_size, n_inputs=embeddings.shape[1],
                                      dropout=hdropout)]
            for i in range(1,n_layers):
                lstms.append(BidirectionalGRU(lstm_size, n_inputs=feature_size, dropout=hdropout))
        else:
            feature_size = lstm_size
            lstms = [GRU(lstm_size, n_inputs=embeddings.shape[1], dropout=hdropout)]
            for i in range(1,n_layers):
                lstms.append(GRU(lstm_size, n_inputs=feature_size, dropout=hdropout))

        # setup crf layer
        if crf_type in 'none':
            self.crf_type = None
            crf_type = 'simple' # it won't actually be used
        else:
            self.crf_type = crf_type

        super(Tagger, self).__init__(
            embed = L.EmbedID(embeddings.shape[0], embeddings.shape[1],
                                     embeddings),
            mlp = L.Linear(feature_size, feature_size),
            out = L.Linear(feature_size, feature_size),
            logit = L.Linear(feature_size, out_size),
            crf = LinearChainCRF(out_size, feature_size, crf_type)
        )
        self.lstms = lstms
        for i, lstm in enumerate(self.lstms):
            self.add_link('lstm_{}'.format(i), lstm)
        self.embedding_size = embeddings.shape[1]
        self.lstm_size = lstm_size
        self.feature_size = feature_size
        self.out_size = out_size
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.use_mlp = use_mlp
        self.use_hdropout = use_hdropout

    def reset_state(self):
        for lstm in self.lstms:
            lstm.reset_state()

    def __call__(self, x_list, train=True, return_logits=False):
        drop = F.dropout
        self.embeds = [ drop(self.embed(x), self.dropout, train) for x in x_list ]

        # run lstm layer over embeddings
        if self.bidirectional:
            # helper function
            def bilstm(inputs, lstm):
                f_lstms, b_lstms = [], []
                for x_f, x_b in zip(inputs, inputs[::-1]):
                    h_f, h_b = lstm(x_f, x_b, train=train)
                    f_lstms.append(h_f)
                    b_lstms.append(h_b)
                b_lstms = b_lstms[::-1]
                return [ F.hstack([f,b]) for f,b in zip(f_lstms, b_lstms) ]

            # run the layers of bilstms
            # don't dropout h twice if using horizontal dropout
            if self.use_hdropout:
                lstms = [ h for h in bilstm(self.embeds, self.lstms[0]) ]
                for lstm in self.lstms[1:]:
                    lstms = [ h for h in bilstm(lstms, lstm) ]
            else:
                lstms = [ drop(h, self.dropout, train) for h in bilstm(self.embeds, self.lstms[0]) ]
                for lstm in self.lstms[1:]:
                    lstms = [ drop(h, self.dropout, train) for h in bilstm(lstms, lstm) ]
        else:
            if self.use_hdropout:
                # lstms = []
                # for i, x in enumerate(self.embeds):
                #     print i
                #     lstms.append(drop(self.lstms[0](x), self.dropout, train))
                # print
                lstms = [ drop(self.lstms[0](x), self.dropout, train) for x in self.embeds ]
                for lstm in self.lstms[1:]:
                    lstms = [ lstm(h, train=train) for h in lstms ]
            else:
                lstms = [ drop(self.lstms[0](x, train=train), self.dropout, train) for x in self.embeds ]
                for lstm in self.lstms[1:]:
                    lstms = [ drop(lstm(h, train=train), self.dropout, train) for h in lstms ]


        f = F.leaky_relu
        # rnn output layer
        lstms = [ drop(f(self.out(h)), self.dropout, train) for h in lstms]

        # hidden layer
        if self.use_mlp:
            lstms = [ drop(f(self.mlp(h)) , self.dropout, train) for h in lstms ]

        if return_logits: # no crf layer, so do simple logit layer
            return [ self.logit(h) for h in lstms ]
        else:
            return lstms

    def predict(self, x_list, reset=True, return_features=False, train=False, **kwds):
        if reset:
            self.reset_state()
        if self.crf_type:
            features = self(x_list, train=train, return_logits=False)
            _, preds = self.crf.argmax(features)
        else:
            features = self(x_list, train=train, return_logits=True)
            preds = [ ch.functions.argmax(logit, axis=1) for logit in features ]
        if return_features:
            return preds, features
        else:
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
    def __init__(self, tagger,
                 loss_func=ch.functions.softmax_cross_entropy):
        super(TaggerLoss, self).__init__(
            tagger = tagger
        )
        self.loss_func = loss_func

    def __call__(self, x_list, b_list, features=None):
        # inputting features skips evaluation of the network
        if self.tagger.crf_type:
            if features is None:
                features = self.tagger(x_list)
            return self.tagger.crf(features, b_list)

        elif self.loss_func == ch.functions.softmax_cross_entropy:
            loss = 0
            if features is None:
                features = self.tagger(x_list, return_logits=True)
            for logit, b in zip(features, b_list):
                loss += self.loss_func(logit, b)
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
    """ We extract mentions approximately according to the BIO or BILOU schemes

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
