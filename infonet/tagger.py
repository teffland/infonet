import chainer as ch
from util import SequenceIterator, sequences2arrays
class Tagger(ch.Chain):
    def __init__(self, embed, lstm_size, out_size, use_crf=False):
        super(Tagger, self).__init__(
            embed = embed,
            lstm = ch.links.LSTM(embed.W.shape[1], lstm_size),
            out = ch.links.Linear(lstm_size, out_size),
            crf = ch.links.CRF1d(out_size)
        )
        self.use_crf = use_crf

    def reset_state(self):
        self.lstm.reset_state()

    def __call__(self, x_list):
        embeds = [ self.embed(x) for x in x_list ]
        lstms = [ self.lstm(x) for x in embeds ]
        outs = [ self.out(h) for h in lstms ]
        return outs, lstms

    def predict(self, x_list, y_list, return_proba=False):
        # make a single batch out of the data
        x_iter = SequenceIterator(zip(x_list, y_list), len(x_list))
        x_list, y_list = zip(*x_iter.next())

        # run the model
        self.reset_state()
        logits_list, _ = self(sequences2arrays(x_list))
        # logits_list = [ logits.data for logits in logits_list ]

        if return_proba:
            if self.use_crf:
                raise NotImplementedError, "CRF sum-product decoder not implemented..."
            else:
                probs = [ ch.functions.softmax(logit) for logit in logits_list ]
                probs = [ prob.data for prob in ch.functions.transpose_sequence(probs) ]
                return probs, x_list, y_list
        else:
            if self.use_crf:
                _, preds = self.crf.argmax(logits_list)
                preds = [ pred.data for pred in ch.functions.transpose_sequence(preds) ]
                return preds, x_list, y_list
            else:
                preds = [ ch.functions.argmax(logit, axis=1) for logit in logits_list ]
                preds = [ pred.data for pred in ch.functions.transpose_sequence(preds) ]
                return preds, x_list, y_list

class TaggerLoss(ch.Chain):
    def __init__(self, tagger,
                 loss_func=ch.functions.softmax_cross_entropy):
        super(TaggerLoss, self).__init__(
            tagger = tagger
        )
        self.loss_func = loss_func

    def __call__(self, x_list, y_list):
        if self.tagger.use_crf:
            yhat_list, _ = self.tagger(x_list)
            return self.tagger.crf(yhat_list, y_list)

        elif self.loss_func == ch.functions.softmax_cross_entropy:
            loss = 0
            yhat_list,_ = self.tagger(x_list)
            for yhat, y in zip(yhat_list, y_list):
                loss += self.loss_func(yhat, y)
            return loss

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
                     in_tags=('B', 'I', 'L', 'U'),
                     out_tags=('O'),
                     typemap=None):
    """ We extract mentions approximately according to the BIO or BILOU schemes
    with some relaxations.

    We start mentions when we see anything but an 'O'.
    We end them when we see an 'O'.

    When computing the type of the mention
    we simply take the mode of the types of it's constituent tokens.
    """
    if type(seq[0]) == ch.Variable:
        seq = [ s.data for s in seq ]

    mentions = []
    in_mention = False
    mention_start = mention_end = 0
    for i, s in enumerate(seq):
        if not in_mention and s in in_tags:
            mention_start = i
            in_mention = True
        elif in_mention and s in out_tags:
            if typemap:
                mention_type = mode([ typemap[s] for s in seq[mention_start:i] ])
            else:
                mention_type = None
            mentions.append((mention_start, i, mention_type))
            in_mention=False
    if in_mention: # we end on a mention
        if typemap:
            mention_type = mode([ typemap[s] for s in seq[mention_start:i] ])
        else:
            mention_type = None
        mentions.append((mention_start, i+1, mention_type))
    return mentions

def extract_all_mentions(seqs, **kwds):
    return [extract_mentions(seq, **kwds) for seq in seqs]
