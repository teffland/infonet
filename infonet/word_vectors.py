""" Load in word vectors and return an embedding matrix using a vocab.

NOTE: make sure to call this after dropping infrequent words, if needed.
"""
import numpy as np
import numpy.random as npr

def get_pretrained_vectors(vocab, vec_fname, trim=True):
    """ Generate embedding matrix according to idx mapping from vocab.

    If trim is True, also write out the trimmed vector file,
    so we can load the vectorrs much faster next time.
    """
    # load in vectors in vocab
    token2vec = {}
    if trim:
        trim_fname = '/'.join(vec_fname.split('/')[:-1]+['trimmed_'+vec_fname.split('/')[-1]])
        trim_f = open(trim_fname, 'w')
    for line in open(vec_fname, 'r'):
        if line: # sometime they're empty
            split = line.split()
            token, vec = split[0], np.array([float(s) for s in split[1:]], dtype=np.float32)
            if token in vocab.vocabset:
                token2vec[token] = vec
                if trim:
                    trim_f.write(line)
    if trim:
        trim_f.close()

    # put them in a matrix
    d = token2vec.values()[0].shape[0]
    e = npr.normal(size=(vocab.v, d))
    for token, vec in token2vec.items():
        e[vocab.idx(token), :] = vec
    print "Pretrained coverage: {0}/{1} = {2:2.2f}%".format(
        len(token2vec), vocab.v, float(len(token2vec))/vocab.v * 100.)
    return e
