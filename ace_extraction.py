import json
import io
import time
import argparse
import numpy as np
import numpy.random as npr
from scipy.misc import comb # to check for num relations
import chainer as ch

from infonet.vocab import Vocab
from infonet.preprocess import (Entity_BIO_map, Entity_BILOU_map,
                                BIO_map, BILOU_map,
                                typed_BIO_map, typed_BILOU_map,
                                compute_flat_mention_labels,
                                compute_mentions, compute_relations,
                                resolve_annotations)
from infonet.util import (convert_sequences, sequences2arrays,
                          SequenceIterator, print_epoch_loss, sec2hms)
from infonet.tagger import Tagger, TaggerLoss
from infonet.extractor import Extractor, ExtractorLoss
from infonet.evaluation import plot_learning_curve, mention_boundary_stats

def get_ace_extraction_data(fname, count, valid, test, **kwds):
    print "Loading data..."
    # load data
    data = json.loads(io.open(fname, 'r').read())

    # get vocabs
    token_vocab = Vocab(min_count=count)
    for doc in data.values():
        token_vocab.add(doc['tokens'])

    boundary_vocab = Vocab(min_count=0)
    mention_vocab = Vocab(min_count=0)
    relation_vocab = Vocab(min_count=0)
    for doc in data.values():
        doc['annotations'] = resolve_annotations(doc['annotations'])

        # boundary labels
        doc['boundary_labels'] = compute_flat_mention_labels(doc, typed_BIO_map)
        boundary_vocab.add(doc['boundary_labels'])

        # mention labels
        doc['mentions'] = compute_mentions(doc)
        mention_vocab.add([m[2] for m in doc['mentions']])

        # relation labels
        relations = compute_relations(doc)
        print '{} mentions, {} relations'.format(len(doc['mentions']), len(relations))

        # add in NULL relations for all mention pairs that don't have a relation
        rel_mentions = set([ r[:4] for r in relations])
        seen_mentions = set()
        n_null, n_real = 0, 0
        for i, m1 in enumerate(doc['mentions']):
            for m2 in doc['mentions'][i+1:]:
                r = m1[:2] + m2[:2]
                if r not in rel_mentions:
                    relations.append(r+(u'--NULL--',))
                    n_null += 1
                else:
                    n_real += 1
                    seen_mentions |= set([r])
        assert n_real+n_null == int(comb(len(doc['mentions']), 2)),\
                "There should always be m choose 2 relations"
        doc['relations'] = relations
        relation_vocab.add([r[4] for r in doc['relations']])

    # compute the typing stats for extract all mentions
    tag_map = {
        'start_tags':tuple([ t for t in boundary_vocab.vocabset
                             if t.startswith(('B', 'U'))]),
        'in_tags':tuple([ t for t in boundary_vocab.vocabset
                             if t.startswith(('B', 'U', 'I', 'L'))]),
        'out_tags':tuple([ t for t in boundary_vocab.vocabset
                             if t.startswith('O')]),
        'type_map':{t:t.split('-')[1]
                    if len(t.split('-')) > 1
                    else None
                    for t in boundary_vocab.vocabset
                    }
    }

    # create datasets
    dataset = [(doc['tokens'], doc['boundary_labels'], doc['mentions'], doc['relations']) for doc in data.values()]
    npr.shuffle(dataset)
    test = 1-test # eg, .1 -> .9
    valid = test-valid # eg, .1 -> .9
    valid_split = int(len(dataset)*valid)
    test_split = int(len(dataset)*test)
    dataset_train, dataset_valid, dataset_test = (dataset[:valid_split],
                                                  dataset[valid_split:test_split],
                                                  dataset[test_split:])

    x_train = [d[0] for d in dataset_train]
    b_train = [d[1] for d in dataset_train]
    m_train = [d[2] for d in dataset_train]
    r_train = [d[3] for d in dataset_train]

    x_valid = [d[0] for d in dataset_valid]
    b_valid = [d[1] for d in dataset_valid]
    m_valid = [d[2] for d in dataset_valid]
    r_valid = [d[3] for d in dataset_valid]

    x_test = [d[0] for d in dataset_test]
    b_test = [d[1] for d in dataset_test]
    m_test = [d[2] for d in dataset_test]
    r_test = [d[3] for d in dataset_test]

    print '{} train, {} validation, and {} test documents'.format(len(x_train), len(x_valid), len(x_test))

    dataset = { 'data':data,
                'token_vocab':token_vocab,
                'boundary_vocab':boundary_vocab,
                'mention_vocab':mention_vocab,
                'relation_vocab':relation_vocab,
                'tag_map':tag_map,
                'x_train':x_train,
                'b_train':b_train,
                'm_train':m_train,
                'r_train':r_train,
                'x_valid':x_valid,
                'b_valid':b_valid,
                'm_valid':m_valid,
                'r_valid':r_valid,
                'x_test':x_test,
                'b_test':b_test,
                'm_test':m_test,
                'r_test':r_test
            }
    return dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fname',
                        default='data/ace_05_head_yaat.json',
                        help='Relative location of data file',
                        type=str)
    parser.add_argument('-c', '--count',
                        default=5,
                        help='Minimum number of token count to not be UNK',
                        type=int)
    parser.add_argument('-v', '--valid',
                        default=.1,
                        help='Percent of data for validation (as float < 1)',
                        type=float)
    parser.add_argument('-t', '--test',
                        default=.1,
                        help='Percent of data for testing (as float < 1)',
                        type=float)
    parser.add_argument('-b', '--batch_size',
                        default=256,
                        help='Number of docs per minibatch',
                        type=int)
    parser.add_argument('-n', '--n_epoch',
                        default=50,
                        help='Max number of epochs to train',
                        type=int)
    parser.add_argument('-d', '--embedding_size',
                        default=50,
                        help='Size of token embeddings',
                        type=int)
    parser.add_argument('-l', '--learning_rate',
                        default=.01,
                        help='Learning rate of Adam optimizer',
                        type=float)
    parser.add_argument('-w', '--wait',
                        default=5,
                        help='Number of epochs to wait for early stopping')
    parser.add_argument('-m', '--model_fname',
                        default='best_tagger.model',
                        help='Name of file to save best model to',
                        type=str)
    parser.add_argument('-p', '--plot_fname',
                        default='learning_curve.png',
                        help='Name of file to save learning curve plot to',
                        type=str)
    parser.add_argument('--dropout',
                        default=.25,
                        type=float)
    parser.add_argument('--lstm_size',
                        default=50,
                        type=int)
    parser.add_argument('--use_crf', action='store_true', default=False)
    parser.add_argument('--eval_only', action='store_true', default=False)
    parser.add_argument('--rseed', type=int, default=42,
                        help='Sets the random seed')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    npr.seed(args.rseed)
    dataset = get_ace_extraction_data(**vars(args))
    print dataset['relation_vocab'].vocabset
    # train(dataset, **vars(args))
