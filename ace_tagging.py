import os
import argparse
import yaml
from datetime import datetime
from pprint import pprint

import numpy as np
import numpy.random as npr
import chainer as ch

from infonet.preprocess import get_ace_extraction_data
from infonet.util import SequenceIterator
from infonet.tagger import Tagger, TaggerLoss, TaggerEvaluator
from infonet.word_vectors import get_pretrained_vectors
from infonet.experiment import train, print_f1_stats, dump_stats

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', help='YAML file containing configuration')
    parser.add_argument('--eval', type=str, default='',
                        help='Override experiment to evaluate model with this name')
    parser.add_argument('-v', '--verbosity', type=int, default=2,
                        help='0 for nothing, 1 for stages, 2 for most, 3 for way too much')

    args = parser.parse_args()
    args.v = args.verbosity
    return args

def setup_data(dataset, batch_size):
    train_iter = SequenceIterator(zip(dataset['ix_train'],
                                      dataset['ip_train'],
                                      dataset['ib_train'],
                                      dataset['f_train'],
                                      dataset['x_train'],
                                      dataset['m_train']),
                                  batch_size,
                                  repeat=True)
    dev_iter = SequenceIterator(zip(dataset['ix_dev'],
                                    dataset['ip_dev'],
                                    dataset['ib_dev'],
                                    dataset['f_dev'],
                                    dataset['x_dev'],
                                    dataset['m_dev']),
                                batch_size,
                                repeat=True)
    test_iter = SequenceIterator(zip(dataset['ix_test'],
                                     dataset['ip_test'],
                                     dataset['ib_test'],
                                     dataset['f_test'],
                                     dataset['x_test'],
                                     dataset['m_test']),
                                 batch_size,
                                 repeat=True)
    return train_iter, dev_iter, test_iter

def setup_tagger(dataset, tagger_config, v=1):
    # allows for decoding of sequences to mention boundaries
    token_vocab = dataset['token_vocab']
    boundary_vocab = dataset['boundary_vocab']
    tag_map = dataset['tag_map']
    start_tags = [ boundary_vocab.idx(tag) for tag in tag_map['start_tags'] ]
    in_tags = [ boundary_vocab.idx(tag) for tag in tag_map['in_tags'] ]
    out_tags = [ boundary_vocab.idx(tag) for tag in tag_map['out_tags'] ]
    tag2mtype = { boundary_vocab.idx(tag):mtype
                  for tag, mtype in tag_map['tag2mtype'].items() }

    # get pretrained vectors if we speficied a file instead of size
    w2v = tagger_config['word_vector_size']
    if type(w2v) is str:
        if v > 0:
            print "Loading pretrained embeddings...",
        trim_fname = '/'.join(w2v.split('/')[:-1]+['trimmed_'+w2v.split('/')[-1]])
        if os.path.isfile(trim_fname):
            if v > 1:
                print "Already trimmed..."
            embeddings = get_pretrained_vectors(token_vocab, trim_fname, trim=False)
        else:
            if v > 1:
                print "Trimming..."
            embeddings = get_pretrained_vectors(token_vocab, w2v, trim=True)
        embedding_size = embeddings.shape[1]
        print "Embedding size overwritten to {}".format(embedding_size)
    else:
        embeddings = npr.normal(size=(token_vocab.v, w2v))

    tagger = Tagger(embeddings, boundary_vocab.v,
                    start_tags=start_tags,
                    in_tags=in_tags,
                    out_tags=out_tags,
                    tag2mtype=tag2mtype,
                    **tagger_config)
    tagger_loss = TaggerLoss(tagger)
    tagger_evaluator = TaggerEvaluator(tagger,
                                       dataset['token_vocab'],
                                       dataset['pos_vocab'],
                                       dataset['boundary_vocab'],
                                       dataset['mention_vocab'],
                                       dataset['tag_map'])
    return tagger, tagger_loss, tagger_evaluator

if __name__ == '__main__':
    # read input args
    args = parse_args()
    config = yaml.load(open(args.config_file))
    if args.eval:
        config = yaml.load(open(config['experiment_dir']+args.eval+'/config.yaml'))
        tagger_name = args.eval

    else:
        tagger_name = 'tagger_'+datetime.strftime(datetime.now(), '%b-%d-%Y-%H:%M:%f')

    print "Config: "
    pprint(config)
    save_prefix = config['experiment_dir']+tagger_name+'/'
    if not os.path.exists(save_prefix):
        os.makedirs(save_prefix)

    # save experiment configuration
    yaml.dump(config, open(save_prefix+'config.yaml', 'w'),
              default_flow_style=False)

    # setup everything
    npr.seed(config['random_seed'])
    dataset = get_ace_extraction_data(v=args.v, **config['preprocess'])
    train_iter, dev_iter, test_iter = setup_data(dataset, config['train']['batch_size'])
    tagger, tagger_loss, tagger_evaluator = setup_tagger(dataset, config['tagger'],
                                                         v=args.v)

    # train
    if not args.eval:
        train(tagger_loss, train_iter,
              tagger_evaluator, dev_iter,
              config, save_prefix, v=args.v)
        tagger.rescale_Us() # for recurrent dropout

    # restore best and evaluate
    if args.v > 0:
        print 'Restoring best model...',
    ch.serializers.load_npz(save_prefix+'tagger.model', tagger)
    if args.v > 0:
        print 'Done'

    doc_save_prefix = save_prefix+'docs/'
    if not os.path.exists(doc_save_prefix):
        os.makedirs(doc_save_prefix)

    if args.v > 0:
        print 'Evaluating...'
    STATS = {}
    STATS['train_eval'] = tagger_evaluator.evaluate(train_iter, doc_save_prefix)
    print_f1_stats('Training', STATS['train_eval'])
    STATS['dev_eval'] = tagger_evaluator.evaluate(dev_iter, doc_save_prefix)
    print_f1_stats('Dev', STATS['dev_eval'])
    STATS['test_eval'] = tagger_evaluator.evaluate(test_iter, doc_save_prefix)
    print_f1_stats('Test', STATS['test_eval'])
    dump_stats(STATS, save_prefix+'eval')
    if args.v > 0:
        print "Finished Experiment"

    # save the total precision, recall, and f1s in global file
    all_stats_fname = config['experiment_dir']+'tagger_experiments.csv'
    line = tagger_name
    line += ', {s[precision]:2.4f}, {s[recall]:2.4f}, {s[f1]:2.4f},'.format(s=STATS['train_eval'])
    line += ' {s[precision]:2.4f}, {s[recall]:2.4f}, {s[f1]:2.4f},'.format(s=STATS['dev_eval'])
    line += ' {s[precision]:2.4f}, {s[recall]:2.4f}, {s[f1]:2.4f}'.format(s=STATS['test_eval'])
    if os.path.exists(all_stats_fname):
        with open(all_stats_fname, 'a') as f:
            f.write(line+'\n')
    else:
        header = 'name, train p, train r, train f1, dev p, dev r, dev f1, test p, test r, test f1'
        with open(all_stats_fname, 'w') as f:
            f.write(header+'\n')
            f.write(line+'\n')
