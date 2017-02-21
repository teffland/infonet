import os
import argparse
import yaml
from datetime import datetime
from pprint import pprint

import numpy as np
import numpy.random as npr
import chainer as ch

from ace_tagging import setup_tagger

from infonet.preprocess import get_ace_extraction_data
from infonet.util import SequenceIterator
from infonet.extractor import Extractor, ExtractorLoss, ExtractorEvaluator
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
                                      dataset['im_train'],
                                      dataset['ir_train'],
                                      dataset['f_train'],
                                      dataset['x_train']),
                                  batch_size,
                                  repeat=True)
    dev_iter = SequenceIterator(zip(dataset['ix_dev'],
                                      dataset['ip_dev'],
                                      dataset['ib_dev'],
                                      dataset['im_dev'],
                                      dataset['ir_dev'],
                                      dataset['f_dev'],
                                      dataset['x_dev']),
                                batch_size,
                                repeat=True)
    test_iter = SequenceIterator(zip(dataset['ix_test'],
                                      dataset['ip_test'],
                                      dataset['ib_test'],
                                      dataset['im_test'],
                                      dataset['ir_test'],
                                      dataset['f_test'],
                                      dataset['x_test']),
                                 batch_size,
                                 repeat=True)
    return train_iter, dev_iter, test_iter

def setup_extractor(dataset, tagger_config, extractor_config, v=1):
    # setup tagger
    tagger, _, _ = setup_tagger(dataset, tagger_config, v=v)

    # convert type constraint mappings from tokens to indices
    mention_vocab = dataset['mention_vocab']
    relation_vocab = dataset['relation_vocab']
    mtype2msubtype = { mtype:[ mention_vocab.idx(msubtype) for msubtype in s ]
                       for mtype, s in dataset['mtype2msubtype'].items() }
    msubtype2rtype = dataset['msubtype2rtype']
    msubtype2rtype['left'] = { mention_vocab.idx(msubtype):
                               [ relation_vocab.idx(r) for r in s ]
                               for msubtype, s in msubtype2rtype['left'].items()}
    msubtype2rtype['right'] = { mention_vocab.idx(msubtype):
                                [ relation_vocab.idx(r) for r in s ]
                                for msubtype, s in msubtype2rtype['right'].items()}

    # setup extractor
    max_r_dist = extractor_config['relation_options'].pop('max_r_dist', 'infer')
    if max_r_dist == 'infer':
        max_r_dist = dataset['max_r_dist']
    print 'Max r dist: {}'.format(max_r_dist)
    extractor_config['relation_options']['max_r_dist'] = max_r_dist

    extractor = Extractor(tagger,
                          n_mention_class=mention_vocab.v,
                          n_relation_class=relation_vocab.v,
                          null_idx=relation_vocab.idx('--NULL--'),
                          coref_idx=relation_vocab.idx('--SameAs--'),
                          mtype2msubtype=mtype2msubtype,
                          msubtype2rtype=msubtype2rtype,
                          **extractor_config)
    extractor_loss = ExtractorLoss(extractor,
                                   use_gold_boundaries=extractor_config['use_gold_boundaries'])
    extractor_evaluator = ExtractorEvaluator(extractor,
                                             dataset['token_vocab'],
                                             dataset['pos_vocab'],
                                             dataset['boundary_vocab'],
                                             dataset['mention_vocab'],
                                             dataset['relation_vocab'],
                                             dataset['tag_map'],
                                             use_gold_boundaries=extractor_config['use_gold_boundaries'])
    return extractor, extractor_loss, extractor_evaluator

if __name__ == '__main__':
    # read input args
    args = parse_args()
    config = yaml.load(open(args.config_file))
    tagger_config = yaml.load(open(os.path.join(config['tagger'], 'config.yaml')))
    if args.eval:
        config = yaml.load(open(config['experiment_dir']+args.eval+'/config.yaml'))
        tagger_config = yaml.load(open(config['experiment_dir']+args.eval+'/tagger_config.yaml'))
        extractor_name = args.eval

    else:
        extractor_name = 'extractor_'+datetime.strftime(datetime.now(), '%b-%d-%Y-%H:%M:%f')
    print "Config: "
    pprint(config)
    save_prefix = config['experiment_dir']+extractor_name+'/'
    if not os.path.exists(save_prefix):
        os.makedirs(save_prefix)

    # save experiment configuration
    yaml.dump(config, open(save_prefix+'config.yaml', 'w'),
              default_flow_style=False)
    yaml.dump(tagger_config, open(save_prefix+'tagger_config.yaml', 'w'),
              default_flow_style=False)

    # setup everything
    npr.seed(tagger_config['random_seed'])
    dataset = get_ace_extraction_data(v=args.v, **tagger_config['preprocess'])
    train_iter, dev_iter, test_iter = setup_data(dataset, config['train']['batch_size'])
    extractor, extractor_loss, extractor_evaluator = setup_extractor(dataset,
        tagger_config['tagger'], config['extractor'], v=args.v)

    # train
    extract_conf = config['extractor']
    b_loss = (extract_conf['build_on_tagger_features']
              or not extract_conf['use_gold_boundaries'])
    if not args.eval:
        train(extractor_loss, train_iter,
              extractor_evaluator, dev_iter,
              config, save_prefix, v=args.v,
              reweight_relations=extract_conf['relation_options']['reweight'],
              **config['train'])
        extractor.rescale_Us() # for recurrent dropout

    # restore best and evaluate
    if args.v > 0:
        print 'Restoring best model...',
    ch.serializers.load_npz(save_prefix+'extractor.model', extractor)
    if args.v > 0:
        print 'Done'

    doc_save_prefix = save_prefix+'docs/'
    if not os.path.exists(doc_save_prefix):
        os.makedirs(doc_save_prefix)

    if args.v > 0:
        print 'Evaluating...'
    STATS = {}
    STATS['train_eval'] = extractor_evaluator.evaluate(train_iter, doc_save_prefix)
    print_f1_stats('Training', STATS['train_eval'])
    STATS['dev_eval'] = extractor_evaluator.evaluate(dev_iter, doc_save_prefix)
    print_f1_stats('Dev', STATS['dev_eval'])
    STATS['test_eval'] = extractor_evaluator.evaluate(test_iter, doc_save_prefix)
    print_f1_stats('Test', STATS['test_eval'])
    dump_stats(STATS, save_prefix+'eval')
    if args.v > 0:
        print "Finished Experiment"

    # save the total precision, recall, and f1s in global file
    all_stats_fname = config['experiment_dir']+'extractor_experiments.csv'
    line = extractor_name
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
