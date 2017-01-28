import json
import io
import time
import argparse
import numpy as np
import numpy.random as npr
import chainer as ch

from infonet.vocab import Vocab
from infonet.preprocess import get_ace_extraction_data
from infonet.util import (convert_sequences, sequences2arrays,
                          SequenceIterator,
                          print_epoch_loss, print_batch_loss,
                          sec2hms)
from infonet.tagger import Tagger, TaggerLoss
from infonet.extractor import Extractor, ExtractorLoss
from infonet.evaluation import mention_boundary_stats, mention_relation_stats

def dump_stats(STATS, model_name):
    print "Dumping stats for {}...".format(model_name),
    # write out stats that how to configure (instantiate) a model
    stats = {k:v for k,v in STATS.items()
             if k in ('args', 'model_name')}
    with open('experiments/{}_config.json'.format(model_name), 'w') as f:
        f.write(json.dumps(stats)+'\n')
    # write out stats that involve evaluation of model
    stats = {k:v for k,v in STATS.items()
             if 'stats' in k}
    with open('experiments/{}_eval_stats.json'.format(model_name), 'a') as f:
        f.write(json.dumps(stats)+'\n')
    # write out stats that are monitored model statistics
    stats = {k:v for k,v in STATS.items()
             if 'stats' not in k}
    with open('experiments/{}_report_stats.json'.format(model_name), 'a') as f:
        f.write(json.dumps(stats)+'\n')
    print "Done"

def train(dataset, tagger,
          STATS, model_name,
          batch_size, n_epoch, wait,
          lstm_size, use_mlp, bidirectional, shortcut_embeds,
          learning_rate, dropout, backprop,
          eval_only=False,
          max_dist=500,
          **kwds):
    # unpack dataset
    token_vocab = dataset['token_vocab']
    boundary_vocab = dataset['boundary_vocab']
    mention_vocab = dataset['mention_vocab']
    relation_vocab = dataset['relation_vocab']
    tag_map = dataset['tag_map']
    ## convert to indices in vocabs
    start_tags = [ boundary_vocab.idx(tag) for tag in tag_map['start_tags'] ]
    in_tags = [ boundary_vocab.idx(tag) for tag in tag_map['in_tags'] ]
    out_tags = [ boundary_vocab.idx(tag) for tag in tag_map['out_tags'] ]
    tag2mtype = { boundary_vocab.idx(tag):mtype
                  for tag, mtype in tag_map['tag2mtype'].items() }
    mtype2msubtype = { mtype:[ mention_vocab.idx(msubtype) for msubtype in s ]
                       for mtype, s in dataset['mtype2msubtype'].items() }
    msubtype2rtype = dataset['msubtype2rtype']
    msubtype2rtype['left'] = { mention_vocab.idx(msubtype):
                               [ relation_vocab.idx(r) for r in s ]
                               for msubtype, s in msubtype2rtype['left'].items()}
    msubtype2rtype['right'] = { mention_vocab.idx(msubtype):
                                [ relation_vocab.idx(r) for r in s ]
                                for msubtype, s in msubtype2rtype['right'].items()}
    ix_train = dataset['ix_train']
    ix_dev = dataset['ix_dev']
    ix_test = dataset['ix_test']
    ib_train = dataset['ib_train']
    ib_dev = dataset['ib_dev']
    ib_test = dataset['ib_test']
    im_train = dataset['im_train']
    im_dev = dataset['im_dev']
    im_test = dataset['im_test']
    ir_train = dataset['ir_train']
    ir_dev = dataset['ir_dev']
    ir_test = dataset['ir_test']

    # print tag_map

    # data
    train_iter = SequenceIterator(zip(ix_train, ib_train, im_train, ir_train),
        batch_size, repeat=True)
    dev_iter = SequenceIterator(zip(ix_dev, ib_dev, im_dev, ir_dev),
        batch_size, repeat=True)
    test_iter = SequenceIterator(zip(ix_test, ib_test, im_test, ir_test),
        batch_size, repeat=True)

    # model
    extractor = Extractor(tagger,
                          mention_vocab.v, relation_vocab.v,
                          null_idx=relation_vocab.idx(u'--NULL--'),
                          lstm_size=lstm_size,
                          use_mlp=use_mlp,
                          bidirectional=bidirectional,
                          shortcut_embeds=shortcut_embeds,
                          start_tags=start_tags, in_tags=in_tags, out_tags=out_tags,
                          tag2mtype=tag2mtype,
                          mtype2msubtype=mtype2msubtype,
                          msubtype2rtype=msubtype2rtype,
                          max_rel_dist=max_dist)
    extractor_loss = ExtractorLoss(extractor)
    optimizer = ch.optimizers.Adam(learning_rate)
    optimizer.setup(extractor_loss)

    # evalutation subroutine
    def evaluate(extractor, batch_iter,
                 token_vocab=token_vocab,
                 mention_vocab=mention_vocab,
                 relation_vocab=relation_vocab,
                 keep_raw=False):
        all_xs = []
        all_bs, all_bpreds = [], []
        all_ms, all_mpreds = [], []
        all_rs, all_rpreds = [], []
        for batch in batch_iter:
            x_list, b_list, m_list, r_list = zip(*batch)
            all_xs.extend(x_list)
            all_bs.extend(b_list)
            all_ms.extend(m_list)
            all_rs.extend(r_list)
            b_preds, m_preds, r_preds, m_spans, r_spans = extractor.predict(sequences2arrays(x_list))
            b_preds = [ pred.data for pred in ch.functions.transpose_sequence(b_preds) ]
            all_bpreds.extend(b_preds)
            mp_list = [ [ (s[0],s[1], p) for p,s in zip(preds, spans)]
                        for preds,spans in zip(m_preds, m_spans)]
            all_mpreds.extend(mp_list)
            rp_list = [ [ (s[0],s[1],s[2],s[3], p) for p,s in zip(preds, spans)]
                        for preds,spans in zip(r_preds, r_spans)]
            all_rpreds.extend(rp_list)
            if batch_iter.is_new_epoch:
                break

        # print [ (len([r for r in rpreds if r[-1] == 0]), len(rpreds))
        #         for rpreds in all_rpreds]
        # print [(len(rs), len(rpreds)) for rs, rpreds in zip(all_rs, all_rpreds)]
        all_xs = convert_sequences(all_xs, token_vocab.token)
        all_bs = convert_sequences(all_bs, boundary_vocab.token)
        all_bpreds = convert_sequences(all_bpreds, boundary_vocab.token)
        convert_mention = lambda x: x[:-1]+(mention_vocab.token(x[-1]),) # type is last
        all_ms = convert_sequences(all_ms, convert_mention)
        all_mpreds = convert_sequences(all_mpreds, convert_mention)
        convert_relation = lambda x: x[:-1]+(relation_vocab.token(x[-1]),) # type is last
        all_rs= convert_sequences(all_rs, convert_relation)
        all_rpreds = convert_sequences(all_rpreds, convert_relation)
        # m_f1_stats = mention_stats(all_mpreds, all_ms)
        # r_f1_stats = relation_stats(all_rpreds, all_rs)
        f1_stats = mention_relation_stats(all_ms, all_mpreds, all_rs, all_rpreds)
        f1_stats.update({'tag-'+k:v for k,v in
                         mention_boundary_stats(all_bs, all_bpreds, **tag_map).items()})
        if keep_raw:
            # m_f1_stats['xs'] = all_xs
            # m_f1_stats['m_preds'] = all_mpreds
            # m_f1_stats['m_trues'] = all_ms
            # r_f1_stats['xs'] = all_xs
            # r_f1_stats['r_preds'] = all_rpreds
            # r_f1_stats['r_trues'] = all_rs
            f1_stats['xs'] = all_xs
            f1_stats['b_preds'] = all_bpreds
            f1_stats['b_trues'] = all_bs
            f1_stats['m_preds'] = all_mpreds
            f1_stats['m_trues'] = all_ms
            f1_stats['r_preds'] = all_rpreds
            f1_stats['r_trues'] = all_rs
        return f1_stats #m_f1_stats, r_f1_stats

    def reset_stats(STATS):
        """ Reset the monitor statistics of STATS """
        STATS['epoch'] = None
        STATS['epoch_losses'] = []
        STATS['dev_f1'] = None
        STATS['dev_stats'] = []
        STATS['forward_times'] = []
        STATS['backward_times'] = []
        STATS['seq_lengths'] = []
        STATS['reports'] = []
        return STATS

    def print_stats(name, f1_stats):
        print "{} :: P: {s[precision]:2.4f} R: {s[recall]:2.4f} F1: {s[f1]:2.4f} \
 tp:{s[tp]} fp:{s[fp]} fn:{s[fn]} support:{s[support]}".format(
                name, s=f1_stats)
        for t,s in sorted(f1_stats.items(), key= lambda x:x[0]):
            if type(s) is dict:
                print "{} :{}: P: {s[precision]:2.4f} R: {s[recall]:2.4f} F1: {s[f1]:2.4f}\
 tp:{s[tp]} fp:{s[fp]} fn:{s[fn]} support:{s[support]}".format(
                        name, t, s=s)

    # training
    if not eval_only:
        print "Training"
        best_dev_f1 = 0.0
        n_dev_down = 0
        STATS = reset_stats(STATS)
        fit_start = time.time()
        for batch in train_iter:
            STATS['epoch'] = train_iter.epoch
            # prepare data and model
            x_list, b_list, m_list, r_list = zip(*batch)
            x_list = sequences2arrays(x_list)
            extractor.reset_state()
            extractor_loss.cleargrads()
            STATS['seq_lengths'].append(len(x_list))

            # run model
            start = time.time()
            loss = extractor_loss(x_list, m_list, r_list,
                                  backprop_to_tagger=backprop)
            STATS['forward_times'].append(time.time()-start)
            loss_val = np.asscalar(loss.data)
            print_batch_loss(loss_val,
                             train_iter.epoch+1,
                             train_iter.current_position,
                             train_iter.n_batches)
            STATS['epoch_losses'].append(loss_val)

            # backprop
            start = time.time()
            loss.backward()
            optimizer.update()
            STATS['backward_times'].append(time.time()-start)

            # report params and grads
            reports = {k:v.tolist() for k,v in extractor_loss.report().items()}
            STATS['reports'].append(reports)

            # validation routine
            if train_iter.is_new_epoch:
                dev_stats = evaluate(extractor, dev_iter)
                dev_f1 = dev_stats['f1']
                STATS['dev_f1'] = dev_f1
                print_epoch_loss(train_iter.epoch,
                                 np.mean(STATS['epoch_losses']),
                                 dev_f1,
                                 time=np.sum(STATS['forward_times']+STATS['backward_times']))
                print_stats('Dev', dev_stats)
                STATS['dev_stats'].append(dev_stats)

                # save best
                if dev_f1 >= best_dev_f1:
                    best_dev_f1 = dev_f1
                    n_dev_down = 0
                    print "Saving model...",
                    ch.serializers.save_npz('experiments/'+model_name+'.model', extractor)
                    print "Done"
                else:
                    n_dev_down += 1
                    if n_dev_down > wait:
                        print "Stopping early"
                        break
                if train_iter.epoch == n_epoch:
                    break
                dump_stats(STATS, model_name)
                STATS = reset_stats(STATS)


        STATS['fit_time'] = time.time()-fit_start
        print "Training finished. {} epochs in {}".format(
              n_epoch, sec2hms(STATS['fit_time']))

    # restore and evaluate
    print 'Restoring best model...',
    ch.serializers.load_npz('experiments/'+model_name+'.model', extractor)
    print 'Done'

    print 'Evaluating...'
    f1_stats = evaluate(extractor, train_iter, keep_raw=True)
    print_stats('Training', f1_stats)
    STATS['train_stats'] = f1_stats

    f1_stats = evaluate(extractor, dev_iter, keep_raw=True)
    print_stats('Dev', f1_stats)
    STATS['dev_stats'] = f1_stats

    f1_stats = evaluate(extractor, test_iter, keep_raw=True)
    print_stats('Test', f1_stats)
    STATS['test_stats'] = f1_stats

    dump_stats(STATS, model_name)
    print "Finished Experiment"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tagger_f', type=str,
                        help='Name of pretrained tagger model to use')
    parser.add_argument('-b', '--batch_size',
                        default=256,
                        help='Number of docs per minibatch',
                        type=int)
    parser.add_argument('-n', '--n_epoch',
                        default=50,
                        help='Max number of epochs to train',
                        type=int)
    parser.add_argument('-l', '--learning_rate',
                        default=.01,
                        help='Learning rate of Adam optimizer',
                        type=float)
    parser.add_argument('-w', '--wait',
                        default=20,
                        help='Number of epochs to wait for early stopping')
    parser.add_argument('--dropout',
                        default=.25,
                        type=float)
    parser.add_argument('--lstm_size',
                        default=50,
                        type=int)
    parser.add_argument('--backprop', action='store_true', default=False)
    parser.add_argument('--use_mlp', action='store_true', default=False)
    parser.add_argument('--bidirectional', action='store_true', default=False)
    parser.add_argument('--shortcut_embeds', action='store_true', default=False)
    parser.add_argument('--eval_only', action='store_true', default=False)
    parser.add_argument('--rseed', type=int, default=42,
                        help='Sets the random seed')
    parser.add_argument('--max_dist', type=int, default=200,
                        help="""Maximum distance in document to try and classify
                             relations. Model speed/memory usage declines
                             as this increases""")
    return parser.parse_args()

def load_dataset_and_tagger(arg_dict):
    # load in the tagger options
    print "Loading stats"
    model_name = arg_dict['tagger_f']
    with open('experiments/{}_config.json'.format(model_name), 'r') as f:
        for line in f:
            tagger_stats = json.loads(line)
            break # only need first log
    tagger_args = tagger_stats['args']
    print "Tagger args: ", tagger_args
    # load in dataset (need this to create the tagger)
    dataset = get_ace_extraction_data(**tagger_args)
    # create and load the actual tagger
    embeddings = np.zeros((dataset['token_vocab'].v, tagger_args['embedding_size']))
    tagger = Tagger(embeddings,
                    tagger_args['lstm_size'],
                    dataset['boundary_vocab'].v,
                    dropout=tagger_args['dropout'],
                    crf_type=tagger_args['crf_type'],
                    bidirectional=tagger_args['bidirectional'],
                    use_mlp=tagger_args['use_mlp'],
                    n_layers=tagger_args['n_layers'])
    ch.serializers.load_npz('experiments/'+model_name+'.model', tagger)
    return dataset, tagger

def name_extractor(args):
    model_name = 'extractor_{a.lstm_size}_{a.dropout}_\
{a.n_epoch}_{a.batch_size}'.format(a=args)
    return model_name

if __name__ == '__main__':
    args = parse_args()
    npr.seed(args.rseed)
    arg_dict = vars(args)
    model_name = name_extractor(args)
    STATS = {'args': arg_dict,
             'model_name':model_name,
             'start_time':time.time()}

    dataset, tagger = load_dataset_and_tagger(arg_dict)
    train(dataset, tagger, STATS=STATS, model_name=model_name, **arg_dict)
