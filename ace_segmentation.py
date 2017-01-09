import json
import io
import time
import os
import argparse
import numpy as np
import numpy.random as npr
import chainer as ch

from infonet.vocab import Vocab
from infonet.preprocess import get_ace_extraction_data
from infonet.util import (convert_sequences, sequences2arrays,
                          SequenceIterator, print_epoch_loss, sec2hms)
from infonet.tagger import Tagger, TaggerLoss
from infonet.evaluation import mention_boundary_stats
from infonet.word_vectors import get_pretrained_vectors

def train(dataset, STATS, model_name,
          batch_size, n_epoch, wait,
          embedding_size, lstm_size, learning_rate,
          crf_type, dropout,
          weight_decay, grad_clip,
          bidirectional, use_mlp,
          w2v_fname='',
          eval_only=False,
          **kwds):
    # unpack dataset
    token_vocab = dataset['token_vocab']
    boundary_vocab = dataset['boundary_vocab']
    tag_map = dataset['tag_map']
    ix_train = dataset['ix_train']
    ix_dev = dataset['ix_dev']
    ix_test = dataset['ix_test']
    ib_train = dataset['ib_train']
    ib_dev = dataset['ib_dev']
    ib_test = dataset['ib_test']

    x_train = dataset['x_train']
    x_dev = dataset['x_dev']
    x_test = dataset['x_test']
    b_train = dataset['b_train']
    b_dev = dataset['b_dev']
    b_test = dataset['b_test']

    print tag_map

    train_iter = SequenceIterator(zip(ix_train, ib_train), batch_size, repeat=True)
    dev_iter = SequenceIterator(zip(ix_dev, ib_dev), batch_size, repeat=True)
    test_iter = SequenceIterator(zip(ix_test, ib_test), batch_size, repeat=True)

    # get pretrained vectors
    if w2v_fname:
        print "Loading pretrained embeddings...",
        trim_fname = '/'.join(w2v_fname.split('/')[:-1]+['trimmed_'+w2v_fname.split('/')[-1]])
        if os.path.isfile(trim_fname):
            print "Already trimmed..."
            embeddings = get_pretrained_vectors(token_vocab, trim_fname, trim=False)
        else:
            print "Trimming..."
            embeddings = get_pretrained_vectors(token_vocab, w2v_fname, trim=True)
        embedding_size = embeddings.shape[1]
        STATS['args']['embedding_size'] = embedding_size
        print "Embedding size overwritten to {}".format(embedding_size)
    else:
        embeddings = None

    # model
    embed = ch.functions.EmbedID(token_vocab.v, embedding_size,
                                 initialW=embeddings)
    tagger = Tagger(embed, lstm_size, boundary_vocab.v,
                    crf_type=crf_type,
                    dropout=dropout,
                    bidirectional=bidirectional,
                    use_mlp=use_mlp)
    model_loss = TaggerLoss(tagger)
    optimizer = ch.optimizers.Adam(learning_rate)
    optimizer.setup(model_loss)
    optimizer.add_hook(ch.optimizer.WeightDecay(weight_decay))
    optimizer.add_hook(ch.optimizer.GradientClipping(grad_clip))
    print "Done"

    # evalutation subroutine
    def evaluate(batch_iter):
        all_preds, all_xs, all_bs = [], [], []
        for batch in batch_iter:
            x_list, b_list = zip(*batch)
            preds = tagger.predict(sequences2arrays(x_list))
            preds = [pred.data for pred in ch.functions.transpose_sequence(preds) ]
            all_preds.extend(preds)
            all_xs.extend(x_list)
            all_bs.extend(b_list)
            if batch_iter.is_new_epoch:
                break
        all_preds = convert_sequences(all_preds, boundary_vocab.token)
        all_xs = convert_sequences(all_xs, token_vocab.token)
        all_bs = convert_sequences(all_bs, boundary_vocab.token)
        f1_stats = mention_boundary_stats(all_bs, all_preds, **tag_map)
        return f1_stats

    def print_stats(name, f1_stats):
        print "{}:: P: {s[precision]:2.4f}, R: {s[recall]:2.4f}, F1: {s[f1]:2.4f}".format(
                name, s=f1_stats)
        for t,s in f1_stats.items():
            if type(s) is dict:
                print "{}:{}: P: {s[precision]:2.4f}, R: {s[recall]:2.4f}, F1: {s[f1]:2.4f}".format(
                    name, t, s=s)

    if not eval_only:
        # training
        # n_epoch = 50
        best_dev_f1 = 0
        n_dev_down = 0
        epoch_losses = [[]]
        dev_f1s = []
        dev_statss = []
        forward_times = [[]]
        backward_times = [[]]
        seq_lengths = [[]]
        fit_start = time.time()
        for batch in train_iter:
            # prepare data and model
            x_list, b_list = zip(*batch)
            x_list = sequences2arrays(x_list)
            b_list = sequences2arrays(b_list)
            model_loss.cleargrads()
            tagger.reset_state()
            seq_lengths[-1].append(len(x_list))

            # run model
            start = time.time()
            loss = model_loss(x_list, b_list)
            forward_times[-1].append(time.time()-start)
            loss_val = np.asscalar(loss.data)
            # print_batch_loss(loss_val,
            #                  train_iter.epoch+1,
            #                  train_iter.current_position,
            #                  train_iter.n_batches)
            epoch_losses[-1].append(loss_val)

            # backprop
            start = time.time()
            loss.backward()
            optimizer.update()
            backward_times[-1].append(time.time()-start)

            # devation routine
            if train_iter.is_new_epoch:
                dev_stats = evaluate(dev_iter)
                dev_f1 = dev_stats['f1']
                print_epoch_loss(train_iter.epoch,
                                 np.mean(epoch_losses[-1]),
                                 dev_f1,
                                 time=np.sum(forward_times[-1]+backward_times[-1]))
                print_stats('Dev', dev_stats)
                dev_f1s.append(dev_f1)
                dev_statss.append(dev_stats)
                # save best
                if dev_f1 >= best_dev_f1:
                    best_dev_f1 = dev_f1
                    n_dev_down = 0
                    ch.serializers.save_npz('experiments/'+model_name+'.model', tagger)
                else:
                    n_dev_down += 1
                    if n_dev_down > wait:
                        print "Stopping early"
                        break
                if train_iter.epoch == n_epoch:
                    break
                epoch_losses.append([])
                seq_lengths.append([])
                forward_times.append([])
                backward_times.append([])

        fit_time = time.time()-fit_start
        print "Training finished. {} epochs in {}".format(
              n_epoch, sec2hms(fit_time))

        STATS['epoch_losses'] = epoch_losses
        STATS['dev_f1s'] = dev_f1s
        STATS['dev_stats'] = dev_statss
        STATS['seq_lengths'] = seq_lengths
        STATS['forward_times'] = forward_times
        STATS['backward_times'] = backward_times
        STATS['fit_time'] = fit_time

    # restore and evaluate
    print 'Restoring best model...',
    ch.serializers.load_npz('experiments/'+model_name+'.model', tagger)
    print 'Done'

    def print_stats(name, f1_stats):
        print "{}:: P: {s[precision]:2.4f}, R: {s[recall]:2.4f}, F1: {s[f1]:2.4f}".format(
                name, s=f1_stats)
        for t,s in f1_stats.items():
            if type(s) is dict:
                print "{}:{}: P: {s[precision]:2.4f}, R: {s[recall]:2.4f}, F1: {s[f1]:2.4f}".format(
                    name, t, s=s)

    print 'Evaluating...'
    f1_stats = evaluate(train_iter)
    print_stats('Training', f1_stats)
    STATS['train_stats'] = f1_stats
    # print "Training:: P: {s[precision]:2.4f}, R: {s[recall]:2.4f}, F1: {s[f1]:2.4f}".format(s=f1_stats)

    f1_stats = evaluate(dev_iter)
    print_stats('Dev', f1_stats)
    STATS['dev_stats'] = f1_stats
    # print "Dev:: P: {s[precision]:2.4f}, R: {s[recall]:2.4f}, F1: {s[f1]:2.4f}".format(s=f1_stats)

    f1_stats = evaluate(test_iter)
    print_stats('Test', f1_stats)
    STATS['test_stats'] = f1_stats
    # print "Test:: P: {s[precision]:2.4f}, R: {s[recall]:2.4f}, F1: {s[f1]:2.4f}".format(s=f1_stats)

    stats_fname = 'experiments/'+model_name+'_stats.json'
    print "Dumping run stats to {}".format(stats_fname)
    with open(stats_fname, 'w') as f:
        f.write(json.dumps(STATS))
    print "Finished Experiment"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--count',
                        default=0,
                        help='Minimum number of token count to not be UNK',
                        type=int)
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
    parser.add_argument('--w2v_fname', type=str, default='',
                        help='Location of word vectors file')
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
    parser.add_argument('--bidirectional', action='store_true', default=False)
    parser.add_argument('--use_mlp', action='store_true', default=False)
    parser.add_argument('--weight_decay',
                        default=.0001,
                        type=float)
    parser.add_argument('--grad_clip',
                        default=50.,
                        type=float)
    parser.add_argument('--crf_type', type=str, default='none',
                        help='Choose from none, simple, linear, simple_bilinear, and bilinear')
    parser.add_argument('--eval_only', action='store_true', default=False)
    parser.add_argument('--rseed', type=int, default=42,
                        help='Sets the random seed')
    return parser.parse_args()

def name_tagger(args):
    w2v = args.w2v_fname.split('/')[-1] if args.w2v_fname else ''
    bi = 'bi' if args.bidirectional else ''
    model_name = 'tagger_{bi}_{a.embedding_size}_{a.lstm_size}_{a.crf_type}_{a.dropout}_\
{a.n_epoch}_{a.batch_size}_{a.count}_{w2v}'.format(a=args, w2v=w2v, bi=bi)
    return model_name

if __name__ == '__main__':
    # read input args
    args = parse_args()
    npr.seed(args.rseed)
    arg_dict = vars(args)

    # setup stats and model name
    model_name = name_tagger(args)
    STATS = {'args': arg_dict,
             'model_name':model_name,
             'start_time':time.time()}

    # run it
    dataset = get_ace_extraction_data(**arg_dict)
    train(dataset, STATS=STATS, model_name=model_name, **arg_dict)
