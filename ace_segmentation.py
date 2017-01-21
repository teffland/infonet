import json
import io
import time
import os
import argparse
import numpy as np
import numpy.random as npr
import chainer as ch
from chainer import function_hooks

from infonet.vocab import Vocab
from infonet.preprocess import get_ace_extraction_data
from infonet.util import (convert_sequences, sequences2arrays,
                          SequenceIterator, print_epoch_loss, sec2hms)
from infonet.tagger import Tagger, TaggerLoss
from infonet.evaluation import mention_boundary_stats
from infonet.word_vectors import get_pretrained_vectors

def dump_stats(STATS, model_name):
    print "Dumping stats for {}...".format(model_name),
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

def train(dataset, STATS, model_name,
          batch_size, n_epoch, wait,
          embedding_size, lstm_size, learning_rate,
          crf_type, dropout,
          weight_decay, grad_clip,
          bidirectional, use_mlp, n_layers,
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
    im_train = dataset['im_train']
    im_dev = dataset['im_dev']
    im_test = dataset['im_test']

    x_train = dataset['x_train']
    x_dev = dataset['x_dev']
    x_test = dataset['x_test']
    b_train = dataset['b_train']
    b_dev = dataset['b_dev']
    b_test = dataset['b_test']
    m_train = dataset['m_train']
    m_dev = dataset['m_dev']
    m_test = dataset['m_test']

    print tag_map

    train_iter = SequenceIterator(zip(ix_train, ib_train, im_train), batch_size, repeat=True)
    dev_iter = SequenceIterator(zip(ix_dev, ib_dev, im_dev), batch_size, repeat=True)
    test_iter = SequenceIterator(zip(ix_test, ib_test, im_test), batch_size, repeat=True)

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
        embeddings = npr.normal(size=(token_vocab.v, embedding_size))

    # model
    # embeddings = np.zeros((token_vocab.v, embedding_size))
    tagger = Tagger(embeddings, lstm_size, boundary_vocab.v,
                    crf_type=crf_type,
                    dropout=dropout,
                    bidirectional=bidirectional,
                    use_mlp=use_mlp,
                    n_layers=n_layers)
    model_loss = TaggerLoss(tagger)

    class MyAdam(ch.optimizers.Adam):
        def update(self, lossfun=None, *args, **kwds):
            """Updates parameters based on a loss function or computed gradients.

            This method runs in two ways.

            - If ``lossfun`` is given, then use it as a loss function to compute
              gradients.
            - Otherwise, this method assumes that the gradients are already
              computed.

            In both cases, the computed gradients are used to update parameters.
            The actual update routines are defined by the :meth:`update_one`
            method (or its CPU/GPU versions, :meth:`update_one_cpu` and
            :meth:`update_one_gpu`).

            """
            if lossfun is not None:
                use_cleargrads = getattr(self, '_use_cleargrads', False)
                loss = lossfun(*args, **kwds)
                if use_cleargrads:
                    self.target.cleargrads()
                else:
                    self.target.zerograds()
                loss.backward()
                del loss

            # TODO(unno): Some optimizers can skip this process if they does not
            # affect to a parameter when its gradient is zero.
            for name, param in self.target.namedparams():
                if param.grad is None:
                    print name, param.name, 'has no grad'
                    # with cuda.get_device(param.data):
                    #     xp = cuda.get_array_module(param.data)
                    #     param.grad = xp.zeros_like(param.data)
                    param.grad = np.zeros_like(param.data)

            self.call_hooks()
            self.prepare()

            self.t += 1
            states = self._states
            for name, param in self.target.namedparams():
                # with cuda.get_device(param.data):
                print name,
                self.update_one(param, states[name])

        def update_one_cpu(self, param, state):
            m, v = state['m'], state['v']
            grad = param.grad
            print param.name
            print 'data', param.data[:5]
            print 'grad', grad[:5]
            m += (1 - self.beta1) * (grad - m)
            v += (1 - self.beta2) * (grad * grad - v)
            param.data -= self.lr * m / (np.sqrt(v) + self.eps)
    # optimizer = MyAdam(learning_rate)
    optimizer = ch.optimizers.Adam(learning_rate)
    # optimizer = ch.optimizers.AdaDelta()
    optimizer.setup(model_loss)
    optimizer.add_hook(ch.optimizer.WeightDecay(weight_decay))
    optimizer.add_hook(ch.optimizer.GradientClipping(grad_clip))
    print "Done"

    # evalutation subroutine
    def evaluate(tagger, batch_iter,
                 token_vocab=token_vocab,
                 boundary_vocab=boundary_vocab,
                 tag_map=tag_map,
                 keep_raw=False):
        all_preds, all_xs, all_bs, all_ms = [], [], [], []
        for batch in batch_iter:
            x_list, b_list, m_list = zip(*batch)
            preds = tagger.predict(sequences2arrays(x_list))
            preds = [pred.data for pred in ch.functions.transpose_sequence(preds) ]
            # for p in preds:
            #     print p.shape, p
            all_preds.extend(preds)
            all_xs.extend(x_list)
            all_bs.extend(b_list)
            if batch_iter.is_new_epoch:
                break
        all_preds = convert_sequences(all_preds, boundary_vocab.token)
        all_xs = convert_sequences(all_xs, token_vocab.token)
        all_bs = convert_sequences(all_bs, boundary_vocab.token)
        f1_stats = mention_boundary_stats(all_bs, all_preds, **tag_map)
        # keep raw predictions for error analysis
        if keep_raw:
            f1_stats['xs'] = all_xs
            f1_stats['b_trues'] = all_bs
            f1_stats['b_preds'] = all_preds
        return f1_stats

    def reset_stats(STATS):
        """Reset the monitor statistics of stats"""
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
        STATS = reset_stats(STATS)
        fit_start = time.time()
        for batch in train_iter:
            STATS['epoch'] = train_iter.epoch
            # prepare data and model
            x_list, b_list, m_list = zip(*batch)
            x_list = sequences2arrays(x_list)
            b_list = sequences2arrays(b_list)
            model_loss.cleargrads()
            tagger.reset_state()
            STATS['seq_lengths'].append(len(x_list))

            # run model
            start = time.time()
            # with ch.function_hooks.PrintHook():
            loss = model_loss(x_list, b_list)
            STATS['forward_times'].append(time.time()-start)
            loss_val = np.asscalar(loss.data)
            # print_batch_loss(loss_val,
            #                  train_iter.epoch+1,
            #                  train_iter.current_position,
            #                  train_iter.n_batches)
            STATS['epoch_losses'].append(loss_val)

            # backprop
            start = time.time()
            loss.backward()
            optimizer.update()
            STATS['backward_times'].append(time.time()-start)

            # report params and grads
            reports = {k:v.tolist() for k,v in model_loss.report().items()}
            STATS['reports'].append(reports)

            # devation routine
            if train_iter.is_new_epoch:
                dev_stats = evaluate(tagger, dev_iter)
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
                    ch.serializers.save_npz('experiments/'+model_name+'.model', tagger)
                else:
                    n_dev_down += 1
                    if n_dev_down > wait:
                        print "Stopping early"
                        break
                if train_iter.epoch == n_epoch:
                    break
                dump_stats(STATS, model_name)
                STATS = reset_stats(STATS)
                # STATS['epoch_losses'].append([])
                # STATS['seq_lengths'].append([])
                # STATS['forward_times'].append([])
                # STATS['backward_times'].append([])
                # STATS['reports'].append([])
                # if (train_iter.epoch % 10) == 0:
                #     dump_stats(STATS, model_name)

        STATS['fit_time'] = time.time()-fit_start
        print "Training finished. {} epochs in {}".format(
              n_epoch, sec2hms(STATS['fit_time']))


    # restore and evaluate
    print 'Restoring best model...',
    # tagger = tagger.copy()
    # tagger = Tagger(embeddings, lstm_size, boundary_vocab.v,
    #                 crf_type=crf_type,
    #                 dropout=dropout,
    #                 bidirectional=bidirectional,
    #                 use_mlp=use_mlp,
    #                 n_layers=n_layers)
    # embeddings = np.zeros((token_vocab.v, embedding_size))
    # tagger = Tagger(embeddings, lstm_size, boundary_vocab.v,
    #                 crf_type=crf_type,
    #                 dropout=dropout,
    #                 bidirectional=bidirectional,
    #                 use_mlp=use_mlp,
    #                 n_layers=n_layers)
    ch.serializers.load_npz('experiments/'+model_name+'.model', tagger)
    print 'Done'

    print 'Evaluating...'
    train_iter = SequenceIterator(zip(ix_train, ib_train, im_train), batch_size, repeat=True, shuffle=False)
    dev_iter = SequenceIterator(zip(ix_dev, ib_dev, im_dev), batch_size, repeat=True, shuffle=False)
    test_iter = SequenceIterator(zip(ix_test, ib_test, im_test), batch_size, repeat=True, shuffle=False)

    f1_stats = evaluate(tagger, train_iter, keep_raw=True)
    print_stats('Training', f1_stats)
    STATS['train_stats'] = f1_stats

    f1_stats = evaluate(tagger, dev_iter, keep_raw=True)
    print_stats('Dev', f1_stats)
    STATS['dev_stats'] = f1_stats

    f1_stats = evaluate(tagger, test_iter, keep_raw=True)
    print_stats('Test', f1_stats)
    STATS['test_stats'] = f1_stats

    dump_stats(STATS, model_name)
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
    parser.add_argument('--n_layers',
                        default=1,
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
    parser.add_argument('--model_f', type=str, default='',
                        help='Overrides name of the model output files')
    parser.add_argument('--map_func_name', type=str, default='E_BIO_map',
                        choices=['NoVal_BIO_map', 'E_BIO_map'])
    return parser.parse_args()

def name_tagger(args):
    w2v = args.w2v_fname.split('/')[-1] if args.w2v_fname else ''
    bi = 'bi' if args.bidirectional else ''
    model_name = 'tagger_{bi}_{a.embedding_size}_{a.lstm_size}_{a.crf_type}_{a.dropout}_\
{a.n_epoch}_{a.batch_size}_{a.weight_decay}_{w2v}'.format(a=args, w2v=w2v, bi=bi)
    return model_name

if __name__ == '__main__':
    # read input args
    args = parse_args()
    npr.seed(args.rseed)
    arg_dict = vars(args)

    # setup stats and model name
    if args.model_f:
        model_name = args.model_f
    else:
        model_name = name_tagger(args)
    STATS = {'args': arg_dict,
             'model_name':model_name,
             'start_time':time.time()}

    # run it
    dataset = get_ace_extraction_data(**arg_dict)
    train(dataset, STATS=STATS, model_name=model_name, **arg_dict)
