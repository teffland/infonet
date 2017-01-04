import json
import io
import time
import os
import argparse
import numpy as np
import numpy.random as npr
import chainer as ch

from infonet.vocab import Vocab
from infonet.preprocess import (Entity_BIO_map, Entity_BILOU_map,
                                BIO_map, BILOU_map,
                                typed_BIO_map, typed_BILOU_map,
                                compute_flat_mention_labels,
                                compute_tag_map, resolve_annotations)
from infonet.util import (convert_sequences, sequences2arrays,
                          SequenceIterator, print_epoch_loss, sec2hms)
from infonet.tagger import Tagger, TaggerLoss
from infonet.evaluation import plot_learning_curve, mention_boundary_stats
from infonet.word_vectors.word_vectors import get_pretrained_vectors

def get_ace_segmentation_data(count, **kwds):
    print "Loading data..."
    # load data
    train_data = json.loads(io.open('data/ace_05_head_yaat_train.json', 'r').read())
    dev_data = json.loads(io.open('data/ace_05_head_yaat_dev.json', 'r').read())
    test_data = json.loads(io.open('data/ace_05_head_yaat_test.json', 'r').read())
    all_data_values = train_data.values() + dev_data.values() + test_data.values()

    # get vocabs
    token_vocab = Vocab(min_count=count)
    for doc in all_data_values:
        token_vocab.add(doc['tokens'])

    boundary_vocab = Vocab(min_count=0)
    for doc in all_data_values:
        doc['annotations'] = resolve_annotations(doc['annotations'])
        doc['boundary_labels'] = compute_flat_mention_labels(doc, typed_BIO_map)
        boundary_vocab.add(doc['boundary_labels'])
    print boundary_vocab.vocabset

    # compute the typing stats for extract all mentions
    tag_map = compute_tag_map(boundary_vocab)

    # create datasets
    x_train = [ doc['tokens'] for doc in train_data.values() ]
    x_dev = [ doc['tokens'] for doc in dev_data.values() ]
    x_test = [ doc['tokens'] for doc in test_data.values() ]
    b_train = [ doc['boundary_labels'] for doc in train_data.values() ]
    b_dev = [ doc['boundary_labels'] for doc in dev_data.values() ]
    b_test = [ doc['boundary_labels'] for doc in test_data.values() ]
    print '{} train, {} dev, and {} test documents'.format(len(x_train), len(x_dev), len(x_test))

    dataset = { 'token_vocab':token_vocab,
                'boundary_vocab':boundary_vocab,
                'tag_map':tag_map,
                'x_train':x_train,
                'b_train':b_train,
                'x_dev':x_dev,
                'b_dev':b_dev,
                'x_test':x_test,
                'b_test':b_test }
    return dataset

def train(dataset, STATS, model_name,
          batch_size, n_epoch, wait,
          embedding_size, lstm_size, learning_rate,
          crf_type, dropout,
          use_w2v=False,
          eval_only=False,
          plot_fit_curve=False,
          **kwds):
    # unpack dataset
    token_vocab = dataset['token_vocab']
    boundary_vocab = dataset['boundary_vocab']
    tag_map = dataset['tag_map']
    x_train = dataset['x_train']
    x_dev = dataset['x_dev']
    x_test = dataset['x_test']
    b_train = dataset['b_train']
    b_dev = dataset['b_dev']
    b_test = dataset['b_test']

    # convert dataset to idxs
    # before we do conversions, we need to drop infrequent words from the vocab and reindex it
    print "Setting up...",
    token_vocab.drop_infrequent()
    boundary_vocab.drop_infrequent()

    ix_train = convert_sequences(x_train, token_vocab.idx)
    ix_dev = convert_sequences(x_dev, token_vocab.idx)
    ix_test = convert_sequences(x_test, token_vocab.idx)
    ib_train = convert_sequences(b_train, boundary_vocab.idx)
    ib_dev = convert_sequences(b_dev, boundary_vocab.idx)
    ib_test = convert_sequences(b_test, boundary_vocab.idx)

    # data
    train_iter = SequenceIterator(zip(ix_train, ib_train), batch_size, repeat=True)
    dev_iter = SequenceIterator(zip(ix_dev, ib_dev), batch_size, repeat=True)

    # get pretrained vectors
    if use_w2v:
        print "Loading pretrained embeddings...",
        vec_fname = 'infonet/word_vectors/GoogleNews-vectors-negative300.txt'
        trim_fname = '/'.join(vec_fname.split('/')[:-1]+['trimmed_'+vec_fname.split('/')[-1]])
        if os.path.isfile(trim_fname):
            print "Already trimmed..."
            embeddings = get_pretrained_vectors(token_vocab, trim_fname, trim=False)
        else:
            print "Trimming..."
            embeddings = get_pretrained_vectors(token_vocab, vec_fname, trim=True)
        embedding_size = embeddings.shape[1]
        print "Embedding size overwritten to {}".format(embedding_size)
    else:
        embeddings = None

    # model
    embed = ch.functions.EmbedID(token_vocab.v, embedding_size,
                                 initialW=embeddings)
    tagger = Tagger(embed, lstm_size, boundary_vocab.v,
                    crf_type=crf_type,
                    dropout=dropout)
    model_loss = TaggerLoss(tagger)
    optimizer = ch.optimizers.Adam(learning_rate)
    optimizer.setup(model_loss)
    print "Done"

    if not eval_only:
        # training
        # n_epoch = 50
        best_dev_loss = 1e50
        n_dev_up = 0
        epoch_losses = [[]]
        dev_losses = []
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
                dev_loss = 0
                for dev_batch in dev_iter:
                    x_list, b_list = zip(*dev_batch)
                    x_list = sequences2arrays(x_list)
                    b_list = sequences2arrays(b_list)
                    tagger.reset_state()
                    dev_loss += np.asscalar(model_loss(x_list, b_list).data)
                    if dev_iter.is_new_epoch:
                        break
                print_epoch_loss(train_iter.epoch,
                                 np.mean(epoch_losses[-1]),
                                 dev_loss,
                                 time=np.sum(forward_times[-1]+backward_times[-1]))
                dev_losses.append(dev_loss)
                # save best
                if dev_loss < best_dev_loss:
                    best_dev_loss = dev_loss
                    n_dev_up = 0
                    ch.serializers.save_npz('experiments/'+model_name+'.model', tagger)
                else:
                    n_dev_up += 1
                    if n_dev_up > wait:
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
        if plot_fit_curve:
            plot_learning_curve(epoch_losses, dev_losses, savename='experiments/'+model_name+'_fitcurve.pdf')

        STATS['epoch_losses'] = epoch_losses
        STATS['dev_losses'] = dev_losses
        STATS['seq_lengths'] = seq_lengths
        STATS['forward_times'] = forward_times
        STATS['backward_times'] = backward_times
        STATS['fit_time'] = fit_time

    # restore and evaluate
    print 'Restoring best model...',
    ch.serializers.load_npz('experiments/'+model_name+'.model', tagger)
    print 'Done'

    print 'Evaluating...'
    preds, x_list, b_list = tagger.predict(ix_train, ib_train)
    preds = convert_sequences(preds, boundary_vocab.token)
    xs = convert_sequences(x_list, token_vocab.token)
    bs = convert_sequences(b_list, boundary_vocab.token)
    f1_stats = mention_boundary_stats(bs, preds, **tag_map)
    STATS['train_stats'] = f1_stats
    print "Training:: P: {s[precision]:2.4f}, R: {s[recall]:2.4f}, F1: {s[f1]:2.4f}".format(s=f1_stats)

    preds, x_list, b_list = tagger.predict(ix_test, ib_test)
    preds = convert_sequences(preds, boundary_vocab.token)
    xs = convert_sequences(x_list, token_vocab.token)
    bs = convert_sequences(b_list, boundary_vocab.token)
    f1_stats = mention_boundary_stats(bs, preds, **tag_map)
    STATS['test_stats'] = f1_stats
    print "Testing:: P: {s[precision]:2.4f}, R: {s[recall]:2.4f}, F1: {s[f1]:2.4f}".format(s=f1_stats)

    print "Dumping run stats"
    with open('experiments/'+model_name+'_stats.json', 'w') as f:
        f.write(json.dumps(STATS))

    print "Finished Experiment"
    # print zip(xs[-1], preds[-1], ys[-1])
    # print 'Transition matrix:'
    # print ' '.join([ v for k,v in sorted(boundary_vocab._idx2vocab.items(), key=lambda x:x[0]) ])
    # if tagger.crf_type == 'simple':
    #     print tagger.crf.cost.data
    # print boundary_vocab.vocabset

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
    parser.add_argument('--use_w2v', action='store_true', default=False,
                        help='Whether or not to use pretrained word vectors')
    parser.add_argument('-l', '--learning_rate',
                        default=.01,
                        help='Learning rate of Adam optimizer',
                        type=float)
    parser.add_argument('-w', '--wait',
                        default=5,
                        help='Number of epochs to wait for early stopping')
    parser.add_argument('--dropout',
                        default=.25,
                        type=float)
    parser.add_argument('--lstm_size',
                        default=50,
                        type=int)
    parser.add_argument('--crf_type', type=str, default='none',
                        help='Choose from none, simple, linear, simple_bilinear, and bilinear')
    parser.add_argument('--eval_only', action='store_true', default=False)
    parser.add_argument('--rseed', type=int, default=42,
                        help='Sets the random seed')
    parser.add_argument('--plot_fit_curve', action='store_true', default=False,
                        help='Whether to plot the learning curve (needs matplotlib)')
    return parser.parse_args()

if __name__ == '__main__':
    # read input args
    args = parse_args()
    npr.seed(args.rseed)
    arg_dict = vars(args)

    # setup stats and model name
    w2v = 'w2v' if args.use_w2v else ''
    model_name = 'tagger_{a.embedding_size}_{w2v}_{a.lstm_size}_{a.crf_type}_{a.dropout}_\
{a.n_epoch}_{a.batch_size}_{a.count}'.format(a=args, w2v=w2v)
    STATS = {'args': arg_dict,
             'model_name':model_name}

    # run it
    dataset = get_ace_segmentation_data(**arg_dict)
    train(dataset, STATS=STATS, model_name=model_name, **arg_dict)
