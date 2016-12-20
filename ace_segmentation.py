import json
import io
import time
import argparse
import numpy as np
import numpy.random as npr
import chainer as ch

from infonet.vocab import Vocab
from infonet.preprocess import (Entity_BIO_map, Entity_typed_BILOU_map,
                                compute_flat_mention_labels)
from infonet.util import (convert_sequences, sequences2arrays,
                          SequenceIterator, print_epoch_loss, sec2hms)
from infonet.tagger import Tagger, TaggerLoss
from infonet.evaluation import plot_learning_curve, mention_boundary_stats

def get_ace_segmentation_data(fname, count, valid, test, **kwds):
    print "Loading data..."
    # load data
    data = json.loads(io.open(fname, 'r').read())

    # get vocabs
    token_vocab = Vocab(min_count=count)
    for doc in data.values():
        token_vocab.add(doc['tokens'])

    boundary_vocab = Vocab(min_count=0)
    for doc in data.values():
        doc['boundary_labels'] = compute_flat_mention_labels(doc, Entity_BIO_map)
        boundary_vocab.add(doc['boundary_labels'])

    # create datasets
    xy = [(doc['tokens'], doc['boundary_labels']) for doc in data.values()]
    npr.shuffle(xy)
    test = 1-test # eg, .1 -> .9
    valid = test-valid # eg, .1 -> .9
    valid_split = int(len(xy)*valid)
    test_split = int(len(xy)*test)
    xy_train, xy_valid, xy_test = xy[:valid_split], xy[valid_split:test_split], xy[test_split:]

    x_train = [d[0] for d in xy_train]
    y_train = [d[1] for d in xy_train]
    x_valid = [d[0] for d in xy_valid]
    y_valid = [d[1] for d in xy_valid]
    x_test = [d[0] for d in xy_test]
    y_test = [d[1] for d in xy_test]
    print '{} train, {} validation, and {} test documents'.format(len(x_train), len(x_valid), len(x_test))

    dataset = { 'data':data,
                'token_vocab':token_vocab,
                'boundary_vocab':boundary_vocab,
                'x_train':x_train,
                'y_train':y_train,
                'x_valid':x_valid,
                'y_valid':y_valid,
                'x_test':x_test,
                'y_test':y_test }
    return dataset

def train(dataset,
          batch_size, n_epoch, wait,
          embedding_size, learning_rate, use_crf,
          model_fname, plot_fname,
          **kwds):
    # unpack dataset
    token_vocab = dataset['token_vocab']
    boundary_vocab = dataset['boundary_vocab']
    x_train = dataset['x_train']
    x_valid = dataset['x_valid']
    x_test = dataset['x_test']
    y_train = dataset['y_train']
    y_valid = dataset['y_valid']
    y_test = dataset['y_test']

    # convert dataset to idxs
    # before we do conversions, we need to drop unfrequent words from the vocab and reindex it
    print "Setting up...",
    token_vocab.drop_infrequent()
    boundary_vocab.drop_infrequent()

    ix_train = convert_sequences(x_train, token_vocab.idx)
    ix_valid = convert_sequences(x_valid, token_vocab.idx)
    ix_test = convert_sequences(x_test, token_vocab.idx)
    iy_train = convert_sequences(y_train, boundary_vocab.idx)
    iy_valid = convert_sequences(y_valid, boundary_vocab.idx)
    iy_test = convert_sequences(y_test, boundary_vocab.idx)

    # data
    train_iter = SequenceIterator(zip(ix_train, iy_train), batch_size, repeat=True)
    valid_iter = SequenceIterator(zip(ix_valid, iy_valid), batch_size, repeat=True)

    # hyperparams
    # embedding_size = 50
    # learning_rate = .05

    # model
    embed = ch.functions.EmbedID(token_vocab.v, embedding_size)
    tagger = Tagger(embed, embedding_size, boundary_vocab.v, use_crf)
    model_loss = TaggerLoss(tagger)
    optimizer = ch.optimizers.Adam(learning_rate)
    optimizer.setup(model_loss)
    print "Done"

    # training
    # n_epoch = 50
    best_valid_loss = 1e50
    n_valid_up = 0
    epoch_losses = [[]]
    valid_losses = []
    forward_times = [[]]
    backward_times = [[]]
    seq_lengths = [[]]
    fit_start = time.time()
    for batch in train_iter:
        # prepare data and model
        x_list, y_list = zip(*batch)
        x_list = sequences2arrays(x_list)
        y_list = sequences2arrays(y_list)
        model_loss.cleargrads()
        tagger.reset_state()
        seq_lengths[-1].append(len(x_list))

        # run model
        start = time.time()
        loss = model_loss(x_list, y_list)
        forward_times[-1].append(time.time()-start)
        loss_val = loss.data
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

        # validation routine
        if train_iter.is_new_epoch:
            valid_loss = 0
            for valid_batch in valid_iter:
                x_list, y_list = zip(*valid_batch)
                x_list = sequences2arrays(x_list)
                y_list = sequences2arrays(y_list)
                tagger.reset_state()
                valid_loss += model_loss(x_list, y_list).data
                if valid_iter.is_new_epoch:
                    break
            print_epoch_loss(train_iter.epoch,
                             np.mean(epoch_losses[-1]),
                             valid_loss,
                             time=np.sum(forward_times[-1]+backward_times[-1]))
            valid_losses.append(valid_loss)
            # save best
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                n_valid_up = 0
                ch.serializers.save_npz(model_fname, tagger)
            else:
                n_valid_up += 1
                if n_valid_up > wait:
                    print "Stopping early"
                    break
            if train_iter.epoch == n_epoch:
                break
            epoch_losses.append([])
            seq_lengths.append([])
            forward_times.append([])
            backward_times.append([])

    print "Training finished. {} epochs in {}".format(
          n_epoch, sec2hms(time.time()-fit_start))
    print 'Restoring best model...',
    ch.serializers.load_npz(model_fname, tagger)
    print 'Done'
    plot_learning_curve(epoch_losses, valid_losses, savename=plot_fname)

    print 'Evaluating...'
    preds, x_list, y_list = tagger.predict(ix_train, iy_train)
    preds = convert_sequences(preds, boundary_vocab.token)
    xs = convert_sequences(x_list, token_vocab.token)
    ys = convert_sequences(y_list, boundary_vocab.token)
    f1_stats = mention_boundary_stats(ys, preds)
    print "Training:: P: {s[precision]:2.4f}, R: {s[recall]:2.4f}, F1: {s[f1]:2.4f}".format(s=f1_stats)

    preds, x_list, y_list = tagger.predict(ix_test, iy_test)
    preds = convert_sequences(preds, boundary_vocab.token)
    xs = convert_sequences(x_list, token_vocab.token)
    ys = convert_sequences(y_list, boundary_vocab.token)
    f1_stats = mention_boundary_stats(ys, preds)
    print "Testing:: P: {s[precision]:2.4f}, R: {s[recall]:2.4f}, F1: {s[f1]:2.4f}".format(s=f1_stats)

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
    parser.add_argument('--use_crf', action='store_true', default=False)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    dataset = get_ace_segmentation_data(**vars(args))
    train(dataset, **vars(args))
