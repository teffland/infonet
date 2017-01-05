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
from infonet.evaluation import plot_learning_curve, mention_boundary_stats

def train(dataset,
          batch_size, n_epoch, wait,
          embedding_size, lstm_size, learning_rate,
          use_crf, dropout,
          model_fname, plot_fname, eval_only=False,
          max_dist=500,
          **kwds):
    # unpack dataset
    token_vocab = dataset['token_vocab']
    boundary_vocab = dataset['boundary_vocab']
    mention_vocab = dataset['mention_vocab']
    relation_vocab = dataset['relation_vocab']
    tag_map = dataset['tag_map']
    start_tags = [ boundary_vocab.idx(tag) for tag in tag_map['start_tags'] ]
    in_tags = [ boundary_vocab.idx(tag) for tag in tag_map['start_tags'] ]
    out_tags = [ boundary_vocab.idx(tag) for tag in tag_map['start_tags'] ]
    x_train = dataset['x_train']
    x_dev = dataset['x_dev']
    x_test = dataset['x_test']
    b_train = dataset['b_train']
    b_dev = dataset['b_dev']
    b_test = dataset['b_test']
    m_train = dataset['m_train']
    m_dev = dataset['m_dev']
    m_test = dataset['m_test']
    r_train = dataset['r_train']
    r_dev = dataset['r_dev']
    r_test = dataset['r_test']

    # convert dataset to idxs
    # before we do conversions, we need to drop infrequent words from the vocab and reindex it
    print "Setting up...",
    token_vocab.drop_infrequent()
    boundary_vocab.drop_infrequent()
    mention_vocab.drop_infrequent()
    relation_vocab.drop_infrequent()

    ix_train = convert_sequences(x_train, token_vocab.idx)
    ix_dev = convert_sequences(x_dev, token_vocab.idx)
    ix_test = convert_sequences(x_test, token_vocab.idx)
    ib_train = convert_sequences(b_train, boundary_vocab.idx)
    ib_dev = convert_sequences(b_dev, boundary_vocab.idx)
    ib_test = convert_sequences(b_test, boundary_vocab.idx)
    convert_mention = lambda x: x[:-1]+(mention_vocab.idx(x[-1]),) # type is last
    im_train = convert_sequences(m_train, convert_mention)
    im_dev = convert_sequences(m_dev, convert_mention)
    im_test = convert_sequences(m_test, convert_mention)
    convert_relation = lambda x: x[:-1]+(relation_vocab.idx(x[-1]),) # type is last
    ir_train = convert_sequences(r_train, convert_relation)
    ir_dev = convert_sequences(r_dev, convert_relation)
    ir_test = convert_sequences(r_test, convert_relation)

    # data
    train_iter = SequenceIterator(zip(ix_train, ib_train, im_train, ir_train), batch_size, repeat=True)
    dev_iter = SequenceIterator(zip(ix_dev, ib_dev, im_dev, ir_dev), batch_size, repeat=True)

    # model
    embed = ch.functions.EmbedID(token_vocab.v, embedding_size)
    tagger = Tagger(embed, lstm_size, boundary_vocab.v,
                    dropout=dropout, use_crf=use_crf)
    ch.serializers.load_npz('best_tagger.model', tagger)
    # start_tags = [ boundary_vocab._vocab2idx[s] for s in ('B') ]
    # in_tags = [ boundary_vocab._vocab2idx[s] for s in ('B', 'I') ]
    # out_tags = [ boundary_vocab._vocab2idx[s] for s in ('O',) ]
    extractor = Extractor(tagger,
                          mention_vocab.v, relation_vocab.v, lstm_size=lstm_size,
                          start_tags=start_tags, in_tags=in_tags, out_tags=out_tags,
                          max_rel_dist=max_dist)
    extractor_loss = ExtractorLoss(extractor)
    optimizer = ch.optimizers.Adam(learning_rate)
    optimizer.setup(extractor_loss)
    print "Done"

    # training
    if not eval_only:
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
            x_list, b_list, m_list, r_list = zip(*batch)
            x_list = sequences2arrays(x_list)
            b_list = sequences2arrays(b_list)
            extractor.reset_state()
            extractor_loss.cleargrads()
            seq_lengths[-1].append(len(x_list))

            # run model
            start = time.time()
            loss = extractor_loss(x_list, b_list, m_list, r_list)
            forward_times[-1].append(time.time()-start)
            loss_val = loss.data
            print_batch_loss(loss_val,
                             train_iter.epoch+1,
                             train_iter.current_position,
                             train_iter.n_batches)
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
                    x_list, b_list, m_list, r_list = zip(*dev_batch)
                    x_list = sequences2arrays(x_list)
                    b_list = sequences2arrays(b_list)
                    extractor.reset_state()
                    dev_loss += extractor_loss(x_list, b_list, m_list, r_list).data
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
                    ch.serializers.save_npz(model_fname, extractor)
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


        print "Training finished. {} epochs in {}".format(
              n_epoch, sec2hms(time.time()-fit_start))
        plot_learning_curve(epoch_losses, dev_losses, savename=plot_fname)

    # restore and evaluate
    print 'Restoring best model...',
    ch.serializers.load_npz(model_fname, extractor)
    print 'Done'

    print 'Evaluating...'
    preds, x_list, y_list = tagger.predict(ix_train, iy_train)
    preds = convert_sequences(preds, boundary_vocab.token)
    xs = convert_sequences(x_list, token_vocab.token)
    ys = convert_sequences(y_list, boundary_vocab.token)
    f1_stats = mention_boundary_stats(ys, preds, **tag_map)
    print "Training:: P: {s[precision]:2.4f}, R: {s[recall]:2.4f}, F1: {s[f1]:2.4f}".format(s=f1_stats)

    preds, x_list, y_list = tagger.predict(ix_test, iy_test)
    preds = convert_sequences(preds, boundary_vocab.token)
    xs = convert_sequences(x_list, token_vocab.token)
    ys = convert_sequences(y_list, boundary_vocab.token)
    f1_stats = mention_boundary_stats(ys, preds, **tag_map)
    print "Testing:: P: {s[precision]:2.4f}, R: {s[recall]:2.4f}, F1: {s[f1]:2.4f}".format(s=f1_stats)

    print zip(xs[-1], preds[-1], ys[-1])
    print 'Transition matrix:'
    print ' '.join([ v for k,v in sorted(boundary_vocab._idx2vocab.items(), key=lambda x:x[0]) ])
    print tagger.crf.cost.data
    print boundary_vocab.vocabset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tagger', type=str,
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
                        default=5,
                        help='Number of epochs to wait for early stopping')
    parser.add_argument('--dropout',
                        default=.25,
                        type=float)
    parser.add_argument('--lstm_size',
                        default=50,
                        type=int)
    parser.add_argument('--eval_only', action='store_true', default=False)
    parser.add_argument('--rseed', type=int, default=42,
                        help='Sets the random seed')
    parser.add_argument('--max_dist', type=int, default=500,
                        help="""Maximum distance in document to try and classify
                             relations. Model speed/memory usage declines
                             as this increases""")
    return parser.parse_args()

def load_tagger(args):
    # load in the tagger options
    tagger_stats_f = open('experiments/{}_stats.json'.format(args.tagger, 'r'))
    tagger_args = json.load(tagger_stats_f)['args']
    args.count = tagger_args['count']

    # create and load the actual tagger
    


if __name__ == '__main__':
    args = parse_args()
    npr.seed(args.rseed)
    arg_dict = vars(args)
    dataset = get_ace_extraction_data(**arg_dict)

    model_name = 'extractor_{a.lstm_size}_{a.dropout}_\
{a.n_epoch}_{a.batch_size}_{a.count}'.format(a=args)
    STATS = {'args': arg_dict,
             'model_name':model_name,
             'start_time':time.time()}
    train(dataset, **arg_dict)
