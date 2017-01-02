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
                          SequenceIterator,
                          print_epoch_loss, print_batch_loss,
                          sec2hms)
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
        # print '{} mentions, {} relations'.format(len(doc['mentions']), len(relations))

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

    x_train = [ d[0] for d in dataset_train ]
    b_train = [ d[1] for d in dataset_train ]
    m_train = [ d[2] for d in dataset_train ]
    r_train = [ d[3] for d in dataset_train ]

    x_valid = [ d[0] for d in dataset_valid ]
    b_valid = [ d[1] for d in dataset_valid ]
    m_valid = [ d[2] for d in dataset_valid ]
    r_valid = [ d[3] for d in dataset_valid ]

    x_test = [ d[0] for d in dataset_test ]
    b_test = [ d[1] for d in dataset_test ]
    m_test = [ d[2] for d in dataset_test ]
    r_test = [ d[3] for d in dataset_test ]

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
    x_valid = dataset['x_valid']
    x_test = dataset['x_test']
    b_train = dataset['b_train']
    b_valid = dataset['b_valid']
    b_test = dataset['b_test']
    m_train = dataset['m_train']
    m_valid = dataset['m_valid']
    m_test = dataset['m_test']
    r_train = dataset['r_train']
    r_valid = dataset['r_valid']
    r_test = dataset['r_test']

    # convert dataset to idxs
    # before we do conversions, we need to drop infrequent words from the vocab and reindex it
    print "Setting up...",
    token_vocab.drop_infrequent()
    boundary_vocab.drop_infrequent()
    mention_vocab.drop_infrequent()
    relation_vocab.drop_infrequent()

    ix_train = convert_sequences(x_train, token_vocab.idx)
    ix_valid = convert_sequences(x_valid, token_vocab.idx)
    ix_test = convert_sequences(x_test, token_vocab.idx)
    ib_train = convert_sequences(b_train, boundary_vocab.idx)
    ib_valid = convert_sequences(b_valid, boundary_vocab.idx)
    ib_test = convert_sequences(b_test, boundary_vocab.idx)
    convert_mention = lambda x: x[:-1]+(mention_vocab.idx(x[-1]),) # type is last
    im_train = convert_sequences(m_train, convert_mention)
    im_valid = convert_sequences(m_valid, convert_mention)
    im_test = convert_sequences(m_test, convert_mention)
    convert_relation = lambda x: x[:-1]+(relation_vocab.idx(x[-1]),) # type is last
    ir_train = convert_sequences(r_train, convert_relation)
    ir_valid = convert_sequences(r_valid, convert_relation)
    ir_test = convert_sequences(r_test, convert_relation)

    # data
    train_iter = SequenceIterator(zip(ix_train, ib_train, im_train, ir_train), batch_size, repeat=True)
    valid_iter = SequenceIterator(zip(ix_valid, ib_valid, im_valid, ir_valid), batch_size, repeat=True)

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

            # validation routine
            if train_iter.is_new_epoch:
                valid_loss = 0
                for valid_batch in valid_iter:
                    x_list, b_list, m_list, r_list = zip(*valid_batch)
                    x_list = sequences2arrays(x_list)
                    b_list = sequences2arrays(b_list)
                    extractor.reset_state()
                    valid_loss += extractor_loss(x_list, b_list, m_list, r_list).data
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
        plot_learning_curve(epoch_losses, valid_losses, savename=plot_fname)

    # restore and evaluate
    print 'Restoring best model...',
    ch.serializers.load_npz(model_fname, tagger)
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
                        default='best_extractor.model',
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
    parser.add_argument('--max_dist', type=int, default=500,
                        help="""Maximum distance in document to try and classify
                             relations. Model speed/memory usage declines
                             as this increases""")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    npr.seed(args.rseed)
    dataset = get_ace_extraction_data(**vars(args))
    print dataset['relation_vocab'].vocabset
    train(dataset, **vars(args))
