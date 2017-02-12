""" Base routines for training and evaluating a model """
import json
import time

import numpy as np
import chainer as ch

from util import print_batch_loss, print_epoch_loss, sec2hms

def dump_stats(STATS, save_prefix, v=0):
    if v > 1:
        print "Dumping stats to {}_stats.json ...".format(save_prefix),
    with open(save_prefix+'_stats.json', 'a') as f:
        f.write(json.dumps(STATS)+'\n')
    if v > 1:
        print "Done"

def print_f1_stats(name, f1_stats):
    print "{} :: P: {s[precision]:2.4f} R: {s[recall]:2.4f} F1: {s[f1]:2.4f} \
tp:{s[tp]} fp:{s[fp]} fn:{s[fn]} support:{s[support]}".format(
            name, s=f1_stats)
    for t,s in sorted(f1_stats.items(), key= lambda x:x[0]):
        if type(s) is dict:
            print "{} :{}: P: {s[precision]:2.4f} R: {s[recall]:2.4f} F1: {s[f1]:2.4f} \
tp:{s[tp]} fp:{s[fp]} fn:{s[fn]} support:{s[support]}".format(
                    name, t, s=s)

def reset_stats(STATS):
    """Reset the monitor statistics of stats"""
    STATS['epoch'] = None
    STATS['epoch_losses'] = []
    STATS['dev_score'] = None
    STATS['dev_stats'] = []
    STATS['forward_times'] = []
    STATS['backward_times'] = []
    STATS['reports'] = []
    return STATS

def train(model_loss, train_iter, model_evaluator, dev_iter, config, save_prefix,
          v=1, **loss_kwds):
    if v > 0:
        print "Training..."
    # setup the optimizer
    optimizer = getattr(ch.optimizers, config['optimizer']['type'])(
        config['optimizer']['learning_rate'])
    optimizer.setup(model_loss)
    # extra optimizer bells
    decay = config['optimizer']['weight_decay']
    if decay:
        optimizer.add_hook(ch.optimizer.WeightDecay(decay))
    grad_clip_val = config['optimizer']['grad_clip']
    if grad_clip_val:
        optimizer.add_hook(ch.optimizer.GradientHardClipping(-grad_clip_val, grad_clip_val))

    # training
    best_dev_score = 0
    n_dev_down = 0
    STATS = reset_stats({})
    fit_start = time.time()
    for batch in train_iter:
        STATS['epoch'] = train_iter.epoch

        # reset
        model_loss.cleargrads()
        model_loss.reset_state()

        # run model
        start = time.time()
        loss = model_loss(*zip(*batch), **loss_kwds)
        STATS['forward_times'].append(time.time()-start)
        loss_val = np.asscalar(loss.data)
        if v > 2:
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
        STATS['reports'].append(model_loss.report())
        # print STATS['reports'][-1]

        # validation routine
        if train_iter.is_new_epoch:
            dev_stats = model_evaluator.evaluate(dev_iter)
            STATS['dev_score'] = dev_stats['score']
            if v > 1:
                print_epoch_loss(train_iter.epoch,
                                 np.mean(STATS['epoch_losses']),
                                 dev_stats['score'],
                                 time=np.sum(STATS['forward_times']+STATS['backward_times']))
                print_f1_stats('Dev', dev_stats)
            STATS['dev_stats'].append(dev_stats)

            # save best
            if dev_stats['score'] >= best_dev_score:
                best_dev_score = dev_stats['score']
                n_dev_down = 0
                if v > 1:
                    print "Saving model to {} ...".format(save_prefix),
                model_loss.save_model(save_prefix)
                if v > 1:
                    print "Done"
            else:
                n_dev_down += 1
                if n_dev_down > config['train']['patience']:
                    if v > 1:
                        print "Stopping early"
                    break
            dump_stats(STATS, save_prefix+'train')
            STATS = reset_stats(STATS)
            if train_iter.epoch == config['train']['n_epoch']:
                break
    if v > 0:
        print "Training finished. {} epochs in {}".format(
              config['train']['n_epoch'], sec2hms(time.time()-fit_start))
