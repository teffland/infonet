import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
sb.set_color_codes()

from tagger import extract_all_mentions

def plot_learning_curve(epoch_losses, valid_losses,
                        logx=False,
                        logy=False,
                        title='Training Learning Curve',
                        xlabel='Epoch',
                        ylabel='Loss',
                        figsize=(6,4),
                        savename=None):
    fig, ax = plt.subplots(1, figsize=figsize)
    train_means = np.array([ np.mean(epoch_loss) for epoch_loss in epoch_losses ])
    train_stds = np.array([ np.std(epoch_loss) for epoch_loss in epoch_losses ])
    t = xrange(len(epoch_losses))
    ax.plot(t, train_means, 'bo-', lw=2,label='Average Per Epoch Training Loss', markersize=1)
    ax.fill_between(t, train_means+train_stds, train_means-train_stds, facecolor='b', alpha=0.15)
    ax.plot(range(len(valid_losses)), valid_losses, 'go--', lw=2, label='Per Epoch Validation Loss', markersize=1)
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')
    if savename:
        fig.savefig(savename)
    return fig, ax

def mention_precision_recall(true_mentions, pred_mentions):
    """ This function returns the counts of true positives, false positives, and false negatives
    which are necessary for calculating precision and recall.
    A mention boundary is considered correct if both ends are correct.
    """
    typeset = set([m[2] for m in true_mentions+pred_mentions])
    true_mentions = set(true_mentions)
    pred_mentions = set(pred_mentions)
    stats = dict(
        tp = len(true_mentions & pred_mentions),
        fn = len(true_mentions - pred_mentions),
        fp = len(pred_mentions - true_mentions)
    )
    for t in typeset:
        trues = { m for m in true_mentions if m[2] == t }
        preds = { m for m in pred_mentions if m[2] == t }
        stats[t] = dict(
            tp = len(trues & preds),
            fn = len(trues - preds),
            fp = len(preds - trues)
        )
    return stats

def mention_boundary_stats(true_ys, pred_ys, **kwds):
    """ Calculates the precision/recall of mention boundaries
    for entire dataset according to ACE evaluation
    """
    all_true_mentions = extract_all_mentions(true_ys, **kwds)
    all_pred_mentions = extract_all_mentions(pred_ys, **kwds)
    typeset = { m[2] for seq_mentions in all_true_mentions+all_pred_mentions
                     for m in seq_mentions }
    # print typeset
    # for t in typeset:
    #     print t, len({m for seq_mentions in all_true_mentions+all_pred_mentions
    #                     for m in seq_mentions if m[2] == t})
    stats = {'tp':0,
             'fp':0,
             'fn':0}
    for t in typeset:
        stats[t] = {'tp':0,
                    'fp':0,
                    'fn':0}
    for true_mentions, pred_mentions in zip(all_true_mentions, all_pred_mentions):
        s = mention_precision_recall(true_mentions, pred_mentions)
        stats['tp'] += s['tp']
        stats['fp'] += s['fp']
        stats['fn'] += s['fn']
        for t in typeset:
            if t in s:
                stats[t]['tp'] += s[t]['tp']
                stats[t]['fp'] += s[t]['fp']
                stats[t]['fn'] += s[t]['fn']

    stats['precision'] = stats['tp'] / float(stats['tp'] + stats['fp'] +1e-15)
    stats['recall'] = stats['tp'] / float(stats['tp'] + stats['fn'] +1e-15)
    stats['f1'] = 2*stats['precision']*stats['recall']/(stats['precision']+stats['recall']+1e-15)
    for t in typeset:
        stats[t]['precision'] = stats[t]['tp'] / float(stats[t]['tp'] + stats[t]['fp'] +1e-15)
        stats[t]['recall'] = stats[t]['tp'] / float(stats[t]['tp'] + stats[t]['fn'] +1e-15)
        stats[t]['f1'] = (2*stats[t]['precision']*stats[t]['recall']/
                         (stats[t]['precision']+stats[t]['recall']+1e-15))
    return stats

def mention_stats(m_preds, m_trues):
    stats = {'tp':0, 'fp':0, 'fn':0}
    stats['entity'] = {'tp':0, 'fp':0, 'fn':0}
    stats['event-anchor'] = {'tp':0, 'fp':0, 'fn':0}
    for m_pred, m_true in zip(m_preds, m_trues):
        m_pred = set(m_pred)
        m_true = set(m_true)
        tp = m_pred & m_true
        fp = m_pred - m_true
        fn = m_true - m_pred

        for m in tp:
            msubtype = m[2]
            # print 'tp msubstype:', msubtype
            if msubtype not in stats:
                stats[msubtype] = {'tp':0, 'fp':0, 'fn':0}
            stats[msubtype]['tp'] += 1
            # stats by node-type
            if 'entity' in msubtype:
                # print 'tp entity'
                stats['entity']['tp'] += 1
            elif 'event-anchor' in msubtype:
                # print 'tp event-anchor'
                stats['event-anchor']['tp'] += 1
            else:
                print "invalid m type found {}".format(msubtype)
        for m in fp:
            msubtype = m[2]
            # print 'fp msubstype:', msubtype
            if msubtype not in stats:
                stats[msubtype] = {'tp':0, 'fp':0, 'fn':0}
            stats[msubtype]['fp'] += 1
            # stats by node-type
            if 'entity' in msubtype:
                # print 'fp entity'
                stats['entity']['fp'] += 1
            elif 'event-anchor' in msubtype:
                # print 'fp event-anchor'
                stats['event-anchor']['fp'] += 1
            else:
                print "invalid m type found {}".format(msubtype)
        for m in fn:
            msubtype = m[2]
            # print 'fn msubstype:', msubtype
            if msubtype not in stats:
                stats[msubtype] = {'tp':0, 'fp':0, 'fn':0}
            stats[msubtype]['fn'] += 1
            # stats by node-type
            if 'entity' in msubtype:
                # print 'fn entity'
                stats['entity']['fn'] += 1
            elif 'event-anchor' in msubtype:
                # print 'fn event-anchor'
                stats['event-anchor']['fn'] += 1
            else:
                print "invalid m type found {}".format(msubtype)

        # tp, fp, fn regardless of type
        stats['tp'] += len(tp)
        stats['fp'] += len(fp)
        stats['fn'] += len(fn)
    # micro stats for all
    stats['precision'] = stats['tp'] / float(stats['tp'] + stats['fp'] + 1e-15)
    stats['recall'] = stats['tp'] / float(stats['tp'] + stats['fn'] + 1e-15)
    stats['f1'] = 2*stats['precision']*stats['recall']/(stats['precision']+stats['recall']+1e-15)
    stats['support'] = stats['tp'] + stats['fp'] + stats['fn']
    # micro stats by type
    for t, s in stats.items():
        if type(s) is dict:
            stats[t]['precision'] = s['tp'] / float(s['tp'] + s['fp'] + 1e-15)
            stats[t]['recall'] = s['tp'] / float(s['tp'] + s['fn'] + 1e-15)
            stats[t]['f1'] = (2.*stats[t]['precision']*stats[t]['recall']/
                             (stats[t]['precision']+stats[t]['recall']+1e-15))
            stats[t]['support'] = s['tp'] + s['fp'] + s['fn']
    return stats
