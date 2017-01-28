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
    stats = {'tp':0, 'fp':0,'fn':0}
    stats['entity'] = {'tp':0, 'fp':0, 'fn':0}
    stats['event-anchor'] = {'tp':0, 'fp':0, 'fn':0}
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
                if 'entity' in t:
                    stats['entity']['tp'] += s[t]['tp']
                    stats['entity']['fp'] += s[t]['fp']
                    stats['entity']['fn'] += s[t]['fn']
                elif 'event-anchor' in t:
                    stats['event-anchor']['tp'] += s[t]['tp']
                    stats['event-anchor']['fp'] += s[t]['fp']
                    stats['event-anchor']['fn'] += s[t]['fn']
                else:
                    print "invalid m type found {}".format(t)
    stats['precision'] = stats['tp'] / float(stats['tp'] + stats['fp'] +1e-15)
    stats['recall'] = stats['tp'] / float(stats['tp'] + stats['fn'] +1e-15)
    stats['f1'] = 2*stats['precision']*stats['recall']/(stats['precision']+stats['recall']+1e-15)
    stats['support'] = stats['tp'] + stats['fp'] + stats['fn']
    for t in typeset | set(['entity', 'event-anchor']):
        stats[t]['precision'] = stats[t]['tp'] / float(stats[t]['tp'] + stats[t]['fp'] +1e-15)
        stats[t]['recall'] = stats[t]['tp'] / float(stats[t]['tp'] + stats[t]['fn'] +1e-15)
        stats[t]['f1'] = (2*stats[t]['precision']*stats[t]['recall']/
                         (stats[t]['precision']+stats[t]['recall']+1e-15))
        stats[t]['support'] = stats[t]['tp'] + stats[t]['fp'] + stats[t]['fn']
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

def mention_relation_stats(m_trues, m_preds, r_trues, r_preds):
        stats = {'tp':0, 'fp':0, 'fn':0}
        stats['node'] = {'tp':0, 'fp':0, 'fn':0}
        stats['entity'] = {'tp':0, 'fp':0, 'fn':0}
        stats['event-anchor'] = {'tp':0, 'fp':0, 'fn':0}
        stats['edge'] = {'tp':0, 'fp':0, 'fn':0}
        stats['relation'] = {'tp':0, 'fp':0, 'fn':0}
        stats['event-argument'] = {'tp':0, 'fp':0, 'fn':0}
        stats['coreference'] = {'tp':0, 'fp':0, 'fn':0}
        stats['NULL'] = {'tp':0, 'fp':0, 'fn':0}
        # nodes
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
            stats['node']['tp'] += len(tp)
            stats['node']['fp'] += len(fp)
            stats['node']['fn'] += len(fn)
        # # micro stats for all mentions
        # stats['node']['precision'] = stats['node']['tp'] / float(stats['node']['tp'] + stats['node']['fp'] + 1e-15)
        # stats['node']['recall'] = stats['node']['tp'] / float(stats['node']['tp'] + stats['node']['fn'] + 1e-15)
        # stats['node']['f1'] = 2*stats['node']['precision']*stats['node']['recall']/(stats['node']['precision']+stats['node']['recall']+1e-15)
        # stats['node']['support'] = stats['node']['tp'] + stats['node']['fp'] + stats['node']['fn']
        # micro stats by type
        # for t, s in stats.items():
        #     if type(s) is dict:
        #         stats[t]['precision'] = s['tp'] / float(s['tp'] + s['fp'] + 1e-15)
        #         stats[t]['recall'] = s['tp'] / float(s['tp'] + s['fn'] + 1e-15)
        #         stats[t]['f1'] = (2.*stats[t]['precision']*stats[t]['recall']/
        #                          (stats[t]['precision']+stats[t]['recall']+1e-15))
        #         stats[t]['support'] = s['tp'] + s['fp'] + s['fn']

        # edges
        # NOTE: The evaluation considers an edge correct
        #       if its type and constituent spans are correct
        #       It does NOT consider the correctness of the classes of the constituent mentions.
        # TODO: This might not be the correct evaluation wrt Li and Ji or Yang and Mitchell
        for r_pred, r_true in zip(r_preds, r_trues):
            r_pred = set(r_pred)
            r_true = set(r_true)
            tp = r_pred & r_true
            fp = r_pred - r_true
            fn = r_true - r_pred

            for r in tp:
                rtype = r[4]
                # print 'tp msubstype:', msubtype
                if rtype not in stats:
                    stats[rtype] = {'tp':0, 'fp':0, 'fn':0}
                stats[rtype]['tp'] += 1
                # stats by node-type
                if 'relation' in rtype:
                    # print 'tp entity'
                    stats['relation']['tp'] += 1
                elif 'event-argument' in rtype:
                    # print 'tp event-anchor'
                    stats['event-argument']['tp'] += 1
                elif 'coreference' in rtype:
                    # print 'tp event-anchor'
                    stats['coreference']['tp'] += 1
                elif 'NULL' in rtype:
                    # print 'tp event-anchor'
                    stats['NULL']['tp'] += 1
                else:
                    print "invalid r type found in tp {}".format(rtype)
            for r in fp:
                rtype = r[4]
                # print 'tp msubstype:', msubtype
                if rtype not in stats:
                    stats[rtype] = {'tp':0, 'fp':0, 'fn':0}
                stats[rtype]['fp'] += 1
                # stats by node-type
                if 'relation' in rtype:
                    # print 'tp entity'
                    stats['relation']['fp'] += 1
                elif 'event-argument' in rtype:
                    # print 'tp event-anchor'
                    stats['event-argument']['fp'] += 1
                elif 'coreference' in rtype:
                    # print 'tp event-anchor'
                    stats['coreference']['fp'] += 1
                elif 'NULL' in rtype:
                    # print 'tp event-anchor'
                    stats['NULL']['fp'] += 1
                else:
                    print "invalid r type found in fp {}".format(rtype)
            for r in fn:
                rtype = r[4]
                # print 'tp msubstype:', msubtype
                if rtype not in stats:
                    stats[rtype] = {'tp':0, 'fp':0, 'fn':0}
                stats[rtype]['fn'] += 1
                # stats by node-type
                if 'relation' in rtype:
                    # print 'tp entity'
                    stats['relation']['fn'] += 1
                elif 'event-argument' in rtype:
                    # print 'tp event-anchor'
                    stats['event-argument']['fn'] += 1
                elif 'coreference' in rtype:
                    # print 'tp event-anchor'
                    stats['coreference']['fn'] += 1
                elif 'NULL' in rtype:
                    # print 'tp event-anchor'
                    stats['NULL']['fn'] += 1
                else:
                    print "invalid r type found in fn {}".format(rtype)

            # tp, fp, fn regardless of type
            stats['tp'] += len(tp)
            stats['fp'] += len(fp)
            stats['fn'] += len(fn)
            stats['edge']['tp'] += len(tp)
            stats['edge']['fp'] += len(fp)
            stats['edge']['fn'] += len(fn)
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
