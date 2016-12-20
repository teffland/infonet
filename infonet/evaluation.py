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
    true_mentions = set(true_mentions)
    pred_mentions = set(pred_mentions)
    tp = len(true_mentions & pred_mentions)
    fn = len(true_mentions - pred_mentions)
    fp = len(pred_mentions - true_mentions)
    return tp, fp, fn

def mention_boundary_stats(true_ys, pred_ys, **kwds):
    """ Calculates the precision/recall of mention boundaries
    for entire dataset according to ACE evaluation
    """
    all_true_mentions = extract_all_mentions(true_ys, **kwds)
    all_pred_mentions = extract_all_mentions(pred_ys, **kwds)
    stats = {'tp':0,
             'fp':0,
             'fn':0}
    for true_mentions, pred_mentions in zip(all_true_mentions, all_pred_mentions):
        tp, fp, fn = mention_precision_recall(true_mentions, pred_mentions)
        stats['tp'] += tp
        stats['fp'] += fp
        stats['fn'] += fn
    stats['precision'] = tp / float(tp + fp +1e-15)
    stats['recall'] = tp / float(tp + fn +1e-15)
    stats['f1'] = 2*stats['precision']*stats['recall']/(stats['precision']+stats['recall']+1e-15)
    return stats
