""" Implements a mixin class that reports useful training statistics """
import numpy as np
import chainer as ch

class ReporterMixin():
    """ TODO: Document this class.

    NOTE: Always init this superclass after the Link or Chain superclass
    """
    def __init__(self):
        param_sizes = { 'total':0 }
        for link in self.links(skipself=True):
            if issubclass(type(link), ch.Chain): continue # drill down to links only
            param_sizes[link.name] = 0
            for param in link.params():
                d = '{}/{}'.format(link.name, param.name)
                size = param.data.size
                param_sizes[d] = size
                param_sizes[link.name] += size
                param_sizes['total'] += size
        self.param_sizes = param_sizes

    def report(self):
        summary = {}
        for name, link in self.namedlinks(skipself=True):
            if issubclass(type(link), ch.Chain): continue # drill down to links only
            for param in link.params():
                d = '{}/{}/{}'.format(link.name, param.name, 'data')
                summary[d] = self.calc_stats(param.data)
                g = '{}/{}/{}'.format(link.name, param.name, 'grad')
                summary[g] =  self.calc_stats(param.grad)
        return summary


    def calc_stats(self, matrix):
        stats = {}
        stats['n'] = matrix.size
        stats['norm'] = np.asscalar(np.linalg.norm(matrix))
        stats['mean'] = np.asscalar(np.mean(matrix))
        stats['stdev'] = np.asscalar(np.std(matrix))
        stats['min'] = np.asscalar(np.min(matrix))
        stats['max'] = np.asscalar(np.max(matrix))
        stats['bin_counts'], stats['bins'] = self._calc_bins(matrix)
        return stats

    def _calc_bins(self,
                   matrix,
                   bins=[-1e10, -1e4, -1e3, -1e2, -1e1, -1e0, -1e-1, -1e-2,
                         -1e-3, -1e-4, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1,
                         1e2, 1e3, 1e4, 1e10]):
        counts, bins = np.histogram(matrix[~np.isnan(matrix)], bins)
        return counts.tolist(), bins.tolist()
