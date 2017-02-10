""" Implementation of bipartite crf with approximate inference """
import numpy as np
import numpy.random as npr
import chainer as ch
import chainer.functions as F
import chainer.links as L

class BipartiteCRF(ch.Link):
    def __init__(self,
                 n_mention_class, n_relation_class,
                 n_mention_feature, n_relation_feature):
        super(BipartiteCRF, self).__init__(
            mention_cost=(n_mention_feature, n_mention_class),
            mention_bias=(n_mention_class,),
            relation_cost=(n_relation_feature, n_relation_class),
            relation_bias=(n_relation_class,),
            trans_cost=(n_mention_class, n_relation_class)
        )
        self.n_mention_class = n_mention_class
        self.n_relation_class = n_relation_class
        self.n_mention_feature = n_mention_feature
        self.n_relation_feature = n_relation_feature

        W_init = ch.initializers.Uniform()
        W_init(self.mention_cost.data)
        W_init(self.relation_cost.data)
        W_init(self.trans_cost.data)

        b_init = ch.initializers.Zero()
        b_init(self.mention_bias.data)
        b_init(self.relation_bias.data)

    def calc_mention_scores(self, mention_features):
        return F.bias(F.matmul(mention_features, self.mention_cost),
                      self.mention_bias)

    def calc_relation_scores(self, relation_features):
        return F.bias(F.matmul(relation_features, self.relation_cost),
                      self.relation_bias)

    def calc_trans_scores(self, R):
        # repeat the trans score matrix for each joint factor (2R of them)
        return F.broadcast_to(self.trans_cost, (2*R,)+self.trans_cost.shape)

    def __call__(self,
                 mention_features, relation_features,
                 mention_nbrs, relation_nbrs):
        # sizes of mention and relation layers
        M, R = mention_features.shape[0], relation_features.shape[0]

        # compute log factor scores
        log_phi_m = F.log(self.calc_mention_scores(mention_features))
        log_phi_r = F.log(self.calc_relation_scores(relation_features))
        log_phi_mr = F.log(self.calc_trans_scores(R))

        # initialize marginal approximations to uniform
        q_m = ch.Variable(np.ones((M, self.n_mention_class))/float(self.n_mention_class))
        q_r = ch.Variable(np.ones((M, self.n_relation_class))/float(self.n_relation_class))

        # do approximate inference
        q_m, q_r = self.mean_field(q_m, q_r,
                                   log_phi_m, log_phi_r, log_phi_mr,
                                   mention_nbrs, relation_nbrs)

    def mean_field(self, q_m, q_r,
                   log_phi_m, log_phi_r, log_phi_mr,
                   mention_nbrs, relation_nbrs,
                   t_max=100, tol=.01):
        M, R = log_phi_m.shape[0], log_phi_r.shape[0]
        Cm, Cr = self.n_mention_class, self.n_relation_class
        flat_log_phi_mr = F.reshape(log_phi_mr, (-1, Cm*Cr))
        t, avg_diff = 0, 1e15
        q_mt_prev = q_m
        q_rt_prev = q_r
        while t < t_max and avg_diff > tol:
            print 'Mean field: T:{0}, Diff:{1:2.4f}'.format(t, avg_diff)
            # calculate next q_m
            q_mt = log_phi_m #F.identity(log_phi_m)
            for m_nbr in mention_nbrs:
                q_mt, q_mt_rest = F.split_axis(q_mt, [m_nbr.shape[0]], axis=0)
                nbr_phis = F.reshape(F.embed_id(m_nbr, flat_log_phi_mr),
                                     (-1, Cm, Cr))
                b_q_mt = F.broadcast_to(q_mt_prev, nbr_phis.shape)
                q_mt += F.sum(b_q_mt * nbr_phis, axis=2)
                q_mt = F.vstack([q_mt, q_mt_rest])
            q_mt = F.softmax(q_mt)

            # calculate next q_r
            q_rt = log_phi_r
            for r_nbr in relation_nbrs:
                nbr_phis = F.reshape(F.embed_id(r_nbr, flat_log_phi_mr),
                                     (-1, Cm, Cr))
                b_q_rt = F.broadcast_to(q_rt_prev, nbr_phis.shape)
                q_rt += F.sum(b_q_rt * nbr_phis, axis=1)
            q_rt = F.softmax(q_rt)

            # average absolute difference between marginals
            avg_diff = np.asscalar(np.mean(np.hstack([
                        np.absolute(q_mt.data-q_mt_prev.data).reshape(-1),
                        np.absolute(q_rt.data-q_rt_prev.data).reshape(-1)])))
            q_mt_prev = q_mt
            q_rt_prev = q_rt
            t += 1
        return q_mt, q_rt


class ApproxBipartiteCRF(F.Function):
    """ This function computes the probability of a labeled bipartite graph
    under an approximation.

    Its derivative is the approximation to the negative conditional log-likelihood
    of the bipartite graph.

    Note that this derivative is NOT the true derivative of the approximate log-likelihood.
    It is an approximation of the gradient of the true log-likelihood.
    """
    def forward_cpu(self, ):
        pass

    def backward_cpu(self):
