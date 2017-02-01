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
                 mention_neighbors, relation_neighbors):
        # sizes of mention and relation layers
        M, R = mention_features.shape[0], relation_features.shape[0]

        # compute log factor scores
        log_phi_m = F.log(self.calc_mention_scores(mention_features))
        log_phi_r = F.log(self.calc_relation_scores(relation_features))
        log_phi_mr = F.log(self.calc_trans_scores(R))

        # initialize marginal approximations
        q_m = ch.Variable(np.zeros((M, self.n_mention_class)))
        q_r = ch.Variable(np.zeros((M, self.n_mention_class)))


    def mean_field(self, q_m, q_r,
                   log_phi_m, log_phi_r, log_phi_mr,
                   mention_neighbors, relation_neighbors,
                   t_max=100, tol=.01):
        M, R = log_phi_m.shape[0], log_phi_r.shape[0]
        t = 0
        while t < t_max and avg_diff > tol:
            # calculate next q_m
            q_mt = log_phi_m #F.identity(log_phi_m)
