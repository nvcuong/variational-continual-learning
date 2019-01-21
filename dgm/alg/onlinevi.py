import numpy as np
import tensorflow as tf
from helper_functions import *
import time

def get_q_theta_params():
    t_vars = tf.trainable_variables()
    var_list = [var for var in t_vars if 'gen_shared' in var.name]
    param_dict = {}
    for var in var_list:
        param_dict[var.name] = var	# make sure here is not a copy!
    return param_dict
    
def get_headnet_params(task):
    t_vars = tf.trainable_variables()
    var_list = [var for var in t_vars if 'gen_%d_head' % task in var.name]
    param_dict = {}
    for var in var_list:
        param_dict[var.name] = var	# make sure here is not a copy!
    return param_dict
    
def init_shared_prior():
    q_params = get_q_theta_params()
    prior_params = {}
    for name in q_params.keys():
        shape = q_params[name].get_shape().as_list()
        prior_params[name] = np.zeros(shape, dtype='f')
        
    return prior_params
    
def update_shared_prior(sess, prior_params):
    q_params = get_q_theta_params()
    for name in prior_params.keys():
        prior_params[name] = sess.run(q_params[name])
    
    return prior_params
    
def update_q_sigma(sess):
    q_params = get_q_theta_params()
    for name in q_params.keys():
        if 'log_sig' in name:
            shape = q_params[name].get_shape().as_list()
            sess.run(tf.assign(q_params[name], np.ones(shape)*-6))
    print 'reset the log sigma of q to -5'
    
def KL_param(shared_prior_params, task, regularise_headnet=False):
    # first get q params
    shared_q_params = get_q_theta_params()
    N_layer = len(shared_q_params.keys()) / 4	# one layer has for params
    # then compute kl between prior and q
    kl_total = 0.0
    # for the shared network
    for l in xrange(N_layer):
        suffices = ['W', 'b']
        for suffix in suffices:
            mu_q = shared_q_params['gen_shared_l%d_mu_' % l + suffix + ':0']
            log_sig_q = shared_q_params['gen_shared_l%d_log_sig_' % l + suffix + ':0']
            mu_p = shared_prior_params['gen_shared_l%d_mu_' %l + suffix + ':0']
            log_sig_p = shared_prior_params['gen_shared_l%d_log_sig_' % l + suffix + ':0']
            kl_total += tf.reduce_sum(KL(mu_q, log_sig_q, mu_p, log_sig_p))

    # for the head network
    if regularise_headnet:
        head_q_params = get_headnet_params(task)
        N_layer = len(head_q_params.keys()) / 4	# one layer has for params
        for l in xrange(N_layer):
            for suffix in ['W', 'b']:
                mu_q = shared_q_params['gen_head%d_l%d_mu_' % (task, l) + suffix + ':0']
                log_sig_q = shared_q_params['gen_head%d_l%d_log_sig_' % (task, l) + suffix + ':0']
                kl_total += tf.reduce_sum(KL(mu_q, log_sig_q, 0.0, 0.0))
            
    return kl_total

def lowerbound(x, enc, dec, ll, K = 1, mu_pz = 0.0, log_sig_pz = 0.0):
    mu_qz, log_sig_qz = enc(x)
    #z = sample_gaussian(mu_qz, log_sig_qz)
    kl_z = KL(mu_qz, log_sig_qz, mu_pz, log_sig_pz)
    if K > 1:
        print 'using K=%d theta samples for onlinevi' % K
    logp = 0.0
    for _ in xrange(K):
        # see bayesian_generator.py, tiling z does not work!
        z = sample_gaussian(mu_qz, log_sig_qz)	# sample different z
        mu_x = dec(z)	# sample different theta
        if ll == 'bernoulli':
            logp += log_bernoulli_prob(x, mu_x) / K
        if ll == 'l2':
            logp += log_l2_prob(x, mu_x) / K
        if ll == 'l1':
            logp += log_l1_prob(x, mu_x) / K
    return logp - kl_z

def construct_optimizer(X_ph, enc, dec, ll, N_data, batch_size_ph, shared_prior_params, task, K):

    # loss function
    bound = tf.reduce_mean(lowerbound(X_ph, enc, dec, ll, K))
    kl_theta = KL_param(shared_prior_params, task)
    loss_total = -bound + kl_theta / N_data
    batch_size = X_ph.get_shape().as_list()[0]

    # now construct optimizers
    lr_ph = tf.placeholder(tf.float32, shape=())
    t_vars = tf.trainable_variables()
    var_list = [var for var in t_vars if 'gen' in var.name]
    N_param = np.sum([np.prod(var.get_shape().as_list()) for var in var_list])
    opt = tf.train.AdamOptimizer(learning_rate=lr_ph).minimize(loss_total)
    
    ops = [opt, bound, kl_theta]
    def train(sess, X, lr):
        _, logp, kl = sess.run(ops, feed_dict={X_ph: X, lr_ph: lr,
                                    batch_size_ph: X.shape[0]})
        return logp, kl / N_param

    def fit(sess, X, n_iter, lr):
        N = X.shape[0]        
        print "training for %d epochs with lr=%.5f" % (n_iter, lr)
        begin = time.time()
        n_iter_vae = N / batch_size
        for iteration in xrange(1, n_iter + 1):
            ind_s = np.random.permutation(range(N))
            bound_total = 0.0
            kl_total = 0.0
            for j in xrange(0, n_iter_vae):
                indl = j * batch_size
                indr = (j+1) * batch_size
                ind = ind_s[indl:min(indr, N)]
                if indr > N:
                    ind = np.concatenate((ind, ind_s[:(indr-N)]))
                logp, kl = train(sess, X[ind], lr)    
                bound_total += logp / n_iter_vae
                kl_total += kl / n_iter_vae
            end = time.time()
            print "Iter %d, bound=%.2f, kl=%.2f, time=%.2f" \
                  % (iteration, bound_total, kl_total, end - begin)
            begin = end

    return fit

