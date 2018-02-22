import numpy as np
import tensorflow as tf
from helper_functions import *
import time

def compute_fisher(X_ph, batch_size_ph, bound, N_data):
    """
    Copy-paste from Cuong's EWC implementation
    """
    # initialize Fisher information for most recent task
    t_vars = tf.trainable_variables()
    var_list = [var for var in t_vars if 'gen_shared' in var.name]
    F_accum = [0.0 for v in range(len(var_list))]
        
    # in this case we only have lower-bounds of log p,
    # so the Fisher information matrix can be inaccurate!
    # assume lowerbound is of shape (batch_size)
    batch_size = X_ph.get_shape().as_list()[0]
    for i in xrange(batch_size):
        grad_square = tf.gradients(bound[i], var_list)
        for v in range(len(F_accum)):
            F_accum[v] += grad_square[v]**2 / batch_size
    
    # correct scaling (in a Bayesian sense)        
    for v in range(len(F_accum)):
        F_accum[v] *= N_data
    
    def fisher(sess, x):
        return sess.run(F_accum, feed_dict={X_ph: x,
                                 batch_size_ph: x.shape[0]})
    
    return fisher, var_list

def init_fisher_accum():
    t_vars = tf.trainable_variables()
    var_list = [var for var in t_vars if 'gen_shared' in var.name]
    F_accum = [0.0 for v in range(len(var_list))]
    return F_accum
    
def update_laplace_loss(sess, F_accum_old, var_list, fisher, lbd, x):
    old_var_list = sess.run(var_list)
    F_accum = fisher(sess, x)
    loss = 0.0
    F_accum_new = []
    for v in range(len(F_accum)):
        hessian = F_accum[v] + F_accum_old[v]
        loss += 0.5 * lbd * tf.reduce_sum(hessian * \
                              (var_list[v] - old_var_list[v])**2)
        F_accum_new.append(hessian)
    return loss, F_accum_new
    
def extract_old_var(sess):
    t_vars = tf.trainable_variables()
    var_list = [var for var in t_vars if 'gen_shared' in var.name]
    return sess.run(var_list)

def lowerbound(x, enc, dec, ll, mu_pz = 0.0, log_sig_pz = 0.0):
    mu_qz, log_sig_qz = enc(x)
    z = sample_gaussian(mu_qz, log_sig_qz)
    mu_x = dec(z)
    kl_z = KL(mu_qz, log_sig_qz, mu_pz, log_sig_pz)
    if ll == 'bernoulli':
        logp = log_bernoulli_prob(x, mu_x)
    if ll == 'l2':
        logp = log_l2_prob(x, mu_x)
    if ll == 'l1':
        logp = log_l1_prob(x, mu_x)
    return logp - kl_z

def construct_optimizer(X_ph, batch_size_ph, bound, N_data, laplace_loss):

    # loss function
    bound = tf.reduce_mean(bound)
    loss_total = -bound + laplace_loss / N_data
    batch_size = X_ph.get_shape().as_list()[0]

    # now construct optimizers
    lr_ph = tf.placeholder(tf.float32, shape=())
    t_vars = tf.trainable_variables()
    var_list = [var for var in t_vars if 'gen' in var.name]
    var_list = var_list + [var for var in t_vars if 'enc' in var.name]
    opt = tf.train.AdamOptimizer(learning_rate=lr_ph).minimize(loss_total, \
                                                     var_list=var_list)
    
    ops = [opt, bound]
    def train(sess, X, lr):
        _, logp = sess.run(ops, feed_dict={X_ph: X, lr_ph: lr, 
                                batch_size_ph: X.shape[0]})
        return logp

    def fit(sess, X, n_iter, lr):
        N = X.shape[0]        
        print "training for %d epochs with lr=%.5f" % (n_iter, lr)
        begin = time.time()
        n_iter_vae = N / batch_size
        for iteration in xrange(1, n_iter + 1):
            ind_s = np.random.permutation(range(N))
            bound_total = 0.0
            for j in xrange(0, n_iter_vae):
                indl = j * batch_size
                indr = (j+1) * batch_size
                ind = ind_s[indl:min(indr, N)]
                if indr > N:
                    ind = np.concatenate((ind, ind_s[:(indr-N)]))
                logp = train(sess, X[ind], lr)    
                bound_total += logp / n_iter_vae
            end = time.time()
            print "Iter %d, bound=%.2f, time=%.2f" \
                  % (iteration, bound_total, end - begin)
            begin = end

    return fit

