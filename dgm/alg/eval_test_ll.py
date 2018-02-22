import numpy as np
import tensorflow as tf
from helper_functions import *
import time

def IS_estimate(x, enc, dec, ll, K = 100, sample_W = True, mu_pz = 0.0, log_sig_pz = 0.0):
    if len(x.get_shape().as_list()) == 4:
        x_rep = tf.tile(x, [K, 1, 1, 1])
    if len(x.get_shape().as_list()) == 2:
        x_rep = tf.tile(x, [K, 1])
    N = x.get_shape().as_list()[0]
    mu_qz, log_sig_qz = enc(x_rep)
    z = sample_gaussian(mu_qz, log_sig_qz)
    mu_x = dec(z, sampling = sample_W)
    log_prior = log_gaussian_prob(z)
    logq = log_gaussian_prob(z, mu_qz, log_sig_qz)
    kl_z = logq - log_prior
    if ll == 'bernoulli':
        logp = log_bernoulli_prob(x_rep, mu_x)
    if ll == 'l2':
        logp = log_l2_prob(x_rep, mu_x)
    if ll == 'l1':
        logp = log_l1_prob(x_rep, mu_x)
    bound = tf.reshape(logp - kl_z, [K, N])
    bound_max = tf.reduce_max(bound, 0)
    bound -= bound_max
    log_norm = tf.log(tf.clip_by_value(tf.reduce_mean(tf.exp(bound), 0), 1e-9, np.inf))
    test_ll = log_norm + bound_max
    test_ll_mean = tf.reduce_mean(test_ll)
    test_ll_var = tf.reduce_mean((test_ll - test_ll_mean)**2)
    
    return test_ll_mean, test_ll_var
    
def construct_eval_func(X_ph, enc, dec, ll, batch_size_ph, K = 100, sample_W = True):
    test_ll_mean, test_ll_var = IS_estimate(X_ph, enc, dec, ll, K, sample_W)
    batch_size = X_ph.get_shape().as_list()[0]
    ops = [test_ll_mean, test_ll_var]
    
    def eval(sess, X):
        N = X.shape[0]        
        n_iter_vae = N / batch_size
        bound_total = 0.0; bound_var = 0.0
        begin = time.time()
        for j in xrange(0, n_iter_vae):
            indl = j * batch_size
            indr = min((j+1) * batch_size, N)
            logp_mean, logp_var = sess.run(ops, feed_dict={X_ph: X[indl:indr], 
                                     batch_size_ph: K*batch_size})   
            bound_total += logp_mean / n_iter_vae
            bound_var += logp_var / n_iter_vae
        end = time.time()
        print "test_ll=%.2f, ste=%.2f, time=%.2f" \
                  % (bound_total, np.sqrt(bound_var / N), end - begin)
        return bound_total, np.sqrt(bound_var / N)
        
    return eval
