import numpy as np
import tensorflow as tf
from helper_functions import *
import time
    
def update_si_reg(sess, si_reg, new_params, old_params, w_params):
    if si_reg is None:
        si_reg = [0.0 for i in xrange(len(old_params))]
    for i in range(len(old_params)):
        delta = new_params[i] - old_params[i]
        si_reg[i] += w_params[i] / (delta ** 2 + 1e-6)
        w_params[i] *= 0.0	# reset things for next task                             
    return si_reg, w_params
    
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

def construct_optimizer(X_ph, batch_size_ph, bound, N_data, si_reg, old_params, lbd=1.0):

    t_vars = tf.trainable_variables()
    var_list = [var for var in t_vars if 'gen' in var.name]
    var_list = var_list + [var for var in t_vars if 'enc' in var.name]
    shared_var_list = [var for var in t_vars if 'gen_shared' in var.name]

    # loss function
    bound = tf.reduce_mean(bound)
    loss_task = -bound #* N_data
    si_loss = 0.0
    if old_params is not None:
        for i in xrange(len(si_reg)):
            si_loss += 0.5 * lbd * tf.reduce_sum(si_reg[i]*(shared_var_list[i] - old_params[i])**2)
    loss_total = (loss_task + si_loss) #/ N_data
    batch_size = X_ph.get_shape().as_list()[0]

    # now construct optimizers
    lr_ph = tf.placeholder(tf.float32, shape=())    
    opt = tf.train.AdamOptimizer(learning_rate=lr_ph).minimize(loss_total, \
                                                     var_list=var_list)
    
    # check eq. (3) in the SI paper
    grad_shared = tf.gradients(-loss_task, shared_var_list)
    ops = [opt, bound, grad_shared]
    def train(sess, X, lr, old_params, w_params):
        _, logp, grad = sess.run(ops, feed_dict={X_ph: X, lr_ph: lr,
                                                 batch_size_ph: X.shape[0]})
        new_params = sess.run(shared_var_list)
        for i in xrange(len(w_params)):
            w_params[i] += grad[i] * (new_params[i] - old_params[i])
        return logp, new_params, w_params

    def fit(sess, X, n_iter, lr, w_params = None):
        N = X.shape[0]        
        print "training for %d epochs with lr=%.5f" % (n_iter, lr)
        begin = time.time()
        n_iter_vae = N / batch_size
        old_params = sess.run(shared_var_list)
        if w_params is None:
            w_params = [np.zeros(old_params[i].shape) \
                        for i in xrange(len(old_params))]        
        for iteration in xrange(1, n_iter + 1):
            ind_s = np.random.permutation(range(N))
            bound_total = 0.0
            for j in xrange(0, n_iter_vae):
                indl = j * batch_size
                indr = (j+1) * batch_size
                ind = ind_s[indl:min(indr, N)]
                if indr > N:
                    ind = np.concatenate((ind, ind_s[:(indr-N)]))
                logp, old_params, w_params = train(sess, X[ind], lr, old_params, w_params)    
                bound_total += logp / n_iter_vae
            end = time.time()
            print "Iter %d, bound=%.2f, time=%.2f" \
                  % (iteration, bound_total, end - begin)
            begin = end
            
        return old_params, w_params

    return fit, shared_var_list

