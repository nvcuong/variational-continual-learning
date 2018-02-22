import numpy as np
import tensorflow as tf
import time
import keras

def KL_generated_images(dec, cla, N, dimZ, task, sample_W = True):
    z = tf.random_normal(shape=(N, dimZ))
    x_gen = dec(z, sampling = sample_W)
    y_gen = tf.clip_by_value(cla(x_gen), 1e-9, 1.0)
    y_true = np.zeros([N, 10]); y_true[:, task] = 1
    y_true = tf.constant(np.asarray(y_true, dtype='f'))
    kl = -tf.reduce_sum(y_true * tf.log(y_gen), 1)
    kl_mean = tf.reduce_mean(kl)
    kl_var = tf.reduce_mean((kl - kl_mean)**2)
    return kl_mean, kl_var
    
def construct_eval_func(dec, cla, batch_size_ph, dimZ, task, sample_W = True):
    N_gen = 100
    kl_mean, kl_var = KL_generated_images(dec, cla, N_gen, dimZ, task, sample_W)
    ops = [kl_mean, kl_var]
    
    def eval(sess):      
        n_iter = 10
        N = n_iter * N_gen
        begin = time.time()
        kl_total = 0.0; kl_var = 0.0
        for j in xrange(0, n_iter):
            a, b = sess.run(ops, feed_dict={batch_size_ph: N_gen,
                                            keras.backend.learning_phase(): 0})   
            kl_total += a / n_iter
            kl_var += b / n_iter
        end = time.time()
        print "kl=%.2f, ste=%.2f, time=%.2f" \
                  % (kl_total, np.sqrt(kl_var / N), end - begin)
        return kl_total, np.sqrt(kl_var / N)
        
    return eval
