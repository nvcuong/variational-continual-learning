import numpy as np
import tensorflow as tf

def sample_gaussian(mu, log_sig):
    return mu + tf.exp(log_sig) * tf.random_normal(mu.get_shape())

def sample_bernoulli(mu):
    shape = mu.get_shape()
    return tf.where(tf.random_uniform(shape) < mu, tf.ones(shape), tf.zeros(shape))

# define log densities
def log_gaussian_prob(x, mu=0.0, log_sig=0.0):
    logprob = -(0.5 * np.log(2 * np.pi) + log_sig) \
                - 0.5 * ((x - mu) / tf.exp(log_sig)) ** 2
    ind = list(range(1, len(x.get_shape().as_list())))
    return tf.reduce_sum(logprob, ind) 

def log_laplace_prob(x, mu=0.0, log_sig=0.0):
    logprob = - log_sig - tf.abs(x - mu) / tf.exp(log_sig) #- np.log(2.0)
    ind = list(range(1, len(x.get_shape().as_list())))
    return tf.reduce_sum(logprob, ind) 
    
def log_bernoulli_prob(x, p=0.5):
    logprob = x * tf.log(tf.clip_by_value(p, 1e-9, 1.0)) \
              + (1 - x) * tf.log(tf.clip_by_value(1.0 - p, 1e-9, 1.0))
    ind = list(range(1, len(x.get_shape().as_list())))
    return tf.reduce_sum(logprob, ind)
    
def log_l2_prob(x, mu):
    logprob = -(x - mu) ** 2
    ind = list(range(1, len(x.get_shape().as_list())))
    return tf.reduce_sum(logprob, ind)
    
def log_l1_prob(x, mu):
    logprob = -tf.abs(x - mu)
    ind = list(range(1, len(x.get_shape().as_list())))
    return tf.reduce_sum(logprob, ind) 

def KL(mu_p, log_sig_p, mu_q, log_sig_q):
    # compute KL[p||q]
    precision_q = tf.exp(-2*log_sig_q)
    kl = 0.5 * (mu_p - mu_q)**2 * precision_q - 0.5
    kl += log_sig_q - log_sig_p
    kl += 0.5 * tf.exp(2 * log_sig_p - 2 * log_sig_q)
    ind = list(range(1, len(mu_p.get_shape().as_list())))
    return tf.reduce_sum(kl, ind)

def log_logistic_cdf_prob(x, mu, log_scale):
    binsize = np.asarray(1/256.0, dtype='f')
    scale = tf.exp(log_scale)
    sample = (tf.floor(x / binsize) * binsize - mu) / scale
    
    logprob = tf.log(1 - tf.exp(-binsize / scale)) 
    logprob -= tf.nn.softplus(sample)
    logprob -= tf.nn.softplus(-sample - binsize/scale)
    ind = list(range(1, len(x.get_shape().as_list())))
    return tf.reduce_sum(logprob, ind)

def log_logit_normal_prob(x, mu, log_sig):
    x_logit = tf.log(tf.clip_by_value(x, 1e-5, 1.0 - 1e-5)) \
               - tf.log(tf.clip_by_value(1 - x, 1e-5, 1.0 - 1e-5))
    logprob = -(0.5 * np.log(2 * np.pi) + log_sig) \
                - 0.5 * ((x_logit - mu) / tf.exp(log_sig)) ** 2
    # the following part is a constant wrt params
    #logprob -= tf.log(tf.clip_by_value(x*(1 - x), 1e-5, 1.0))
    ind = list(range(1, len(x.get_shape().as_list())))
    return tf.reduce_sum(logprob, ind)

def log_logistic_prob(x, mu, log_scale):
    s = (x - mu) / tf.exp(g_scale)
    logprob = -s - log_scale - 2 * tf.nn.softplus(-s)
    ind = list(range(1, len(x.get_shape().as_list())))
    return tf.reduce_sum(logprob, ind)
