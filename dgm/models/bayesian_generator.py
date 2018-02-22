import numpy as np
import tensorflow as tf
from mlp import init_weights

"""
A Bayesian MLP generator
"""

def sample_gaussian(mu, log_sig):
    return mu + tf.exp(log_sig) * tf.random_normal(mu.get_shape())

def bayesian_mlp_layer(d_in, d_out, activation, name):
    mu_W = tf.Variable(init_weights(d_in, d_out), name = name+'_mu_W')
    mu_b = tf.Variable(tf.zeros([d_out]), name = name+'_mu_b')
    log_sig_W = tf.Variable(tf.ones([d_in, d_out])*-6, name = name+'_log_sig_W')
    log_sig_b = tf.Variable(tf.ones([d_out])*-6, name = name+'_log_sig_b')
    
    def apply_layer(x, sampling=True):
        if sampling:
            W = sample_gaussian(mu_W, log_sig_W)
            b = sample_gaussian(mu_b, log_sig_b)
        else:
            print 'use mean of q(theta)...'
            W = mu_W; b = mu_b
        a = tf.matmul(x, W) + b
        if activation == 'relu':
            return tf.nn.relu(a)
        if activation == 'sigmoid':
            return tf.nn.sigmoid(a)
        if activation == 'linear':
            return a  
            
    return apply_layer

def generator_head(dimZ, dimH, n_layers, name):
    fc_layer_sizes = [dimZ] + [dimH for i in xrange(n_layers)]
    layers = []
    N_layers = len(fc_layer_sizes) - 1
    for i in xrange(N_layers):
        d_in = fc_layer_sizes[i]; d_out = fc_layer_sizes[i+1]
        name_layer = name + '_head_l%d' % i
        layers.append(bayesian_mlp_layer(d_in, d_out, 'relu', name_layer))
    
    print 'decoder head MLP of size', fc_layer_sizes
    
    def apply(x, sampling=True):
        for layer in layers:
            x = layer(x, sampling)
        return x
        
    return apply

def generator_shared(dimX, dimH, n_layers, last_activation, name):
    # now construct a decoder
    fc_layer_sizes = [dimH for i in xrange(n_layers)] + [dimX]
    layers = []
    N_layers = len(fc_layer_sizes) - 1
    for i in xrange(N_layers):
        d_in = fc_layer_sizes[i]; d_out = fc_layer_sizes[i+1]
        if i < N_layers - 1:
            activation = 'relu'
        else:
            activation = last_activation
        name_layer = name + '_shared_l%d' % i
        layers.append(bayesian_mlp_layer(d_in, d_out, activation, name_layer))
    
    print 'decoder shared MLP of size', fc_layer_sizes
    
    def apply(x, sampling=True):
        for layer in layers:
            x = layer(x, sampling)
        return x
        
    return apply
    
def generator(head_net, shared_net):
    def apply(x, sampling=True):
        x = head_net(x, sampling)
        x = shared_net(x, sampling)
        return x
    
    return apply

def construct_gen(gen, dimZ, sampling=True):
    def gen_data(N):
        # start from sample z_0, generate data
        z = tf.random_normal(shape=(N, dimZ))
        return gen(z, sampling)

    return gen_data
    
