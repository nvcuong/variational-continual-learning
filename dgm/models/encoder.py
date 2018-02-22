import numpy as np
import tensorflow as tf
from mlp import mlp_layer
import time
    
def encoder_shared(dimX, dimH, n_layers, name):
 
    fc_layer_sizes = [dimX] + [dimH for i in xrange(n_layers)]
    layers = []
    N_layers = len(fc_layer_sizes) - 1
    for i in xrange(N_layers):
        d_in = fc_layer_sizes[i]; d_out = fc_layer_sizes[i+1]
        name_layer = name + '_shared_l%d' % i
        layers.append(mlp_layer(d_in, d_out, 'relu', name_layer))
        
    print 'encoder shared MLP of size', fc_layer_sizes
    
    def apply(x):
        for layer in layers:
            x = layer(x)
        return x
        
    return apply

def encoder_head(dimH, dimZ, n_layers, name):
    fc_layer_sizes = [dimH for i in xrange(n_layers)] + [dimZ*2]
    layers = []
    N_layers = len(fc_layer_sizes) - 1
    for i in xrange(N_layers):
        d_in = fc_layer_sizes[i]; d_out = fc_layer_sizes[i+1]
        name_layer = name + '_head_l%d' % i
        if i < N_layers - 1:
            activation = 'relu'
        else:
            activation = 'linear'
        layers.append(mlp_layer(d_in, d_out, activation, name_layer))
        
    print 'encoder head MLP of size', fc_layer_sizes
    
    def apply(x):
        for layer in layers:
            x = layer(x)
        return x
        
    return apply
    
def encoder(head_net, shared_net):
    def apply(x):
        x = shared_net(x)
        mu, log_sig = tf.split(1, 2, head_net(x))
        return mu, log_sig
    
    return apply

def sample_gaussian(mu, log_sig):
    return mu + tf.exp(log_sig) * tf.random_normal(mu.get_shape())

def recon(x, gen, enc, sampling = False):
 
    # then infer z, do bidiretional lstm
    mu, log_sig = enc(x)
    if sampling:
        z = sample_gaussian(mu, log_sig)
    else:
        z = mu
        
    return gen(z)

