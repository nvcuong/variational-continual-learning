import numpy as np
import tensorflow as tf
from mlp import mlp_layer


"""
An MLP generator
"""

def generator_head(dimZ, dimH, n_layers, name):
    fc_layer_sizes = [dimZ] + [dimH for i in xrange(n_layers)]
    layers = []
    N_layers = len(fc_layer_sizes) - 1
    for i in xrange(N_layers):
        d_in = fc_layer_sizes[i]; d_out = fc_layer_sizes[i+1]
        name_layer = name + '_head_l%d' % i
        layers.append(mlp_layer(d_in, d_out, 'relu', name_layer))
    
    print 'decoder head MLP of size', fc_layer_sizes
    
    def apply(x):
        for layer in layers:
            x = layer(x)
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
        layers.append(mlp_layer(d_in, d_out, activation, name_layer))
    
    print 'decoder shared MLP of size', fc_layer_sizes
    
    def apply(x):
        for layer in layers:
            x = layer(x)
        return x
        
    return apply
    
def generator(head_net, shared_net):
    def apply(x, sampling=True):
        x = head_net(x)
        x = shared_net(x)
        return x
    
    return apply

def construct_gen(gen, dimZ, sampling=True):
    def gen_data(N):
        # start from sample z_0, generate data
        z = tf.random_normal(shape=(N, dimZ))
        return gen(z)

    return gen_data
    
