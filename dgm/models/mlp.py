
import tensorflow as tf
import numpy as np

def init_weights(input_size, output_size, constant=1.0, seed=123): 
    """ Glorot and Bengio, 2010's initialization of network weights"""
    scale = constant*np.sqrt(6.0/(input_size + output_size))
    if output_size > 0:
        return tf.random_uniform((input_size, output_size), 
                             minval=-scale, maxval=scale, 
                             dtype=tf.float32, seed=seed)
    else:
        return tf.random_uniform([input_size], 
                             minval=-scale, maxval=scale, 
                             dtype=tf.float32, seed=seed)

def mlp_layer(d_in, d_out, activation, name):
    W = tf.Variable(init_weights(d_in, d_out), name = name+'_W')
    b = tf.Variable(tf.zeros([d_out]), name = name+'_b')
    
    def apply_layer(x):
        a = tf.matmul(x, W) + b
        if activation == 'relu':
            return tf.nn.relu(a)
        if activation == 'sigmoid':
            return tf.nn.sigmoid(a)
        if activation == 'linear':
            return a  
            
    return apply_layer

