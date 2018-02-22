import os
import numpy as np
import time
import cPickle
import tensorflow as tf

def load_data(data_name, path, labels = None, conv = False, seed = 0):
    if data_name == 'mnist':
        from import_data_mnist import load_data_mnist
        data_train, data_test, labels_train, labels_test = \
            load_data_mnist(path, labels, conv, seed)
            
    if data_name == 'omni':
        from import_data_omni import load_data_omni
        data_train, data_test, labels_train, labels_test = \
            load_data_omni(path, labels, conv, seed)
            
    if data_name == 'cifar10':
        from import_data_cifar10 import load_data_cifar10
        data_train, data_test, labels_train, labels_test = \
            load_data_cifar10(path, labels, conv, seed)
        
    return data_train, data_test, labels_train, labels_test

def init_variables(sess, old_var_list = set([])):
    all_var_list = set(tf.all_variables())
    init = tf.initialize_variables(var_list = all_var_list - old_var_list)
    sess.run(init)
    return all_var_list
    
def save_params(sess, filename, checkpoint):
    params = tf.trainable_variables()
    param_dict = dict()
    for v in params:
        param_dict[v.name] = sess.run(v)
    filename = filename + '_' + str(checkpoint)
    f = open(filename + '.pkl', 'w')
    cPickle.dump(param_dict, f, protocol=cPickle.HIGHEST_PROTOCOL)
    print 'parameters saved at ' + filename + '.pkl'    
    f.close()

def load_params(sess, filename, checkpoint, init_all = True):
    params = tf.trainable_variables()
    filename = filename + '_' + str(checkpoint)
    f = open(filename + '.pkl', 'r')
    param_dict = cPickle.load(f)
    print 'param loaded', len(param_dict)
    f.close()
    ops = []
    for v in params:
        if v.name in param_dict.keys():
            ops.append(tf.assign(v, param_dict[v.name]))
    sess.run(ops)
    # init uninitialised params
    if init_all:
        all_var = tf.all_variables()
        var = [v for v in all_var if v not in params]
        sess.run(tf.initialize_variables(var))
    print 'loaded parameters from ' + filename + '.pkl'
