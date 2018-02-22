import numpy as np

def config(data_name, n_channel):
    if data_name == 'mnist':
        labels = [[i] for i in xrange(10)]
        n_iter = 200
        dimX = 28**2
        shape_high = (28, 28)
        ll = 'bernoulli'
    if data_name == 'notmnist':
        labels = [[i] for i in xrange(10)]
        n_iter = 400
        dimX = 28**2
        shape_high = (28, 28)
        ll = 'bernoulli'
        
    return labels, n_iter, dimX, shape_high, ll

