import os, struct
import numpy as np
from scipy.io import loadmat

def load_data_omni(path, labels = None, conv = False, seed = 0):
    # load and split data
    print "Loading data"
    mat = loadmat(path + 'OMNIGLOT/chardata.mat')
    data_train = np.array(mat['data'].T, dtype='f')     # float32
    data_test = np.array(mat['testdata'].T, dtype='f')  # float32
    labels_train = np.array(mat['target'].T, dtype='f') # float32
    labels_test = np.array(mat['testtarget'].T, dtype='f') # float32

    # for some of the classes
    if labels is not None:
        ind_train = []
        ind_test = []
        for label in labels:
            ind_train = ind_train + list(np.where(labels_train[:, label] == 1)[0])
            ind_test = ind_test + list(np.where(labels_test[:, label] == 1)[0])
            
        np.random.seed(seed)
        ind_train = np.random.permutation(ind_train)
        ind_test = np.random.permutation(ind_test)
        data_train = data_train[ind_train]
        labels_train = labels_train[ind_train]
        data_test = data_test[ind_test]
        labels_test = labels_test[ind_test]
    
    # for conv net
    if conv:
        data_train = data_train.reshape(data_train.shape[0], 28, 28, 1)
        data_test = data_test.reshape(data_test.shape[0], 28, 28, 1)
    
    return data_train, data_test, labels_train, labels_test
    
