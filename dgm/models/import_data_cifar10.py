import os, struct
import numpy as np
import cPickle

def to_categorical(y, num_classes=None):
    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical
    
def load_data_cifar10(path, labels = None, conv = True, seed = 0):
    # load and split data
    def unpickle(path, name):
        f = open(path + 'cifar-10-batches-py/' + name,'rb')
        data = cPickle.load(f)
        f.close()
        return data
    def futz(X):
        return X.reshape((10000, 3, 32, 32)).transpose(0, 2, 3, 1)
        
    print "Loading data"
    data_train = np.zeros((50000, 32, 32, 3), dtype='uint8')
    labels_train = np.zeros(50000, dtype='int32')
    fnames = ['data_batch_%i'%i for i in range(1,6)]

    # load train and validation data
    n_loaded = 0
    for i, fname in enumerate(fnames):
        data = unpickle(path, fname)
        assert data['data'].dtype == np.uint8
        data_train[n_loaded:n_loaded + 10000] = futz(data['data'])
        labels_train[n_loaded:n_loaded + 10000] = data['labels']
        n_loaded += 10000
    
    # load test set
    data = unpickle(path, 'test_batch')
    assert data['data'].dtype == np.uint8
    data_test = futz(data['data'])
    labels_test = data['labels']
    
    labels_train = to_categorical(labels_train, 10)
    labels_test = to_categorical(labels_test, 10)
    
    # convert to float
    data_train = np.array(data_train, dtype='f')	# float32
    data_test = np.array(data_test, dtype='f')	# float32
    labels_train = np.array(labels_train, dtype='f')	# float32
    labels_test = np.array(labels_test, dtype='f')
    
    data_train = 1.0 * data_train / 256.
    data_test = 1.0 * data_test / 256.
    
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
    
    return data_train, data_test, labels_train, labels_test
    
