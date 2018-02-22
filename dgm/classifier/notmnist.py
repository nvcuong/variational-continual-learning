import numpy as np
from scipy.io import loadmat
from keras.utils import np_utils

def load_notmnist(path, digits = None, conv = False, seed=0, ratio=0.9):
    # Get MNIST test data
    #X_train, Y_train, X_test, Y_test = data_mnist()
    out = loadmat(path+'notMNIST/'+'notMNIST_small.mat')
    X = out['images']; Y = out['labels']
    X = X.transpose(2, 0, 1)
    X /= 255.0
    # collect the corresponding digits
    if digits is not None:
        ind = []
        for i in digits:
            ind = ind + list(np.where(Y == i)[0])
        X = X[ind]; Y = Y[ind] 
    Y = np_utils.to_categorical(Y, 10)
    np.random.seed(seed)
    N_train = int(X.shape[0]*ratio)
    ind = np.random.permutation(range(X.shape[0]))
    if conv:
        X = X[:, :, :, np.newaxis]
    else:
        X = X.reshape(X.shape[0], -1)
    X_train = X[ind[:N_train]]
    Y_train = Y[ind[:N_train]]
    X_test = X[ind[N_train:]]
    Y_test = Y[ind[N_train:]]

    return X_train, X_test, Y_train, Y_test
