from keras.utils import np_utils
from keras.datasets import mnist
import numpy as np

def load_mnist(digits = None, conv = False):
    # Get MNIST test data
    #X_train, Y_train, X_test, Y_test = data_mnist()
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    
    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)
    
    # collect the corresponding digits
    if digits is not None:
        ind_train = []
        ind_test = []
        for i in digits:
            ind_train = ind_train + list(np.where(Y_train[:, i] == 1)[0])
            ind_test = ind_test + list(np.where(Y_test[:, i] == 1)[0])
        X_train = X_train[ind_train]; Y_train = Y_train[ind_train]
        X_test = X_test[ind_test]; Y_test = Y_test[ind_test]
    
    if conv:
        X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
        X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
    
    return X_train, X_test, Y_train, Y_test
