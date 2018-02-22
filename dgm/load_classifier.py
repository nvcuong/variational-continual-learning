from __future__ import print_function

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
FLAGS = flags.FLAGS

def load_model(data_name = 'mnist'):

    path = 'classifier/save/'
    file_name = path + data_name
    model = Sequential()
    dimH = 1000
    num_classes = 10
    n_layer = 3
    model.add(Dense(dimH, activation='relu', input_shape=(784,)))
    model.add(Dropout(0.2))
    for _ in xrange(n_layer-1):
        model.add(Dense(dimH, activation='relu'))
        model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    model.load_weights(file_name+'_weights.h5', by_name=True)
    print("model loaded from "+file_name+' weights.h5')
    
    return model

