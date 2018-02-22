# for keras 1.2.0

from __future__ import print_function

import keras, os
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import tensorflow as tf
import numpy as np

batch_size = 100
num_classes = 10
epochs = 500

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
keras.backend.set_session(sess)

# the data, shuffled and split between train and test sets
data_name = 'notmnist'
if data_name == 'mnist':
    from mnist import load_mnist
    x_train, x_test, y_train, y_test = load_mnist()
if data_name == 'notmnist':
    from notmnist import load_notmnist
    data_path = # TODO
    x_train, x_test, y_train, y_test = [], [], [], []
    for i in xrange(10):
        a, b, c, d = load_notmnist(data_path, [i], ratio=0.995)
        x_train.append(a); x_test.append(b)
        y_train.append(c); y_test.append(d)
    x_train = np.concatenate(x_train, 0)
    x_test = np.concatenate(x_test, 0)
    y_train = np.concatenate(y_train, 0)
    y_test = np.concatenate(y_test, 0)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

model = Sequential()
dimH = 1000
n_layer = 3
model.add(Dense(dimH, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
for _ in xrange(n_layer-1):
    model.add(Dense(dimH, activation='relu'))
    model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    nb_epoch=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# save model for future test
if not os.path.isdir('save/'):
    os.mkdir('save/')
    print 'create path save/'
file_name = 'save/'+data_name
model.save_weights(file_name+'_weights.h5')
print('model weights saved to file ' + file_name + '_weights.h5')

