import numpy as np
import tensorflow as tf
import gzip
import cPickle
import sys
sys.path.extend(['alg/'])
import vcl
import coreset
import utils


class SplitMnistGenerator():
    def __init__(self):
        # Open data file
        f = gzip.open('data/mnist.pkl.gz', 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()

        # Define train and test data
        self.X_train = np.vstack((train_set[0], valid_set[0]))
        self.X_test = test_set[0]
        self.train_label = np.hstack((train_set[1], valid_set[1]))
        self.test_label = test_set[1]

        # split MNIST
        task1 = [0, 1]
        task2 = [2, 3]
        task3 = [4, 5]
        task4 = [6, 7]
        task5 = [8, 9]
        self.sets = [task1, task2, task3, task4, task5]

        self.max_iter = len(self.sets)

        self.out_dim = 0        # Total number of unique classes
        self.class_list = []    # List of unique classes being considered, in the order they appear
        for task_id in range(self.max_iter):
            for class_index in range(len(self.sets[task_id])):
                if self.sets[task_id][class_index] not in self.class_list:
                    # Convert from MNIST digit numbers to class index number by using self.class_list.index(),
                    # which is done in self.classes
                    self.class_list.append(self.sets[task_id][class_index])
                    self.out_dim = self.out_dim + 1

        # self.classes is the classes (with correct indices for training/testing) of interest at each task_id
        self.classes = []
        for task_id in range(self.max_iter):
            class_idx = []
            for i in range(len(self.sets[task_id])):
                class_idx.append(self.class_list.index(self.sets[task_id][i]))
            self.classes.append(class_idx)

        self.cur_iter = 0

    def get_dims(self):
        # Get data input and output dimensions
        return self.X_train.shape[1], self.out_dim

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception('Number of tasks exceeded!')
        else:
            next_x_train = []
            next_y_train = []
            next_x_test = []
            next_y_test = []

            # Loop over all classes in current iteration
            for class_index in range(np.size(self.sets[self.cur_iter])):

                # Find the correct set of training inputs
                train_id = np.where(self.train_label == self.sets[self.cur_iter][class_index])[0]
                # Stack the training inputs
                if class_index == 0:
                    next_x_train = self.X_train[train_id]
                else:
                    next_x_train = np.vstack((next_x_train, self.X_train[train_id]))

                # Initialise next_y_train to zeros, then change relevant entries to ones, and then stack
                next_y_train_interm = np.zeros((len(train_id), self.out_dim))
                next_y_train_interm[:, self.classes[self.cur_iter][class_index]] = 1
                if class_index == 0:
                    next_y_train = next_y_train_interm
                else:
                    next_y_train = np.vstack((next_y_train, next_y_train_interm))

                # Repeat above process for test inputs
                test_id = np.where(self.test_label == self.sets[self.cur_iter][class_index])[0]
                if class_index == 0:
                    next_x_test = self.X_test[test_id]
                else:
                    next_x_test = np.vstack((next_x_test, self.X_test[test_id]))

                next_y_test_interm = np.zeros((len(test_id), self.out_dim))
                next_y_test_interm[:, self.classes[self.cur_iter][class_index]] = 1
                if class_index == 0:
                    next_y_test = next_y_test_interm
                else:
                    next_y_test = np.vstack((next_y_test, next_y_test_interm))

            self.cur_iter += 1

            return next_x_train, next_y_train, next_x_test, next_y_test

    def reset(self):
        self.cur_iter = 0


store_weights = True    # Store weights after training on each task (for plotting later)
multi_head = True       # Multi-head or single-head network

hidden_size = [200]     # Size and number of hidden layers
batch_size = 256        # Batch size
no_epochs = 600         # Number of training epochs per task


# No coreset
tf.reset_default_graph()
random_seed = 0
tf.set_random_seed(random_seed+1)
np.random.seed(random_seed)

path = 'model_storage/split/'   # Path where to store files
data_gen = SplitMnistGenerator()
coreset_size = 0
vcl_result = vcl.run_vcl_shared(hidden_size, no_epochs, data_gen,
    coreset.rand_from_batch, coreset_size, batch_size, path, multi_head, store_weights=store_weights)

# Store accuracies
np.savez(path + 'test_acc.npz', acc=vcl_result)


# Random coreset
tf.reset_default_graph()
random_seed = 0
tf.set_random_seed(random_seed+1)
np.random.seed(random_seed)

path = 'model_storage/split_coreset/'   # Path where to store files
data_gen = SplitMnistGenerator()
coreset_size = 40
vcl_result_coresets = vcl.run_vcl_shared(hidden_size, no_epochs, data_gen,
    coreset.rand_from_batch, coreset_size, batch_size, path, multi_head, store_weights=store_weights)

# Store accuracies
np.savez(path + 'test_acc.npz', acc=vcl_result_coresets)

# Plot average accuracy
utils.plot('model_storage/split_mnist_', vcl_result, vcl_result_coresets)