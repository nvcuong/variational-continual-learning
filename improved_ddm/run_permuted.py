import numpy as np
import tensorflow as tf
import gzip
import cPickle
import sys
sys.path.extend(['alg/'])
import vcl
import coreset
import utils
from copy import deepcopy


class PermutedMnistGenerator():
    def __init__(self, max_iter=10, random_seed=0):
        # Open data file
        f = gzip.open('data/mnist.pkl.gz', 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()

        # Define train and test data
        self.X_train = np.vstack((train_set[0], valid_set[0]))
        self.Y_train = np.hstack((train_set[1], valid_set[1]))
        self.X_test = test_set[0]
        self.Y_test = test_set[1]
        self.random_seed = random_seed
        self.max_iter = max_iter
        self.cur_iter = 0

        self.out_dim = 10           # Total number of unique classes
        self.class_list = range(10) # List of unique classes being considered, in the order they appear

        # self.classes is the classes (with correct indices for training/testing) of interest at each task_id
        self.classes = []
        for iter in range(self.max_iter):
            self.classes.append(range(0,10))

        self.sets = self.classes

    def get_dims(self):
        # Get data input and output dimensions
        return self.X_train.shape[1], self.out_dim

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception('Number of tasks exceeded!')
        else:
            np.random.seed(self.cur_iter+self.random_seed)
            perm_inds = range(self.X_train.shape[1])
            # First task is (unpermuted) MNIST, subsequent tasks are random permutations of pixels
            if self.cur_iter > 0:
                np.random.shuffle(perm_inds)

            # Retrieve train data
            next_x_train = deepcopy(self.X_train)
            next_x_train = next_x_train[:,perm_inds]

            # Initialise next_y_train to zeros, then change relevant entries to ones, and then stack
            next_y_train = np.zeros((len(next_x_train), 10))
            next_y_train[:,0:10] = np.eye(10)[self.Y_train]

            # Retrieve test data
            next_x_test = deepcopy(self.X_test)
            next_x_test = next_x_test[:,perm_inds]

            next_y_test = np.zeros((len(next_x_test), 10))
            next_y_test[:,0:10] = np.eye(10)[self.Y_test]

            self.cur_iter += 1

            return next_x_train, next_y_train, next_x_test, next_y_test

    def reset(self):
        self.cur_iter = 0


store_weights = True        # Store weights after training on each task (for plotting later)
multi_head = False          # Multi-head or single-head network

hidden_size = [100, 100]    # Size and number of hidden layers
batch_size = 1024           # Batch size
no_epochs = 800             # Number of training epochs per task
permuted_num_tasks = 10


# No coreset
tf.reset_default_graph()
random_seed = 1
tf.set_random_seed(random_seed+1)
np.random.seed(random_seed)

path = 'model_storage/permuted/'    # Path where to store files
data_gen = PermutedMnistGenerator(max_iter=permuted_num_tasks, random_seed=random_seed)
coreset_size = 0
vcl_result = vcl.run_vcl_shared(hidden_size, no_epochs, data_gen,
    coreset.rand_from_batch, coreset_size, batch_size, path, multi_head, store_weights=store_weights)

# Store accuracies
np.savez(path + 'test_acc.npz', acc=vcl_result)


# Random coreset
tf.reset_default_graph()
random_seed = 1
tf.set_random_seed(random_seed+1)
np.random.seed(random_seed)

path = 'model_storage/permuted_coreset/'    # Path where to store files
data_gen = PermutedMnistGenerator(max_iter=permuted_num_tasks, random_seed=random_seed)
coreset_size = 200
vcl_result_coresets = vcl.run_vcl_shared(hidden_size, no_epochs, data_gen,
    coreset.rand_from_batch, coreset_size, batch_size, path, multi_head, store_weights=store_weights)

# Store accuracies
np.savez(path + 'test_acc.npz', acc=vcl_result_coresets)

# Plot average accuracy
utils.plot('model_storage/permuted_mnist_', vcl_result, vcl_result_coresets)