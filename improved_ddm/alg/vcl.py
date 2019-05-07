import numpy as np
import utils
from cla_models_multihead import MFVI_NN
from copy import deepcopy
import time

ide_func = lambda x: np.float32(x)
log_func = lambda x: np.float32(np.log(x))
exp_func = lambda x: np.float32(np.exp(x))


# Stores model weights (previous posterior weights = new prior weights)
class WeightsStorage():
    def __init__(self, no_lower_weights, no_upper_weights, prior_mean=0.0, prior_var=1.0):
        # Initial mean and variance for lower network and upper network
        self.lower_mean = np.ones([no_lower_weights]) * prior_mean
        self.lower_log_var = np.ones([no_lower_weights]) * log_func(prior_var)
        self.upper_mean = [np.ones(no_weights) * prior_mean for no_weights in no_upper_weights]
        self.upper_log_var = [np.ones(no_weights) * log_func(prior_var) for no_weights in no_upper_weights]

    def return_weights(self):
        # Returns lower and upper weights that are currently stored (the previous posterior)
        upper_mv = []
        for class_ind in range(len(self.upper_mean)):
            upper_mv.append([deepcopy(self.upper_mean[class_ind]), deepcopy(self.upper_log_var[class_ind])])

        return (deepcopy(self.lower_mean), deepcopy(self.lower_log_var)), upper_mv

    def store_weights(self, post_l_mv, post_u_mv):
        # Store model weights
        self.lower_mean = deepcopy(post_l_mv[0])
        self.lower_log_var = deepcopy(post_l_mv[1])

        for class_ind in range(len(post_u_mv)):
            self.upper_mean[class_ind] = deepcopy(post_u_mv[class_ind][0])
            self.upper_log_var[class_ind] = deepcopy(post_u_mv[class_ind][1])

# Initialise model weights before training on new data, using small random means and small variances
def initialise_weights(weights):
    weights_mean_init = np.random.normal(size=weights[0].shape, scale=0.1)
    weights_log_var_init = np.ones_like(weights[1]) * (-6.0)
    return [weights_mean_init, weights_log_var_init]

# Run VCL on model; returns accuracies on each task after training on each task
def run_vcl_shared(hidden_size, no_epochs, data_gen, coreset_method, coreset_size=0,
                   batch_size=None, path='sandbox/', multi_head=False, learning_rate=0.005, store_weights=False):
    in_dim, out_dim = data_gen.get_dims()
    x_coresets, y_coresets = [], []
    x_testsets, y_testsets = [], []
    x_trainsets, y_trainsets = [], []

    all_acc = np.array([])
    no_tasks = data_gen.max_iter

    # Store train and test sets (over all tasks)
    for i in range(no_tasks):
        x_train, y_train, x_test, y_test = data_gen.next_task()
        x_trainsets.append(x_train)
        y_trainsets.append(y_train)
        x_testsets.append(x_test)
        y_testsets.append(y_test)

    all_classes = range(data_gen.out_dim)
    training_loss_classes = []  # Training loss function depends on these classes
    training_classes = []       # Which classes' heads' weights change during training
    test_classes = []           # Which classes to compare between at test time
    for task_id in range(no_tasks):
        # The data input classes for this task
        data_classes = data_gen.classes[task_id]

        if multi_head:
            training_loss_classes.append(data_classes)
            training_classes.append(data_classes)
            test_classes.append(data_classes)
        else:
            # Single-head
            training_loss_classes.append(all_classes)
            training_classes.append(all_classes)
            test_classes.append(all_classes)

    # Create model
    no_heads = out_dim
    lower_size = [in_dim] + deepcopy(hidden_size)
    upper_sizes = [[hidden_size[-1], 1] for i in range(no_heads)]
    model = MFVI_NN(lower_size, upper_sizes, training_loss_classes=training_loss_classes,
                    data_classes=data_gen.classes, use_float64=multi_head)
    no_lower_weights = model.lower_net.no_weights
    no_upper_weights = [net.no_weights for net in model.upper_nets]

    # Set up model weights at initial prior
    weights_storage = WeightsStorage(no_lower_weights, no_upper_weights, prior_mean=0.0, prior_var=1.0)

    for task_id in range(no_tasks):
        # tf init model
        model.init_session(task_id, learning_rate, training_classes[task_id])

        # Get data
        x_train, y_train = x_trainsets[task_id], y_trainsets[task_id]

        # Set batch size
        bsize = x_train.shape[0] if (batch_size is None) else batch_size

        # Select coreset if needed
        if coreset_size > 0:
            x_coresets, y_coresets, x_train, y_train = coreset_method(
                x_coresets, y_coresets, x_train, y_train, coreset_size)

        # Prior of weights is previous posterior (or, if first task, already in weights_storage)
        lower_weights_prior, upper_weights_prior = weights_storage.return_weights()

        # Initialise using random means + small variances
        lower_weights = initialise_weights(lower_weights_prior)
        upper_weights = deepcopy(upper_weights_prior)
        for class_id in training_classes[task_id]:
            upper_weights[class_id] = deepcopy(initialise_weights(upper_weights_prior[class_id]))

        # Assign initial weights to the model
        model.assign_weights(range(no_heads), lower_weights, upper_weights)

        # Train on non-coreset data
        model.reset_optimiser()

        start_time = time.time()
        _, _ = model.train(x_train, y_train, task_id, lower_weights_prior, upper_weights_prior, no_epochs, bsize)
        end_time = time.time()
        print 'Time taken to train (s):', end_time - start_time

        # Get weights from model, and store in weights_storage
        lower_weights, upper_weights = model.get_weights(range(no_heads))
        weights_storage.store_weights(lower_weights, upper_weights)

        # Save model weights after training on non-coreset data
        if store_weights:
            np.savez(path + 'weights_%d.npz' % task_id, lower=lower_weights, upper=upper_weights,
                     classes=data_gen.classes,
                     MNISTdigits=data_gen.sets, class_index_conversion=data_gen.class_list)

        model.close_session()

        # Train on coreset data, then calculate test accuracy
        if multi_head:
            acc = np.zeros(no_tasks)
            for test_task_id in range(task_id+1):
                # Initialise session, and load weights into model
                model.init_session(test_task_id, learning_rate, training_classes[test_task_id])
                lower_weights, upper_weights = weights_storage.return_weights()
                model.assign_weights(range(no_heads), lower_weights, upper_weights)
                if len(x_coresets) > 0:
                    print 'Training on coreset data...'
                    # Train on each task's coreset data just before testing on that task
                    x_train_coreset, y_train_coreset = x_coresets[test_task_id], y_coresets[test_task_id]
                    bsize = x_train_coreset.shape[0] if (batch_size is None) else batch_size
                    model.reset_optimiser()
                    _, _ = model.train(x_train_coreset, y_train_coreset, test_task_id,
                                       lower_weights, upper_weights, no_epochs, bsize)

                # Test-time: Calculate test accuracy
                acc_interm = utils.get_scores_output_pred(model, x_testsets, y_testsets, test_classes,
                                                   task_idx=[test_task_id], multi_head=multi_head)
                acc[test_task_id] = acc_interm[0]

                model.close_session()

        else:
            acc = np.zeros(no_tasks)
            # Initialise session, and load weights into model
            model.init_session(task_id, learning_rate, training_classes[task_id])
            lower_weights, upper_weights = weights_storage.return_weights()
            model.assign_weights(range(no_heads), lower_weights, upper_weights)
            if len(x_coresets) > 0:
                print 'Training on coreset data...'
                x_train_coreset, y_train_coreset = utils.merge_coresets(x_coresets, y_coresets)
                bsize = x_train_coreset.shape[0] if (batch_size is None) else batch_size
                _, _ = model.train(x_train_coreset, y_train_coreset, task_id,
                                   lower_weights, upper_weights, no_epochs, bsize)

            # Test-time: Calculate test accuracy
            acc_interm = utils.get_scores_output_pred(model, x_testsets, y_testsets, test_classes,
                                               task_idx=range(task_id+1), multi_head=multi_head)
            acc[:task_id+1] = acc_interm

            model.close_session()

        # Append accuracies to all_acc array
        if task_id == 0:
            all_acc = np.array(acc)
        else:
            all_acc = np.vstack([all_acc, acc])
        print all_acc

    return all_acc