import tensorflow as tf
import numpy as np
from copy import deepcopy

np.random.seed(0)
tf.set_random_seed(0)

# Create network weights
def _create_weights(size, use_float64=False):
    no_layers = len(size) - 1
    no_weights = 0
    for i in range(no_layers):
        no_weights += size[i] * size[i + 1] + size[i + 1]

    if use_float64 == True:
        m = tf.Variable(tf.constant(np.zeros([no_weights]), dtype=tf.float64))
        v = tf.Variable(tf.constant(np.zeros([no_weights]), dtype=tf.float64))
    else:
        m = tf.Variable(tf.constant(np.zeros([no_weights]), dtype=tf.float32))
        v = tf.Variable(tf.constant(np.zeros([no_weights]), dtype=tf.float32))

    return no_weights, m, v

# Unpack network weights by separating the biases
def _unpack_weights(m, v, size):
    start_ind = 0
    end_ind = 0
    m_weights = []
    m_biases = []
    v_weights = []
    v_biases = []
    no_layers = len(size) - 1
    for i in range(no_layers):
        Din = size[i]
        Dout = size[i + 1]
        end_ind += Din * Dout
        m_weights.append(tf.reshape(m[start_ind:end_ind], [Din, Dout]))
        v_weights.append(tf.reshape(v[start_ind:end_ind], [Din, Dout]))
        start_ind = end_ind
        end_ind += Dout
        m_biases.append(m[start_ind:end_ind])
        v_biases.append(v[start_ind:end_ind])
        start_ind = end_ind
    return m_weights, v_weights, m_biases, v_biases

# The overall model (lower and upper network)
class MFVI_NN(object):
    def __init__(
            self, lower_size, upper_sizes,
            no_train_samples=10, no_test_samples=20, training_loss_classes=[], data_classes=[], use_float64=False):
        self.lower_size = lower_size
        self.no_tasks = len(training_loss_classes)
        self.data_classes = data_classes
        self.training_loss_classes = training_loss_classes
        self.upper_sizes = upper_sizes
        self.no_train_samples = no_train_samples if no_train_samples > 0 else 10
        self.no_test_samples = no_test_samples
        self.use_float64 = use_float64

        # Input and output placeholders
        if use_float64 == True:
            self.x = tf.placeholder(tf.float64, [None, lower_size[0]])
            self.ys = [tf.placeholder(tf.float64, [None, upper_size[-1]]) for upper_size in upper_sizes]
        else:
            self.x = tf.placeholder(tf.float32, [None, lower_size[0]])
            self.ys = [tf.placeholder(tf.float32, [None, upper_size[-1]]) for upper_size in upper_sizes]

        # Number of training points
        self.training_size = tf.placeholder(tf.int32)

        # Lower and upper layer of model
        self.lower_net = HalfNet(lower_size, use_float64)
        self.upper_nets = []
        for t, upper_size in enumerate(self.upper_sizes):
            self.upper_nets.append(HalfNet(upper_size, use_float64))

        # Build train loss, likelihood loss, and predictions (for test-time)
        self.training_loss, self.likelihood_loss = self._build_training_loss()
        self.preds = self._build_preds()

    def _build_training_loss(self):
        # Each task cost depends on different classes
        kl_lower = self.lower_net.KL_term()
        costs = []
        likelihood_costs = []

        # Number of training datapoints
        if self.use_float64 == True:
            N = tf.cast(self.training_size, tf.float64)
        else:
            N = tf.cast(self.training_size, tf.float32)

        # Separate for each task
        for task_id in range(self.no_tasks):
            # Likelihood term in variational objective function
            log_pred = self.log_prediction_fn_training_loss(
                self.x, self.ys, self.training_loss_classes[task_id], self.no_train_samples)
            likelihood_costs.append(-log_pred)
            # KL term (of upper network) in variational objective function
            for ind, class_id in enumerate(self.training_loss_classes[task_id]):
                kl_upper_interm = self.upper_nets[class_id].KL_term()
                if ind == 0:
                    kl_upper = kl_upper_interm
                else:
                    kl_upper += kl_upper_interm
            # Overall variational objective function for this task
            cost = tf.div(kl_lower + kl_upper, N) - log_pred
            costs.append(cost)

        return costs, likelihood_costs

    # Build predictions output from model (for test-time)
    def _build_preds(self):
        preds = []
        for t, upper_net in enumerate(self.upper_nets):
            pred = self.prediction_fn(self.x, t, self.no_test_samples)
            preds.append(pred)
        return preds

    # Called by _build_preds() for each upper_net (or, for each class)
    def prediction_fn(self, inputs, class_id, no_samples):
        K = no_samples
        inputs_3d = tf.tile(tf.expand_dims(inputs, 0), [K, 1, 1])
        lower_output = self.lower_net.prediction(inputs_3d, K)
        upper_output = self.upper_nets[class_id].prediction(lower_output, K)
        return upper_output

    # For building training loss
    def log_prediction_fn_training_loss(self, inputs, targets_input, training_loss_class_idx, no_samples):
        for ind, class_id in enumerate(training_loss_class_idx):
            pred_interm = self.prediction_fn(inputs, class_id, no_samples)
            targets_interm = tf.tile(tf.expand_dims(targets_input[class_id], 0), [self.no_train_samples, 1, 1])
            if ind == 0:
                pred = pred_interm
                targets = targets_interm
            else:
                pred = tf.concat([pred, pred_interm], axis=2)
                targets = tf.concat([targets, targets_interm], axis=2)

        # Calculate mean of cross_entropy for training using variational objective function (and Monte Carlo sampling)
        log_lik = - tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=targets))
        return log_lik

    # Initialise tf session
    def init_session(self, task_idx, learning_rate, training_classes=[]):

        # Which variables are optimised (are allowed to change)
        vars_to_optimise = [self.lower_net.m, self.lower_net.v]
        for class_id in training_classes:
            vars_to_optimise.append(self.upper_nets[class_id].m)
            vars_to_optimise.append(self.upper_nets[class_id].v)

        self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.training_loss[task_idx],
                                                                         var_list=vars_to_optimise)

        # Initializing the variables
        init = tf.global_variables_initializer()

        # launch a session
        self.sess = tf.Session()
        self.sess.run(init)

    # Close tf session
    def close_session(self):
        self.sess.close()

    # Train model
    def train(self, x_train, y_train, task_id, prior_lower, prior_upper,
              no_epochs=1000, batch_size=100, display_epoch=100):

        # Batch size
        N = x_train.shape[0]
        if batch_size > N:
            batch_size = N

        sess = self.sess
        costs = []
        lik_costs = []

        feed_dict = {
            self.lower_net.m0: prior_lower[0],
            self.lower_net.v0: prior_lower[1],
            self.training_size: N}

        # Only load (upper) priors for the training loss classes for this task
        for class_id in self.training_loss_classes[task_id]:
            feed_dict[self.upper_nets[class_id].m0] = prior_upper[class_id][0]
            feed_dict[self.upper_nets[class_id].v0] = prior_upper[class_id][1]

        # Training cycle
        for epoch in range(no_epochs):
            # Random order of batches
            perm_inds = range(x_train.shape[0])
            np.random.shuffle(perm_inds)
            cur_x_train = x_train[perm_inds]
            cur_y_train = y_train[perm_inds]

            total_batch = int(np.ceil(N * 1.0 / batch_size))

            avg_cost = 0.
            avg_lik_cost = 0.

            # Loop over all batches
            for i in range(total_batch):
                # Batches
                start_ind = i * batch_size
                end_ind = np.min([(i + 1) * batch_size, N])
                batch_x = cur_x_train[start_ind:end_ind, :]
                batch_y = cur_y_train[start_ind:end_ind, :]
                feed_dict[self.x] = batch_x

                # Only load class labels for the training loss classes for this task
                for class_id in self.training_loss_classes[task_id]:
                    batch_input = np.zeros([end_ind-start_ind, 1])
                    batch_input[:, 0] = batch_y[:, class_id]
                    feed_dict[self.ys[class_id]] = batch_input

                # Run optimisation operation (backprop) and cost operation (to get loss value)
                _, c_total, lik_loss = sess.run(
                    [self.train_step, self.training_loss[task_id], self.likelihood_loss[task_id]],
                    feed_dict=feed_dict)

                # Compute average loss
                avg_cost += c_total / total_batch
                avg_lik_cost += lik_loss / total_batch

            # Display logs every display_epoch
            if epoch == 0 or (epoch+1) % display_epoch == 0 or epoch == no_epochs-1:
                print("Epoch:", '%04d' % (epoch + 1), "total cost=", "{:.9f}".format(avg_cost), "lik term=",
                      "{:.9f}".format(avg_lik_cost), "kl term=", "{:.9f}".format(avg_cost - avg_lik_cost))
            costs.append(avg_cost)
            lik_costs.append(avg_lik_cost)

        print("Optimisation Finished!")

        return costs, lik_costs

    # For outputting predictions at test-time
    def prediction(self, x_test, batch_size=1000):
        # Test model
        N = x_test.shape[0]
        batch_size = N if batch_size > N else batch_size
        total_batch = int(np.ceil(N * 1.0 / batch_size))
        for i in range(total_batch):
            # Batches
            start_ind = i * batch_size
            end_ind = np.min([(i + 1) * batch_size, N])
            batch_x = x_test[start_ind:end_ind, :]

            prediction = self.sess.run(
                [self.preds],
                feed_dict={self.x: batch_x})[0]

            if i == 0:
                predictions = prediction
            else:
                predictions = np.concatenate((predictions, prediction), axis=2)

        return predictions

    # Function that outputs predictions from model at test-time
    def prediction_prob(self, x_test, batch_size=1000):
        prob = self.sess.run(
            [tf.nn.softmax(self.prediction(x_test, batch_size), 0)],
            feed_dict={self.x: x_test})[0]
        return prob

    # Return weights currently in model
    def get_weights(self, class_idx):
        lower = self.sess.run(self.lower_net.params)
        upper = []
        for class_id in class_idx:
            upper_interm = deepcopy(self.sess.run(self.upper_nets[class_id].params))
            upper.append(upper_interm)
        return (lower, upper)

    # Assign weights to model
    def assign_weights(self, class_idx, lower_weights, upper_weights):
        # Lower network weights
        lower_net = self.lower_net
        self.sess.run(
            [lower_net.assign_m_op, lower_net.assign_v_op],
            feed_dict={
                lower_net.new_m: lower_weights[0],
                lower_net.new_v: lower_weights[1]})

        if not isinstance(class_idx, (list,)):
            class_idx = [class_idx]

        # Upper weights: go over each class
        for class_id in class_idx:
            assign_m_op = tf.assign(self.upper_nets[class_id].m, self.upper_nets[class_id].new_m)
            self.sess.run(
                assign_m_op, feed_dict={
                    self.upper_nets[class_id].new_m: upper_weights[class_id][0]})

            assign_v_op = tf.assign(self.upper_nets[class_id].v, self.upper_nets[class_id].new_v)
            self.sess.run(
                assign_v_op, feed_dict={
                    self.upper_nets[class_id].new_v: upper_weights[class_id][1]})

    # Reset tf optimiser
    def reset_optimiser(self):
        optimiser_scope = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,
            "scope/prefix/for/optimizer")
        self.sess.run(tf.initialize_variables(optimiser_scope))

# Either the lower network, or the upper network (top-most layer of model)
class HalfNet():
    def __init__(self, size, use_float64=False, act_func=tf.nn.relu):
        self.size = size                # Size of network (number of units per layer)
        self.no_layers = len(size) - 1  # Number of layers
        self.act_func = act_func        # Activation function
        self.use_float64 = use_float64  # Increased storage precision

        # Create weights (means and variances)
        self.no_weights, self.m, self.v = _create_weights(self.size, use_float64)

        # Unpack weights so that biases are separate
        self.mw, self.vw, self.mb, self.vb = _unpack_weights(self.m, self.v, self.size)
        self.params = [self.m, self.v]

        # Create placeholders for loading weights into HalfNet
        if use_float64:
            self.new_m = tf.placeholder(tf.float64, [self.no_weights])
            self.new_v = tf.placeholder(tf.float64, [self.no_weights])
        else:
            self.new_m = tf.placeholder(tf.float32, [self.no_weights])
            self.new_v = tf.placeholder(tf.float32, [self.no_weights])

        self.assign_m_op = tf.assign(self.m, self.new_m)
        self.assign_v_op = tf.assign(self.v, self.new_v)

        # Prior as placeholder, as these can change
        if use_float64:
            self.m0 = tf.placeholder(tf.float64, [self.no_weights])
            self.v0 = tf.placeholder(tf.float64, [self.no_weights])
        else:
            self.m0 = tf.placeholder(tf.float32, [self.no_weights])
            self.v0 = tf.placeholder(tf.float32, [self.no_weights])

    # This samples a layer at a time, using the local reparameterisation trick
    def prediction(self, inputs, no_samples):
        K = no_samples              # Number of (random) samples per datapoint
        N = tf.shape(inputs)[1]     # Number of datapoints in batch
        Dout = self.size[-1]
        mw, vw, mb, vb = self.mw, self.vw, self.mb, self.vb
        act = inputs
        for i in range(self.no_layers):
            # Means of pre-activation latents
            m_pre = tf.einsum('kni,io->kno', act, mw[i])
            m_pre = m_pre + mb[i]

            # Variances of pre-activation latents
            v_pre = tf.einsum('kni,io->kno', act ** 2.0, tf.exp(vw[i]))
            v_pre = v_pre + tf.exp(vb[i])

            # Sample pre-activation latents
            if self.use_float64:
                eps = tf.random_normal([K, 1, self.size[i + 1]], 0.0, 1.0, dtype=tf.float64)
            else:
                eps = tf.random_normal([K, 1, self.size[i + 1]], 0.0, 1.0, dtype=tf.float32)
            pre = eps * tf.sqrt(1e-9 + v_pre) + m_pre

            act = self.act_func(pre)

        # Return pre-activation from final layer
        pre = tf.reshape(pre, [K, N, Dout])
        return pre

    # Calculate KL term
    def KL_term(self):
        const_term = -0.5 * self.no_weights
        log_std_diff = 0.5 * tf.reduce_sum(self.v0 - self.v)
        mu_diff_term = 0.5 * tf.reduce_sum((tf.exp(self.v) + (self.m0 - self.m) ** 2) / tf.exp(self.v0))
        kl = const_term + log_std_diff + mu_diff_term
        return kl