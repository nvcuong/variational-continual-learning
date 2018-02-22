import numpy as np
import tensorflow as tf
import sys, os
import keras
sys.path.extend(['alg/', 'models/', 'utils/'])
from visualisation import plot_images
from encoder_no_shared import encoder, recon
from utils import init_variables, save_params, load_params, load_data
from eval_test_class import construct_eval_func
from load_classifier import load_model

dimZ = 50
dimH = 500
n_channel = 128
batch_size = 50
lr = 1e-4
K_mc = 10
checkpoint = -1

data_path = # TODO

def main(data_name, method, dimZ, dimH, n_channel, batch_size, K_mc, checkpoint, lbd):
    # set up dataset specific stuff
    from config import config
    labels, n_iter, dimX, shape_high, ll = config(data_name, n_channel)

    # import functionalities
    if method == 'onlinevi':
        from bayesian_generator import generator_head, generator_shared, \
                               generator, construct_gen
    if method in ['ewc', 'noreg', 'si', 'si2', 'laplace']:
        from generator import generator_head, generator_shared, generator, construct_gen

    # then define model
    n_layers_shared = 2
    batch_size_ph = tf.placeholder(tf.int32, shape=(), name='batch_size')
    dec_shared = generator_shared(dimX, dimH, n_layers_shared, 'sigmoid', 'gen')

    # initialise sessions
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)
    string = method
    if method in ['ewc', 'laplace']:
        string = string + '_lbd%d' % lbd
    if method in ['si', 'si2']:
        string = string + '_lbd%.1f' % lbd
    if method == 'onlinevi' and K_mc > 1:
        string = string + '_K%d' % K_mc
    path_name = data_name + '_%s_no_share_enc/' % string
    assert os.path.isdir('save/'+path_name)
    filename = 'save/' + path_name + 'checkpoint'
    # load the classifier
    cla = load_model(data_name)
    # print test error
    X_ph = tf.placeholder(tf.float32, shape=(batch_size, 28**2))
    y_ph = tf.placeholder(tf.float32, shape=(batch_size, 10))
    y_pred = cla(X_ph)
    correct_pred = tf.equal(tf.argmax(y_ph, 1), tf.argmax(y_pred, 1))
    acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    y_pred_ = tf.clip_by_value(y_pred, 1e-9, 1.0)
    kl = tf.reduce_mean(-tf.reduce_sum(y_ph * tf.log(y_pred_), 1))
    
    for task in range(1):
        if data_name == 'mnist':
            from mnist import load_mnist
            _, X_test, _, Y_test = load_mnist([task])
        if data_name == 'notmnist':
            from notmnist import load_notmnist
            _, X_test, _, Y_test = load_notmnist(data_path, [task], conv = False)
        test_acc = 0.0; test_kl = 0.0
        N_test = X_test.shape[0]
        for i in xrange(N_test / batch_size):
            indl = i * batch_size; indr = min((i+1)*batch_size, N_test)
            tmp1, tmp2 = sess.run((acc, kl), feed_dict={X_ph: X_test[indl:indr],
                                       y_ph: Y_test[indl:indr],
                                       keras.backend.learning_phase(): 0})
            test_acc += tmp1 / (N_test / batch_size)
            test_kl += tmp2 / (N_test / batch_size)
        print 'classification accuracy on test data', test_acc
        print 'kl on test data', test_kl

    # now start fitting
    N_task = len(labels)
    eval_func_list = []
    result_list = []

    n_layers_head = 2
    n_layers_enc = n_layers_shared + n_layers_head - 1
    for task in xrange(1, N_task+1):
        
        # define the head net and the generator ops
        dec = generator(generator_head(dimZ, dimH, n_layers_head, 'gen_%d' % task), dec_shared)
        eval_func_list.append(construct_eval_func(dec, cla, batch_size_ph, \
                                                  dimZ, task-1, sample_W = True))
        
        # then load the trained model
        load_params(sess, filename, checkpoint=task-1, init_all = False)
        
        # print test-ll on all tasks
        tmp_list = []
        for i in xrange(len(eval_func_list)):
            print 'task %d' % (i+1),
            kl = eval_func_list[i](sess)
            tmp_list.append(kl)
        result_list.append(tmp_list)
    
    for i in xrange(len(result_list)):
        print result_list[i]
        
    # save results
    fname = 'results/' + data_name + '_%s_gen_class.pkl' % string
    import pickle
    pickle.dump(result_list, open(fname, 'wb'))
    print 'test-ll results saved in', fname

if __name__ == '__main__':
    data_name = str(sys.argv[1])
    method = str(sys.argv[2])
    lbd = float(sys.argv[3])
    main(data_name, method, dimZ, dimH, n_channel, batch_size, K_mc, checkpoint, lbd)
    
