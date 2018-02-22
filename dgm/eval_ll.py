import numpy as np
import tensorflow as tf
import sys, os
sys.path.extend(['alg/', 'models/'])
from visualisation import plot_images
from encoder_no_shared import encoder, recon
from utils import init_variables, save_params, load_params, load_data
from eval_test_ll import construct_eval_func

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
    if data_name == 'mnist':
        from mnist import load_mnist
    if data_name == 'notmnist':
        from notmnist import load_notmnist

    # import functionalities
    if method == 'onlinevi':
        from bayesian_generator import generator_head, generator_shared, \
                               generator, construct_gen
    if method in ['ewc', 'noreg', 'si', 'laplace']:
        from generator import generator_head, generator_shared, generator, construct_gen

    # then define model
    n_layers_shared = 2
    batch_size_ph = tf.placeholder(tf.int32, shape=(), name='batch_size')
    dec_shared = generator_shared(dimX, dimH, n_layers_shared, 'sigmoid', 'gen')

    # initialise sessions
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    string = method
    if method in ['ewc', 'laplace', 'si']:
        string = string + '_lbd%.1f' % lbd
    if method == 'onlinevi' and K_mc > 1:
        string = string + '_K%d' % K_mc
    path_name = data_name + '_%s/' % string
    assert os.path.isdir('save/'+path_name)
    filename = 'save/' + path_name + 'checkpoint'

    # visualise the samples
    N_gen = 10**2
    X_ph = tf.placeholder(tf.float32, shape=(batch_size, dimX), name = 'x_ph')

    # now start fitting
    N_task = len(labels)
    gen_ops = []
    X_valid_list = []
    X_test_list = []
    eval_func_list = []
    result_list = []
    
    n_layers_head = 2
    n_layers_enc = n_layers_shared + n_layers_head - 1
    for task in xrange(1, N_task+1):
        # first load data
        # first load data
        if data_name == 'mnist':
            X_train, X_test, _, _ = load_mnist(digits = labels[task-1], conv = False)
        if data_name == 'notmnist':
            X_train, X_test, _, _ = load_notmnist(data_path, digits = labels[task-1], conv = False)
        N_train = int(X_train.shape[0] * 0.9)
        X_valid_list.append(X_train[N_train:])
        X_train = X_train[:N_train]
        X_test_list.append(X_test)
        
        # define the head net and the generator ops
        dec = generator(generator_head(dimZ, dimH, n_layers_head, 'gen_%d' % task), dec_shared)
        enc = encoder(dimX, dimH, dimZ, n_layers_enc, 'enc_%d' % task)
        gen_ops.append(construct_gen(dec, dimZ, sampling=False)(N_gen))
        eval_func_list.append(construct_eval_func(X_ph, enc, dec, ll, batch_size_ph, 
                                                  K = 5000, sample_W = False))
        
        # then load the trained model
        load_params(sess, filename, checkpoint=task-1, init_all = False)
        
        # plot samples
        x_gen_list = sess.run(gen_ops, feed_dict={batch_size_ph: N_gen})
        x_list = []
        for i in xrange(len(x_gen_list)):
            ind = np.random.randint(len(x_gen_list[i]))
            x_list.append(x_gen_list[i][ind:ind+1])
        x_list = np.concatenate(x_list, 0)
        tmp = np.zeros([10, dimX])
        tmp[:task] = x_list
        if task == 1:
            x_gen_all = tmp
        else:           
            x_gen_all = np.concatenate([x_gen_all, tmp], 0)
        
        # print test-ll on all tasks
        tmp_list = []
        for i in xrange(len(eval_func_list)):
            print 'task %d' % (i+1),
            test_ll = eval_func_list[i](sess, X_test_list[i])
            tmp_list.append(test_ll)
        result_list.append(tmp_list)
    
    #x_gen_all = 1.0 - x_gen_all
    if not os.path.isdir('figs/visualisation/'):
        os.mkdir('figs/visualisation/')
        print 'create path figs/visualisation/'
    plot_images(x_gen_all, shape_high, 'figs/visualisation/', data_name+'_gen_all_'+method)
    
    for i in xrange(len(result_list)):
        print result_list[i]
        
    # save results
    fname = 'results/' + data_name + '_%s.pkl' % string
    import pickle
    pickle.dump(result_list, open(fname, 'wb'))
    print 'test-ll results saved in', fname

if __name__ == '__main__':
    data_name = str(sys.argv[1])
    method = str(sys.argv[2])
    assert method in ['noreg', 'laplace', 'ewc', 'si', 'onlinevi']
    if method == 'onlinevi':
        lbd = 1.0	# some placeholder, doesn't matter
    else:
        lbd = float(sys.argv[3])
    main(data_name, method, dimZ, dimH, n_channel, batch_size, K_mc, checkpoint, lbd)
    
