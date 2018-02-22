import os, struct
from array import array
import numpy as np

def load_data_mnist(path, digits = None, conv = False, seed = 0):
    # load and split data
    print "Loading data"
    path = path + 'MNIST/'
    data_train, labels_train = read(path, 0, "training", seed, digits)
    data_test, labels_test = read(path, 0, "testing", seed, digits)
    data_train /= 255.0	# real-value
    data_test /= 255.0	# real-value
    # transform to float32
    data_train = np.array(data_train.T, dtype='f')	# float32
    data_test = np.array(data_test.T, dtype='f')	# float32
    labels_train = np.array(labels_train.T, dtype='f')	# float32
    labels_test = np.array(labels_test.T, dtype='f')	# float32
    
    # for conv net
    if conv:
        data_train = data_train.reshape(data_train.shape[0], 28, 28, 1)
        data_test = data_test.reshape(data_test.shape[0], 28, 28, 1)
    
    return data_train, data_test, labels_train, labels_test


def read(path, num_per_digit = 0, dataset = "training", seed = 0, digits = None):
    """
    Python function for importing the MNIST data set.
    """

    if dataset is "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError, "dataset must be 'testing' or 'training'"

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = array("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = array("B", fimg.read())
    fimg.close()

    if digits is None: 
        digits = np.arange(10)
    num_digits = len(digits)
    if dataset == 'training':
        num_per_digit = 6000
    if dataset == 'testing':
        num_per_digit = 1000
    if num_per_digit > 0 and num_per_digit < len(img) / (rows * cols) / num_digits:
    	num_data = num_per_digit * num_digits
    else:
	num_data = len(img) / (rows * cols)
    images =  np.zeros([rows*cols, num_data])
    labels = np.zeros([10, num_data])
    for j in xrange(num_digits):
    	ind = [ k for k in xrange(size) if lbl[k] == digits[j] ]
	if len(ind) == 0:
	    raise ValueError, "invalid digits, should be in range 0-9"   
	if num_per_digit == 0 or num_per_digit > len(ind):
		num_per_digit = len(ind)
		if seed is None:
			ind = ind[:num_per_digit]  
    	else:
    		np.random.seed(seed)
    		ind = np.random.permutation(ind)[:num_per_digit]  
    	for i in xrange(num_per_digit):
	    #print i, num_per_digit, len(ind), i* num_digits + j, ind[i]*rows*cols, (ind[i]+1)*rows*cols 
            images[:, i * num_digits + j] = img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]
            labels[lbl[ind[i]], i * num_digits + j] = 1

    # here each column in images corresponds to a datapoint
    return images, labels
    
