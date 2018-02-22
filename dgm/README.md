# Continual Learning for deep generative models

We provide an implementation for continual learning methods applied to training VAEs. 

The code is tested on tensorflow 1.0 and Keras 1.2.0. Using lower or higher version might
cause bugs.

## Train a model
To train a model, you need to modify the code to provide the data path in the TODO part. 
Then you can simply run

    python exp.py data_name method lbd
    
to train a model. Here the arguments are:

data_name: the name of your data, e.g. mnist or notmnist. If you want to test it on other
datasets then you need to write your own data loading codes accordingly.

method: be sure that method is one of the following:

+ noreg: none of the continual learning method is in use, just naive online learning

+ ewc: the EWC method with our implimentation adapted to VAEs

+ laplace: Laplace propagation (LP) method adapted to VAEs

+ SI: the Synaptic Intelligence method adapted to VAEs

+ onlinevi: the VCL method, which essentially runs online variational inference

lbd is the lambda or c parameters for ewc/laplace/si, for other methods lbd is ineffectived

You can also modify the configurations in config.py to determine the total epochs and the split of the tasks.

## To evaluate a model
To evaluate test-LL, run

    python eval_ll.py data_name method lbd
    
To evaluate the classifier uncertainty metric, you need to first train a classifier:

    cd classifier/
    python train_classifier.py
    
with data_name in file train_classifier.py modified accordingly. After training, run

    python eval_kl.py data_name method lbd
