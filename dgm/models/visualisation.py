"""
Visualising animations
"""

import numpy as np

def reshape_and_tile_images(array, shape=(28, 28), n_cols=None):
    if n_cols is None:
        n_cols = int(math.sqrt(array.shape[0]))
    n_rows = int(np.ceil(float(array.shape[0])/n_cols))
    if len(shape) == 2:
        order = 'C'
    else:
        order = 'F'

    def cell(i, j):
        ind = i*n_cols+j
        if i*n_cols+j < array.shape[0]:
            return array[ind].reshape(*shape, order='C')
        else:
            return np.zeros(shape)

    def row(i):
        return np.concatenate([cell(i, j) for j in range(n_cols)], axis=1)

    return np.concatenate([row(i) for i in range(n_rows)], axis=0)

def plot_images(images, shape, path, filename, n_rows = 10, color = True):
     # finally save to file
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    images = reshape_and_tile_images(images, shape, n_rows)
    if color:
        from matplotlib import cm
        plt.imsave(fname=path+filename+".png", arr=images, cmap=cm.Greys_r)
    else:
        plt.imsave(fname=path+filename+".png", arr=images, cmap='Greys')
    #plt.axis('off')
    #plt.tight_layout()
    #plt.savefig(path + filename + ".png", format="png")
    print "saving image to " + path + filename + ".png"
    plt.close()

