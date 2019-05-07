import numpy as np
import matplotlib
import matplotlib.pylab as plt
import matplotlib.mlab as mlab


def visualise_layer_weights(no_hiddens, path="", multi_head=False):
    # Plot lower / upper level weights
    plot_lower = True
    plot_upper = True

    # Load files
    res = np.load(path + 'weights_%d.npz' % 0)
    mnist_digits = res['MNISTdigits']
    classes = res['classes']
    class_index_conversion = res['class_index_conversion']
    no_tasks = len(classes)
    task_range = range(no_tasks)

    for task_id in task_range:
        # Load weights for this task
        res = np.load(path + 'weights_%d.npz' % task_id)
        lower = res['lower']
        upper = res['upper']
        m_upper = upper[:,0]
        var_upper = np.exp(upper[:,1])

        # Lower network
        for layer in range(len(no_hiddens)):
            if layer == 0:
                in_dim = 784
                no_rows = 28
                no_cols = 28
                no_params = 0
            else:
                in_dim = no_hiddens[layer-1]

            # Find mean and variance parameters for this layer
            m_low = lower[0, no_params:no_params + (in_dim + 1) * no_hiddens[layer]]
            var_low = np.exp(lower[1, no_params:no_params+(in_dim+1)*no_hiddens[layer]])
            no_params = no_params + (in_dim + 1) * no_hiddens[layer]
            m_low = m_low.reshape([in_dim+1, no_hiddens[layer]])
            var_low = var_low.reshape([in_dim+1, no_hiddens[layer]])

            # Minimum and maximum values
            m_min, m_max = np.min(m_low), np.max(m_low)
            v_min, v_max = np.min(var_low), np.max(var_low)

            shape_dim = [no_rows, no_cols]

            # Number of rows and columns in figure plots
            if no_hiddens[layer] == 100:
                no_cols = 20
                no_rows = 5
            elif no_hiddens[layer] == 200:
                no_cols = 25
                no_rows = 8
            else:
                no_cols = int(np.sqrt(no_hiddens[layer]))
                no_rows = int(no_hiddens[layer] / no_cols)

            if plot_lower:
                print "task %d lower figures ..." % task_id

                if layer == 1:
                    fig0, axs0 = plt.subplots(no_rows, no_cols, figsize=(no_cols, no_rows))
                    fig1, axs1 = plt.subplots(no_rows, no_cols, figsize=(no_cols, no_rows))
                else:
                    fig0, axs0 = plt.subplots(no_rows, no_cols, figsize=(no_cols, no_rows))
                    fig1, axs1 = plt.subplots(no_rows, no_cols, figsize=(no_cols, no_rows))

                # Go over all units in this layer and plot input weights
                for i in range(no_rows):
                    for j in range(no_cols):
                        k = i * no_cols + j
                        ma = m_low[:in_dim, k].reshape(shape_dim)
                        va = var_low[:in_dim, k].reshape(shape_dim)

                        axs0[i, j].matshow(ma, cmap=matplotlib.cm.binary, vmin=m_min, vmax=m_max)
                        axs0[i, j].set_xticks(np.array([]))
                        axs0[i, j].set_yticks(np.array([]))

                        axs1[i, j].matshow(va, cmap=matplotlib.cm.binary, vmin=v_min, vmax=v_max)
                        axs1[i, j].set_xticks(np.array([]))
                        axs1[i, j].set_yticks(np.array([]))

                # Save figures
                fig0.savefig(path + 'task%d_layer%d_mean.png' % (task_id+1, layer), bbox_inches='tight')
                fig1.savefig(path + 'task%d_layer%d_var.png' % (task_id+1, layer), bbox_inches='tight')

        # Upper weights: plot as Gaussians
        no_params = no_hiddens[-1] * len(mnist_digits)
        x_max = np.max(np.abs(m_upper[:,:no_hiddens[-1]]) + np.sqrt(var_upper[:,:no_hiddens[-1]]))

        # Number of rows and columns in figure plots
        if no_hiddens[-1] == 100:
            no_cols = 20
            no_rows = 5
        elif no_hiddens[-1] == 200:
            no_cols = 25
            no_rows = 8
        else:
            no_cols = int(np.sqrt(no_hiddens[-1]))
            no_rows = int(no_hiddens[-1] / no_cols)

        # x-axis for Gaussian plots
        x = np.linspace(-x_max, x_max, 1000)

        if plot_upper:
            print "task %d upper figures ..." % task_id

            fig, axs = plt.subplots(no_rows, no_cols, figsize=(no_cols, no_rows))
            no_plot_digits = (task_id+1)*2 if multi_head else 10    # How many classes' upper level weights to plot
            for i in range(no_rows):
                for j in range(no_cols):
                    k = i * no_cols + j
                    for digit in range(no_plot_digits):
                        axs[i, j].plot(x, mlab.normpdf(x, m_upper[digit][k], np.sqrt(var_upper[digit][k])),
                                       label='%d' % class_index_conversion[digit])
                    axs[i, j].set_xticks(np.array([]))
                    axs[i, j].set_yticks(np.array([]))
                    axs[i,j].set_ylim([0, 1.0])

            axs[i,0].legend()
            fig.savefig(path + 'task%d_upper.png' % (task_id+1), bbox_inches='tight')

            # Plot bias of upper weights
            x_bias_max = np.max(np.abs(m_upper[:,-1])) + np.sqrt(np.max(var_upper[:,-1]))*2
            x_bias = np.linspace(-x_bias_max, x_bias_max, 1000)
            plt.figure()
            for digit in range(no_plot_digits):
                plt.plot(x_bias, mlab.normpdf(x_bias, m_upper[digit][-1], np.sqrt(var_upper[digit][-1])),
                         label='%d' % class_index_conversion[digit])
                plt.ylim([0, 2.0])
            plt.legend()
            plt.savefig(path + 'task%d_upper_bias.png' % (task_id+1))


if __name__ == "__main__":
    print 'Plotting weights for split MNIST'
    no_hiddens = [200]
    path = 'model_storage/split/'
    visualise_layer_weights(no_hiddens, path=path, multi_head=True)
    path = 'model_storage/split_coreset/'
    visualise_layer_weights(no_hiddens, path=path, multi_head=True)

    print 'Plotting weights for permuted MNIST'
    no_hiddens = [100, 100]
    path = 'model_storage/permuted/'
    visualise_layer_weights(no_hiddens, path=path, multi_head=False)
    path = 'model_storage/permuted_coreset/'
    visualise_layer_weights(no_hiddens, path=path, multi_head=False)