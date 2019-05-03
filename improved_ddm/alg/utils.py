import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def merge_coresets(x_coresets, y_coresets):
    merged_x, merged_y = x_coresets[0], y_coresets[0]
    for i in range(1, len(x_coresets)):
        merged_x = np.vstack((merged_x, x_coresets[i]))
        merged_y = np.vstack((merged_y, y_coresets[i]))
    return merged_x, merged_y


# Get test accuracies from model
def get_scores_output_pred(model, x_testsets, y_testsets, test_classes, task_idx=[0], multi_head=False):
    acc = []

    # Go over each task's testset
    for i in task_idx:
        x_test, y_test = x_testsets[i], y_testsets[i]

        # Output from model
        pred = model.prediction_prob(x_test)

        # Mean over the different Monte Carlo models
        pred_mean_total = np.mean(pred, axis=1)

        heads = i if multi_head else 0  # Different for multi-head and single-head

        # test_classes[heads] holds which classes we are predicting between
        # We are only interested in finding which of these classes has maximum prediction output from model
        # We therefore set all the other classes to have a large negative value: this is pred_mean
        pred_mean = -10000000000*np.ones(np.shape(pred_mean_total))
        pred_mean[test_classes[heads], :, :] = pred_mean_total[test_classes[heads], :, :]

        # Predicted class
        pred_y = np.argmax(pred_mean, axis=0)
        pred_y = pred_y[:, 0]

        # True class
        y = np.argmax(y_test, axis=1)

        # Calculate test accuracy
        cur_acc = len(np.where((pred_y - y) == 0)[0]) * 1.0 / y.shape[0]
        acc.append(cur_acc)

    return acc


def plot(path, vcl_result, vcl_result_coreset):
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    matplotlib.rcParams.update({'font.size': 22})

    no_tasks = len(vcl_result)
    result_avg = np.zeros(no_tasks)
    result_coreset_avg = np.zeros(no_tasks)

    plot_every_task = False

    # Plot performance on every task
    for i in range(no_tasks):
        if plot_every_task:
            fig = plt.figure(figsize=(5,3))
            ax = plt.gca()
            plt.plot(np.arange(i, no_tasks)+1, vcl_result_coreset[i:,i], label='VCL+coreset', marker='o', color='y')
            plt.plot(np.arange(i, no_tasks) + 1, vcl_result[i:, i], label='VCL', marker='o', color='b')
            ax.set_xticks(range(1, no_tasks+1))
            ax.set_ylim([0.85, 1.01])
            ax.set_yticks([0.85, 0.9, 0.95, 1.0])
            ax.set_ylabel('Accuracy')
            ax.set_xlabel('Tasks')
            ax.set_title('Task %d (%d or %d)' % (i+1, 2*i, 2*i+1))
            ax.legend()

            fig.savefig(path+'accuracy_task_%d.svg' % (i), bbox_inches='tight')
            plt.close()

        result_avg[i] = np.average(vcl_result[i,:i+1])
        result_coreset_avg[i] = np.average(vcl_result_coreset[i,:i+1])

    # Plot average accuracy
    fig = plt.figure(figsize=(5,3))
    ax = plt.gca()
    plt.plot(np.arange(no_tasks) + 1, result_coreset_avg, label='VCL+coreset', marker='o', color='y')
    plt.plot(np.arange(no_tasks) + 1, result_avg, label='VCL', marker='o', color='b')
    ax.set_xticks(range(1, no_tasks + 1))
    ax.set_ylim([0.85, 1.01])
    ax.set_yticks([0.85, 0.9, 0.95, 1.0])
    ax.set_ylabel('Average Accuracy')
    ax.set_xlabel('Tasks')
    ax.set_title('Average')
    ax.legend()

    fig.savefig(path + 'average_accuracy.svg', bbox_inches='tight')
    plt.close()