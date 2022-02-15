import matplotlib.pyplot as plt
import numpy as np


def plot_learning_curve(scores, time, fig_file):
    running_avg = np.zeros(len(scores))

    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - 1000):(i + 1)])
    plt.plot(time, running_avg)
    plt.title("Running Average of Previous 100 Scores")
    plt.savefig(fig_file)
