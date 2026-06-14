# Delay + Energy

import numpy as np
import matplotlib.pyplot as plt
import os




def plot_graphs(axs, train_cost, train_dropped, train_delay, train_energy, show=False, save=False,
                path=None):
    if save and path is not None:
        plots_dir = os.path.join(path, "plots")
        os.makedirs(plots_dir, exist_ok=True)

    x = np.arange(len(train_cost)).tolist()

    axs[0].clear()
    axs[0].plot(x, train_cost, color='red', label='Training')
    axs[0].set(title='Avg. Cost')
    axs[0].set(ylabel='Avg. Cost')
    axs[0].set(xlabel='Episode')
    axs[0].legend(loc='upper right')

    axs[1].clear()
    axs[1].plot(x, train_dropped, color='blue', label='Training')
    axs[1].set(title='Ratio of Dropped Tasks')
    axs[1].set(ylabel='Dropped Ratio')
    axs[1].set(xlabel='Episode')
    axs[1].legend(loc='upper right')

    axs[2].clear()
    axs[2].plot(x, train_delay, color='green', label='Training')
    axs[2].set(title='Avg. Task Delay')
    axs[2].set(ylabel='Avg. Delay (Sec)')
    axs[2].set(xlabel='Episode')
    axs[2].legend(loc='upper right')

    axs[3].clear()
    axs[3].plot(x, train_energy, color='yellow', label='Training')
    axs[3].set(title='Avg. Task Energy')
    axs[3].set(ylabel='Avg. energy')
    axs[3].set(xlabel='Episode')
    axs[3].legend(loc='upper right')


    if save and path is not None:
       plt.savefig(os.path.join(plots_dir, "learning_curves.png"))

       np.save(
        os.path.join(plots_dir, "avg_cost.npy"),
        np.array(train_cost)
       )

       np.save(
        os.path.join(plots_dir, "dropped_ratio.npy"),
        np.array(train_dropped)
       )

       np.save(
        os.path.join(plots_dir, "avg_delay.npy"),
        np.array(train_delay)
       )

       np.save(
        os.path.join(plots_dir, "avg_energy.npy"),
        np.array(train_energy)
       )

    if show:
        plt.show(block=False)
        plt.pause(0.01)
