import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import seaborn as sns

sns.set_theme(style='darkgrid')

def plot_comparison(files, labels=None, save_path=None):
    plt.figure(figsize=(10, 5))

    for idx, file in enumerate(files):
        cost = np.load(file)
        episodes = np.arange(len(cost))
        label = labels[idx] if labels and idx < len(labels) else os.path.basename(file)
        linestyle = ['-', '--', '-.', ':'][idx % 4]
        plt.plot(episodes, cost, label=label, linestyle=linestyle)

    plt.xlabel('Episode')
    plt.ylabel('Avg. Cost')
    plt.title('Comparison of Training Costs')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare multiple training cost plots from npy files.")
    parser.add_argument("files", type=str, nargs='+', help="Paths to the .npy files")
    parser.add_argument("--labels", type=str, nargs='*', help="Labels for the experiments (in order)")
    parser.add_argument("--save", type=str, default=None, help="Path to save the output plot")

    args = parser.parse_args()
    plot_comparison(args.files, args.labels, args.save)
