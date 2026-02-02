import os
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = "training/quantum/original env"

def plot_kl(run_dir):
    results_dir = os.path.join(run_dir, "results")
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    metrics_path = os.path.join(results_dir, "metrics_dump.npz")

    if not os.path.exists(metrics_path):
        print(f"[SKIP] metrics_dump.npz not found in {run_dir}")
        return

    data = np.load(metrics_path)
    if "kl_drifts" not in data:
        print(f"[SKIP] kl_drifts missing in {metrics_path}")
        return

    kl = data["kl_drifts"]

    if kl.ndim != 1:
        print(f"[SKIP] kl_drifts not 1D in {run_dir}")
        return

    plt.figure(figsize=(8, 4))
    plt.plot(kl, linewidth=1.5)
    plt.xlabel("Episode")
    plt.ylabel("KL Divergence")
    plt.title("KL Divergence vs Episode")
    plt.grid(True, alpha=0.3)

    save_path = os.path.join(plots_dir, "kl_vs_episode.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"[OK] Saved KL plot → {save_path}")


def main():
    if not os.path.isdir(BASE_DIR):
        raise RuntimeError(f"Base directory not found: {BASE_DIR}")

    run_folders = sorted([
        os.path.join(BASE_DIR, d)
        for d in os.listdir(BASE_DIR)
        if os.path.isdir(os.path.join(BASE_DIR, d))
    ])

    print(f"Found {len(run_folders)} runs")

    for run in run_folders:
        plot_kl(run)


if __name__ == "__main__":
    main()

