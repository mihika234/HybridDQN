import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# GLOBAL PLOT STYLE
# ============================================================
plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 14
})

BASE_DIR = "training/new environment final/baseline/quantum"
WINDOW = 30


# ============================================================
# HELPERS
# ============================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def rolling_var(x, w):
    return pd.Series(x).rolling(w).var()


def rolling_slope(x, w):
    x = pd.Series(x)
    return x.diff().rolling(w).mean()


def extract_config(folder_name):
    try:
        parts = folder_name.split("_")
        return f"{parts[-2]}_{parts[-1]}"
    except:
        return folder_name


# ============================================================
# CORE PROCESSING
# ============================================================
def process_results(res_dir, config_name):
    plot_dir = os.path.join(res_dir, "plots")
    ensure_dir(plot_dir)

    # ================= LOAD =================
    try:
        episode_df = pd.read_csv(os.path.join(res_dir, "episode_metrics.csv"))
        reward = episode_df["reward"].values

        metrics = np.load(os.path.join(res_dir, "metrics_dump.npz"))
        entropy = metrics["entropies"]
        kl = metrics["kl_drifts"]
        switch = metrics["switch_rates"]

        grad_norm = np.load(os.path.join(res_dir, "grad_norm.npy"))

        epi_metrics = np.load(
            os.path.join(res_dir, "episode_level_metrics.npz"),
            allow_pickle=True
        )

        congestion = [np.mean(c) for c in epi_metrics["congestion"]]
        actions = epi_metrics["actions"]

    except Exception as e:
        print(f"⚠️ Failed loading {res_dir}: {e}")
        return

    # ================= PLOT HELPER =================
    def save_plot(y, title, xlabel, ylabel, filename):
        plt.figure(figsize=(6, 4))
        plt.plot(y)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, filename), dpi=300)
        plt.close()

    # ================= ROLLING METRICS =================
    save_plot(rolling_var(reward, WINDOW),
              f"Rolling Reward Variance ({config_name})",
              "Episode", "Variance",
              "rolling_reward_variance.png")

    save_plot(rolling_var(entropy, WINDOW),
              f"Rolling Entropy Variance ({config_name})",
              "Episode", "Variance",
              "rolling_entropy_variance.png")

    save_plot(rolling_var(switch, WINDOW),
              f"Rolling Switch Variance ({config_name})",
              "Episode", "Variance",
              "rolling_switch_variance.png")

    save_plot(rolling_var(kl, WINDOW),
              f"KL Variance ({config_name})",
              "Episode", "Variance",
              "rolling_kl_variance.png")

    # ================= DIRECT METRICS =================
    save_plot(switch,
              f"Switch Rate ({config_name})",
              "Episode", "Rate",
              "switching_vs_episode.png")

    entropy_slope = rolling_slope(entropy, WINDOW)
    save_plot(entropy_slope,
              f"Entropy Slope ({config_name})",
              "Episode", "dH/dt",
              "entropy_slope.png")

    # ============================================================
    # 🔥 NORMALIZED GRAD NORM VARIANCE (FIXED)
    # ============================================================
    try:
        grad_norm = np.array(grad_norm)

        grad_norm_norm = grad_norm / (np.mean(grad_norm) + 1e-8)

        save_plot(rolling_var(grad_norm_norm, WINDOW),
                  f"Normalized Grad Norm Variance ({config_name})",
                  "Step", "Variance",
                  "grad_norm_variance_normalized.png")

    except Exception as e:
        print(f"⚠️ Grad norm normalization failed for {config_name}: {e}")

    # ================= CONGESTION =================
    plt.figure(figsize=(6, 4))
    plt.plot(congestion)
    plt.axhline(np.mean(congestion), linestyle="--")
    plt.title(f"Congestion ({config_name})")
    plt.xlabel("Episode")
    plt.ylabel("Mean Congestion")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "queue_stability_congestion.png"), dpi=300)
    plt.close()

    # ============================================================
    # 🔥 TV DISTANCE (FIXED ROBUST VERSION)
    # ============================================================
    try:
        tv = []

        for e in range(1, len(actions)):
            A_prev = np.array(actions[e - 1]).ravel()
            A_curr = np.array(actions[e]).ravel()

            # safe cast
            A_prev = np.array([int(x) for x in A_prev])
            A_curr = np.array([int(x) for x in A_curr])

            if len(A_prev) == 0 or len(A_curr) == 0:
                continue

            K = int(max(A_prev.max(), A_curr.max())) + 1

            p_prev = np.bincount(A_prev, minlength=K) / len(A_prev)
            p_curr = np.bincount(A_curr, minlength=K) / len(A_curr)

            tv.append(0.5 * np.sum(np.abs(p_prev - p_curr)))

        if len(tv) > 0:
            save_plot(tv,
                      f"TV Distance ({config_name})",
                      "Episode", "TV Distance",
                      "tv_distance.png")

    except Exception as e:
        print(f"⚠️ TV computation failed for {config_name}: {e}")

    print(f"✅ Done: {config_name}")


# ============================================================
# MAIN LOOP
# ============================================================
for config in sorted(os.listdir(BASE_DIR)):

    config_path = os.path.join(BASE_DIR, config)
    if not os.path.isdir(config_path):
        continue

    config_name = extract_config(config)
    print(f"\n=== {config_name} ===")

    # CASE 1: results directly in config
    direct_results = os.path.join(config_path, "results")
    if os.path.isdir(direct_results):
        process_results(direct_results, config_name)
        continue

    # CASE 2: multiple runs inside
    for run in os.listdir(config_path):
        run_path = os.path.join(config_path, run)
        res_dir = os.path.join(run_path, "results")

        if os.path.isdir(res_dir):
            process_results(res_dir, f"{config_name}_{run}")


print("\n✔ ALL plots generated.")
