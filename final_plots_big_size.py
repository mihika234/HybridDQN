import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# UNIFORM GLOBAL STYLE
# ============================================================
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 14,
    "lines.linewidth": 2
})

DEFAULT_FIGSIZE = (6, 4)
DEFAULT_DPI = 300

BASE_DIR = "training/new environment final/baseline/classical"
WINDOW = 30

def rolling_var(x, w):
    return pd.Series(x).rolling(w).var()

def rolling_slope(x, w):
    x = pd.Series(x)
    return x.diff().rolling(w).mean()

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

# ============================================================
# MAIN LOOP
# ============================================================
for run in sorted(os.listdir(BASE_DIR)):
    if "dont use" in run.lower():
        continue

    run_dir = os.path.join(BASE_DIR, run)
    res_dir = os.path.join(run_dir, "results")
    plot_dir = os.path.join(res_dir, "plots")

    if not os.path.isdir(res_dir):
        continue

    print(f"[INFO] Processing {run}")
    ensure_dir(plot_dir)

    # ================= Load =================
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

    # ============================================================
    # 1. Rolling reward variance
    # ============================================================
    plt.figure(figsize=DEFAULT_FIGSIZE)
    plt.plot(rolling_var(reward, WINDOW))
    plt.title("Rolling Reward Variance")
    plt.xlabel("Episode")
    plt.ylabel("Variance")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/rolling_reward_variance.png", dpi=DEFAULT_DPI)
    plt.close()

    # ============================================================
    # 2. Rolling entropy variance
    # ============================================================
    plt.figure(figsize=DEFAULT_FIGSIZE)
    plt.plot(rolling_var(entropy, WINDOW))
    plt.title("Rolling Entropy Variance")
    plt.xlabel("Episode")
    plt.ylabel("Variance")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/rolling_entropy_variance.png", dpi=DEFAULT_DPI)
    plt.close()

    # ============================================================
    # 3. Rolling switch variance
    # ============================================================
    plt.figure(figsize=DEFAULT_FIGSIZE)
    plt.plot(rolling_var(switch, WINDOW))
    plt.title("Rolling Action Switching Variance")
    plt.xlabel("Episode")
    plt.ylabel("Variance")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/rolling_switch_variance.png", dpi=DEFAULT_DPI)
    plt.close()

    # ============================================================
    # 4. KL drift rolling variance
    # ============================================================
    plt.figure(figsize=DEFAULT_FIGSIZE)
    plt.plot(rolling_var(kl, WINDOW))
    plt.title("KL Drift Rolling Variance")
    plt.xlabel("Episode")
    plt.ylabel("Variance")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/rolling_kl_variance.png", dpi=DEFAULT_DPI)
    plt.close()

    # ============================================================
    # 5. KL Drift vs Congestion  ✅ (YOU ASKED FOR THIS)
    # ============================================================
    plt.figure(figsize=DEFAULT_FIGSIZE)
    plt.scatter(congestion[:len(kl)], kl, alpha=0.6, s=20)
    plt.title("Policy Drift vs Congestion")
    plt.xlabel("Mean Congestion")
    plt.ylabel("KL Drift")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/kl_vs_congestion.png", dpi=DEFAULT_DPI)
    plt.close()

    # ============================================================
    # 6. Switching vs episode
    # ============================================================
    plt.figure(figsize=DEFAULT_FIGSIZE)
    plt.plot(switch)
    plt.title("Action Switching Rate vs Episode")
    plt.xlabel("Episode")
    plt.ylabel("Switching Rate")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/switching_vs_episode.png", dpi=DEFAULT_DPI)
    plt.close()

    # ============================================================
    # 7. Entropy slope
    # ============================================================
    entropy_slope = rolling_slope(entropy, WINDOW)
    plt.figure(figsize=DEFAULT_FIGSIZE)
    plt.plot(entropy_slope)
    plt.title("Policy Entropy Slope (Rolling)")
    plt.xlabel("Episode")
    plt.ylabel("dH/dt")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/entropy_slope.png", dpi=DEFAULT_DPI)
    plt.close()

    # ============================================================
    # 8. Gradient variance
    # ============================================================
    plt.figure(figsize=DEFAULT_FIGSIZE)
    plt.plot(rolling_var(grad_norm, WINDOW))
    plt.title("Gradient Norm Rolling Variance")
    plt.xlabel("Training Step")
    plt.ylabel("Variance")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/grad_norm_variance.png", dpi=DEFAULT_DPI)
    plt.close()

    # ============================================================
    # 9. Congestion curve
    # ============================================================
    plt.figure(figsize=DEFAULT_FIGSIZE)
    plt.plot(congestion)
    plt.axhline(np.mean(congestion), linestyle="--")
    plt.title("Mean Congestion (Lyapunov Proxy)")
    plt.xlabel("Episode")
    plt.ylabel("Mean Fog Congestion")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/queue_stability_congestion.png", dpi=DEFAULT_DPI)
    plt.close()

    # ============================================================
    
print("✔ ALL stability and congestion plots completed.")
