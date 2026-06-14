import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
# Force non-interactive backend to prevent display errors on servers
matplotlib.use('Agg') 

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = "training/new environment final/baseline/quantum hamiltonian"
WINDOW = 30  # Rolling window size

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def load_file(base_path, filename, allow_pickle=False):
    path = os.path.join(base_path, filename)
    if not os.path.exists(path):
        return None
    try:
        if filename.endswith('.npy'):
            data = np.load(path, allow_pickle=allow_pickle)
            # Handle 0-d arrays or empty arrays
            if data.size == 0: return None
            return data
        elif filename.endswith('.npz'):
            data = np.load(path, allow_pickle=allow_pickle)
            return data
        elif filename.endswith('.csv'):
            return pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] Corrupt file {filename}: {e}")
        return None
    return None

def rolling_var(x, w):
    """Safely calculates rolling variance."""
    if x is None or len(x) < 2: return None
    # Convert to pandas series
    s = pd.Series(x)
    # Check if we have enough data points relative to window
    if len(s) == 0: return None
    return s.rolling(w).var()

def rolling_slope(x, w):
    if x is None or len(x) < 2: return None
    s = pd.Series(x)
    return s.diff().rolling(w).mean()

def action_switching_rate(A):
    if A.shape[0] < 2: return np.nan
    return np.mean(A[1:] != A[:-1])

def action_entropy_fn(actions, n_actions):
    actions = np.asarray(actions, dtype=np.int64)
    if len(actions) == 0: return 0.0
    counts = np.bincount(actions, minlength=int(n_actions)) + 1e-12
    p = counts / counts.sum()
    return -np.sum(p * np.log(p))

def safe_plot(data, title, ylabel, save_path, xlabel="Episode"):
    """Wrapper to plot only if data is valid."""
    if data is None: return
    
    # Check for empty pandas series or all NaNs
    if isinstance(data, (pd.Series, pd.DataFrame)):
        if data.isna().all(): return
    elif np.all(np.isnan(data)): return

    try:
        plt.figure(figsize=(8, 5))
        plt.plot(data)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"[WARN] Failed to plot {title}: {e}")
        plt.close()

# ============================================================
# MAIN LOOP
# ============================================================
def process_run(run_name, run_path):
    res_dir = os.path.join(run_path, "results")
    if not os.path.isdir(res_dir):
        return

    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"[INFO] Processing Run: {run_name}")
    plot_dir = os.path.join(res_dir, "plots")
    ensure_dir(plot_dir)

    # --- LOAD DATA ---
    action_hist = load_file(res_dir, "action_hist.npy")
    buffer_size = load_file(res_dir, "buffer_size.npy")
    grad_norm   = load_file(res_dir, "grad_norm.npy")
    loss        = load_file(res_dir, "loss.npy")
    q_stats     = load_file(res_dir, "q_stats.npy")
    entropy_npy = load_file(res_dir, "action_entropy.npy")
    
    drop_iot    = load_file(res_dir, "drop_iot.npy")
    drop_trans  = load_file(res_dir, "drop_trans.npy")
    drop_fog    = load_file(res_dir, "drop_fog.npy")

    metrics_dump = load_file(res_dir, "metrics_dump.npz", allow_pickle=True)
    kl_drifts = None
    switch_rates = None
    
    if metrics_dump:
        if "kl_drifts" in metrics_dump: kl_drifts = metrics_dump["kl_drifts"]
        if "switch_rates" in metrics_dump: switch_rates = metrics_dump["switch_rates"]
        # Fallback for entropy
        if entropy_npy is None and "entropies" in metrics_dump:
            entropy_npy = metrics_dump["entropies"]

    episode_df = load_file(res_dir, "episode_metrics.csv")
    rewards = episode_df["reward"].values if episode_df is not None and "reward" in episode_df.columns else None

    ep_level = load_file(res_dir, "episode_level_metrics.npz", allow_pickle=True)
    actions_grid = ep_level["actions"] if ep_level and "actions" in ep_level else None
    congestion_grid = ep_level["congestion"] if ep_level and "congestion" in ep_level else None
    
    mean_congestion = []
    if congestion_grid is not None and len(congestion_grid) > 0:
        mean_congestion = [np.mean(c) for c in congestion_grid]

    # --- PLOTTING ---

    # 1. Reward Variance
    safe_plot(rolling_var(rewards, WINDOW), "Rolling Reward Variance", "Variance", f"{plot_dir}/rolling_reward_variance.png")

    # 2. Entropy Variance & Slope
    safe_plot(rolling_var(entropy_npy, WINDOW), "Rolling Entropy Variance", "Variance", f"{plot_dir}/rolling_entropy_variance.png")
    safe_plot(rolling_slope(entropy_npy, WINDOW), "Policy Entropy Slope", "dH/dt", f"{plot_dir}/entropy_slope.png")

    # 3. Switching Variance & Rate
    safe_plot(rolling_var(switch_rates, WINDOW), "Rolling Action Switching Variance", "Variance", f"{plot_dir}/rolling_switch_variance.png")
    safe_plot(switch_rates, "Action Switching Rate", "Rate", f"{plot_dir}/switching_vs_episode.png")

    # 4. KL Variance
    safe_plot(rolling_var(kl_drifts, WINDOW), "KL Drift Rolling Variance", "Variance", f"{plot_dir}/rolling_kl_variance.png")

    # 7. Gradient Norm Variance (Use 'Training Step' as label)
    safe_plot(rolling_var(grad_norm, WINDOW), "Gradient Norm Rolling Variance", "Variance", f"{plot_dir}/grad_norm_variance.png", xlabel="Training Step")

    # 8. Congestion Stability
    if len(mean_congestion) > 0:
        try:
            plt.figure(figsize=(8, 5))
            plt.plot(mean_congestion)
            plt.axhline(np.mean(mean_congestion), linestyle="--", color='r', label='Mean')
            plt.title("Mean Congestion (Lyapunov Proxy)")
            plt.xlabel("Episode"); plt.ylabel("Mean Fog Congestion")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{plot_dir}/queue_stability_congestion.png")
            plt.close()
        except: plt.close()

    # 12. Gradient Norm Raw
    if grad_norm is not None and len(grad_norm) > 0:
        try:
            plt.figure(figsize=(8, 5))
            plt.plot(grad_norm)
            plt.axhline(1.0, color="red", linestyle="--")
            plt.title("Gradient Norm (Raw)")
            plt.xlabel("Training Step"); plt.ylabel("Norm")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{plot_dir}/grad_norm_raw.png")
            plt.close()
        except: plt.close()

    # 17. Episode Metrics Grid
    if episode_df is not None:
        try:
            cols = ["reward", "drop", "delay", "energy"]
            valid_cols = [c for c in cols if c in episode_df.columns]
            if len(valid_cols) == 4:
                fig, axs = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
                axs[0,0].plot(episode_df["reward"]); axs[0,0].set_title("Reward")
                axs[0,1].plot(episode_df["drop"]);   axs[0,1].set_title("Drop Ratio")
                axs[1,0].plot(episode_df["delay"]);  axs[1,0].set_title("Delay")
                axs[1,1].plot(episode_df["energy"]); axs[1,1].set_title("Energy")
                for ax in axs.flat: ax.grid(alpha=0.3)
                plt.tight_layout()
                plt.savefig(f"{plot_dir}/episode_metrics_grid.png")
                plt.close()
        except: plt.close()

    # Correlations (Entropy vs Congestion)
    if actions_grid is not None and len(mean_congestion) > 0:
        try:
            calc_entropies = []
            for A in actions_grid:
                A_flat = A.flatten()
                if len(A_flat) == 0: calc_entropies.append(0)
                else:
                    K = int(np.max(A_flat)) + 1
                    calc_entropies.append(action_entropy_fn(A_flat, K))
            
            L = min(len(calc_entropies), len(mean_congestion))
            if L > 5:
                plt.figure(figsize=(6,4))
                plt.scatter(mean_congestion[:L], calc_entropies[:L], alpha=0.6)
                plt.xlabel("Mean Congestion"); plt.ylabel("Action Entropy")
                plt.title("Entropy vs Congestion")
                plt.grid(alpha=0.3)
                plt.tight_layout()
                plt.savefig(f"{plot_dir}/entropy_vs_congestion.png")
                plt.close()
        except: plt.close()

    print(f"   -> Plots saved to {plot_dir}")

if __name__ == "__main__":
    if not os.path.exists(BASE_DIR):
        print(f"[ERROR] BASE_DIR not found: {BASE_DIR}")
    else:
        runs = sorted(os.listdir(BASE_DIR))
        for run in runs:
            run_path = os.path.join(BASE_DIR, run)
            if os.path.isdir(run_path) and "dont use" not in run.lower():
                process_run(run, run_path)
