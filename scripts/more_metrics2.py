import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = "training/new environment final/classical"
WINDOW = 50  # rolling window

def rolling_var(x, w):
    return pd.Series(x).rolling(w).var()

def rolling_slope(x, w):
    """Rolling slope via finite difference"""
    x = pd.Series(x)
    return x.diff().rolling(w).mean()

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def action_switching_rate(A):
    """A: [T, N_iot]"""
    return np.mean(A[1:] != A[:-1], axis=1)

for run in sorted(os.listdir(BASE_DIR)):
    if "dont use" in run.lower():
        print(f"[SKIP] Skipping folder: {run}")
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
    actions = epi_metrics["actions"]  # [episode][time][iot]

    # ================= 1. Rolling reward variance =================
    plt.figure()
    plt.plot(rolling_var(reward, WINDOW))
    plt.title("Rolling Reward Variance")
    plt.xlabel("Episode")
    plt.ylabel("Variance")
    plt.savefig(f"{plot_dir}/rolling_reward_variance.png")
    plt.close()

    # ================= 2. Rolling entropy variance =================
    plt.figure()
    plt.plot(rolling_var(entropy, WINDOW))
    plt.title("Rolling Entropy Variance")
    plt.xlabel("Episode")
    plt.ylabel("Variance")
    plt.savefig(f"{plot_dir}/rolling_entropy_variance.png")
    plt.close()

    # ================= 3. Rolling action switching variance =================
    plt.figure()
    plt.plot(rolling_var(switch, WINDOW))
    plt.title("Rolling Action Switching Variance")
    plt.xlabel("Episode")
    plt.ylabel("Variance")
    plt.savefig(f"{plot_dir}/rolling_switch_variance.png")
    plt.close()

    # ================= 4. KL drift rolling variance =================
    plt.figure()
    plt.plot(rolling_var(kl, WINDOW))
    plt.title("KL Drift Rolling Variance")
    plt.xlabel("Episode")
    plt.ylabel("Variance")
    plt.savefig(f"{plot_dir}/rolling_kl_variance.png")
    plt.close()

    # ================= 5. Action switching vs episode =================
    plt.figure()
    plt.plot(switch)
    plt.title("Action Switching Rate vs Episode")
    plt.xlabel("Episode")
    plt.ylabel("Switching Rate")
    plt.savefig(f"{plot_dir}/switching_vs_episode.png")
    plt.close()

    # ================= 6. Policy entropy slope =================
    entropy_slope = rolling_slope(entropy, WINDOW)
    plt.figure()
    plt.plot(entropy_slope)
    plt.title("Policy Entropy Slope (Rolling)")
    plt.xlabel("Episode")
    plt.ylabel("dH/dt")
    plt.savefig(f"{plot_dir}/entropy_slope.png")
    plt.close()

    # ================= 7. Gradient norm variance =================
    plt.figure()
    plt.plot(rolling_var(grad_norm, WINDOW))
    plt.title("Gradient Norm Rolling Variance")
    plt.xlabel("Training Step")
    plt.ylabel("Variance")
    plt.savefig(f"{plot_dir}/grad_norm_variance.png")
    plt.close()

    # ================= 8. Lyapunov-style queue stability =================
    plt.figure()
    plt.plot(congestion)
    plt.axhline(np.mean(congestion), linestyle="--")
    plt.title("Mean Congestion (Lyapunov Proxy)")
    plt.xlabel("Episode")
    plt.ylabel("Mean Fog Congestion")
    plt.savefig(f"{plot_dir}/queue_stability_congestion.png")
    plt.close()

    # ================= 9. Total Variation Distance (if possible) =================
    try:
        # ================= 9. Total Variation (TV) distance =================
        tv = []

        for e in range(1, len(actions)):
            try:
                A_prev = np.array(actions[e - 1], dtype=np.int64).reshape(-1)
                A_curr = np.array(actions[e], dtype=np.int64).reshape(-1)
                K = int(max(A_prev.max(), A_curr.max())) + 1
                p_prev = np.bincount(A_prev, minlength=K) / len(A_prev)
                p_curr = np.bincount(A_curr, minlength=K) / len(A_curr)
                tv.append(0.5 * np.sum(np.abs(p_prev - p_curr)))
            except Exception as ex:
                print(f"[WARN] TV distance skipped for {run}, episode {e}: {ex}")
                if len(tv) > 0:
                    plt.figure()
                    plt.plot(tv)
                    plt.title("Total Variation Distance vs Episode")
                    plt.xlabel("Episode")
                    plt.ylabel("TV Distance")
                    plt.savefig(f"{plot_dir}/tv_distance.png")
                    plt.close()


    except Exception as e:
        print(f"[WARN] TV distance skipped for {run}: {e}")

print("✔ All requested stability metrics computed and plotted.")

