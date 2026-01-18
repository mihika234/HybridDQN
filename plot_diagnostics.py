import os
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# CONFIG
# -------------------------------
RESULTS_DIR = ""   # set to your results folder if needed
SAVE_DIR = "./plots"
os.makedirs(SAVE_DIR, exist_ok=True)

def load(name):
    path = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(path):
        print(f"[WARN] {name} not found")
        return None
    return np.load(path)

# -------------------------------
# LOAD FILES
# -------------------------------
action_hist = load("action_hist.npy")      # (n_actions,)
buffer_size = load("buffer_size.npy")      # (T,)
grad_norm   = load("grad_norm.npy")        # (T,)
loss        = load("loss.npy")              # (T,)
q_stats     = load("q_stats.npy")           # (2, T)
entropy     = load("action_entropy.npy")

# NEW: drop breakdown
drop_iot   = load("drop_iot.npy")
drop_trans = load("drop_trans.npy")
drop_fog   = load("drop_fog.npy")
drop_soft  = load("drop_soft.npy")

# -------------------------------
# 1. ACTION DISTRIBUTION
# -------------------------------
if action_hist is not None:
    plt.figure(figsize=(6,4))
    plt.bar(range(len(action_hist)), action_hist)
    plt.xlabel("Action (0=Local, 1..Fog)")
    plt.ylabel("Fraction")
    plt.title("Overall Action Distribution")
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/action_distribution.png", dpi=300)
    plt.close()

# -------------------------------
# 2. REPLAY BUFFER SIZE
# -------------------------------
if buffer_size is not None:
    plt.figure(figsize=(6,4))
    plt.plot(buffer_size)
    plt.xlabel("Training Step")
    plt.ylabel("Transitions Stored")
    plt.title("Replay Buffer Size")
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/buffer_size.png", dpi=300)
    plt.close()

# -------------------------------
# 3. GRADIENT NORM
# -------------------------------
if grad_norm is not None:
    plt.figure(figsize=(6,4))
    plt.plot(grad_norm, alpha=0.7)
    plt.axhline(1.0, color="red", linestyle="--", label="Clip Threshold")
    plt.xlabel("Training Step")
    plt.ylabel("Gradient Norm")
    plt.title("Gradient Norm (Clipped)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/grad_norm.png", dpi=300)
    plt.close()

# -------------------------------
# 4. LOSS CURVE (LOG SCALE)
# -------------------------------
if loss is not None:
    plt.figure(figsize=(6,4))
    plt.plot(loss)
    plt.yscale("log")
    plt.xlabel("Training Step")
    plt.ylabel("DQN Loss (log)")
    plt.title("Training Loss")
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/loss.png", dpi=300)
    plt.close()

# -------------------------------
# 5. Q-VALUE STATISTICS
# -------------------------------
if q_stats is not None and q_stats.ndim == 2 and q_stats.shape[0] == 2:
    q_mean = q_stats[0]
    q_std  = q_stats[1]

    plt.figure(figsize=(6,4))
    plt.plot(q_mean, label="Q mean")
    plt.plot(q_std, label="Q std")
    plt.xlabel("Training Step")
    plt.ylabel("Q-value")
    plt.title("Q-value Statistics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/q_stats.png", dpi=300)
    plt.close()

# -------------------------------
# 6. ACTION ENTROPY
# -------------------------------
if entropy is not None:
    plt.figure(figsize=(8,4))
    plt.plot(entropy, label="Action Entropy")
    plt.xlabel("Episode")
    plt.ylabel("Entropy")
    plt.title("Policy Action Entropy Over Training")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/action_entropy.png", dpi=300)
    plt.close()

# ============================================================
# 7. DROP LOCATION BREAKDOWN (NEW)
# ============================================================
def plot_drop_breakdown(
    drop_iot, drop_trans, drop_fog, drop_soft,
    save_path, window=10
):
    """
    Stacked bar plot of drop locations.
    window: smoothing window (episodes)
    """

    def smooth(x, w):
        return np.convolve(x, np.ones(w)/w, mode="valid")

    drop_iot   = smooth(drop_iot, window)
    drop_trans = smooth(drop_trans, window)
    drop_fog   = smooth(drop_fog, window)
    drop_soft  = smooth(drop_soft, window)

    x = np.arange(len(drop_iot))

    plt.figure(figsize=(10, 5))

    plt.bar(x, drop_iot, label="IoT drops")
    plt.bar(x, drop_trans, bottom=drop_iot, label="Transmission drops")
    plt.bar(x, drop_fog, bottom=drop_iot + drop_trans, label="Fog drops")
    plt.bar(
        x,
        drop_soft,
        bottom=drop_iot + drop_trans + drop_fog,
        label="Soft drops",
        alpha=0.8
    )

    plt.xlabel("Episode")
    plt.ylabel("Number of Dropped Tasks")
    plt.title("Drop Location Breakdown (Smoothed)")
    plt.legend()
    plt.grid(alpha=0.3)

    os.makedirs(save_path, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "drop_breakdown.png"), dpi=300)
    plt.close()

# Call drop breakdown plot
if all(x is not None for x in [drop_iot, drop_trans, drop_fog, drop_soft]):
    plot_drop_breakdown(
        drop_iot, drop_trans, drop_fog, drop_soft,
        SAVE_DIR, window=10
    )

print("✅ Diagnostic plots saved to:", SAVE_DIR)
