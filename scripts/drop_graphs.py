import os
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# =========================
# CONFIG (LOCK THESE)
# =========================
BASE_DIR = "training"
ENV_DIR = "4nl env"

TASK_PROBS = [0.08, 0.10, 0.12, 0.15, 0.17, 0.19, 0.20]
MAX_DELAYS = [10, 15, 20]

WINDOW = 50        # rolling window (episodes)
DELTA = 1.0        # ≤ 1 drop ≈ zero-drop regime

OUT_FIG = "risk_stabilization_from_npy.png"

# =========================
# STABILITY METRIC (COUNT-BASED)
# =========================
def compute_stable_episode_from_counts(total_drops):
    if len(total_drops) < WINDOW:
        return np.nan

    rolling = np.convolve(
        total_drops,
        np.ones(WINDOW) / WINDOW,
        mode="valid"
    )

    for i in range(len(rolling)):
        if np.all(rolling[i:] <= DELTA):
            return i + WINDOW

    return np.nan


# =========================
# LOAD TOTAL DROPS
# =========================
def load_total_drops(run_path):
    try:
        di = np.load(os.path.join(run_path, "results", "drop_iot.npy"))
        df = np.load(os.path.join(run_path, "results", "drop_fog.npy"))
        dt = np.load(os.path.join(run_path, "results", "drop_trans.npy"))
        ds = np.load(os.path.join(run_path, "results", "drop_soft.npy"))
    except FileNotFoundError:
        return None

    return di + df + dt + ds


# =========================
# COLLECT RUNS
# =========================
def collect_agent(agent):
    root = os.path.join(BASE_DIR, agent, ENV_DIR)
    data = defaultdict(list)

    if not os.path.exists(root):
        raise FileNotFoundError(root)

    for run in os.listdir(root):
        run_path = os.path.join(root, run)
        if not os.path.isdir(run_path):
            continue

        # folder name contains: <prob> <maxDelay>
        match = re.search(r"([0-9.]+)\s+(\d+)", run)
        if not match:
            continue

        p = float(match.group(1))
        md = int(match.group(2))

        total_drops = load_total_drops(run_path)
        if total_drops is None:
            continue

        t_stable = compute_stable_episode_from_counts(total_drops)
        data[(p, md)].append(t_stable)

    return data


# =========================
# AGGREGATE (MEDIAN)
# =========================
def summarize(data):
    summary = {}
    for key, vals in data.items():
        vals = np.array(vals)
        finite = vals[~np.isnan(vals)]
        summary[key] = np.median(finite) if len(finite) > 0 else np.nan
    return summary


# =========================
# PLOT (ONE GRAPH, BOTH AGENTS)
# =========================
def plot_comparison(classical, quantum):
    plt.figure(figsize=(9, 5))

    styles = {
        ("classical", 10): ("--", "tab:blue"),
        ("classical", 15): ("--", "tab:orange"),
        ("classical", 20): ("--", "tab:green"),
        ("quantum",   10): ("-",  "tab:blue"),
        ("quantum",   15): ("-",  "tab:orange"),
        ("quantum",   20): ("-",  "tab:green"),
    }

    for agent, summary in [("classical", classical), ("quantum", quantum)]:
        for md in MAX_DELAYS:
            y = []
            for p in TASK_PROBS:
                y.append(summary.get((p, md), np.nan))

            linestyle, color = styles[(agent, md)]
            label = f"{agent.capitalize()}, MaxDelay={md}"

            plt.plot(
                TASK_PROBS,
                y,
                marker="o",
                linestyle=linestyle,
                color=color,
                label=label
            )

    plt.xlabel("Task Arrival Probability")
    plt.ylabel("Episodes to Stable Zero Drops")
    plt.title("Risk Stabilization Time vs Load (4-NL MEC)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(OUT_FIG, dpi=300)
    plt.close()

    print(f"[INFO] Saved: {OUT_FIG}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    print("[INFO] Loading Classical runs...")
    classical_raw = collect_agent("classical")

    print("[INFO] Loading Quantum runs...")
    quantum_raw = collect_agent("quantum")

    classical_summary = summarize(classical_raw)
    quantum_summary = summarize(quantum_raw)

    print("[INFO] Plotting...")
    plot_comparison(classical_summary, quantum_summary)

