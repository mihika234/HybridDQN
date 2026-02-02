import os
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

# ======================================================
# CONFIG
# ======================================================
BASE_DIR = "training"
CLASSICAL_DIR = os.path.join(BASE_DIR, "classical")
QUANTUM_DIR   = os.path.join(BASE_DIR, "quantum")

TAIL_FRAC = 0.2
OUT_DIR = "analysis_outputs_final_2"
os.makedirs(OUT_DIR, exist_ok=True)

EPS = 1e-8

# ======================================================
# METRIC DEFINITIONS
# ======================================================
def relative_variability(x):
    return np.std(x) / (np.abs(np.mean(x)) + EPS)

def reward_iqr(x):
    return np.percentile(x, 75) - np.percentile(x, 25)

def jains_fairness(counts):
    return (counts.sum() ** 2) / (len(counts) * (counts ** 2).sum() + EPS)

def action_concentration(counts):
    return np.max(counts) / (np.sum(counts) + EPS)

def gradient_zero_fraction(g):
    if len(g) == 0:
        return np.nan
    return np.mean(np.abs(g) < 1e-6)

def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    gt = sum(x > y for x in a for y in b)
    lt = sum(x < y for x in a for y in b)
    return (gt - lt) / (len(a) * len(b))

def mann_whitney(a, b):
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    _, p = mannwhitneyu(a, b, alternative="two-sided")
    return p

# ======================================================
# SINGLE RUN ANALYSIS
# ======================================================
def analyze_run(run_dir):
    results_dir = os.path.join(run_dir, "results")
    plots_dir   = os.path.join(run_dir, "plots")

    out = {}

    cost = np.load(os.path.join(plots_dir, "avg_cost.npy"))
    tail_len = max(2, int(len(cost) * TAIL_FRAC))
    tail_cost = cost[-tail_len:]

    out["rv"] = relative_variability(tail_cost)
    out["iqr"] = reward_iqr(tail_cost)
    out["reward_mean"] = np.mean(tail_cost)

    grad = np.load(os.path.join(results_dir, "grad_norm.npy"))
    if len(grad) >= 2:
        out["grad_var"] = np.var(grad)
        out["grad_zero_frac"] = gradient_zero_fraction(grad)
    else:
        out["grad_var"] = np.nan
        out["grad_zero_frac"] = np.nan

    action_hist = np.load(os.path.join(results_dir, "action_hist.npy"))
    out["jain"] = jains_fairness(action_hist)
    out["action_conc"] = action_concentration(action_hist)

    return out

# ======================================================
# GROUP ANALYSIS
# ======================================================
def analyze_group(base_dir):
    metrics = {
        "rv": [], "iqr": [], "reward_mean": [],
        "jain": [], "action_conc": [],
        "grad_var": [], "grad_zero_frac": []
    }

    for run in sorted(glob(os.path.join(base_dir, "*"))):
        if run.endswith("*new"):
            continue
        try:
            m = analyze_run(run)
            for k in metrics:
                metrics[k].append(m[k])
        except Exception as e:
            print(f"Skipping {run}: {e}")

    for k in metrics:
        metrics[k] = np.array(metrics[k])

    return metrics

# ======================================================
# LOAD DATA
# ======================================================
C = analyze_group(CLASSICAL_DIR)
Q = analyze_group(QUANTUM_DIR)

# ======================================================
# SUMMARY PRINT
# ======================================================
def summarize(name, c, q, lower_better=True):
    delta = cliffs_delta(q, c) if lower_better else cliffs_delta(c, q)
    p = mann_whitney(c, q)

    print(f"\n{name}")
    print("-" * len(name))
    print(f"Classical median: {np.nanmedian(c):.4f}")
    print(f"Quantum   median: {np.nanmedian(q):.4f}")
    print(f"Mann–Whitney p : {p}")
    print(f"Cliff’s delta  : {delta:.3f}")

print("\n=== FINAL STATISTICAL SUMMARY ===")

summarize("Relative Variability (↓ better)", C["rv"], Q["rv"], True)
summarize("Reward IQR (↓ better)", C["iqr"], Q["iqr"], True)
summarize("Load Balancing – Jain (↑ better)", C["jain"], Q["jain"], False)
summarize("Action Concentration (↓ better)", C["action_conc"], Q["action_conc"], True)
summarize("Gradient Variance (↓ better)", C["grad_var"], Q["grad_var"], True)
summarize("Gradient Zero Fraction (↑ regularized)", C["grad_zero_frac"], Q["grad_zero_frac"], False)

# ======================================================
# PLOTTING
# ======================================================
def boxplot(c, q, ylabel, title, fname, log=False):
    c, q = c[~np.isnan(c)], q[~np.isnan(q)]
    plt.figure(figsize=(5,4))
    plt.boxplot([c, q], tick_labels=["Classical", "Quantum"], widths=0.6)
    if log:
        plt.yscale("log")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=300)
    plt.close()

boxplot(C["jain"], Q["jain"],
        "Jain’s Fairness Index (↑ better)",
        "Load Balancing",
        "jain_boxplot.png")

boxplot(C["action_conc"], Q["action_conc"],
        "Action Concentration (↓ better)",
        "Mode Collapse",
        "action_concentration.png")

boxplot(C["grad_var"] + EPS, Q["grad_var"] + EPS,
        "Gradient Variance (log, ↓ better)",
        "Optimization Stability",
        "gradvar_boxplot.png",
        log=True)

boxplot(C["rv"], Q["rv"],
        "Relative Reward Variability (↓ better)",
        "Convergence Stability",
        "rv_boxplot.png")

print(f"\nPlots saved to: {OUT_DIR}/")

