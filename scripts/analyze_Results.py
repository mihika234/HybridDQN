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

TAIL_FRAC = 0.2      # last 20% episodes = convergence window
SAVE_DIR  = "analysis_outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

# ======================================================
# METRIC DEFINITIONS
# ======================================================
def coefficient_of_variation(x):
    x = np.asarray(x)
    return np.std(x) / (np.mean(x) + 1e-8)

def jains_fairness(counts):
    counts = np.asarray(counts)
    return (counts.sum() ** 2) / (len(counts) * (counts ** 2).sum() + 1e-8)

def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    gt = sum(x > y for x in a for y in b)
    lt = sum(x < y for x in a for y in b)
    return (gt - lt) / (len(a) * len(b))

def mann_whitney(a, b):
    _, p = mannwhitneyu(a, b, alternative="two-sided")
    return p

# ======================================================
# SINGLE RUN ANALYSIS
# ======================================================
def analyze_run(run_dir):
    results_dir = os.path.join(run_dir, "results")
    plots_dir   = os.path.join(run_dir, "plots")

    metrics = {}

    # ---- Stability (CV of avg_cost)
    cost = np.load(os.path.join(plots_dir, "avg_cost.npy"))
    tail_len = int(len(cost) * TAIL_FRAC)
    metrics["cv"] = coefficient_of_variation(cost[-tail_len:])

    # ---- Gradient norm variance
    grad_norms = np.load(os.path.join(results_dir, "grad_norm.npy"))
    metrics["grad_var"] = np.var(grad_norms)

    # ---- Jain’s fairness (action histogram)
    action_hist = np.load(os.path.join(results_dir, "action_hist.npy"))
    metrics["jain"] = jains_fairness(action_hist)

    return metrics

# ======================================================
# GROUP ANALYSIS (ONLY folders ending with "new new")
# ======================================================
def analyze_group(base_path):
    cvs, jains, grad_vars = [], [], []

    for run in sorted(glob(os.path.join(base_path, "*"))):
        if not run.endswith("new new"):
            continue
        try:
            m = analyze_run(run)
            cvs.append(m["cv"])
            jains.append(m["jain"])
            grad_vars.append(m["grad_var"])
        except Exception as e:
            print(f"Skipping {run}: {e}")

    return np.array(cvs), np.array(jains), np.array(grad_vars)

# ======================================================
# LOAD DATA
# ======================================================
c_cv, c_jain, c_grad = analyze_group(CLASSICAL_DIR)
q_cv, q_jain, q_grad = analyze_group(QUANTUM_DIR)

# ======================================================
# STATISTICS
# ======================================================
print("\n=== STATISTICS ===")

print("\nConvergence Stability (CV)")
print("Mann–Whitney p:", mann_whitney(c_cv, q_cv))
print("Cliff's Delta :", cliffs_delta(q_cv, c_cv))

print("\nLoad Balancing (Jain)")
print("Mann–Whitney p:", mann_whitney(c_jain, q_jain))
print("Cliff's Delta :", cliffs_delta(c_jain, q_jain))

print("\nGradient Norm Variance")
print("Mann–Whitney p:", mann_whitney(c_grad, q_grad))
print("Cliff's Delta :", cliffs_delta(q_grad, c_grad))

# ======================================================
# PLOTTING
# ======================================================
def boxplot(data_c, data_q, ylabel, title, filename):
    plt.figure(figsize=(5,4))
    plt.boxplot([data_c, data_q], labels=["Classical", "Quantum"], widths=0.6)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, filename), dpi=300)
    plt.close()

boxplot(
    c_cv, q_cv,
    "Coefficient of Variation (↓ better)",
    "Convergence Stability",
    "cv_boxplot.png"
)

boxplot(
    c_jain, q_jain,
    "Jain’s Fairness Index (↑ better)",
    "Load Balancing",
    "jain_boxplot.png"
)

boxplot(
    c_grad, q_grad,
    "Gradient Norm Variance (↓ better)",
    "Optimization Stability",
    "gradvar_boxplot.png"
)

print(f"\nPlots saved to: {SAVE_DIR}/")

