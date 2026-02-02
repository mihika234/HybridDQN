import os
import numpy as np
from glob import glob
from scipy.stats import mannwhitneyu

# ================= CONFIG =================
BASE_DIR = "training"
CLASSICAL_DIR = os.path.join(BASE_DIR, "classical")
QUANTUM_DIR   = os.path.join(BASE_DIR, "quantum")

# Change this if needed:
# examples: "*new new", "*new1"
RUN_FILTER = "*new1"

TAIL_FRAC = 0.2     # last 20% defines final value
TOL = 0.10          # 10% tolerance
EPS = 1e-8

# ============== METRICS ===================

def convergence_step(curve, tail_frac=TAIL_FRAC, tol=TOL):
    """
    First index after which curve stays within ±tol of final value.
    Returns np.nan if never converges.
    """
    curve = np.asarray(curve)
    T = len(curve)
    tail_len = max(2, int(T * tail_frac))
    final_val = np.mean(curve[-tail_len:])
    band = tol * abs(final_val)

    for t in range(T):
        if np.all(np.abs(curve[t:] - final_val) <= band):
            return t
    return np.nan


def total_variation(curve, normalize=True):
    """
    Sum of absolute differences between consecutive steps.
    """
    curve = np.asarray(curve)
    tv = np.sum(np.abs(np.diff(curve)))
    if normalize:
        tv /= (len(curve) - 1)
    return tv

# ============== DATA LOADING ==============

def analyze_group(base_dir):
    conv_steps = []
    tvs = []

    for run in glob(os.path.join(base_dir, RUN_FILTER)):
        try:
            curve = np.load(os.path.join(run, "plots", "avg_cost.npy"))
            conv_steps.append(convergence_step(curve))
            tvs.append(total_variation(curve))
        except:
            pass

    return np.array(conv_steps), np.array(tvs)

C_conv, C_tv = analyze_group(CLASSICAL_DIR)
Q_conv, Q_tv = analyze_group(QUANTUM_DIR)

# Remove NaNs for stats
C_conv_clean = C_conv[~np.isnan(C_conv)]
Q_conv_clean = Q_conv[~np.isnan(Q_conv)]

# ============== STATISTICS ================

def stats_report(name, C, Q, lower_better=True):
    if len(C) < 2 or len(Q) < 2:
        print(f"\n{name}: insufficient data")
        return

    c_med = np.median(C)
    q_med = np.median(Q)

    _, p = mannwhitneyu(C, Q, alternative="two-sided")

    print(f"\n{name}")
    print("-" * len(name))
    print(f"Classical median: {c_med:.4f}")
    print(f"Quantum   median: {q_med:.4f}")
    print(f"Better when: {'LOWER' if lower_better else 'HIGHER'}")
    print(f"Mann–Whitney p: {p}")

# ============== OUTPUT ====================

print("\n=== CONVERGENCE & SMOOTHNESS METRICS (Env A) ===")

stats_report(
    "Convergence Step (learning speed)",
    C_conv_clean, Q_conv_clean,
    lower_better=True
)

stats_report(
    "Total Variation (oscillation)",
    C_tv, Q_tv,
    lower_better=True
)

