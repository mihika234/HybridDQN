import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from scipy.stats import mannwhitneyu

BASE = "training"
GROUPS = {
    "Classical": os.path.join(BASE, "classical"),
    "Quantum": os.path.join(BASE, "quantum"),
}

OUT = "analysis_envB_partial"
os.makedirs(OUT, exist_ok=True)

TAIL_FRAC = 0.2
EPS = 1e-8

# ---------------- helpers ----------------

def tail(x):
    n = max(2, int(len(x) * TAIL_FRAC))
    return x[-n:]

def action_concentration(hist):
    return np.max(hist) / (np.sum(hist) + EPS)

def effective_support(ent):
    return np.exp(np.mean(ent))

def safe_load(path):
    return np.load(path) if os.path.exists(path) else None

# ---------------- per-run ----------------

def analyze_run(run):
    R = os.path.join(run, "results")
    out = {}

    grad = safe_load(os.path.join(R, "grad_norm.npy"))
    if grad is not None and len(grad) > 1:
        out["grad_var"] = np.var(grad)
    else:
        out["grad_var"] = np.nan

    hist = safe_load(os.path.join(R, "action_hist.npy"))
    if hist is not None:
        out["action_conc"] = action_concentration(hist)

    ent = safe_load(os.path.join(R, "action_entropy.npy"))
    if ent is not None:
        out["eff_support"] = effective_support(ent)

    loss = safe_load(os.path.join(R, "loss.npy"))
    if loss is not None:
        t = tail(loss)
        out["loss_iqr"] = np.percentile(t, 75) - np.percentile(t, 25)

    drops = []
    for f in ["drop_fog.npy", "drop_iot.npy", "drop_soft.npy", "drop_trans.npy"]:
        d = safe_load(os.path.join(R, f))
        if d is not None:
            drops.append(np.mean(d))
    if drops:
        out["drop_total"] = np.mean(drops)

    return out

# ---------------- group ----------------

def analyze_group(base):
    data = {}
    for run in glob(os.path.join(base, "*new new")):
        try:
            m = analyze_run(run)
            for k, v in m.items():
                data.setdefault(k, []).append(v)
        except:
            pass
    return {k: np.array(v) for k, v in data.items()}

DATA = {k: analyze_group(v) for k, v in GROUPS.items()}

# ---------------- summary ----------------

def summarize(metric, lower_better=True):
    c = DATA["Classical"].get(metric, np.array([]))
    q = DATA["Quantum"].get(metric, np.array([]))

    c = c[~np.isnan(c)]
    q = q[~np.isnan(q)]

    if len(c) < 2 or len(q) < 2:
        print(f"{metric}: insufficient data")
        return

    med_c, med_q = np.median(c), np.median(q)
    _, p = mannwhitneyu(c, q, alternative="two-sided")

    print(f"\n{metric}")
    print("-" * len(metric))
    print(f"Classical median: {med_c:.4f}")
    print(f"Quantum   median: {med_q:.4f}")
    print(f"Direction: {'↓' if lower_better else '↑'} better")
    print(f"Mann–Whitney p: {p}")

# ---------------- run summaries ----------------

print("\n=== ENV B (PARTIAL DATA) SUMMARY ===")

summarize("grad_var", lower_better=True)
summarize("action_conc", lower_better=True)
summarize("eff_support", lower_better=False)
summarize("loss_iqr", lower_better=True)
summarize("drop_total", lower_better=True)

# ---------------- plots ----------------

def boxplot(metric, ylabel, log=False):
    c = DATA["Classical"].get(metric, np.array([]))
    q = DATA["Quantum"].get(metric, np.array([]))
    c, q = c[~np.isnan(c)], q[~np.isnan(q)]

    if len(c) == 0 or len(q) == 0:
        return

    plt.figure(figsize=(5,4))
    plt.boxplot([c, q], tick_labels=["Classical", "Quantum"])
    if log:
        plt.yscale("log")
    plt.ylabel(ylabel)
    plt.title(metric.replace("_", " ").title())
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{OUT}/{metric}.png", dpi=300)
    plt.close()

boxplot("grad_var", "Gradient Variance (log)", log=True)
boxplot("action_conc", "Action Concentration (↓)")
boxplot("eff_support", "Effective Action Support (↑)")
boxplot("loss_iqr", "Loss IQR (↓)")
boxplot("drop_total", "Constraint Violation Rate (↓)")

print(f"\nPlots saved in `{OUT}/`")

