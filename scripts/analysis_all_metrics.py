import os
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

BASE_DIR = "training"
ENV_DIRS = {
    "Classical": os.path.join(BASE_DIR, "classical"),
    "Quantum": os.path.join(BASE_DIR, "quantum")
}

TAIL_FRAC = 0.2
EPS = 1e-8
OUT = "analysis_all_metrics"
os.makedirs(OUT, exist_ok=True)

# ------------------ helpers ------------------

def tail(x):
    n = max(2, int(len(x) * TAIL_FRAC))
    return x[-n:]

def cvar(x, q=5):
    return np.mean(np.percentile(x, np.arange(0, q)))

def jain(x):
    return (np.sum(x) ** 2) / (len(x) * np.sum(x**2) + EPS)

def action_concentration(x):
    return np.max(x) / (np.sum(x) + EPS)

def effective_support(entropy):
    return np.exp(entropy)

# ------------------ extraction ------------------

def analyze_run(run):
    R = os.path.join(run, "results")
    P = os.path.join(run, "plots")

    out = {}

    # returns
    ret = np.load(os.path.join(R, "rv_return.npy"))
    out["reward_iqr"] = np.percentile(tail(ret), 75) - np.percentile(tail(ret), 25)
    out["reward_cvar"] = cvar(ret)

    # energy / delay
    energy = np.load(os.path.join(R, "rv_energy.npy"))
    delay = np.load(os.path.join(P, "avg_delay.npy"))

    success = 1.0 - np.mean(np.load(os.path.join(P, "dropped_ratio.npy")))
    out["energy_per_success"] = np.mean(energy) / (success + EPS)
    out["delay_per_success"] = np.mean(delay) / (success + EPS)

    # constraint violations
    out["drop_total"] = np.mean([
        np.mean(np.load(os.path.join(R, f)))
        for f in ["drop_fog.npy", "drop_iot.npy", "drop_soft.npy", "drop_trans.npy"]
    ])

    # policy shape
    hist = np.load(os.path.join(R, "action_hist.npy"))
    out["action_conc"] = action_concentration(hist)

    ent = np.load(os.path.join(R, "action_entropy.npy"))
    out["eff_support"] = effective_support(np.mean(ent))

    # optimization
    grad = np.load(os.path.join(R, "grad_norm.npy"))
    out["grad_var"] = np.var(grad) if len(grad) > 1 else np.nan

    return out

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

DATA = {name: analyze_group(path) for name, path in ENV_DIRS.items()}

from scipy.stats import mannwhitneyu

def print_summary(metric, higher_better=True):
    c = np.array(DATA["Classical"].get(metric, []))
    q = np.array(DATA["Quantum"].get(metric, []))

    # clean NaNs
    c = c[~np.isnan(c)]
    q = q[~np.isnan(q)]

    if len(c) == 0 or len(q) == 0:
        print(f"{metric}: no data")
        return

    c_med = np.median(c)
    q_med = np.median(q)

    try:
        _, p = mannwhitneyu(c, q, alternative="two-sided")
    except:
        p = np.nan

    direction = "↑" if higher_better else "↓"

    print(f"\n{metric}")
    print("-" * len(metric))
    print(f"Classical median: {c_med:.4f}")
    print(f"Quantum   median: {q_med:.4f}")
    print(f"Direction: {direction} better")
    print(f"Mann–Whitney p: {p}")


# ------------------ plotting ------------------

def boxplot(metric, ylabel, log=False):
    plt.figure(figsize=(5,4))
    vals = [DATA["Classical"][metric], DATA["Quantum"][metric]]
    plt.boxplot(vals, tick_labels=["Classical", "Quantum"])
    if log:
        plt.yscale("log")
    plt.ylabel(ylabel)
    plt.title(metric.replace("_", " ").title())
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{OUT}/{metric}.png", dpi=300)
    plt.close()

PLOTS = {
    "reward_iqr": ("Reward IQR (↓)", False),
    "reward_cvar": ("Worst-case reward (↑)", False),
    "drop_total": ("Constraint violation rate (↓)", False),
    "energy_per_success": ("Energy per success (↓)", True),
    "delay_per_success": ("Delay per success (↓)", True),
    "action_conc": ("Action concentration (↓)", False),
    "eff_support": ("Effective action support (↑)", False),
    "grad_var": ("Gradient variance (log)", True),
}

for k, (lab, lg) in PLOTS.items():
    boxplot(k, lab, lg)

# ------------------ win rate ------------------

wins = 0
total = 0
for c, q in zip(DATA["Classical"]["reward_iqr"], DATA["Quantum"]["reward_iqr"]):
    if q < c:
        wins += 1
    total += 1

print("\n=== WIN RATE (Quantum better) ===")
print(f"{wins}/{total}  =  {wins/total:.2%}")
print(f"\nPlots saved in `{OUT}/`")

print("\n=== METRIC SUMMARY ===")

print_summary("reward_iqr", higher_better=False)
print_summary("reward_cvar", higher_better=True)

print_summary("drop_total", higher_better=False)

print_summary("energy_per_success", higher_better=False)
print_summary("delay_per_success", higher_better=False)

print_summary("action_conc", higher_better=False)
print_summary("eff_support", higher_better=True)

print_summary("grad_var", higher_better=False)


