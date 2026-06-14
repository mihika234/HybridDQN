import os
import numpy as np

BASE_DIR = "training/new environment final"
N_LAST = 50
TOP_N = 8

def load_mean_kl(path):
    data = np.load(path)
    kl = data["kl_drifts"]
    kl_last = kl[-N_LAST:] if len(kl) >= N_LAST else kl
    return kl_last.mean()

def collect(family):
    out = {}
    family_dir = os.path.join(BASE_DIR, family)
    for exp in os.listdir(family_dir):
        p = os.path.join(family_dir, exp, "results", "metrics_dump.npz")
        if not os.path.isfile(p):
            continue

        # strip prefix: classical_ / quantum_
        key = exp.split("_", 1)[1]   # e.g. "0.15_10"
        out[key] = load_mean_kl(p)
    return out

classical = collect("classical")
quantum   = collect("quantum")

rows = []

for key, c_kl in classical.items():
    if key in quantum:
        q_kl = quantum[key]
        pct = 100 * (c_kl - q_kl) / max(c_kl, 1e-12)
        rows.append((key, c_kl, q_kl, pct))

rows.sort(key=lambda x: x[3], reverse=True)

print(f"{'Config':<15} {'Classical':>12} {'Quantum':>12} {'Δ%':>8}")
print("-"*55)

for r in rows[:TOP_N]:
    print(f"{r[0]:<15} {r[1]:>12.3e} {r[2]:>12.3e} {r[3]:>7.1f}%")

