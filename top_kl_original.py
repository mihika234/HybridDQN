import os
import numpy as np

BASE_DIR = "training"
ENV = "original env"
N_LAST = 100
TOP_N = 10


def mean_kl(run_dir):
    npz = os.path.join(run_dir, "results", "metrics_dump.npz")
    if not os.path.isfile(npz):
        return None
    data = np.load(npz)
    if "kl_drifts" not in data:
        return None
    kl = data["kl_drifts"]
    kl_last = kl[-N_LAST:] if len(kl) >= N_LAST else kl
    return kl_last.mean()


def collect(algo):
    """
    algo ∈ {'classical', 'quantum'}
    folders like: 'classical 0.15 10' / 'quantum 0.15 10'
    """
    root = os.path.join(BASE_DIR, algo, ENV)
    runs = {}

    if not os.path.isdir(root):
        print(f"[ERROR] Missing directory: {root}")
        return runs

    for folder in os.listdir(root):
        parts = folder.split()
        if len(parts) != 3:
            continue

        tag, lam, N = parts
        if tag.lower() != algo:
            continue

        key = f"{lam}_{N}"
        run_path = os.path.join(root, folder)
        val = mean_kl(run_path)

        if val is not None:
            runs[key] = val
        else:
            print(f"[WARN] Missing metrics_dump.npz in {run_path}")

    return runs


# --- collect ---
classical = collect("classical")
quantum   = collect("quantum")

print(f"[INFO] Classical runs: {len(classical)}")
print(f"[INFO] Quantum runs:   {len(quantum)}")

# --- pair ---
rows = []
for key in classical:
    if key in quantum:
        c = classical[key]
        q = quantum[key]
        pct = 100 * (c - q) / max(c, 1e-12)
        rows.append((key, c, q, pct))

if not rows:
    print("[ERROR] No matching classical–quantum pairs found")
    print("Classical keys:", sorted(classical.keys()))
    print("Quantum keys:", sorted(quantum.keys()))
    exit(1)

rows.sort(key=lambda x: x[3], reverse=True)

# --- print ---
print(f"\n{'Config (λ_N)':<15} {'Classical':>12} {'Quantum':>12} {'Δ%':>8}")
print("-" * 55)

for key, c, q, pct in rows[:TOP_N]:
    print(f"{key:<15} {c:>12.3e} {q:>12.3e} {pct:>7.1f}%")

