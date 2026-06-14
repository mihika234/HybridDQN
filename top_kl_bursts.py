import os
import numpy as np

BASE_DIR = "training/new environment final"
N_LAST = 100
TOP_N = 10

def mean_kl_from_folder(folder_path):
    npz = os.path.join(folder_path, "results", "metrics_dump.npz")
    if not os.path.isfile(npz):
        return None
    data = np.load(npz)
    if "kl_drifts" not in data:
        return None
    kl = data["kl_drifts"]
    kl_last = kl[-N_LAST:] if len(kl) >= N_LAST else kl
    return kl_last.mean()


def collect_runs(algo):
    """
    algo = 'classical' or 'quantum'
    expects folders like: classical 0.15 10
    """
    runs = {}
    algo_root = os.path.join(BASE_DIR, algo)
    if not os.path.isdir(algo_root):
        print(f"[ERROR] Missing directory: {algo_root}")
        return runs

    for folder in os.listdir(algo_root):
        parts = folder.split()
        if len(parts) != 3:
            continue

        tag, lam, N = parts
        if tag.lower() != algo:
            continue

        key = f"{lam}_{N}"
        folder_path = os.path.join(algo_root, folder)
        kl_mean = mean_kl_from_folder(folder_path)

        if kl_mean is not None:
            runs[key] = kl_mean
        else:
            print(f"[WARN] No metrics_dump.npz in {folder_path}")

    return runs


classical = collect_runs("classical")
quantum   = collect_runs("quantum")

print(f"[INFO] Found {len(classical)} classical runs")
print(f"[INFO] Found {len(quantum)} quantum runs")

rows = []
for key in classical:
    if key in quantum:
        c = classical[key]
        q = quantum[key]
        pct = 100 * (c - q) / max(c, 1e-12)
        rows.append((key, c, q, pct))

if not rows:
    print("[ERROR] No matched (classical, quantum) pairs found.")
    print("Classical keys:", sorted(classical.keys()))
    print("Quantum keys:", sorted(quantum.keys()))
    exit(1)

rows.sort(key=lambda x: x[3], reverse=True)

print(f"\n{'Config (λ_N)':<15} {'Classical':>12} {'Quantum':>12} {'Δ%':>8}")
print("-" * 55)

for key, c, q, pct in rows[:TOP_N]:
    print(f"{key:<15} {c:>12.3e} {q:>12.3e} {pct:>7.1f}%")

