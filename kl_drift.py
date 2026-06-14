import os
import numpy as np
import pandas as pd

BASE_DIR = "training/new environment final/baseline"
N_LAST = 100  # last episodes window

FAMILIES = ["classical", "quantum", "hamiltonian"]


# ============================================================
# Extract config (task_prob, delay)
# ============================================================
def extract_config(name):
    try:
        parts = name.split("_")
        return f"{parts[-2]}_{parts[-1]}"
    except:
        return name


# ============================================================
# Process one family
# ============================================================
def process_family(family_name):
    family_dir = os.path.join(BASE_DIR, family_name)
    rows = []

    for exp in sorted(os.listdir(family_dir)):
        exp_dir = os.path.join(family_dir, exp)
        metrics_path = os.path.join(exp_dir, "results", "metrics_dump.npz")

        if not os.path.isfile(metrics_path):
            continue

        data = np.load(metrics_path)

        if "kl_drifts" not in data:
            print(f"[WARN] kl_drifts missing in {metrics_path}")
            continue

        kl = data["kl_drifts"]
        kl_last = kl[-N_LAST:] if len(kl) >= N_LAST else kl

        rows.append({
            "family": family_name,
            "experiment": exp,
            "config": extract_config(exp),
            "mean_kl": kl_last.mean(),
            "std_kl": kl_last.std(),
            "max_kl": kl_last.max()
        })

    return rows


# ============================================================
# Collect all data
# ============================================================
all_rows = []

for fam in FAMILIES:
    all_rows.extend(process_family(fam))

df = pd.DataFrame(all_rows)

# ============================================================
# Save raw CSV
# ============================================================
out_csv = os.path.join(BASE_DIR, "kl_summary_all.csv")
df.to_csv(out_csv, index=False)
print(f"\n✅ Saved: {out_csv}")


# ============================================================
# Compute % differences vs classical
# ============================================================
summary_rows = []

configs = df["config"].unique()

for cfg in configs:
    sub = df[df["config"] == cfg]

    try:
        c = sub[sub["family"] == "classical"]["mean_kl"].values[0]
        q = sub[sub["family"] == "quantum"]["mean_kl"].values[0]
        h = sub[sub["family"] == "hamiltonian"]["mean_kl"].values[0]
    except IndexError:
        continue

    # % difference: (other - classical) / classical * 100
    pct_q = ((q - c) / c) * 100 if c != 0 else np.nan
    pct_h = ((h - c) / c) * 100 if c != 0 else np.nan

    summary_rows.append({
        "config": cfg,
        "classical_mean": c,
        "quantum_mean": q,
        "hamiltonian_mean": h,
        "%diff_quantum_vs_classical": pct_q,
        "%diff_hamiltonian_vs_classical": pct_h
    })

summary_df = pd.DataFrame(summary_rows)

# ============================================================
# Save comparison CSV
# ============================================================
cmp_csv = os.path.join(BASE_DIR, "kl_comparison_vs_classical.csv")
summary_df.to_csv(cmp_csv, index=False)
print(f"✅ Saved: {cmp_csv}")


# ============================================================
# Pretty print
# ============================================================
print("\n=== KL Comparison (last 100 episodes) ===")
print(summary_df.to_string(index=False, float_format="%.3e"))
