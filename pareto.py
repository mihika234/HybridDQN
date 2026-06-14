import os
import matplotlib.pyplot as plt
import pandas as pd

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = "training/new environment final/baseline"
FAMILIES = ["classical", "quantum", "hamiltonian"]

OUT_DIR = os.path.join(BASE_DIR, "energy_delay_analysis")
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================
def extract_config(name):
    try:
        parts = name.split("_")
        return f"{parts[-2]}_{parts[-1]}"
    except:
        return name


def parse_eval_file(filepath):
    try:
        with open(filepath, "r") as f:
            lines = f.readlines()

        delay = None
        energy = None

        for line in lines:
            if "avg_delay" in line:
                delay = float(line.split(":")[1].strip())
            if "avg_energy" in line:
                energy = float(line.split(":")[1].strip())

        return delay, energy

    except Exception as e:
        print(f"⚠️ Failed reading {filepath}: {e}")
        return None, None


# ============================================================
# COLLECT DATA
# ============================================================
data = {}

for fam in FAMILIES:
    fam_dir = os.path.join(BASE_DIR, fam)

    if not os.path.isdir(fam_dir):
        continue

    for exp in os.listdir(fam_dir):
        eval_path = os.path.join(fam_dir, exp, "results", "eval_results.txt")

        if not os.path.isfile(eval_path):
            continue

        cfg = extract_config(exp)
        delay, energy = parse_eval_file(eval_path)

        if delay is None:
            continue

        if cfg not in data:
            data[cfg] = {}

        data[cfg][fam] = {
            "delay": delay,
            "energy": energy
        }


# ============================================================
# FILTER COMPLETE CONFIGS
# ============================================================
filtered_data = {}

for cfg, methods in data.items():
    if all(m in methods for m in FAMILIES):
        filtered_data[cfg] = methods
    else:
        print(f"⚠️ Skipping {cfg} (missing: {set(FAMILIES) - set(methods.keys())})")

print(f"\n✅ Using {len(filtered_data)} complete configs")


# ============================================================
# SAVE CSV
# ============================================================
rows = []

for cfg, methods in filtered_data.items():
    rows.append({
        "config": cfg,
        "classical_delay": methods["classical"]["delay"],
        "classical_energy": methods["classical"]["energy"],
        "quantum_delay": methods["quantum"]["delay"],
        "quantum_energy": methods["quantum"]["energy"],
        "hamiltonian_delay": methods["hamiltonian"]["delay"],
        "hamiltonian_energy": methods["hamiltonian"]["energy"],
    })

df = pd.DataFrame(rows)
csv_path = os.path.join(OUT_DIR, "energy_delay_summary.csv")
df.to_csv(csv_path, index=False)

print(f"✅ Saved CSV: {csv_path}")


# ============================================================
# PER-CONFIG PLOTS
# ============================================================
for cfg, methods in filtered_data.items():

    plt.figure(figsize=(6,5))

    for method, vals in methods.items():
        plt.scatter(vals["delay"], vals["energy"], s=120, label=method.capitalize())

    plt.xlabel("Average Delay")
    plt.ylabel("Average Energy")
    plt.title(f"Energy–Delay Tradeoff ({cfg})")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{cfg}_tradeoff.png"), dpi=300)
    plt.close()

print("✅ Per-config plots saved.")


# ============================================================
# NORMALIZED GLOBAL PLOT
# ============================================================
norm_rows = []

for cfg, methods in filtered_data.items():

    base_delay = methods["classical"]["delay"]
    base_energy = methods["classical"]["energy"]

    if base_delay == 0 or base_energy == 0:
        continue

    for method, vals in methods.items():
        norm_rows.append({
            "method": method,
            "delay": vals["delay"] / base_delay,
            "energy": vals["energy"] / base_energy
        })

if len(norm_rows) == 0:
    print("⚠️ No normalized data available")
else:
    df_norm = pd.DataFrame(norm_rows)

    plt.figure(figsize=(6,5))

    for method in df_norm["method"].unique():
        sub = df_norm[df_norm["method"] == method]
        plt.scatter(sub["delay"], sub["energy"], s=80, label=method.capitalize())

    plt.axvline(1, linestyle="--")
    plt.axhline(1, linestyle="--")

    plt.xlabel("Normalized Delay (vs Classical)")
    plt.ylabel("Normalized Energy (vs Classical)")
    plt.title("Normalized Energy–Delay Tradeoff")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "normalized_tradeoff.png"), dpi=300)
    plt.close()

    print("✅ Normalized plot saved.")


print("\n🎯 DONE — clean, correct, paper-ready.")
