import os
import numpy as np
from glob import glob
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt

# =========================
# CONFIGURATION
# =========================
# Adjust these paths to match your actual folder structure
BASE_CLASSICAL = os.path.expanduser("~/HybridDQN/training/classical")
BASE_QUANTUM   = os.path.expanduser("~/HybridDQN/training/quantum")

OUT = "aggregated_analysis_matplotlib"
os.makedirs(OUT, exist_ok=True)

# =========================
# DATA LOADER
# =========================
def get_run_metrics(root_path, label):
    """
    Walks through all subfolders, finds .npz files, 
    and computes the AVERAGE for each run to treat it as a single sample.
    """
    metrics = {
        "latency": [],
        "energy": [],
        "entropy": [],
        "kl_drift": [],
        "switch_rate": []
    }
    
    # Recursive search for all .npz files
    print(f"🔍 Searching in {label} path: {root_path}...")
    files = glob(os.path.join(root_path, "**", "*.npz"), recursive=True)
    
    if not files:
        print(f"   ❌ No files found for {label}!")
        return metrics

    print(f"   ✅ Found {len(files)} experiments (scenarios). Processing...")

    for f in files:
        try:
            with np.load(f, allow_pickle=True) as data:
                # 1. LATENCY
                if "latency" in data and len(data["latency"]) > 0:
                    metrics["latency"].append(np.mean(data["latency"]))
                
                # 2. ENERGY
                if "energy" in data and len(data["energy"]) > 0:
                    metrics["energy"].append(np.mean(data["energy"]))

                # 3. ENTROPY (Mechanism)
                if "entropies" in data and len(data["entropies"]) > 0:
                    metrics["entropy"].append(np.mean(data["entropies"]))
                
                # 4. KL DRIFT (Stability)
                if "kl_drifts" in data and len(data["kl_drifts"]) > 0:
                    metrics["kl_drift"].append(np.mean(data["kl_drifts"]))
                
                # 5. SWITCH RATE (Smoothness)
                if "switch_rates" in data and len(data["switch_rates"]) > 0:
                    metrics["switch_rate"].append(np.mean(data["switch_rates"]))

        except Exception as e:
            print(f"   ⚠️ Error reading {os.path.basename(f)}: {e}")

    return metrics

# =========================
# EXECUTE DATA LOADING
# =========================
c_data = get_run_metrics(BASE_CLASSICAL, "Classical")
q_data = get_run_metrics(BASE_QUANTUM,   "Quantum")

# =========================
# STATS & PLOTTING FUNCTION
# =========================
def analyze_metric(key, pretty_name, higher_better=False):
    c_vals = c_data[key]
    q_vals = q_data[key]
    
    # Check if we have data
    if len(c_vals) == 0 or len(q_vals) == 0:
        print(f"\n⚠️ Skipping {pretty_name}: No data found.")
        return

    print(f"\n📊 ANALYSIS: {pretty_name}")
    print("-" * 40)
    print(f"   Samples (Experiments): Classical={len(c_vals)}, Quantum={len(q_vals)}")
    print(f"   Classical Mean: {np.mean(c_vals):.4f}")
    print(f"   Quantum   Mean: {np.mean(q_vals):.4f}")

    # Mann-Whitney U Test
    try:
        stat, p = mannwhitneyu(c_vals, q_vals, alternative='two-sided')
        print(f"   Mann-Whitney p-value: {p:.4e}")
        
        if p < 0.05:
            c_m = np.median(c_vals)
            q_m = np.median(q_vals)
            if higher_better:
                winner = "Quantum" if q_m > c_m else "Classical"
            else:
                winner = "Quantum" if q_m < c_m else "Classical"
            print(f"   ✅ SIGNIFICANT RESULT! Winner: {winner}")
        else:
            print(f"   ❌ Not statistically significant.")
    except:
        print("   ⚠️ Stats failed (too few samples?)")

    # --- MATPLOTLIB PLOTTING ---
    plt.figure(figsize=(6, 5))
    
    data_to_plot = [c_vals, q_vals]
    labels = ["Classical", "Quantum"]

    # Create Boxplot
    box = plt.boxplot(data_to_plot, labels=labels, patch_artist=True,
                      medianprops=dict(color="black", linewidth=1.5))

    # Color the boxes (Light Blue vs Light Green, for example)
    colors = ['#add8e6', '#90ee90'] # Light Blue, Light Green
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    # Add Jittered Scatter Plots (The "Red Dots")
    # This shows the individual experiment distribution
    for i, vals in enumerate(data_to_plot):
        y = vals
        # Create random x-offsets (jitter) centered around index 1 and 2
        x = np.random.normal(1 + i, 0.04, size=len(y))
        plt.plot(x, y, 'r.', alpha=0.6, markersize=8)

    plt.title(f"Average {pretty_name}\n(Across {len(c_vals)} Scenarios)")
    plt.ylabel(f"Avg {pretty_name}")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Save Plot
    save_path = os.path.join(OUT, f"{key}_comparison.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"   Plot saved to: {save_path}")

# =========================
# RUN ANALYSIS FOR ALL METRICS
# =========================
metrics_map = {
    "latency":     ("Latency (s)", False),
    "energy":      ("Energy (J)", False),
    "entropy":     ("Action Entropy", True),
    "kl_drift":    ("KL Drift", False),
    "switch_rate": ("Switch Rate", False)
}

print("\n========================================")
for k, (label, hb) in metrics_map.items():
    analyze_metric(k, label, higher_better=hb)
print(f"\n✅ Analysis Complete. Check '{OUT}/' folder.")
