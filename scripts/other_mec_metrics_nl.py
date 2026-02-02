import os
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import re

# =========================
# CONFIGURATION
# =========================
# 1. Define where your data lives
BASE_PATHS = {
    "Classical": os.path.expanduser("~/HybridDQN/training/classical/4nl env"),
    # Pointing this to the specific "4nl env" folder you mentioned
    "Quantum":   os.path.expanduser("~/HybridDQN/training/quantum/4nl env") 
}

# 2. Assumption for normalization (only used if drops are raw counts, not rates)
# If your folder names are like "classical 0.15 20", we use 0.15 * 100 = 15 tasks
STEPS_PER_EPISODE = 100 

def extract_arrival_prob(folder_name):
    """ Tries to find the arrival probability (e.g., 0.15) in the folder name. """
    try:
        parts = re.split(r'[ _]', folder_name)
        for p in parts:
            try:
                val = float(p)
                # Probabilities are typically between 0.05 and 0.95
                if 0.05 <= val <= 0.95: 
                    return val
            except:
                continue
    except:
        pass
    return None

def analyze_run(run_folder):
    """ Computes all metrics for a single run folder. """
    res_dir = os.path.join(run_folder, "results")
    metrics_path = os.path.join(res_dir, "episode_metrics.npy")
    
    # We need the metrics file at minimum
    if not os.path.exists(metrics_path):
        return None

    try:
        # Load Main Metrics
        m = np.load(metrics_path, allow_pickle=True).item()
        
        # --- 1. NORMALIZED EFFICIENCY (The "New Thing") ---
        if 'drop' not in m or 'energy' not in m: return None
        
        drops_raw = np.array(m['drop'])
        energy = np.array(m['energy'])
        
        # Adaptive Logic: Is 'drop' a Count (>1) or Rate (0-1)?
        mean_drop_val = np.mean(drops_raw)
        
        if mean_drop_val > 1.0:
            # It is a COUNT. Normalize it.
            prob = extract_arrival_prob(os.path.basename(run_folder))
            if prob and STEPS_PER_EPISODE:
                estimated_total = STEPS_PER_EPISODE * prob
                drop_rate = np.clip(drops_raw / estimated_total, 0, 1)
            else:
                return None # Cannot normalize safely
        else:
            # It is already a RATE
            drop_rate = drops_raw

        reliability = 1.0 - drop_rate
        
        # Efficiency = Reliability / Energy
        # "How much reliability do I get for 1 Joule?"
        mean_rel = np.mean(reliability)
        mean_eng = np.mean(energy)
        norm_efficiency = mean_rel / mean_eng if mean_eng > 1e-6 else 0

        # --- 2. THROUGHPUT (Goodput) ---
        # Throughput = Reliability / Latency
        if 'delay' in m:
            mean_lat = np.mean(m['delay'])
            throughput = mean_rel / mean_lat if mean_lat > 1e-6 else 0
        else:
            throughput = 0

        # --- 3. BUFFER STABILITY ---
        buffer_file = os.path.join(res_dir, "buffer_size.npy")
        buffer_mean = np.mean(np.load(buffer_file)) if os.path.exists(buffer_file) else 0

        # --- 4. FAILURE MODE BREAKDOWN ---
        # Who is causing the drops?
        drop_files = {"Fog": "drop_fog.npy", "Trans": "drop_trans.npy", "Soft": "drop_soft.npy"}
        failures = {}
        total_fail_sum = 0
        
        for key, fname in drop_files.items():
            fpath = os.path.join(res_dir, fname)
            if os.path.exists(fpath):
                s = np.sum(np.load(fpath))
                failures[key] = s
                total_fail_sum += s
        
        # Convert to percentages (e.g., 80% Fog drops, 20% Soft drops)
        fail_dist = {k: (v/total_fail_sum)*100 for k,v in failures.items()} if total_fail_sum > 0 else {k:0 for k in failures}

        return {
            "efficiency": norm_efficiency,
            "throughput": throughput,
            "buffer": buffer_mean,
            "fail_dist": fail_dist
        }

    except Exception as e:
        print(f"⚠️ Error reading {os.path.basename(run_folder)}: {e}")
        return None

# =========================
# EXECUTION LOOP
# =========================
print(f"{'Agent':<10} | {'Scenario':<25} | {'Eff (Rel/J)':<12} | {'Thruput':<10} | {'Buff':<6} | {'Fog%':<5} | {'Soft%':<5}")
print("-" * 90)

plot_data = {
    "Classical": {"eff": [], "thru": [], "buff": []},
    "Quantum":   {"eff": [], "thru": [], "buff": []}
}

for agent_name, base_path in BASE_PATHS.items():
    if not os.path.exists(base_path):
        print(f"❌ Path not found: {base_path}")
        continue
    
    # Search for runs (handles nested folders like '4nl env')
    # We look 2 levels deep to be safe
    all_runs = glob(os.path.join(base_path, "*")) + glob(os.path.join(base_path, "*", "*"))
    # Filter only directories that contain 'results'
    valid_runs = [r for r in all_runs if os.path.isdir(os.path.join(r, "results"))]
    # Remove duplicates
    valid_runs = sorted(list(set(valid_runs)))

    for run in valid_runs:
        stats = analyze_run(run)
        if stats:
            name = os.path.basename(run)
            
            # Print Data
            fog_p = stats['fail_dist'].get('Fog', 0)
            soft_p = stats['fail_dist'].get('Soft', 0)
            print(f"{agent_name:<10} | {name:<25} | {stats['efficiency']:<12.2f} | {stats['throughput']:<10.4f} | {stats['buffer']:<6.2f} | {fog_p:<5.0f} | {soft_p:<5.0f}")
            
            # Store for Plotting
            plot_data[agent_name]["eff"].append(stats['efficiency'])
            plot_data[agent_name]["thru"].append(stats['throughput'])
            plot_data[agent_name]["buff"].append(stats['buffer'])

# =========================
# PLOTTING
# =========================
metrics_cfg = [
    ("eff", "Normalized Energy Efficiency\n(Reliability / Joule)"),
    ("thru", "Effective Throughput\n(Reliability / Latency)"),
    ("buff", "Buffer Occupancy\n(Queue Length)")
]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for i, (key, title) in enumerate(metrics_cfg):
    c_vals = plot_data["Classical"][key]
    q_vals = plot_data["Quantum"][key]
    
    if len(c_vals) > 0 and len(q_vals) > 0:
        # Boxplot
        axes[i].boxplot([c_vals, q_vals], labels=["Classical", "Quantum"], patch_artist=True,
                        boxprops=dict(facecolor='#a6cee3' if i%2==0 else '#1f78b4'))
        
        # Jitter (Red Dots)
        for j, vals in enumerate([c_vals, q_vals]):
            x = np.random.normal(1+j, 0.04, size=len(vals))
            axes[i].plot(x, vals, 'r.', alpha=0.5)
            
    axes[i].set_title(title)
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("final_mec_comparison.png", dpi=300)
print("\n✅ Saved 'final_mec_comparison.png'")
