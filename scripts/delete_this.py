import numpy as np
import os
import pandas as pd

# UPDATE THIS PATH to the specific folder shown in your screenshot
TARGET_DIR = os.path.expanduser("~/HybridDQN/training/classical/classical 0.15 20 new new/results")

print(f"📂 Inspecting: {TARGET_DIR}")

# 1. Load Episode Metrics (Energy is likely here)
npy_path = os.path.join(TARGET_DIR, "episode_metrics.npy")
csv_path = os.path.join(TARGET_DIR, "episode_metrics.csv")

if os.path.exists(npy_path):
    print("\n🔍 content of 'episode_metrics.npy':")
    try:
        data = np.load(npy_path, allow_pickle=True)
        # Check if it's a dictionary or a structured array
        if isinstance(data.item(), dict):
            keys = data.item().keys()
            print(f"   Keys found: {list(keys)}")
            # Print first value of energy if it exists
            if 'avg_energy' in keys:
                print(f"   Sample Energy: {data.item()['avg_energy'][:5]}")
        else:
            print(f"   Array shape: {data.shape}")
    except Exception as e:
        print(f"   ❌ Error reading .npy: {e}")

# 2. Check CSV columns (backup)
if os.path.exists(csv_path):
    print("\n🔍 columns in 'episode_metrics.csv':")
    try:
        df = pd.read_csv(csv_path)
        print(f"   Columns: {list(df.columns)}")
    except Exception as e:
        print(f"   ❌ Error reading .csv: {e}")
