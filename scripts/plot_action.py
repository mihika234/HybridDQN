import os
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.expanduser("~/HybridDQN/training")
SAVE_NAME = "action_distribution_expected.png"

# -----------------------------
# Helpers
# -----------------------------
def classify_env(folder):
    name = folder.lower()
    if name.endswith("new new"):
        return "Env C (4-NL)"
    elif name.endswith("new"):
        return "Env B (2-NL)"
    else:
        return "Env A (0-NL)"

def extract_params(folder):
    return folder.replace("classical", "") \
                 .replace("quantum", "") \
                 .replace("class", "") \
                 .replace("quan", "") \
                 .replace("new new", "") \
                 .replace("new", "") \
                 .strip()

def is_mode(folder, mode):
    f = folder.lower()
    return (mode == "classical" and f.startswith("class")) or \
           (mode == "quantum" and f.startswith("quan"))

# -----------------------------
# Main loop
# -----------------------------
for mode in ["classical", "quantum"]:
    mode_dir = os.path.join(BASE_DIR, mode)
    if not os.path.isdir(mode_dir):
        continue

    for run in sorted(os.listdir(mode_dir)):
        if not is_mode(run, mode):
            continue

        run_path = os.path.join(mode_dir, run)
        results_dir = os.path.join(run_path, "results")
        hist_path = os.path.join(results_dir, "action_hist.npy")

        if not os.path.isfile(hist_path):
            continue

        action_hist = np.load(hist_path)  
        # shape: [T, N_iot, N_actions] OR [*, N_actions]

        if action_hist.ndim == 3:
            expected_counts = action_hist.sum(axis=(0, 1))
        elif action_hist.ndim == 2:
            expected_counts = action_hist.sum(axis=0)
        else:
            raise ValueError("Unexpected action_hist shape")

        n_actions = len(expected_counts)
        total_tasks = expected_counts.sum()
        fractions = expected_counts / total_tasks

        # -----------------------------
        # Plot
        # -----------------------------
        plt.figure(figsize=(7, 4))
        plt.bar(range(n_actions), expected_counts)
        plt.xlabel("Action")
        plt.ylabel("Expected #Tasks")
        plt.title(
            f"{mode.capitalize()} | {classify_env(run)}\n"
            f"Params: {extract_params(run)}"
        )

        for i, v in enumerate(expected_counts):
            plt.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)

        save_path = os.path.join(run_path, SAVE_NAME)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()

        print(f"✅ Saved: {save_path}")

import os
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.expanduser("~/HybridDQN/training")
SAVE_NAME = "action_distribution_expected.png"

# -----------------------------
# Helpers
# -----------------------------
def classify_env(folder):
    name = folder.lower()
    if name.endswith("new new"):
        return "Env C (4-NL)"
    elif name.endswith("new"):
        return "Env B (2-NL)"
    else:
        return "Env A (0-NL)"

def extract_params(folder):
    return folder.replace("classical", "") \
                 .replace("quantum", "") \
                 .replace("class", "") \
                 .replace("quan", "") \
                 .replace("new new", "") \
                 .replace("new", "") \
                 .strip()

def is_mode(folder, mode):
    f = folder.lower()
    return (mode == "classical" and f.startswith("class")) or \
           (mode == "quantum" and f.startswith("quan"))

# -----------------------------
# Main loop
# -----------------------------
for mode in ["classical", "quantum"]:
    mode_dir = os.path.join(BASE_DIR, mode)
    if not os.path.isdir(mode_dir):
        continue

    for run in sorted(os.listdir(mode_dir)):
        if not is_mode(run, mode):
            continue

        run_path = os.path.join(mode_dir, run)
        results_dir = os.path.join(run_path, "results")
        hist_path = os.path.join(results_dir, "action_hist.npy")

        if not os.path.isfile(hist_path):
            continue

        action_hist = np.load(hist_path)  
        # shape: [T, N_iot, N_actions] OR [*, N_actions]

        if action_hist.ndim == 3:
            expected_counts = action_hist.sum(axis=(0, 1))
        elif action_hist.ndim == 2:
            expected_counts = action_hist.sum(axis=0)
        else:
            raise ValueError("Unexpected action_hist shape")

        n_actions = len(expected_counts)
        total_tasks = expected_counts.sum()
        fractions = expected_counts / total_tasks

        # -----------------------------
        # Plot
        # -----------------------------
        plt.figure(figsize=(7, 4))
        plt.bar(range(n_actions), expected_counts)
        plt.xlabel("Action")
        plt.ylabel("Expected #Tasks")
        plt.title(
            f"{mode.capitalize()} | {classify_env(run)}\n"
            f"Params: {extract_params(run)}"
        )

        for i, v in enumerate(expected_counts):
            plt.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)

        save_path = os.path.join(run_path, SAVE_NAME)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()

        print(f"✅ Saved: {save_path}")

