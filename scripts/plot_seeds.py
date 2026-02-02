import os
import re
import matplotlib.pyplot as plt

BASE_DIR = "training/seeds"
CLASSICAL_DIR = os.path.join(BASE_DIR, "classical")
QUANTUM_DIR   = os.path.join(BASE_DIR, "quantum")

METRICS = ["avg_rewards", "avg_dropped", "avg_delay", "avg_energy"]


def parse_eval_file(path):
    data = {}
    with open(path, "r") as f:
        for line in f:
            for k in METRICS:
                if line.startswith(k):
                    data[k] = float(line.split(":")[1].strip())
    return data


def collect_results(root_dir):
    results = {}
    for folder in os.listdir(root_dir):
        match = re.search(r"seed\s*(\d+)", folder)
        if not match:
            continue

        seed = int(match.group(1))
        eval_path = os.path.join(
            root_dir, folder, "results", "eval_results.txt"
        )

        if os.path.isfile(eval_path):
            results[seed] = parse_eval_file(eval_path)

    return results


# -----------------------------
# Load data
# -----------------------------
classical = collect_results(CLASSICAL_DIR)
quantum   = collect_results(QUANTUM_DIR)

common_seeds = sorted(set(classical) & set(quantum))
if not common_seeds:
    raise RuntimeError("No matching seeds between classical and quantum.")


# -----------------------------
# Plot & save
# -----------------------------
for metric in METRICS:
    plt.figure(figsize=(7, 4))

    c_vals = [classical[s][metric] for s in common_seeds]
    q_vals = [quantum[s][metric] for s in common_seeds]

    plt.plot(common_seeds, q_vals, marker="o", color="pink", label="Quantum")
    plt.plot(common_seeds, c_vals, marker="o", color="green", label="Classical")

    plt.xlabel("Seed")
    plt.ylabel(metric)
    plt.title(metric.replace("_", " ").title())
    plt.grid(True)
    plt.legend()

    save_path = os.path.join(
        BASE_DIR, f"{metric}_vs_seed.png"
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

print(f"Saved plots to: {BASE_DIR}")

