import numpy as np
import matplotlib.pyplot as plt
import os

BASE_DIR = "training/entanglement ablations"

folders = [
    "no entanglement",
    "1 cnot",
    "2 cnot",
    "3 cnot"
]

titles = [
    "1. No entanglement",
    "2. 1 CNOT gate",
    "3. 2 CNOT gates (linear topology)",
    "4. 3 CNOT gates (ring topology)"
]

threshold = None   # or set e.g. 1e5

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
axes = axes.flatten()

for i, folder in enumerate(folders):
    path = os.path.join(BASE_DIR, folder, "results")

    ret = np.load(os.path.join(path, "rolling_return_var.npy"))
    drop = np.load(os.path.join(path, "rolling_drop_var.npy"))

    ax = axes[i]
    ax.plot(ret, label="Return variance", linewidth=1.5)
    ax.plot(drop, label="Drop-rate variance", linewidth=1.5)

    if threshold is not None:
        ax.axhline(threshold, linestyle="--", color="gray",
                   label="Stability threshold")

    ax.set_title(titles[i])
    ax.set_xlabel("Episode")
    ax.set_ylabel("Rolling variance")
    ax.legend()

plt.tight_layout()
plt.savefig("quantum_entanglement_stability.png", dpi=300)
plt.show()
