import numpy as np
import matplotlib.pyplot as plt
import os

BASE_DIR = "training/entanglement ablations"
path = os.path.join(BASE_DIR, "classical", "results")

ret = np.load(os.path.join(path, "rolling_return_var.npy"))
drop = np.load(os.path.join(path, "rolling_drop_var.npy"))

plt.figure(figsize=(6, 4))
plt.plot(ret, label="Return variance")
plt.plot(drop, label="Drop-rate variance")
plt.xlabel("Episode")
plt.ylabel("Rolling variance")
plt.title("Classical DQN (No Quantum)")
plt.legend()
plt.tight_layout()
plt.savefig("classical_stability.png", dpi=300)
plt.show()
