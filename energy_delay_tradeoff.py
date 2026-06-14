import os
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = "training/new environment final/baseline"

classical_dir = os.path.join(BASE_DIR, "classical")
quantum_dir   = os.path.join(BASE_DIR, "quantum")


# =========================
# Find runs
# =========================
def find_runs(model_dir):
    runs = []

    for run in os.listdir(model_dir):
        run_path = os.path.join(model_dir, run)
        plots_path = os.path.join(run_path, "plots")

        if os.path.isdir(plots_path):
            if os.path.exists(os.path.join(plots_path, "avg_delay.npy")) and \
               os.path.exists(os.path.join(plots_path, "avg_energy.npy")):
                runs.append(plots_path)

    return runs


# =========================
# Load + clean
# =========================
def load_clean(path):
    delay = np.load(os.path.join(path, "avg_delay.npy"))
    energy = np.load(os.path.join(path, "avg_energy.npy"))

    mask = ~np.isnan(delay) & ~np.isnan(energy)
    return delay[mask], energy[mask]


def final_point(x, window=50):
    return np.mean(x[-window:])


# =========================
# Get runs
# =========================
classical_runs = find_runs(classical_dir)
quantum_runs   = find_runs(quantum_dir)

print(f"Found {len(classical_runs)} classical runs")
print(f"Found {len(quantum_runs)} quantum runs")


# =========================
# Plot
# =========================
plt.figure()

c_final = []
q_final = []

for run in classical_runs:
    d, e = load_clean(run)
    #plt.plot(d, e, alpha=0.1)
    c_final.append([final_point(d), final_point(e)])

for run in quantum_runs:
    d, e = load_clean(run)
    #plt.plot(d, e, alpha=0.1)
    q_final.append([final_point(d), final_point(e)])

c_final = np.array(c_final)
q_final = np.array(q_final)


def plot_mean(points, label):
    mean = points.mean(axis=0)
    std  = points.std(axis=0)

    plt.scatter(mean[0], mean[1], s=200, label=label)
    plt.errorbar(mean[0], mean[1],
                 xerr=std[0], yerr=std[1],
                 capsize=6)


plot_mean(c_final, "Classical")
plot_mean(q_final, "Quantum")


plt.xlabel("Average Delay")
plt.ylabel("Average Energy")
plt.title("Energy–Delay Tradeoff (All Runs)")
plt.legend()
plt.grid()

plt.savefig("energy_delay_all_runs.png", dpi=300)
plt.show()
