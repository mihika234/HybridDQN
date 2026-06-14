import subprocess
import os

import sys

PYTHON = sys.executable
TRAIN_SCRIPT = "train.py"

# =========================
# EXACT experiments you want
# =========================

EXPERIMENTS = [
    (0.2, 15),
    (0.3, 20),
    #(0.27, 20),
    #(0.1, 10),
    #(0.1, 20),
    #(0.15, 20),
    #(0.32, 20),
]

NUM_EPISODES = 1000
QUBITS = 3
BASE_RESULTS_DIR = "results"

# =========================
# Helper
# =========================

def run(cmd):
    print("\n" + "=" * 100)
    print("RUNNING:")
    print(" ".join(cmd))
    print("=" * 100 + "\n")
    subprocess.run(cmd, check=True)

# =========================
# Main
# =========================

def main():

    os.makedirs(BASE_RESULTS_DIR, exist_ok=True)

    for prob, delay in EXPERIMENTS:

        classical_dir = f"{BASE_RESULTS_DIR}/classical_train_p{prob}_d{delay}"
        quantum_dir   = f"{BASE_RESULTS_DIR}/quantum_train_p{prob}_d{delay}"

        # -------------------------
        # Classical Training
        # -------------------------
        run([
            PYTHON, TRAIN_SCRIPT,
            "--num_episodes", str(NUM_EPISODES),
            "--task_arrival_prob", str(prob),
            "--max_delay", str(delay),
            "--path", classical_dir
        ])

        # -------------------------
        # Quantum Training
        # -------------------------
        run([
            PYTHON, TRAIN_SCRIPT,
            "--hybrid",
            "--qubits", str(QUBITS),
            "--num_episodes", str(NUM_EPISODES),
            "--task_arrival_prob", str(prob),
            "--max_delay", str(delay),
            "--path", quantum_dir
        ])


if __name__ == "__main__":
    main()




