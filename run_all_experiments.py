import subprocess
import sys

PYTHON = sys.executable  # use current venv/python

# TASK_PROBS = ["0.05", "0.08", "0.10", "0.12", "0.15"]
# TASK_PROBS = ["0.22", "0.24", "0.25", "0.27", "0.3", "0.35"]
TASK_PROBS = ["0.3"]
MAX_DELAYS = ["25"]
SEED = ['8', '19', '77', '65', '1975']

def run(name, args):
    print(f"\n{'='*80}")
    print(f"RUN: {name}")
    print(f"{'='*80}")
    print("Command:", " ".join([PYTHON, "train.py"] + args), "\n")
    subprocess.run([PYTHON, "train.py"] + args, check=True)


if __name__ == "__main__":

    # =========================
    # Classical DQN runs
    # =========================
    
    for p in [p for p in TASK_PROBS]:
        for d in MAX_DELAYS:
            run(
                name=f"CLASSICAL | task_arrival_prob={p}, max_delay={d}",
                args=[
                    "--task_arrival_prob", p,
                    "--max_delay", d
                ]
            )
            run(
                name=f"HYBRID | task_arrival_prob={p}, max_delay={d}",
                args=[
                    "--hybrid",
                    "--task_arrival_prob", p,
                    "--max_delay", d
                ]
            )
            
            

    # =========================
    # Hybrid / Quantum DQN runs
    # =========================
    # for p in TASK_PROBS:
    #     for d in MAX_DELAYS:
            

    # for s in SEED:
    #     run(
    #         name=f"CLASSICAL | seed={s}",
    #         args=[
    #                 "--seed", s
    #             ]
    #         )
    # for s in SEED:
    #     run(
    #         name=f"HYBRID | seed={s}",
    #         args=[
    #                 "--hybrid",
    #                 "--seed", s
    #             ]
    #         )
