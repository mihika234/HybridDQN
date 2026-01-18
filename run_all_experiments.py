import subprocess
import sys

PYTHON = sys.executable  # use current venv/python

def run(name, args):
    print(f"\n{'='*70}")
    print(f"RUN: {name}")
    print(f"{'='*70}")
    print("Command:", " ".join([PYTHON, "train.py"] + args), "\n")

    subprocess.run([PYTHON, "train.py"] + args, check=True)


if __name__ == "__main__":

    # =========================
    # Classical DQN runs
    # =========================

    # run(
    #     name="CLASSICAL | task_arrival_prob=0.25, max_delay=15",
    #     args=[
    #         "--task_arrival_prob", "0.25",
    #         "--max_delay", "15"
    #     ]
    # )

    # run(
    #     name="CLASSICAL | task_arrival_prob=0.25, max_delay=12",
    #     args=[
    #         "--task_arrival_prob", "0.25",
    #         "--max_delay", "15"
    #     ]
    # )

    # run(
    #     name="CLASSICAL | task_arrival_prob=0.275, max_delay=19",
    #     args=[
    #         "--task_arrival_prob", "0.275",
    #         "--max_delay", "19"
    #     ]
    # )

    run(
        name="CLASSICAL | task_arrival_prob=0.2, max_delay=15",
        args=[
            "--task_arrival_prob", "0.2",
            "--max_delay", "15"
        ]
    )

    # =========================
    # Hybrid DQN runs
    # =========================

    # run(
    #     name="HYBRID | task_arrival_prob=0.1, max_delay=20",
    #     args=[
    #         "--hybrid",
    #         "--task_arrival_prob", "0.1",
    #         "--max_delay", "20"
    #     ]
    # )

    # run(
    #     name="HYBRID | task_arrival_prob=0.25, max_delay=15",
    #     args=[
    #         "--hybrid",
    #         "--task_arrival_prob", "0.25",
    #         "--max_delay", "15"
    #     ]
    # )
    
    # run(
    #     name="HYBRID | task_arrival_prob=0.25, max_delay=12",
    #     args=[
    #         "--hybrid",
    #         "--task_arrival_prob", "0.25",
    #         "--max_delay", "12"
    #     ]
    # )

    # run(
    #     name="HYBRID | task_arrival_prob=0.275, max_delay=19",
    #     args=[
    #         "--hybrid",
    #         "--task_arrival_prob", "0.275",
    #         "--max_delay", "19"
    #     ]
    # )

    run(
        name="HYBRID | task_arrival_prob=0.2, max_delay=15",
        args=[
            "--hybrid",
            "--task_arrival_prob", "0.2",
            "--max_delay", "15"
        ]
    )
