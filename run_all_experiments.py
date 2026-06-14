# import subprocess
# import sys

# PYTHON = sys.executable  # use current venv/python

# TASK_PROBS = ["0.1", "0.15", "0.2", "0.25", "0.3"]
# TASK_PROBS = ["0.10", "0.15", "0.2", "0.25"]
# SEED = ['8', '19', '77', '65', '1975']

# def run(name, args):
#     print(f"\n{'='*80}")
#     print(f"RUN: {name}")
#     print(f"{'='*80}")
#     print("Command:", " ".join([PYTHON, "train.py"] + args), "\n")
#     subprocess.run([PYTHON, "train.py"] + args, check=True)


# if __name__ == "__main__":

#     # =========================
#     # Classical DQN runs
#     # =========================
    
#     for p in [p for p in TASK_PROBS]:
#         for d in MAX_DELAYS:
#             run(
#                 name=f"CLASSICAL | task_arrival_prob={p}, max_delay={d}",
#                 args=[
#                     "--task_arrival_prob", p,
#                     "--max_delay", d
#                 ]
#             )
#             run(
#                 name=f"HYBRID | task_arrival_prob={p}, max_delay={d}",
#                 args=[
#                     "--hybrid",
#                     "--task_arrival_prob", p,
#                     "--max_delay", d
#                 ]
#             )
            
            

#     # =========================
#     # Hybrid / Quantum DQN runs
#     # =========================
#     # for p in TASK_PROBS:
#     #     for d in MAX_DELAYS:
            

#     # for s in SEED:
#     #     run(
#     #         name=f"CLASSICAL | seed={s}",
#     #         args=[
#     #                 "--seed", s
#     #             ]
#     #         )
#     # for s in SEED:
#     #     run(
#     #         name=f"HYBRID | seed={s}",
#     #         args=[
#     #                 "--hybrid",
#     #                 "--seed", s
#     #             ]
#     #         )


# import subprocess
# import sys

# PYTHON = sys.executable  # use current venv/python


# def run(name, args):
#     print(f"\n{'='*80}")
#     print(f"RUN: {name}")
#     print(f"{'='*80}")
#     print("Command:", " ".join([PYTHON, "train.py"] + args), "\n")
#     subprocess.run([PYTHON, "train.py"] + args, check=True)


# if __name__ == "__main__":

#     # =========================================
#     # CLASSICAL ONLY — requested configurations
#     # =========================================

#     configs = [
#         ("0.287",  "19"),
#         ("0.288",  "19"),
        
#     ]

#     for p, d in configs:
#         # run(
#         #         name=f"CLASSICAL | task_arrival_prob={p}, max_delay={d}",
#         #         args=[
#         #             "--task_arrival_prob", p,
#         #             "--max_delay", d
#         #         ]
#         #     )
#         run(
#             name=f"QUANTUM | task_arrival_prob={p}, max_delay={d}",
#             args=[
#                 "--hybrid",
#                 "--task_arrival_prob", p,
#                 "--max_delay", d
#             ]
#         )

import subprocess
import sys

PYTHON = sys.executable


def run(name, args):
    print("\n" + "=" * 80)
    print(f"RUN: {name}")
    print("=" * 80)
    print("Command:", " ".join([PYTHON, "train.py"] + args))
    print()

    subprocess.run(
        [PYTHON, "train.py"] + args,
        check=True
    )


if __name__ == "__main__":

    configs = [
        ("0.20", "10"),

        ("0.15", "10"),
        ("0.15", "15"),
        ("0.15", "20"),

        ("0.22", "20"),
        ("0.22", "25"),
        ("0.22", "30"),

        ("0.24", "20"),
        ("0.24", "25"),
        ("0.24", "30"),

        ("0.25", "20"),
        ("0.25", "25"),
        ("0.25", "30"),
    ]

    for p, d in configs:

        # Classical
        run(
            name=f"CLASSICAL | λ={p}, D={d}",
            args=[
                "--task_arrival_prob", p,
                "--max_delay", d,
            ]
        )

        # Quantum
        run(
            name=f"QUANTUM | λ={p}, D={d}",
            args=[
                "--hybrid",
                "--task_arrival_prob", p,
                "--max_delay", d,
            ]
        )
