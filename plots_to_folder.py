import os
import re
import matplotlib.pyplot as plt

BASE_CLASSICAL = "baseline/classical"
BASE_QUANTUM = "baseline/quantum"
OUT_BASE = "eval_graphhhhhhhhhhhhs"

METRICS = {
    "avg_dropped": "Average Drop Rate",
    "avg_delay": "Average Delay",
    "avg_reward": "Average Reward",
    "avg_energy": "Average Energy"
}

DELAYS = [10, 15, 20, 25]

MODELS = [
    {
        "name": "classical",
        "path": BASE_CLASSICAL,
        "color": "green",
        "label": "Classical"
    },
    {
        "name": "quantum",
        "path": BASE_QUANTUM,
        "color": "deeppink",
        "label": "Quantum"
    }
]

def read_metric(file_path, metric):
    with open(file_path, "r") as f:
        for line in f:
            if metric in line:
                return float(line.split(":")[-1].strip())
    return None


os.makedirs(OUT_BASE, exist_ok=True)

for metric, y_label in METRICS.items():
    metric_dir = os.path.join(OUT_BASE, metric)
    os.makedirs(metric_dir, exist_ok=True)

    for delay in DELAYS:
        plt.figure(figsize=(6, 4))

        for model in MODELS:
            xs, ys = [], []

            if not os.path.exists(model["path"]):
                continue

            for folder in os.listdir(model["path"]):
                match = re.match(
                    rf"{model['name']}_(\d*\.?\d+)_{delay}", folder
                )
                if not match:
                    continue

                p = float(match.group(1))
                eval_file = os.path.join(
                    model["path"], folder, "results", "eval_results.txt"
                )

                if not os.path.isfile(eval_file):
                    continue

                val = read_metric(eval_file, metric)
                if val is None:
                    continue

                xs.append(p)
                ys.append(val)

            if len(xs) < 2:
                # still plot single point, but line will be trivial
                plt.scatter(xs, ys, color=model["color"], s=70, zorder=3)
                continue

            # 🔑 SORT explicitly
            xs, ys = zip(*sorted(zip(xs, ys)))

            # 🔑 DOTTED LINE FIRST
            plt.plot(
                xs,
                ys,
                linestyle="--",
                linewidth=1.8,
                marker="o",
                markersize=6,
                color=model["color"],
                label=model["label"],
                zorder=2
            )

        plt.xlabel("Task arrival probability (p)")
        plt.ylabel(y_label)
        plt.title(f"{y_label} (Max Delay = {delay})")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()

        out_path = os.path.join(metric_dir, f"delay_{delay}.png")
        plt.savefig(out_path, dpi=200)
        plt.close()

print("✅ All plots regenerated with dotted lines connecting points.")

