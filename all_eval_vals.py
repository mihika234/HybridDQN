import os
import pandas as pd

BASE_PATH = "training/new environment final/baseline"

FOLDERS = {
    "quantum": "quantum",
    "classical": "classical",
    "hamiltonian": "hamiltonian"
}

METRICS = ["avg_rewards", "avg_dropped", "avg_delay", "avg_energy"]


def extract_params(folder_name):
    """
    Extract (task_prob, max_delay) from:
    quantum_0.2_10
    classical_0.1_20
    """
    try:
        parts = folder_name.split("_")
        task_prob = float(parts[-2])
        max_delay = int(parts[-1])
        return task_prob, max_delay
    except:
        return None, None


def read_eval_file(path):
    vals = {}
    try:
        with open(path, "r") as f:
            for line in f:
                for m in METRICS:
                    if line.startswith(m):
                        vals[m] = float(line.split(":")[1].strip())
    except:
        return None
    return vals


def collect_all():
    data = {}

    for model, folder in FOLDERS.items():
        full_path = os.path.join(BASE_PATH, folder)

        if not os.path.exists(full_path):
            continue

        for run in os.listdir(full_path):
            run_path = os.path.join(full_path, run)

            if not os.path.isdir(run_path):
                continue

            task_prob, max_delay = extract_params(run)
            if task_prob is None:
                continue

            key = (task_prob, max_delay)

            eval_file = os.path.join(run_path, "results", "eval_results.txt")

            vals = read_eval_file(eval_file)
            if vals is None:
                continue

            if key not in data:
                data[key] = {
                    "task_prob": task_prob,
                    "max_delay": max_delay
                }

            for m in METRICS:
                data[key][f"{model}_{m}"] = vals.get(m, "-")

    return list(data.values())


def save_tables(rows):
    df = pd.DataFrame(rows)

    # Fill missing combinations with "-"
    df = df.fillna("-")

    df = df.sort_values(by=["task_prob", "max_delay"])

    # create 4 separate tables
    for metric in METRICS:
        cols = ["task_prob", "max_delay",
                f"quantum_{metric}",
                f"classical_{metric}",
                f"hamiltonian_{metric}"]

        sub_df = df[cols]

        filename = f"{metric}_comparison.csv"
        sub_df.to_csv(filename, index=False)
        print(f"Saved {filename}")


if __name__ == "__main__":
    rows = collect_all()
    save_tables(rows)
