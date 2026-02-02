import pandas as pd
import numpy as np

# ---------------- paths ----------------
CLASSICAL_PATH = "training/classical/original env/summary_metrics.csv"
QUANTUM_PATH   = "training/quantum/original env/summary_metrics_quantum.csv"
OUT_PATH       = "training/metric_comparison_classical_vs_quantum.csv"

# ---------------- load ----------------
cl = pd.read_csv(CLASSICAL_PATH)
qt = pd.read_csv(QUANTUM_PATH)

# ---------------- parse run ----------------
def parse_run(df):
    # expected format: "classical 0.1 10" or "quantum 0.1 10"
    split = df["run"].str.split(" ", expand=True)
    df["algo"] = split[0]
    df["epsilon"] = split[1].astype(float)
    df["load"] = split[2].astype(int)
    return df

cl = parse_run(cl)
qt = parse_run(qt)

# ---------------- numeric metrics only ----------------
metric_cols = cl.select_dtypes(include=np.number).columns
metric_cols = metric_cols.drop(["epsilon", "load"], errors="ignore")

# ---------------- pair classical & quantum ----------------
paired = pd.merge(
    cl,
    qt,
    on=["epsilon", "load"],
    suffixes=("_classical", "_quantum"),
    how="inner"
)

if paired.empty:
    raise RuntimeError("No matching (epsilon, load) pairs found")

# ---------------- compute per-pair % change ----------------
records = []

for m in metric_cols:
    c = paired[f"{m}_classical"]
    q = paired[f"{m}_quantum"]

    pct = (q - c) / c * 100

    records.append({
        "metric": m,
        "mean_classical": c.mean(),
        "mean_quantum": q.mean(),
        "median_classical": c.median(),
        "median_quantum": q.median(),
        "mean_pct_change": pct.mean(),
        "median_pct_change": pct.median()
    })

result = pd.DataFrame(records).set_index("metric")

# ---------------- sort by median effect size ----------------
result = result.sort_values(
    by="median_pct_change",
    key=lambda x: np.abs(x),
    ascending=False
)

# ---------------- save ----------------
result.to_csv(OUT_PATH)

pd.set_option("display.float_format", "{:.4f}".format)

print("\n=== Paired Classical vs Quantum Comparison ===\n")
print(result)
print(f"\nSaved to: {OUT_PATH}")
