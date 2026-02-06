import pandas as pd
import numpy as np

# ================= PATHS =================
CLASSICAL_PATH = "training/new environment final/baseline/classical/summary_metrics.csv"
QUANTUM_PATH   = "training/new environment final/baseline/quantum/summary_metrics.csv"
OUT_PATH       = "training/new environment final/per_run_classical_vs_quantum_new_new.csv"

# ================= LOAD =================
cl = pd.read_csv(CLASSICAL_PATH)
qt = pd.read_csv(QUANTUM_PATH)

# ================= PARSE RUN =================
def parse_run(df):
    """
    Expected run format:
    classical_0.2_15
    quantum_0.2_15
    """
    parts = df["run"].str.split("_", expand=True)
    df["algo"] = parts[0]
    df["epsilon"] = parts[1].astype(float)
    df["load"] = parts[2].astype(int)
    return df

cl = parse_run(cl)
qt = parse_run(qt)

# ================= NUMERIC METRICS =================
metric_cols = cl.select_dtypes(include=np.number).columns.tolist()
metric_cols = [c for c in metric_cols if c not in ["epsilon", "load"]]

# ================= PAIR RUNS =================
paired = pd.merge(
    cl,
    qt,
    on=["epsilon", "load"],
    suffixes=("_classical", "_quantum"),
    how="inner"
)

if paired.empty:
    raise RuntimeError("No matched classical–quantum runs found")

# ================= COMPUTE DELTAS =================
rows = []

for _, row in paired.iterrows():
    eps = row["epsilon"]
    load = row["load"]

    for m in metric_cols:
        c = row[f"{m}_classical"]
        q = row[f"{m}_quantum"]

        if pd.isna(c) or pd.isna(q) or c == 0:
            pct = np.nan
        else:
            pct = (q - c) / c * 100.0

        rows.append({
            "epsilon": eps,
            "load": load,
            "metric": m,
            "classical": c,
            "quantum": q,
            "pct_change": pct
        })

result = pd.DataFrame(rows)

# ================= SORT =================
result["abs_pct_change"] = result["pct_change"].abs()
result = result.sort_values("abs_pct_change", ascending=False)

# ================= SAVE =================
result.to_csv(OUT_PATH, index=False)

pd.set_option("display.float_format", "{:.3f}".format)

print("\n=== Per-run Classical vs Quantum % Change ===\n")
print(result.head(20))
print(f"\nSaved to: {OUT_PATH}")

