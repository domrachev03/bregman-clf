"""Per-cell table of the exploratory stress test (sweep `configs/stress_test.yaml`): mean fraction of
QP-infeasible steps, trials leaving the keep-in set by more than 1 mm, deepest excursion [mm],
and trials exceeding a joint position limit by more than 1 mrad.

    python experiments/manipulator/stress_test_table.py

Reads the summary of a fresh sweep in runs/stress_test/ if there is one, else the stored
results/manipulator/stress_test_summary.parquet; writes results/manipulator/stress_test_table.csv.
"""

import pandas as pd
from _paths import RESULTS, summary_path

df = pd.read_parquet(summary_path("stress_test"))
df["scenario"] = df["scenario"].str.removeprefix("iiwa_e8_pilot_")
table = df.groupby(["scenario", "method"], sort=False).agg(
    trials=("seed", "count"),
    infeasible_pct=("fallback_rate", lambda s: round(100 * s.mean(), 1)),
    left_set=("keep_in_violated_trial", lambda s: int(s.sum())),
    depth_mm=("keep_in_violation_depth", lambda s: round(1000 * s.max(), 1)),
    joint_limit=("qpos_violated_trial", lambda s: int(s.sum())),
)
table.to_csv(RESULTS / "stress_test_table.csv")
print(table.to_string())
