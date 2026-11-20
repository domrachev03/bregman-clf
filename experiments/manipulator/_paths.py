"""Locations shared by the analysis scripts (all relative to the repository root)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = Path(__file__).resolve().parent / "configs"
RUNS = ROOT / "runs"  # sweep outputs (episode logs, summaries); not tracked
RESULTS = ROOT / "results" / "manipulator"  # stored summaries and derived tables


def summary_path(name: str) -> Path:
    """`runs/<name>/summary.parquet` of a fresh sweep if present, else the stored copy in `results/`."""
    fresh = RUNS / name / "summary.parquet"
    path = fresh if fresh.exists() else RESULTS / f"{name}_summary.parquet"
    print(f"[{name}] reading {path.relative_to(ROOT)}")
    return path
