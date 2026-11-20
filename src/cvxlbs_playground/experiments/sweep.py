"""Sweeps: YAML grid -> parallel episodes -> `summary.parquet`.

    python -m cvxlbs_playground.experiments.sweep <config.yaml> [--workers N] [--limit K]

Config keys: `name`, `output_dir`, `scenarios` (each `{name, options}`), `methods` (each
`{name, label, kw}`), `seeds` (`{start, n}` or a list), `metrics` (keyword arguments of
`metrics.compute_metrics`). A method's `name` selects the registered builder; its `label` (default:
the name) names the output directory and the `method` column, so one builder can appear several
times with different `kw`.

Each episode is saved as `<output_dir>/<scenario>/<label>/<seed>.npz`. Episodes whose file exists
are skipped, so an interrupted sweep resumes. Episodes run in a `spawn` process pool, one thread
per worker; each worker builds its own scenario and method from the config data.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from cvxlbs_playground.scenarios import make_scenario

from .logs import EpisodeLog
from .metrics import compute_metrics
from .runner import run_episode

_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


@dataclass(frozen=True)
class SweepTask:
    scenario_name: str
    scenario_options: dict[str, Any]
    method_name: str
    method_label: str
    method_kw: dict[str, Any]
    seed: int
    out_path: str


def _expand_seeds(spec: Any) -> tuple[int, ...]:
    if isinstance(spec, dict):
        start = int(spec.get("start", 0))
        return tuple(range(start, start + int(spec["n"])))
    return tuple(int(s) for s in spec)


def build_tasks(cfg: dict[str, Any]) -> list[SweepTask]:
    """Scenarios outer, methods middle, seeds inner: `--limit K` always takes the same prefix."""
    out_dir = Path(cfg.get("output_dir") or f"runs/{cfg['name']}")
    labels = [m.get("label", m["name"]) for m in cfg["methods"]]
    if len(set(labels)) != len(labels):
        raise ValueError(f"sweep: method labels must be distinct, got {labels}")
    tasks = []
    for scenario in cfg["scenarios"]:
        for method, label in zip(cfg["methods"], labels, strict=True):
            for seed in _expand_seeds(cfg["seeds"]):
                tasks.append(
                    SweepTask(
                        scenario_name=scenario["name"],
                        scenario_options=dict(scenario.get("options") or {}),
                        method_name=method["name"],
                        method_label=label,
                        method_kw=dict(method.get("kw") or {}),
                        seed=seed,
                        out_path=str(out_dir / scenario["name"] / label / f"{seed}.npz"),
                    )
                )
    return tasks


_ROBOT_LOAD_LOCK: Any = None


def _worker_init(robot_load_lock: Any) -> None:
    global _ROBOT_LOAD_LOCK
    _ROBOT_LOAD_LOCK = robot_load_lock
    os.environ.update(_THREAD_ENV)


def _run_one(task: SweepTask) -> None:
    if os.path.exists(task.out_path):
        return
    scenario = make_scenario(task.scenario_name, **task.scenario_options)
    with _ROBOT_LOAD_LOCK:  # the first robot load of a process rewrites the shared converted-MJCF cache
        _ = scenario.entry
    log = run_episode(scenario, task.method_name, task.seed, method_kw=task.method_kw)
    # write to a temp file and rename, so a killed worker never leaves a partial episode behind
    final = Path(task.out_path)
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.with_name(f".{final.stem}.{os.getpid()}.{uuid.uuid4().hex}.tmp.npz")
    log.save(tmp)
    os.replace(tmp, final)


def _summary_row(task: SweepTask, metrics_kw: dict[str, Any]) -> dict[str, Any]:
    log = EpisodeLog.load(task.out_path)
    scenario = make_scenario(task.scenario_name, **task.scenario_options)
    scenario.reset(np.random.default_rng(task.seed))  # reproduces the episode's keep-in set and target
    metrics = compute_metrics(log, scenario.sets(), scenario.target_position(), **metrics_kw)
    return {
        "scenario": task.scenario_name,
        "method": task.method_label,
        "method_name": task.method_name,
        "seed": task.seed,
        "npz_path": task.out_path,
        **metrics,
    }


def run_sweep(
    config: str | Path | dict[str, Any],
    *,
    workers: int | None = None,
    limit: int | None = None,
    recompute_summary: bool = False,
) -> pd.DataFrame:
    """Run the pending episodes (the first `limit` of them, none with `recompute_summary`), then
    rewrite `<output_dir>/summary.parquet` from every episode of this config that is on disk."""
    cfg = config if isinstance(config, dict) else yaml.safe_load(Path(config).read_text())
    tasks = build_tasks(cfg)
    pending = [t for t in tasks if not os.path.exists(t.out_path)]
    to_run = [] if recompute_summary else (pending if limit is None else pending[: max(0, limit)])

    if to_run:
        for scenario in cfg["scenarios"]:  # fail fast on a bad config; fetch and convert the robot once
            _ = make_scenario(scenario["name"], **dict(scenario.get("options") or {})).entry
        n_workers = workers if workers is not None else max(1, (os.cpu_count() or 1) - 4)
        os.environ.update(_THREAD_ENV)
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_workers, initializer=_worker_init, initargs=(ctx.Lock(),)) as pool:
            for _ in pool.imap_unordered(_run_one, to_run):
                pass

    metrics_kw = dict(cfg.get("metrics") or {})
    df = pd.DataFrame([_summary_row(t, metrics_kw) for t in tasks if os.path.exists(t.out_path)])
    out_dir = Path(cfg.get("output_dir") or f"runs/{cfg['name']}")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "summary.parquet", index=False)
    return df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m cvxlbs_playground.experiments.sweep")
    parser.add_argument("config", help="path to a sweep YAML config")
    parser.add_argument("--workers", type=int, default=None, help="worker processes (default: CPU count - 4)")
    parser.add_argument("--limit", type=int, default=None, help="run at most this many pending episodes")
    parser.add_argument(
        "--recompute-summary", action="store_true", help="run no episodes; rebuild summary.parquet from the .npz files"
    )
    args = parser.parse_args(argv)
    df = run_sweep(args.config, workers=args.workers, limit=args.limit, recompute_summary=args.recompute_summary)
    print(f"{len(df)} episode(s) in summary")
    return 0


if __name__ == "__main__":
    sys.exit(main())
