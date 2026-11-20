"""Per-step wall time of the controllers on one core.

Each method runs with its gains from `configs/cbf_comparison.yaml` and its QP data compiled to C
(`jit=True`, needs `gcc`; compiled functions are cached in `~/.cache/cvxlbs` or `$CVXLBS_CACHE`),
sequentially and single-threaded, on the box scenario for `T = 2 s` (2000 steps per seed). Reported:
median, 99th percentile and maximum of the controller's step time (QP data evaluation + solve) and
of the solve alone; the statistics are those of the repeat with the smallest median, the maximum
is also given over all repeats.

    taskset -c 2 python experiments/manipulator/timing.py [--repeats 3] [--seeds 0 1 2]

Writes results/manipulator/timing.json.
"""

from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"

import argparse
import json
import platform
import subprocess
import warnings

import numpy as np
import yaml
from _paths import CONFIGS, RESULTS

warnings.simplefilter("ignore")

from cvxlbs_playground.experiments.runner import run_episode  # noqa: E402
from cvxlbs_playground.scenarios import make_scenario  # noqa: E402


def cpu_name() -> str:
    try:
        out = subprocess.run(["lscpu"], capture_output=True, text=True, check=False).stdout
        for line in out.splitlines():
            if line.startswith("Model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def time_method(name: str, kw: dict, repeats: int, seeds: list[int]) -> dict:
    best, worst = None, 0.0
    for _ in range(repeats):
        tot, sol = [], []
        for seed in seeds:
            sc = make_scenario("iiwa_constraints_keep_in_box", T=2.0)
            log = run_episode(sc, name, seed=seed, method_kw=dict(kw))
            tot.append(log.t_total[20:])  # drop the warm-up steps
            sol.append(log.t_solve[20:])
        tot, sol = np.concatenate(tot), np.concatenate(sol)
        res = {
            "total_p50_us": 1e6 * float(np.percentile(tot, 50)),
            "total_p99_us": 1e6 * float(np.percentile(tot, 99)),
            "solve_p50_us": 1e6 * float(np.percentile(sol, 50)),
            "solve_p99_us": 1e6 * float(np.percentile(sol, 99)),
            "total_p999_us": 1e6 * float(np.percentile(tot, 99.9)),
            "total_max_us": 1e6 * float(tot.max()),
            "solve_max_us": 1e6 * float(sol.max()),
            "n_steps": int(tot.size),
        }
        worst = max(worst, res["total_max_us"])
        if best is None or res["total_p50_us"] < best["total_p50_us"]:
            best = res
    best["total_max_all_repeats_us"] = worst
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--methods", nargs="*", default=["bregman_id_level", "clf_aecbf", "tsid_cbf"], help="method labels")
    ap.add_argument("--out", default=str(RESULTS / "timing.json"))
    args = ap.parse_args()
    methods = {m["label"]: m for m in yaml.safe_load((CONFIGS / "cbf_comparison.yaml").read_text())["methods"]}
    out = {"cpu": cpu_name(), "affinity": sorted(os.sched_getaffinity(0)), "methods": {}}
    for label in args.methods:
        kw = {**methods[label]["kw"], "jit": True}
        r = out["methods"][label] = time_method(methods[label]["name"], kw, args.repeats, args.seeds)
        print(
            f"{label:18s} step p50/p99/max {r['total_p50_us']:6.1f}/{r['total_p99_us']:6.1f}/{r['total_max_all_repeats_us']:6.1f} us"
            f"   solve p50/p99 {r['solve_p50_us']:5.1f}/{r['solve_p99_us']:5.1f} us",
            flush=True,
        )
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
