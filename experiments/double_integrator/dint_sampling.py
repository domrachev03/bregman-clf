"""Sampling-period sweep of the double-integrator closed loop.

Same 100 starts and targets as dint_study.study_closed_loop, same input rule (closest point to the
backstepping law satisfying the rate row and ||a|| <= a_max; infeasible -> -a_max Gamma s/|Gamma s|),
but the acceleration is held over each period dt and the state is integrated exactly
(x += dt v + dt^2/2 a, v += dt a). The signed margin min_j slack_j(x(t)) is minimized analytically
over every inter-sample interval, so crossings between updates are counted. A trial ends at the
first crossing (exit) or once V2 < 1e-10 (converged; the margin can no longer shrink).
Writes dint_sampling.json.

    python dint_sampling.py [--amax 100 1000] [--dts 1e-4 2e-4 5e-4 1e-3 2e-3]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp

import numpy as np

from dint import RESULTS, A, Model, slacks
from dint_recover import classify, starts
from dint_study import solve_row_ball

METRICS = [("hess", 1.0), ("I", 1.0), ("target", 1.0), ("I", 1e2), ("I", 1e4)]


def interval_min_slack(x, v, a, dt):
    """min over t in [0, dt] of min_j slack_j(x + t v + t^2/2 a)."""
    sl = slacks(x)
    nv, na = A @ v, A @ a
    best = np.minimum(sl, sl - dt * nv - 0.5 * dt**2 * na)
    with np.errstate(divide="ignore", invalid="ignore"):
        ts = np.where(na < 0, -nv / na, np.nan)  # interior minimum of a convex slack
    ok = np.isfinite(ts) & (ts > 0) & (ts < dt)
    if np.any(ok):
        t = ts[ok]
        best[ok] = np.minimum(best[ok], sl[ok] - t * nv[ok] - 0.5 * t**2 * na[ok])
    return float(best.min())


def run_trial(m: Model, x0, v0, amax, dt, T=4.0):
    x, v = x0.copy(), v0.copy()
    infeas, steps, min_margin = 0, int(round(T / dt)), np.inf
    for n in range(steps):
        c, r = m.row(x, v)
        a = solve_row_ball(m.law(x, v), c, r, amax)
        if a is None:
            infeas += 1
            nc = np.linalg.norm(c)
            a = -amax * c / nc if nc > 0 else np.zeros(2)
        mm = interval_min_slack(x, v, a, dt)
        min_margin = min(min_margin, mm)
        if mm <= 0:
            return dict(exit=True, infeas=infeas, min_margin=min_margin)
        x, v = x + dt * v + 0.5 * dt**2 * a, v + dt * a
        if m.V2(x, v - m.xdot_r(x)) < 1e-10:
            break
    return dict(exit=False, infeas=infeas, min_margin=min_margin)


def _cell(args):
    key, lam, amax, dt = args
    out = []
    for xd, x0, v0 in starts():
        m = Model(mu=1e-2, metric=key, xd=xd, lam=lam)
        o = run_trial(m, x0, v0, amax, dt)
        o["start"] = classify(x0, v0, amax)
        out.append(o)
    name = key if lam == 1.0 else f"I{lam:g}"
    return f"{name}_a{amax:g}_dt{dt:g}", out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amax", type=float, nargs="*", default=[100.0, 1000.0])
    ap.add_argument("--dts", type=float, nargs="*", default=[1e-4, 2e-4, 5e-4, 1e-3, 2e-3])
    ap.add_argument("--procs", type=int, default=10)
    args = ap.parse_args()
    jobs = [(k, lam, amax, dt) for amax in args.amax for dt in args.dts for k, lam in METRICS]
    with mp.Pool(args.procs) as pool:
        res = dict(pool.imap_unordered(_cell, jobs))
    path = RESULTS / "dint_sampling.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update(res)
    path.write_text(json.dumps(old, indent=1))
    for k in sorted(res):
        out = res[k]
        mm = np.array([o["min_margin"] for o in out])
        print(f"{k:24s} exits {sum(o['exit'] for o in out):3d} (from recoverable "
              f"{sum(o['exit'] and o['start'] == 'recoverable' for o in out):3d})  "
              f"inf-trials {sum(o['infeas'] > 0 for o in out):3d}  min margin {mm.min():.2e}  "
              f"p10 {np.percentile(mm, 10):.2e}", flush=True)


if __name__ == "__main__":
    main()
