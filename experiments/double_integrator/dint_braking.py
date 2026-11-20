"""Braking-aware per-face CBF baseline on the double integrator.

Same 100 starts/targets and the same a_max as the closed-loop study. Per face i, with normal speed
v_n = n_i.v and slack l_i, h_i = l_i - max(v_n, 0)^2 / (2 beta), beta = 0.9 a_max (braking reserve),
and the row  dh_i/dt + alpha h_i >= 0  is linear in a (only faces with v_n > 0 give a row). The input
is argmin |a - a_pd|^2 over these rows and the inscribed 64-gon of {|a| <= a_max} (DAQP), with
a_pd = -kp (x - x_d) - kd v; infeasible -> brake along the velocity, a = -a_max v/|v|. Exact
zero-order-hold integration and inter-sample margins as in dint_sampling.py.
Writes dint_braking.json.

    python dint_braking.py
"""

from __future__ import annotations

import json
import multiprocessing as mp

import daqp
import numpy as np

from dint import RESULTS, A, slacks
from dint_recover import classify, starts
from dint_sampling import interval_min_slack

NPOLY = 64
_ANG = 2 * np.pi * (np.arange(NPOLY) + 0.5) / NPOLY
POLY = np.stack([np.cos(_ANG), np.sin(_ANG)], axis=1)  # POLY a <= amax cos(pi/NPOLY): inscribed


def control(x, v, xd, amax, alpha, kp, kd):
    a_pd = -kp * (x - xd) - kd * v
    beta = 0.9 * amax
    sl, vn = slacks(x), A @ v
    rows, ub = [POLY], [np.full(NPOLY, amax * np.cos(np.pi / NPOLY))]
    act = vn > 0
    if np.any(act):
        h = sl[act] - vn[act] ** 2 / (2 * beta)
        # dh/dt = -vn - (vn/beta) n.a  >=  -alpha h   <=>   (vn/beta) n.a <= -vn + alpha h
        rows.append((vn[act] / beta)[:, None] * A[act])
        ub.append(-vn[act] + alpha * h)
    Aq, bu = np.vstack(rows), np.concatenate(ub)
    bl = np.full_like(bu, -1e30)
    sense = np.zeros(len(bu), dtype=np.int32)
    xstar, _, exitflag, _ = daqp.solve(np.eye(2), -a_pd, Aq, bu, bl, sense)
    if exitflag < 1:
        sp = np.linalg.norm(v)
        return (-amax * v / sp if sp > 0 else np.zeros(2)), True
    return np.asarray(xstar), False


def run_trial(xd, x0, v0, amax, dt=2e-4, T=4.0, alpha=20.0, kp=4.0, kd=4.0):
    x, v = x0.copy(), v0.copy()
    infeas, min_margin = 0, np.inf
    for _ in range(int(round(T / dt))):
        a, bad = control(x, v, xd, amax, alpha, kp, kd)
        infeas += bad
        mm = interval_min_slack(x, v, a, dt)
        min_margin = min(min_margin, mm)
        if mm <= 0:
            return dict(exit=True, infeas=infeas, min_margin=min_margin, err=float(np.linalg.norm(x - xd)))
        x, v = x + dt * v + 0.5 * dt**2 * a, v + dt * a
    return dict(exit=False, infeas=infeas, min_margin=min_margin, err=float(np.linalg.norm(x - xd)))


def _cell(amax):
    out = []
    for xd, x0, v0 in starts():
        o = run_trial(xd, x0, v0, amax)
        o["start"] = classify(x0, v0, amax)
        out.append(o)
    return f"braking_a{amax:g}", out


def main():
    amaxes = [10.0, 30.0, 100.0, 1000.0, 3000.0]
    with mp.Pool(len(amaxes)) as pool:
        res = dict(pool.map(_cell, amaxes))
    (RESULTS / "dint_braking.json").write_text(json.dumps(res, indent=1))
    for k, out in res.items():
        mm = np.array([o["min_margin"] for o in out])
        print(f"{k:16s} exits {sum(o['exit'] for o in out):3d} (recoverable "
              f"{sum(o['exit'] and o['start'] == 'recoverable' for o in out)}, unrecoverable "
              f"{sum(o['exit'] and o['start'] == 'unrecoverable' for o in out)})  inf-trials "
              f"{sum(o['infeas'] > 0 for o in out)}  p10 margin {np.percentile(mm, 10):.2e}  "
              f"median err {np.median([o['err'] for o in out]):.1e}", flush=True)


if __name__ == "__main__":
    main()
