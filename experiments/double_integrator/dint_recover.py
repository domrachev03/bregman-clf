"""Recoverability of the closed-loop starts under ||a|| <= a_max, corner-aware.

- unrecoverable: some face is approached at normal speed v_n with v_n^2/(2 l) > a_max (no admissible
  input avoids that face; sufficient for an unavoidable exit, per face);
- recoverable: braking along the velocity, a = -a_max v/||v||, stops on a straight segment of length
  ||v||^2/(2 a_max) that stays inside C with every slack >= 1e-6; failing that, a constant thrust
  a_max (cos th, sin th) for a time tau, followed by braking along the velocity, does so (grid over
  th and tau, exact minimum over the thrust arc). Both are sufficient: an admissible input that keeps
  x in C for all t;
- undetermined: neither test decides.
Each start is reported with the exits of every metric. Reads the closed-loop results of
dint_study.py (modes loop, loop_high, loop_scaled, loop_scaled_high); writes dint_recover.json.

    python dint_recover.py
"""

from __future__ import annotations

import json

import numpy as np

from dint import RESULTS, A, inside, slacks

FILES = ["dint_closed_loop_mu0.01.json", "dint_closed_loop_mu0.01_high.json",
         "dint_closed_loop_mu0.01_scaled.json", "dint_closed_loop_mu0.01_scaled_high.json"]


def starts(seed=1, n=100):
    """Same generator as dint_study.study_closed_loop."""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        xd = rng.uniform(-1, 1, 2)
        x0 = rng.uniform(-1, 1, 2)
        if not (inside(xd, 0.02) and inside(x0, 0.002)):
            continue
        j = int(np.argmin(slacks(x0)))
        v0 = A[j] * rng.uniform(0.5, 1.5) + 0.3 * rng.normal(size=2)
        out.append((xd, x0, v0))
    return out


def classify(x0, v0, amax):
    sl = slacks(x0)
    vn = A @ v0
    if np.any((vn > 0) & (vn**2 / (2 * sl) > amax)):
        return "unrecoverable"
    sp = np.linalg.norm(v0)
    d = sp**2 / (2 * amax)
    # slacks are affine along the segment, so checking the stopping point suffices
    if np.all(slacks(x0 + d * v0 / sp) >= 1e-6):
        return "recoverable"
    from dint_sampling import interval_min_slack

    for th in np.linspace(0, 2 * np.pi, 144, endpoint=False):
        a = amax * np.array([np.cos(th), np.sin(th)])
        for tau in np.linspace(0.005, 0.5, 100):
            if interval_min_slack(x0, v0, a, tau) < 1e-6:
                break  # longer thrusts in this direction cross the same face
            x1, v1 = x0 + tau * v0 + 0.5 * tau**2 * a, v0 + tau * a
            s1 = np.linalg.norm(v1)
            if s1 == 0 or np.all(slacks(x1 + s1 / (2 * amax) * v1) >= 1e-6):
                return "recoverable"
    return "undetermined"


def main():
    tr = starts()
    loops = {}
    for f in FILES:
        loops.update(json.loads((RESULTS / f).read_text()))
    for k, out in loops.items():  # same starts as the stored trials
        assert all(np.allclose(o["xd"], t[0]) for o, t in zip(out, tr)), k
    amaxes = sorted({float(k.split("_a")[1]) for k in loops})
    res = {}
    for amax in amaxes:
        cls = [classify(x0, v0, amax) for _, x0, v0 in tr]
        row = {c: cls.count(c) for c in ("unrecoverable", "recoverable", "undetermined")}
        for k, out in loops.items():
            if float(k.split("_a")[1]) != amax:
                continue
            metric = k.split("_a")[0]
            row[metric] = {c: sum(o["exit"] and cl == c for o, cl in zip(out, cls))
                           for c in ("unrecoverable", "recoverable", "undetermined")}
        res[f"{amax:g}"] = row
        print(f"a_max={amax:g}", json.dumps(row))
    (RESULTS / "dint_recover.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
