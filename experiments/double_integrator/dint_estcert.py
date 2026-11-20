"""Estimated certificates for the closed-loop trials of Table I (Hessian metric).

Cor. 1(i) needs only that the rate constraint is feasible under |a| <= a_max on a sublevel set
{V2 <= c} with c >= V2(0), i.e. that the worst demand of the Hessian metric on that set is below
a_max. The closed-form bound K(c) guarantees this but is conservative. Here the worst demand is
estimated numerically for every trial, on its own sublevel set {V2 <= V2(0)} and for its own target:
random search (as in dint_study.sample_states) followed by Nelder-Mead refinement from the best
samples. The result is a LOWER estimate of the true supremum, so "estimate below a_max" is evidence,
not a proof.

Run:  python dint_estcert.py  -> writes dint_estcert.json (16 processes, a few minutes).
"""

from __future__ import annotations

import json
import multiprocessing as mp

import numpy as np
from scipy.optimize import minimize

from dint import RESULTS, Model, inside
from dint_study import k_bound, sample_states

N_SAMPLES = 30000  # random states per trial and seed
N_REFINE = 12  # Nelder-Mead starts (the best samples)
SEEDS = (0, 1, 2, 3)  # independent repeats; the estimate is their maximum


def worst_demand_multistart(m: Model, c: float, n: int, rng) -> float:
    """Largest demand found on {V2 <= c}: n random states, then local refinement from the best ones."""
    found = []
    for x, s in sample_states(m, c, n, rng):
        d = m.demand(x, s + m.xdot_r(x))
        if np.isfinite(d):
            found.append((d, np.concatenate([x, s])))
    found.sort(key=lambda t: -t[0])
    best = found[0][0] if found else 0.0

    def neg(z):  # stay inside {V2 <= c}
        x, s = z[:2], z[2:]
        if not inside(x, 0.0):
            return 1e30
        v = m.V2(x, s)
        if not np.isfinite(v) or v > c:
            return 1e30
        d = m.demand(x, s + m.xdot_r(x))
        return -d if np.isfinite(d) else 1e30

    for _, z in found[:N_REFINE]:
        r = minimize(neg, z, method="Nelder-Mead", options={"xatol": 1e-12, "fatol": 1e-9, "maxiter": 4000})
        best = max(best, -r.fun)
    return float(best)


def _trial(args):
    i, xd, v20 = args
    m = Model(mu=1e-2, metric="hess", xd=np.array(xd))
    per_seed = [worst_demand_multistart(m, v20, N_SAMPLES, np.random.default_rng(1000 * (sd + 1) + i)) for sd in SEEDS]
    return dict(trial=i, xd=xd, V20=v20, est_per_seed=per_seed, est=max(per_seed),
                K=float(k_bound(m, np.sqrt(2 * v20), v20)))


def main():
    loop = json.loads((RESULTS / "dint_closed_loop_mu0.01.json").read_text())
    trials = loop["hess_a10"]  # the same 100 starts and targets are used for every a_max
    jobs = [(i, t["xd"], t["V20"]) for i, t in enumerate(trials)]
    with mp.Pool(min(16, mp.cpu_count())) as pool:
        out = pool.map(_trial, jobs, chunksize=1)
    (RESULTS / "dint_estcert.json").write_text(json.dumps(dict(n_samples=N_SAMPLES, n_refine=N_REFINE, seeds=list(SEEDS), trials=out), indent=1))
    est = np.array([o["est"] for o in out])
    spread = np.array([(max(o["est_per_seed"]) - min(o["est_per_seed"])) / o["est"] for o in out])
    print(f"estimated worst demand: median {np.median(est):.2f}, max {est.max():.1f}; "
          f"relative spread between seeds: median {np.median(spread):.1e}, max {spread.max():.1e}")
    print("K / estimate: median %.0f, min %.0f, max %.0f" % tuple(np.percentile([o["K"] / o["est"] for o in out], [50, 0, 100])))


if __name__ == "__main__":
    main()
