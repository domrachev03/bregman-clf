"""(1) Worst-case row demand on sublevel sets {V2 <= c}, per metric; (2) closed loop under |a| <= a_max.

Run:  python dint_study.py <mode>, with mode one of
  demand            worst demand of Gamma = H(x), I, H(x_d) at mu = 1e-2, 1e-3      -> dint_demand.json
  demand_scaled     worst demand of Gamma = 1e2 I, 1e4 I                           -> dint_demand_scaled.json
  loop              closed loop, a_max = 10, 30, 100, Gamma = H(x), I, H(x_d)     -> dint_closed_loop_mu0.01.json
  loop_high         same at a_max = 1000, 3000                                    -> ..._high.json
  loop_scaled       closed loop, a_max = 10, 30, 100, Gamma = 1e2 I, 1e4 I        -> ..._scaled.json
  loop_scaled_high  same at a_max = 1000, 3000                                    -> ..._scaled_high.json
  loop_dt           Gamma = 1e2 I, 1e4 I at a_max = 3000 with a 4x finer step     -> ..._scaled_dt5e-5.json
Results are written to results/double_integrator/.
"""

from __future__ import annotations

import json
import sys

import numpy as np
from scipy.optimize import minimize

from dint import RESULTS as OUT
from dint import A, Model, inside, slacks
from dint import B as B_

METRICS = ["hess", "I", "target"]


def sample_states(m: Model, c: float, n: int, rng) -> list[tuple[np.ndarray, np.ndarray]]:
    """States on/inside {V2 <= c}: x uniform in C or close to a face, s on the level V2 = c."""
    out = []
    while len(out) < n:
        if rng.uniform() < 0.5:
            x = rng.uniform(-1.2, 1.2, 2)
        else:  # ray from x_d toward a random direction, stop log-uniformly close to the boundary
            u = rng.normal(size=2)
            u /= np.linalg.norm(u)
            den = A @ u
            t_exit = np.min((slacks(m.xd))[den > 0] / den[den > 0])
            gap = 10 ** rng.uniform(-7, 0) * t_exit
            x = m.xd + (t_exit - gap) * u
        if not inside(x, 0.0):
            continue
        W = m.D(x)
        if not np.isfinite(W) or W > c:
            continue
        G = m.Gamma(x)
        u = rng.normal(size=2)
        if rng.uniform() < 0.5:  # bias s toward the active face normal
            j = int(np.argmin(slacks(x)))
            u = A[j] * rng.choice([1.0, -1.0]) + 0.3 * u
        q = u @ G @ u
        s = u * np.sqrt(2 * (c - W) / q) * rng.uniform(0.5, 1.0) ** 0.5
        out.append((x, s))
    return out


def worst_demand(m: Model, c: float, n: int, rng) -> float:
    best, best_z = 0.0, None
    for x, s in sample_states(m, c, n, rng):
        d = m.demand(x, s + m.xdot_r(x))
        if np.isfinite(d) and d > best:
            best, best_z = d, np.concatenate([x, s])

    def neg(z):  # refine with a penalty for leaving {V2 <= c}
        x, s = z[:2], z[2:]
        if not inside(x, 0.0):
            return 1e30
        v = m.V2(x, s)
        if not np.isfinite(v) or v > c:
            return 1e30
        d = m.demand(x, s + m.xdot_r(x))
        return -d if np.isfinite(d) else 1e30

    if best_z is not None:
        r = minimize(neg, best_z, method="Nelder-Mead", options={"xatol": 1e-12, "fatol": 1e-9, "maxiter": 4000})
        best = max(best, -r.fun)
    return best


def k_bound(m: Model, sigma: float, c: float = np.inf) -> float:
    """Explicit bound K(sigma, c) on the acceleration demanded by the Hessian-metric law, P = p I."""
    D = diameter()
    nu = len(A)
    gF = A.T @ (1.0 / slacks(m.xd))
    lmin = m.p + 4 * m.mu / D**2  # lower bound on the smallest eigenvalue of H
    r_c = min(D, np.sqrt(2 * c / m.p))
    alpha = min(np.sqrt(m.p) * D, np.sqrt(2 * c)) + np.sqrt(m.mu * nu) + m.mu / np.sqrt(lmin) * np.linalg.norm(gF)
    beta = sigma + m.k * alpha
    return r_c + (m.k * beta + m.ks * sigma + beta * (2 * m.k * alpha + sigma) / np.sqrt(m.mu)) / np.sqrt(lmin)


def diameter() -> float:
    from itertools import combinations

    verts = []
    for i, j in combinations(range(len(A)), 2):
        M = np.stack([A[i], A[j]])
        if abs(np.linalg.det(M)) < 1e-9:
            continue
        v = np.linalg.solve(M, np.array([B_[i], B_[j]]))
        if np.all(slacks(v) > -1e-9):
            verts.append(v)
    V = np.array(verts)
    return max(np.linalg.norm(a - b) for a in V for b in V)


def study_demand(seed=0, n=4000):
    rng = np.random.default_rng(seed)
    res = {"c": [], "mu": []}
    cs = np.logspace(-2, 2, 9)
    for mu in [1e-2, 1e-3]:
        for metric in METRICS:
            m = Model(mu=mu, metric=metric, xd=np.array([0.3, -0.2]))
            vals = [worst_demand(m, c, n, rng) for c in cs]
            res[f"{metric}_mu{mu:g}"] = vals
            print(metric, mu, np.array(vals), flush=True)
        m = Model(mu=mu, metric="hess", xd=np.array([0.3, -0.2]))
        res[f"bound_mu{mu:g}"] = [k_bound(m, np.sqrt(2 * c)) for c in cs]
    res["c"] = cs.tolist()
    (OUT / "dint_demand.json").write_text(json.dumps(res, indent=1))


def study_demand_scaled(seed=0, n=4000):
    rng = np.random.default_rng(seed)
    cs = np.logspace(-2, 2, 9)
    res = {"c": cs.tolist()}
    for lam in [1e2, 1e4]:
        m = Model(mu=1e-2, metric="I", xd=np.array([0.3, -0.2]), lam=lam)
        res[f"I{lam:g}_mu0.01"] = [worst_demand(m, c, n, rng) for c in cs]
        print(lam, np.array(res[f"I{lam:g}_mu0.01"]), flush=True)
    (OUT / "dint_demand_scaled.json").write_text(json.dumps(res, indent=1))


def solve_row_ball(a_nom, c, r, amax):
    """argmin |a - a_nom| s.t. c.a <= r, |a| <= amax (2-D); None if empty."""
    nc = np.linalg.norm(c)
    if nc == 0:
        if r < 0:
            return None
        n = np.linalg.norm(a_nom)
        return a_nom if n <= amax else a_nom * amax / n
    if r < -amax * nc - 1e-12:
        return None
    cands = []
    n = np.linalg.norm(a_nom)
    p = a_nom if n <= amax else a_nom * amax / n
    if c @ p <= r + 1e-12:
        cands.append(p)
    q = a_nom - max(0.0, (c @ a_nom - r)) / nc**2 * c
    if np.linalg.norm(q) <= amax + 1e-12:
        cands.append(q)
    # line-circle intersections
    u = c / nc
    t0 = r / nc
    if abs(t0) <= amax:
        w = np.array([-u[1], u[0]])
        h = np.sqrt(max(amax**2 - t0**2, 0.0))
        cands += [t0 * u + h * w, t0 * u - h * w]
    if not cands:
        return None
    return min(cands, key=lambda z: np.linalg.norm(z - a_nom))


def run_trial(m: Model, x0, v0, amax, T=4.0, dt=2e-4):
    dt = float(globals().get("DT_OVERRIDE", dt))  # set by the `loop_dt` mode
    x, v = x0.copy(), v0.copy()
    infeas = 0
    steps = int(T / dt)
    min_slack = np.inf
    V20 = m.V2(x, v - m.xdot_r(x))
    for _ in range(steps):
        sl = slacks(x)
        min_slack = min(min_slack, float(sl.min()))
        if np.any(sl <= 0):
            return dict(exit=True, infeas=infeas / steps, err=float(np.linalg.norm(x - m.xd)), V20=V20, min_slack=min_slack)
        c, r = m.row(x, v)
        a_nom = m.law(x, v)
        a = solve_row_ball(a_nom, c, r, amax)
        if a is None:
            infeas += 1
            nc = np.linalg.norm(c)
            a = -amax * c / nc if nc > 0 else np.zeros(2)
        v = v + dt * a
        x = x + dt * v
    return dict(exit=False, infeas=infeas / steps, err=float(np.linalg.norm(x - m.xd)), V20=V20, min_slack=min_slack)


def _cell(args):
    metric, amax, trials, mu = args
    lam = 1.0
    if metric.startswith("I") and metric != "I":
        lam = float(metric[1:])  # "I1e+02" -> lambda = 100
    out = []
    for xd, x0, v0 in trials:
        m = Model(mu=mu, metric="I" if metric.startswith("I") else metric, xd=xd, lam=lam)
        o = run_trial(m, x0, v0, amax)
        o["xd"] = xd.tolist()
        out.append(o)
    return f"{metric}_a{amax:g}", out


def study_closed_loop(seed=1, n=100, mu=1e-2, amaxes=(10.0, 30.0, 100.0), tag="", metrics=None):
    rng = np.random.default_rng(seed)
    trials = []
    while len(trials) < n:
        xd = rng.uniform(-1, 1, 2)
        x0 = rng.uniform(-1, 1, 2)
        if not (inside(xd, 0.02) and inside(x0, 0.002)):
            continue
        j = int(np.argmin(slacks(x0)))
        v0 = A[j] * rng.uniform(0.5, 1.5) + 0.3 * rng.normal(size=2)  # moving toward the nearest face
        trials.append((xd, x0, v0))
    import multiprocessing as mp

    jobs = [(metric, amax, trials, mu) for amax in amaxes for metric in (metrics or METRICS)]
    with mp.Pool(len(jobs)) as pool:
        res = dict(pool.map(_cell, jobs))
    for key, out in res.items():
        ex = sum(o["exit"] for o in out)
        inf_tr = sum(o["infeas"] > 0 for o in out)
        print(f"{key:12s} exits {ex}/{n}  trials with infeasible steps {inf_tr}/{n}  "
              f"median err {np.median([o['err'] for o in out]):.2e}", flush=True)
    (OUT / f"dint_closed_loop_mu{mu:g}{tag}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("demand", "all"):
        study_demand()
    if which in ("loop", "all"):
        study_closed_loop()
    if which == "loop_scaled":
        study_closed_loop(amaxes=(10.0, 30.0, 100.0), tag="_scaled", metrics=["I1e+02", "I1e+04"])
    if which == "loop_dt":
        globals()["DT_OVERRIDE"] = 5e-5
        study_closed_loop(amaxes=(3000.0,), tag="_scaled_dt5e-5", metrics=["I1e+04", "I1e+02"])
    if which == "loop_scaled_high":
        study_closed_loop(amaxes=(1000.0, 3000.0), tag="_scaled_high", metrics=["I1e+02", "I1e+04"])
    if which == "demand_scaled":
        study_demand_scaled()
    if which == "loop_high":
        study_closed_loop(amaxes=(1000.0, 3000.0), tag="_high")
