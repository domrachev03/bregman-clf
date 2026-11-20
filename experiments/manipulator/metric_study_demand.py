"""Offline minimum-norm demand d_Gamma on the logged states of the manipulator backstepping-metric
study (`runs/metric_study`, produced by the sweep `configs/metric_study.yaml`).

For every episode: rebuild the scenario from its seed (keep-in set, target), take the logged
end-effector positions and the exact task velocity J(q) v from the logged joint states, and
evaluate the rate constraint  V2_dot + gamma V2 <= 0  of each metric at every step:
d_Gamma = max(0, -r) / ||Gamma s||. Self-concordant psi only (polytope, ellipsoid).

The demand is recomputed here in plain numpy, independently of the controller code; the logged
V2 is compared with the recomputed one as a consistency check.

    python experiments/manipulator/metric_study_demand.py   (16 processes, about 1 minute)

Writes results/manipulator/metric_study_demand.json and runs/metric_study/demand.parquet.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import yaml
from _paths import CONFIGS, RESULTS
from _paths import RUNS as _RUNS
from cvxlbs_playground.scenarios import make_scenario

RUNS = _RUNS / "metric_study"
CFG = CONFIGS / "metric_study.yaml"
MU, P, K, KS = 1e-3, 1e-2, 5.0, 5.0
GAMMA = min(K, 2 * KS)


class Barrier:
    def __init__(self, keep_in):
        self.kind = keep_in["type"]
        if self.kind == "polytope":
            self.A, self.b = np.asarray(keep_in["A"]), np.asarray(keep_in["b"])
        else:
            self.c = np.asarray(keep_in["center"])
            self.E = np.linalg.inv(np.asarray(keep_in["shape"]))

    def margin(self, x):
        if self.kind == "polytope":
            return float(np.min(self.b - self.A @ x))
        u = x - self.c
        return 1.0 - u @ self.E @ u

    def dist(self, x):
        """Euclidean distance to the boundary (the ellipsoids here are balls)."""
        if self.kind == "polytope":
            return float(np.min(self.b - self.A @ x))
        r = 1.0 / np.sqrt(self.E[0, 0])
        assert np.allclose(self.E, np.eye(3) / r**2)
        return r - float(np.linalg.norm(x - self.c))

    def val(self, x):
        if self.kind == "polytope":
            return -np.sum(np.log(self.b - self.A @ x))
        u = x - self.c
        return -np.log(1.0 - u @ self.E @ u)

    def grad(self, x):
        if self.kind == "polytope":
            return self.A.T @ (1.0 / (self.b - self.A @ x))
        u = x - self.c
        g = 1.0 - u @ self.E @ u
        return 2 * self.E @ u / g

    def hess(self, x):
        if self.kind == "polytope":
            s = self.b - self.A @ x
            return (self.A.T * (1.0 / s**2)) @ self.A
        u = x - self.c
        g = 1.0 - u @ self.E @ u
        Eu = self.E @ u
        return 2 * self.E / g + 4 * np.outer(Eu, Eu) / g**2

    def dhess(self, x, w):  # directional derivative of the Hessian along w
        if self.kind == "polytope":
            s = self.b - self.A @ x
            return (self.A.T * (2.0 * (self.A @ w) / s**3)) @ self.A
        u = x - self.c
        g = 1.0 - u @ self.E @ u
        gd = -2 * u @ self.E @ w
        Eu, Ew = self.E @ u, self.E @ w
        return (-2 * self.E * gd / g**2 + 4 * (np.outer(Ew, Eu) + np.outer(Eu, Ew)) / g**2
                - 8 * np.outer(Eu, Eu) * gd / g**3)


class Row:
    def __init__(self, bar: Barrier, xd, metric):
        self.bar, self.xd, self.metric = bar, np.asarray(xd), metric
        self.gd = self.grad(self.xd)
        self.Hd = self.hess(self.xd)

    def psi(self, x):
        return 0.5 * P * x @ x + MU * self.bar.val(x)

    def grad(self, x):
        return P * x + MU * self.bar.grad(x)

    def hess(self, x):
        return P * np.eye(3) + MU * self.bar.hess(x)

    def D(self, x):
        return self.psi(self.xd) - self.psi(x) - self.grad(x) @ (self.xd - x)

    def demand(self, x, xdot):
        H = self.hess(x)
        e = np.linalg.solve(H, self.grad(x) - self.gd)
        xr = -K * e
        s = xdot - xr
        Hdot = MU * self.bar.dhess(x, xdot)
        xr2 = -K * (xdot - np.linalg.solve(H, Hdot @ e))
        G = {"hessian": H, "constant": np.eye(3), "target": self.Hd}[self.metric]
        V2 = self.D(x) + 0.5 * s @ G @ s
        r = -GAMMA * V2 - (x - self.xd) @ H @ xdot + s @ G @ xr2
        if self.metric == "hessian":
            r -= 0.5 * s @ Hdot @ s
        c = G @ s
        if r >= 0:
            return 0.0
        n = np.linalg.norm(c)
        return np.inf if n == 0 else -r / n


def _episode(args):
    scen, opts, label, seed = args
    metric_run = label.split("_")[1]
    f = RUNS / scen / label / f"{seed}.npz"
    if not f.exists():
        return [], []
    sc = make_scenario(scen, **dict(opts))
    sc.reset(np.random.default_rng(seed))
    keep_in = sc.sets()["keep_in"]
    xd = np.asarray(sc.target_position(), dtype=float)
    model, frame = sc.entry.model, sc.entry.meta["ee_frame"]
    d = np.load(f, allow_pickle=True)
    Q, Vq = np.asarray(d["q"], float), np.asarray(d["v"], float)
    fb = np.asarray(d["fallback"], dtype=bool)
    V2log = np.asarray(d["monitors__bregman_clf/V2"], float)
    bar = Barrier(keep_in)
    rows_eval = {m: Row(bar, xd, m) for m in ("constant", "target", "hessian")}
    rows, verr = [], []
    for i in range(len(Q)):
        x = np.asarray(model.numeric("frame_position", frame, Q[i]), float).ravel()
        if fb[i] or bar.margin(x) <= 0:
            continue
        J = np.asarray(model.numeric("frame_jacobian", frame, Q[i]), float)[:3, :]
        xdot = J @ Vq[i]
        dist = bar.dist(x)
        r = rows_eval[metric_run]
        H = r.hess(x)
        s_ = xdot + K * np.linalg.solve(H, r.grad(x) - r.gd)
        G = {"hessian": H, "constant": np.eye(3), "target": r.Hd}[metric_run]
        V2 = r.D(x) + 0.5 * s_ @ G @ s_
        if np.isfinite(V2log[i]) and V2log[i] > 1e-9:
            verr.append(abs(V2 - V2log[i]) / V2log[i])
        rows.append((scen, metric_run, label.split("_")[2], seed, dist,
                     rows_eval["constant"].demand(x, xdot), rows_eval["target"].demand(x, xdot),
                     rows_eval["hessian"].demand(x, xdot)))
    return rows, verr


def main():
    import multiprocessing as mp

    cfg = yaml.safe_load(CFG.read_text())
    opts = {s["name"]: s["options"] for s in cfg["scenarios"]}
    jobs = [(scen, opts[scen], f"bs_{m}_{form}", seed)
            for scen in sorted(opts) if "entropy" not in scen
            for m in ("constant", "target", "hessian") for form in ("hard", "level")
            for seed in range(cfg["seeds"]["start"], cfg["seeds"]["start"] + cfg["seeds"]["n"])]
    with mp.Pool(16) as pool:
        res = pool.map(_episode, jobs, chunksize=4)
    rows = [r for rr, _ in res for r in rr]
    verr = np.array([e for _, ee in res for e in ee])
    df = pd.DataFrame(rows, columns=["scenario", "run_metric", "form", "seed", "dist", "d_constant", "d_target", "d_hessian"])
    df.to_parquet(RUNS / "demand.parquet")
    out = {"V2_rel_err_median": float(np.median(verr)), "V2_rel_err_p99": float(np.percentile(verr, 99)), "n_V2": int(len(verr))}
    near = df[df.dist < 1e-3]
    far = df[df.dist > 3e-2]
    for run in ("constant", "target", "hessian"):
        sub, subf = near[near.run_metric == run], far[far.run_metric == run]
        out[f"states_{run}"] = {"n_near": int(len(sub)), "n_trials": int(sub.groupby(["scenario", "form", "seed"]).ngroups),
                                **{f"near_p99_{m}": float(np.percentile(sub[f"d_{m}"], 99)) for m in ("constant", "target", "hessian")},
                                **{f"far_p99_{m}": float(np.percentile(subf[f"d_{m}"], 99)) for m in ("constant", "target", "hessian")}}
    (RESULTS / "metric_study_demand.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
