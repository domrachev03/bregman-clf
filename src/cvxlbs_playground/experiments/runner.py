"""`run_episode`: one closed-loop episode of a method on a scenario in the MuJoCo torque simulator."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from cvxlbs_playground.methods.registry import make_method
from cvxlbs_playground.scenarios import Scenario, keep_in_margin
from cvxlbs_playground.sim import MujocoSim

from .logs import EpisodeLog, status_to_code


def run_episode(scenario: Scenario, method_name: str, seed: int, *, method_kw: dict[str, Any] | None = None) -> EpisodeLog:
    """Reset the scenario with `default_rng(seed)`, build the method, and run `T / dt` control steps
    (controller every simulator step, zero-order hold). Deterministic given `(scenario, method, seed)`
    up to the logged wall times."""
    method_kw = dict(method_kw or {})
    q0, v0 = scenario.reset(np.random.default_rng(seed))
    q0 = np.asarray(q0, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    entry = scenario.entry
    method = make_method(method_name, scenario, **method_kw)
    method.reset(q0, v0)

    dt = float(scenario.spec.dt)
    T = float(scenario.spec.T)
    keep_in = scenario.sets()["keep_in"]
    site_id = mujoco.mj_name2id(entry.mj_model, mujoco.mjtObj.mjOBJ_SITE, entry.meta["ee_frame"])

    sim = MujocoSim(entry, dt=dt)
    sim.reset(q0, v0)

    cols: dict[str, list[Any]] = {
        name: []
        for name in ("t", "q", "v", "u", "tau_req", "tau_applied", "status", "fallback", "t_eval", "t_solve", "t_total")
    }
    monitors: dict[str, list[float]] = {}
    ee_pos: list[np.ndarray] = []
    margin: list[float] = []
    for _ in range(round(T / dt)):
        q, v, t = sim.q, sim.v, sim.t
        p, pd, pdd = scenario.reference(t)
        u, info = method.step(q, v, t, p, pd, pdd)
        u = np.asarray(u, dtype=float)
        sim.step(u)

        cols["t"].append(float(t))
        cols["q"].append(q)
        cols["v"].append(v)
        cols["u"].append(u)
        cols["tau_req"].append(sim.last_requested_torque)
        cols["tau_applied"].append(sim.last_applied_torque)
        cols["status"].append(status_to_code(info["status"]))
        cols["fallback"].append(bool(info["fallback"]))
        cols["t_eval"].append(float(info["t_eval"]))
        cols["t_solve"].append(float(info["t_solve"]))
        cols["t_total"].append(float(info["t_total"]))
        for key, val in info["monitors"].items():
            monitors.setdefault(key, []).append(float(val))
        # ground truth from the simulator: `mj_step` evaluates the kinematics before integrating, so
        # this is the end-effector position at `q[k]`
        pos = np.array(sim.data.site_xpos[site_id], dtype=float)
        ee_pos.append(pos)
        margin.append(float(keep_in_margin(pos, keep_in)))

    return EpisodeLog(
        **cols,
        monitors={k: np.asarray(v, dtype=float) for k, v in monitors.items()},
        ground_truth={"ee_pos": np.asarray(ee_pos), "keep_in_margin": np.asarray(margin)},
        meta={
            "scenario": scenario.spec.name,
            "robot": scenario.spec.robot,
            "method": method_name,
            "method_kw": method_kw,
            "seed": int(seed),
            "dt": dt,
            "T": T,
        },
    )
