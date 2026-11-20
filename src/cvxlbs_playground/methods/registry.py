"""`make_method(name, scenario, **kw) -> Method`: the controllers of the experiments behind one
interface. Every method outputs joint torques; on a step where its QP is infeasible, the
controller applies the shared fallback law `tau = g(q) - 15 M(q) v`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from cvxlbs.controller import Controller

from cvxlbs_playground.scenarios import Scenario

from . import baselines, ours

_NAMES = ("bregman_id", "tsid", "tsid_cbf", "clf_aecbf")


class Method:
    """A built `cvxlbs.Controller` with the `reset`/`step` interface of the episode runner."""

    def __init__(self, name: str, controller: Controller) -> None:
        self.name = name
        self.controller = controller

    def reset(self, q0: np.ndarray, v0: np.ndarray) -> None:
        pass

    def step(self, q, v, t, p=None, pd=None, pdd=None) -> tuple[np.ndarray, dict[str, Any]]:
        u, info = self.controller.step(q, v, t, p, pd, pdd)
        return u, {
            "status": info.status,
            "fallback": info.fallback,
            "reason": info.reason,
            "t_eval": info.t_eval,
            "t_solve": info.t_solve,
            "t_total": info.t_total,
            "monitors": info.monitors,
        }


class LevelMethod(Method):
    """Sets the level of the level constraint (`ours._LevelRow`) to `level_factor * V2(0)`, with
    `V2(0)` read from the certificate at the first step of the episode (one extra controller
    evaluation, with the level constraint inactive)."""

    def __init__(self, name: str, controller: Controller, level_factor: float) -> None:
        super().__init__(name, controller)
        self._level_factor = float(level_factor)
        self._level: float | None = None

    def reset(self, q0: np.ndarray, v0: np.ndarray) -> None:
        self._level = None

    def _params(self, p: dict | None) -> dict:
        p = dict(p or {})
        p["cert_level"] = np.array([self._level if self._level is not None else 1e6])
        return p

    def step(self, q, v, t, p=None, pd=None, pdd=None) -> tuple[np.ndarray, dict[str, Any]]:
        if self._level is None:
            _, info = super().step(q, v, t, self._params(p), pd, pdd)
            v2 = [val for key, val in info["monitors"].items() if key.endswith("_level/V2")]
            self._level = max(self._level_factor * float(v2[0]), 1e-9) if v2 else 1e6
        return super().step(q, v, t, self._params(p), pd, pdd)


def make_method(name: str, scenario: Scenario, *, level_factor: float = 10.0, **kw: Any) -> Method:
    """Build the controller `name` for `scenario` (already `reset`); `kw` goes to the builder:
    `ours.build_bregman_id` or `baselines.build_tsid` / `build_tsid_cbf` / `build_clf_aecbf`.
    `level_factor` applies to `bregman_id` with `clf_mode="level"`."""
    task = scenario.task_obj
    dt = scenario.spec.dt
    if name == "bregman_id":
        ctrl = ours.build_bregman_id(scenario, task=task, dt=dt, **kw)
        if kw.get("clf_mode") == "level":
            return LevelMethod(name, ctrl, level_factor)
        return Method(name, ctrl)
    if name == "tsid":
        return Method(name, baselines.build_tsid(scenario, task=task, dt=dt, **kw))
    if name == "tsid_cbf":
        return Method(name, baselines.build_tsid_cbf(scenario, task=task, dt=dt, **kw))
    if name == "clf_aecbf":
        return Method(name, baselines.build_clf_aecbf(scenario, task=task, dt=dt, **kw))
    raise ValueError(f"unknown method {name!r}; known: {_NAMES}")
