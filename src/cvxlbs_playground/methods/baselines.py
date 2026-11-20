"""Baseline inverse-dynamics QP controllers with one barrier row per constraint.

- `tsid`: task-space PD tracking with naive joint position and velocity bounds, no keep-in rows;
- `tsid_cbf`: `tsid` plus one exponential CBF row per face of the keep-in set;
- `clf_aecbf`: relaxed rapidly exponentially stabilizing CLF plus one exponential CBF row per joint
  limit and per face of the keep-in set (a CLF-CBF QP).

All use the same fallback law, joint-velocity row and posture cost structure as our controller.
"""

from __future__ import annotations

import math
from typing import Any

import casadi as ca
import numpy as np
from cvxlbs.certificates.baselines import RESCLF, HList, PerConstraintECBF
from cvxlbs.certificates.limits import JointAccelerationLimits, JointPositionLimitsID
from cvxlbs.controller import IDController
from cvxlbs.costs import CertificateNominal, Posture, TaskTracking
from cvxlbs.tasks import Task

from cvxlbs_playground.scenarios import Scenario

#: Damping of the fallback law `tau = g(q) - FALLBACK_KD M(q) v` applied on infeasible steps, the
#: same for every method.
FALLBACK_KD = 15.0

#: Rate [rad/s^2] at which the joint-velocity row asks a speed that already exceeds its limit to
#: decrease (instead of an instant return, which would make the QP infeasible).
VELOCITY_RECOVERY = 20.0

#: Default gains of the task-space PD baselines.
DEFAULT_GAINS: dict[str, float] = {
    "kp": 100.0,
    "kd": 20.0,
    "weight": 10.0,
    "posture_kp": 4.0,
    "posture_kd": 2.0,
    "posture_weight": 1e-3,
}


def velocity_row(model, dt: float):
    """The one-step joint-velocity row `|v + dt vdot| <= v_max` every method uses."""
    return JointAccelerationLimits(model, dt, recovery=VELOCITY_RECOVERY)


def _resolve_gains(gains: dict[str, float] | None) -> dict[str, float]:
    """`DEFAULT_GAINS` updated with `gains`; a `kp` given without `kd` gets `kd = 2 sqrt(kp)`."""
    g = dict(DEFAULT_GAINS)
    if gains:
        if "kp" in gains and "kd" not in gains:
            g["kd"] = 2.0 * math.sqrt(float(gains["kp"]))
        g.update(gains)
    return g


def _tsid_controller(scenario: Scenario, *, task: Task, dt: float, gains: dict[str, float] | None) -> IDController:
    model = scenario.entry.model
    g = _resolve_gains(gains)
    ctrl = IDController(model, scenario.params, dt=dt, fallback_kd=FALLBACK_KD, fallback_metric="mass")
    ctrl.add(TaskTracking(task, scenario.target, g["kp"], g["kd"], weight=g["weight"]))
    ctrl.add(Posture("q_home", g["posture_kp"], g["posture_kd"], g["posture_weight"]))
    ctrl.add(velocity_row(model, dt))
    ctrl.add(JointPositionLimitsID(model, dt))
    return ctrl


def _polytope_clearance_h(task: Task, A: np.ndarray, b: np.ndarray) -> HList:
    """`h = b - A x` (one entry per face; distances for unit-norm rows) on `x = task.value(ctx)`."""
    a_sx = ca.SX(A)
    b_sx = ca.SX(b.reshape(-1, 1))

    def h_list(ctx: Any) -> ca.SX:
        return b_sx - ca.mtimes(a_sx, task.value(ctx))

    return h_list


def _ellipsoid_clearance_h(task: Task, center: np.ndarray, shape: np.ndarray) -> HList:
    """`h(x) = 1 - (x-c)^T S^-1 (x-c)` on `x = task.value(ctx)`: a single row."""
    c_sx = ca.SX(np.asarray(center, dtype=float).reshape(-1, 1))
    s_sx = ca.SX(np.asarray(shape, dtype=float))

    def h_list(ctx: Any) -> ca.SX:
        x = task.value(ctx)
        e = x - c_sx
        return 1.0 - ca.mtimes([e.T, ca.solve(s_sx, e)])

    return h_list


def _keep_in_ecbf(scenario: Scenario, task: Task, alpha: float) -> PerConstraintECBF:
    """Order-2 exponential CBF rows keeping `task` in the scenario's keep-in set: one per face of a
    box or polytope, one for an ellipsoid."""
    keep_in = scenario.sets()["keep_in"]
    kind = keep_in["type"]
    if kind == "box":
        lo, hi = np.asarray(keep_in["lo"], dtype=float), np.asarray(keep_in["hi"], dtype=float)
        return PerConstraintECBF.workspace_box(task, lo, hi, alpha=alpha)
    if kind == "polytope":
        A = np.asarray(keep_in["A"], dtype=float)
        b = np.asarray(keep_in["b"], dtype=float)
        norms = np.linalg.norm(A, axis=1)
        h = _polytope_clearance_h(task, A / norms[:, None], b / norms)
        return PerConstraintECBF(h, alpha=alpha, name=f"ecbf_workspace_polytope_{task.name}")
    if kind == "ellipsoid":
        h = _ellipsoid_clearance_h(
            task, np.asarray(keep_in["center"], dtype=float), np.asarray(keep_in["shape"], dtype=float)
        )
        return PerConstraintECBF(h, alpha=alpha, name=f"ecbf_workspace_ellipsoid_{task.name}")
    raise ValueError(f"per-face CBF baselines: unsupported keep_in type {kind!r}")


def build_tsid(
    scenario: Scenario, *, task: Task, dt: float, gains: dict[str, float] | None = None, jit: bool = False
) -> IDController:
    """Inverse dynamics with a task-space PD cost, a posture cost, the joint-velocity row and the
    naive joint-position row `q_min <= q + dt v + dt^2/2 vdot <= q_max`."""
    return _tsid_controller(scenario, task=task, dt=dt, gains=gains).build(jit=jit)


def build_tsid_cbf(
    scenario: Scenario,
    *,
    task: Task,
    dt: float,
    gains: dict[str, float] | None = None,
    alpha: float = 5.0,
    jit: bool = False,
) -> IDController:
    """`tsid` plus hard exponential CBF rows for the keep-in set."""
    ctrl = _tsid_controller(scenario, task=task, dt=dt, gains=gains)
    ctrl.add(_keep_in_ecbf(scenario, task, alpha))
    return ctrl.build(jit=jit)


def build_clf_aecbf(
    scenario: Scenario,
    *,
    task: Task,
    dt: float,
    eps: float = 0.5,
    alpha: float = 5.0,
    clf_slack_weight: float = 1e4,
    nominal_weight: float = 1.0,
    posture_kp: float = 1.0,
    posture_kd: float = 1.0,
    posture_weight: float = 1e-3,
    jit: bool = False,
) -> IDController:
    """CLF-CBF QP: an order-2 RES-CLF row for tracking (rate `gamma / eps`), relaxed with a slack
    of weight `clf_slack_weight` so that the barrier rows stay feasible, with the CLF's own
    stabilizing law as a nominal cost; one exponential CBF row (rate `alpha`) per joint limit and
    per face of the keep-in set."""
    model = scenario.entry.model
    clf = RESCLF(task, scenario.target, eps=eps)
    ctrl = IDController(model, scenario.params, dt=dt, fallback_kd=FALLBACK_KD, fallback_metric="mass")
    ctrl.add(clf, soft_weight=clf_slack_weight)
    ctrl.add(CertificateNominal(clf, weight=nominal_weight))
    ctrl.add(velocity_row(model, dt))
    ctrl.add(PerConstraintECBF.joint_limits(model, alpha=alpha))
    ctrl.add(_keep_in_ecbf(scenario, task, alpha))
    ctrl.add(Posture("q_home", posture_kp, posture_kd, posture_weight))
    return ctrl.build(jit=jit)
