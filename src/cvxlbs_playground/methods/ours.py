"""`bregman_id`: the proposed controller, an inverse-dynamics QP with an order-2 Bregman CLF.

The generating function is `psi = mu * barrier(keep-in set) + quad_weight * 1/2 |x|^2`, so one
Lyapunov function `V2 = D_psi(x_d, x) + 1/2 s^T Gamma s` both attracts the task to `x_d` and grows
without bound at the boundary of the keep-in set. `bs_metric` selects the backstepping metric
`Gamma`: `"hessian"` (`Hess psi(x)`, the proposed one), `"constant"` (`I`) or `"target"`
(`Hess psi(x_d)`).
"""

from __future__ import annotations

import warnings
from typing import Any, Literal

import casadi as ca
import numpy as np
from cvxlbs.certificates.base import Certificate, Row
from cvxlbs.certificates.bregman import BregmanCLF, Metric
from cvxlbs.certificates.viability import ViabilityCalibrationInfeasibleError, ViabilityJointLimits
from cvxlbs.controller import IDController
from cvxlbs.costs import CertificateNominal, Posture
from cvxlbs.psi import Psi, box_log_barrier, ellipsoid_barrier, neg_entropy, polytope_log_barrier, quadratic

from cvxlbs_playground.scenarios import KeepIn, Scenario

from .baselines import FALLBACK_KD, velocity_row

ClfMode = Literal["hard", "level"]

#: Headroom (fraction of each joint's range) used to calibrate the level of the joint-limit
#: certificate; small enough that every sampled start state lies inside its sublevel set.
VIABILITY_MARGIN = 0.005


def _box_entropy_barrier(lo: np.ndarray, hi: np.ndarray, dim: int, *, mu: float) -> Psi:
    """Entropic barrier of a box: `mu * sum_i [h(x_i - lo_i) + h(hi_i - x_i)]`, `h(z) = z log z - z`.
    It is Legendre on the box but not self-concordant."""
    p_lo = neg_entropy(dim, shift=-np.asarray(lo, dtype=float))
    p_hi_raw = neg_entropy(dim, shift=np.asarray(hi, dtype=float))
    p_hi = Psi(
        lambda x: p_hi_raw(-x),
        dim,
        name=f"reflected({p_hi_raw.name})",
        domain=lambda x: p_hi_raw.domain_margins(-x),
    )
    return mu * (p_lo + p_hi)


def _barrier_from_keep_in(keep_in: KeepIn, dim: int, *, mu: float) -> Psi:
    kind = keep_in["type"]
    if kind == "box":
        return box_log_barrier(keep_in["lo"], keep_in["hi"], mu=mu)
    if kind == "box_entropy":
        return _box_entropy_barrier(keep_in["lo"], keep_in["hi"], dim, mu=mu)
    if kind == "polytope":
        return polytope_log_barrier(keep_in["A"], keep_in["b"], mu=mu)
    if kind == "ellipsoid":
        return ellipsoid_barrier(keep_in["center"], keep_in["shape"], mu=mu)
    raise ValueError(f"_barrier_from_keep_in: unknown keep_in type {kind!r}")


class _LevelRow(Certificate):
    """Level constraint on a certificate's `V2`: the hard row `D(V2) + alpha V2 <= alpha c`, with
    the level `c = ctx.P("cert_level")` set per episode (`registry.LevelMethod`). It is inactive
    while `V2` is well below `c` and keeps `V2 <= max(V2(0), c)`, hence the task inside the domain
    of `psi`, without requiring `V2` to decrease."""

    def __init__(self, base: Any, alpha: float, name: str) -> None:
        self.base = base
        self.alpha = float(alpha)
        self.name = name

    def rows(self, ctx: Any) -> list[Row]:
        v2 = self.base.monitors(ctx)["V2"]
        expr = ctx.D(v2) + self.alpha * v2
        return [Row(expr=expr, ub=self.alpha * ctx.P("cert_level")[0], name=self.name)]

    def monitors(self, ctx: Any) -> dict[str, ca.SX]:
        return {"V2": self.base.monitors(ctx)["V2"]}


def build_bregman_id(
    scenario: Scenario,
    *,
    task,
    dt: float,
    k: float = 5.0,
    ks: float | None = None,
    mu: float = 1.0,
    quad_weight: float = 1e-2,
    clf_mode: ClfMode = "hard",
    soft_weight: float = 1e4,
    bs_metric: Metric = "hessian",
    posture_kp: float = 1.0,
    posture_kd: float = 1.0,
    posture_weight: float = 1e-3,
    nominal_weight: float = 1.0,
    monitor_demand: bool = False,
    jit: bool = False,
) -> IDController:
    """Inverse-dynamics QP over the joint acceleration with, in this order:

    - the Lyapunov row(s). `clf_mode="hard"`: the rate constraint `D(V2) + gamma V2 <= 0` as a hard
      row. `clf_mode="level"`: a hard level constraint (`_LevelRow`) plus the rate constraint
      relaxed by a slack of weight `soft_weight`;
    - a posture cost toward `q_home`;
    - braking-aware joint-position rows (`ViabilityJointLimits`) and the one-step joint-velocity
      row. If the torque bounds are too small to calibrate the position rows (e.g. with
      `tau_scale = 0.3`), only the velocity row is used;
    - a cost pulling the task acceleration to the certificate's own backstepping law.

    Gains: `k` (reference velocity), `ks` (velocity error, default `k`), `mu` (barrier weight),
    `quad_weight` (weight `P = quad_weight * I` of the quadratic term). `monitor_demand` logs the
    norm of the nominal task acceleration as `bregman_clf/a_nom`.
    """
    model = scenario.entry.model
    keep_in = scenario.sets()["keep_in"]
    if clf_mode not in ("hard", "level"):
        raise ValueError(f"build_bregman_id: clf_mode must be 'hard' or 'level', got {clf_mode!r}")
    if clf_mode == "level" and "cert_level" not in scenario.params.names:
        scenario.params.add("cert_level", 1, default=np.array([1e6]))

    psi = _barrier_from_keep_in(keep_in, task.dim, mu=mu) + quad_weight * quadratic(task.dim)

    def build_cert(suffix: str, rate_scale: float = 1.0) -> BregmanCLF:
        return BregmanCLF(
            task,
            psi,
            scenario.target,
            k=k,
            ks=ks,
            metric=bs_metric,
            monitor_demand=monitor_demand,
            rate_scale=rate_scale,
            name=f"bregman_clf{suffix}",
        )

    try:
        limits_cert = ViabilityJointLimits(model, margin=VIABILITY_MARGIN)
        viability = True
    except ViabilityCalibrationInfeasibleError as exc:
        warnings.warn(
            f"build_bregman_id: torque bounds too small for the braking-aware joint-position rows ({exc}); "
            "using the joint-velocity row only",
            stacklevel=2,
        )
        limits_cert = velocity_row(model, dt)
        viability = False

    ctrl = IDController(model, scenario.params, dt=dt, fallback_kd=FALLBACK_KD, fallback_metric="mass")
    clf = build_cert("")
    if clf_mode == "hard":
        ctrl.add(clf)
    else:
        alpha = min(k, 2.0 * (ks if ks is not None else k))
        ctrl.add(_LevelRow(build_cert("_level", rate_scale=0.0), alpha, name=f"{clf.name}_level"))
        ctrl.add(clf, soft_weight=soft_weight)
    ctrl.add(Posture("q_home", posture_kp, posture_kd, posture_weight))
    ctrl.add(limits_cert)
    if viability:
        ctrl.add(velocity_row(model, dt))
    ctrl.add(CertificateNominal(clf, weight=nominal_weight))
    return ctrl.build(jit=jit)
