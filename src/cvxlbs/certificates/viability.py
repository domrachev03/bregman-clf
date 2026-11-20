"""Braking-aware joint-position limits as one barrier row on a phase-space Bregman level function.

Per joint `j`, with a deceleration bound `a_max_j`, the margins

    g_up_j(q, v) = q_max_j - q_j - f(v_j)  / (2 a_max_j),
    g_lo_j(q, v) = q_j - q_min_j - f(-v_j) / (2 a_max_j)

are positive iff the joint can still brake to a stop before the limit under `|accel| <= a_max_j`
(Del Prete, "Joint position and velocity bounds in discrete-time acceleration/torque control of
robot manipulators", RA-L 2018). `f(x) = softplus_kappa(x)^2 >= max(x, 0)^2` is a smooth,
conservative replacement of the exact braking term. The generating function

    psi(q, v) = -mu sum_j [log g_up_j + log g_lo_j]

is strictly convex on its domain, and `V(q, v) = D_psi((q_c, 0), (q, v))`, `q_c` the middle of the
joint range, grows without bound at the boundary. `ViabilityJointLimits` keeps `V <= vbar` with the
row `D(h) + alpha h >= 0`, `h = vbar - V`, which is affine in the joint acceleration.
"""

from __future__ import annotations

import casadi as ca
import numpy as np

from ..context import Context
from ..psi import Psi, bregman
from .base import Certificate, Row

#: Seed of the sampling-based calibration: it is a fixed function of the model, not of the episode.
_CALIBRATION_SEED = 0x63767863


class ViabilityCalibrationInfeasibleError(ValueError):
    """The torque limits do not exceed the sampled worst-case bias torque on some joint, so no
    positive deceleration bound `a_max` exists."""


def softplus(x: ca.SX, kappa: float) -> ca.SX:
    """`(1/kappa) log(1 + exp(kappa x)) >= max(x, 0)`, in an overflow-free form."""
    z = kappa * x
    return (ca.fmax(z, 0) + ca.log1p(ca.exp(-ca.fabs(z)))) / kappa


def _margins(x: ca.SX, n: int, q_min: ca.SX, q_max: ca.SX, a_max: ca.SX, kappa: float) -> tuple[ca.SX, ca.SX]:
    q = x[:n]
    v = x[n:]
    g_up = q_max - q - softplus(v, kappa) ** 2 / (2 * a_max)
    g_lo = q - q_min - softplus(-v, kappa) ** 2 / (2 * a_max)
    return g_up, g_lo


def viability_psi(q_min, q_max, a_max, mu: float = 1.0, kappa: float = 50.0) -> Psi:
    """`psi(q, v) = -mu sum_j [log g_up_j + log g_lo_j]` on `x = (q, v)`."""
    n = int(np.asarray(q_min, dtype=float).reshape(-1).shape[0])
    q_min_sx = ca.SX(np.asarray(q_min, dtype=float).reshape(n, 1))
    q_max_sx = ca.SX(np.asarray(q_max, dtype=float).reshape(n, 1))
    a_max_sx = ca.SX(np.asarray(a_max, dtype=float).reshape(n, 1))
    kappa_f = float(kappa)

    def fn(x: ca.SX) -> ca.SX:
        g_up, g_lo = _margins(x, n, q_min_sx, q_max_sx, a_max_sx, kappa_f)
        return -mu * (ca.sum1(ca.log(g_up)) + ca.sum1(ca.log(g_lo)))

    def domain(x: ca.SX) -> list[ca.SX]:
        g_up, g_lo = _margins(x, n, q_min_sx, q_max_sx, a_max_sx, kappa_f)
        return [g_up[i] for i in range(n)] + [g_lo[i] for i in range(n)]

    return Psi(fn, 2 * n, name="viability", domain=domain)


def compute_a_max(
    model,
    *,
    n_samples: int = 2000,
    rng: np.random.Generator,
    bias_safety_factor: float = 1.25,
    v_sample_margin: float = 1.3,
) -> np.ndarray:
    """`a_max_j = (tau_max_j - bias_j) / I_j`, with `I_j = max_q M_jj(q)` and
    `bias_j = bias_safety_factor * max |h_j(q, v)|` estimated from `n_samples` uniform samples of
    `q` in the joint range and `|v| <= v_sample_margin * v_max`."""
    n = model.nv
    q_lo = np.where(np.isfinite(model.q_min), model.q_min, -np.pi)
    q_hi = np.where(np.isfinite(model.q_max), model.q_max, np.pi)
    v_hi = np.where(np.isfinite(model.v_max), model.v_max, 1.0) * float(v_sample_margin)

    i_eff = np.zeros(n)
    bias_abs = np.zeros(n)
    for _ in range(n_samples):
        q = rng.uniform(q_lo, q_hi)
        v = rng.uniform(-v_hi, v_hi)
        mass = model.numeric("mass", q)
        bias = model.numeric("bias", q, v)
        i_eff = np.maximum(i_eff, np.diag(mass))
        bias_abs = np.maximum(bias_abs, np.abs(bias))
    i_eff = np.maximum(i_eff, 1e-9)
    bias_abs = bias_safety_factor * bias_abs

    tau_max = np.asarray(model.tau_max, dtype=float)
    if not np.all(np.isfinite(tau_max)):
        raise ValueError("compute_a_max: model.tau_max must be finite")
    a_max = (tau_max - bias_abs) / i_eff
    if np.any(a_max <= 0):
        bad = np.nonzero(a_max <= 0)[0].tolist()
        raise ViabilityCalibrationInfeasibleError(
            f"joints {bad}: tau_max {tau_max[bad].tolist()} <= worst-case bias torque {bias_abs[bad].tolist()}"
        )
    return a_max


def calibrate_vbar(
    psi: Psi,
    q_c: np.ndarray,
    q_min: np.ndarray,
    q_max: np.ndarray,
    v_max: np.ndarray,
    *,
    n_samples: int = 20_000,
    margin_frac: float = 0.03,
    vbar_margin: float = 1.5,
    rng: np.random.Generator,
    max_draws_factor: int = 50,
) -> float:
    """`vbar = vbar_margin * max V(q, v)` over `n_samples` states sampled uniformly in
    `[q_min, q_max] x [-v_max, v_max]` whose margins `g_up`, `g_lo` are all at least
    `margin_frac * (q_max - q_min)`. (`V` is unbounded at the boundary, so the headroom is needed.)"""
    q_c = np.asarray(q_c, dtype=float).reshape(-1)
    n = q_c.shape[0]
    q_min = np.asarray(q_min, dtype=float).reshape(n)
    q_max = np.asarray(q_max, dtype=float).reshape(n)
    v_bound = np.asarray(v_max, dtype=float).reshape(n)
    v_bound = np.where(np.isfinite(v_bound), v_bound, np.maximum(q_max - q_min, 1.0))
    g_floor = np.concatenate([margin_frac * (q_max - q_min)] * 2)

    x = ca.SX.sym("x", 2 * n)
    y = ca.vertcat(ca.SX(q_c.reshape(n, 1)), ca.SX.zeros(n, 1))
    margins_fn = ca.Function("viability_margins", [x], [ca.vertcat(*psi.domain_margins(x))])
    v_fn = ca.Function("viability_V", [x], [bregman(psi, y, x)])

    v_worst = 0.0
    n_found = 0
    draws = 0
    max_draws = n_samples * max_draws_factor
    while n_found < n_samples and draws < max_draws:
        draws += 1
        q = rng.uniform(q_min, q_max)
        v = rng.uniform(-v_bound, v_bound)
        state = np.concatenate([q, v])
        margins = np.asarray(margins_fn(state)).ravel()
        if np.all(margins >= g_floor):
            n_found += 1
            v_worst = max(v_worst, float(v_fn(state)))
    if n_found == 0:
        raise RuntimeError(f"calibrate_vbar: no viable state with headroom {margin_frac} found in {max_draws} draws")
    return vbar_margin * v_worst


class ViabilityJointLimits(Certificate):
    """Barrier row keeping `V(q, v) = D_psi((q_c, 0), (q, v)) <= vbar` (see the module docstring).

    The deceleration bound `a_max` is computed by `compute_a_max` and the level `vbar` by
    `calibrate_vbar` with headroom `margin`; both use a fixed random seed. Raises
    `ViabilityCalibrationInfeasibleError` if the torque limits are too small.
    """

    def __init__(
        self,
        model,
        *,
        mu: float = 1.0,
        kappa: float = 50.0,
        alpha: float = 20.0,
        margin: float = 0.03,
        name: str = "viability_joint_limits",
    ) -> None:
        q_min = np.asarray(model.q_min, dtype=float)
        q_max = np.asarray(model.q_max, dtype=float)
        if not np.all(np.isfinite(q_min)) or not np.all(np.isfinite(q_max)):
            raise ValueError("ViabilityJointLimits: model.q_min/q_max must be finite for every joint")
        rng = np.random.default_rng(_CALIBRATION_SEED)
        a_max = compute_a_max(model, rng=rng)
        self.psi = viability_psi(q_min, q_max, a_max, mu=mu, kappa=kappa)
        q_c = 0.5 * (q_min + q_max)
        self.center = np.concatenate([q_c, np.zeros(model.nv)])
        self.vbar = calibrate_vbar(self.psi, q_c, q_min, q_max, model.v_max, margin_frac=margin, rng=rng)
        self.alpha = float(alpha)
        self.name = name

    def _level(self, ctx: Context) -> ca.SX:
        x = ca.vertcat(ctx.q, ctx.v)
        c = ca.SX(np.asarray(self.center, dtype=float).reshape(-1, 1))
        return bregman(self.psi, c, x)

    def _h(self, ctx: Context) -> ca.SX:
        return ca.SX(float(self.vbar)) - self._level(ctx)

    def rows(self, ctx: Context) -> list[Row]:
        h = self._h(ctx)
        expr = -(ctx.D(h) + self.alpha * h)
        return [Row(expr=expr, ub=0.0, name=self.name)]

    def monitors(self, ctx: Context) -> dict[str, ca.SX]:
        return {"V": self._level(ctx), "h": self._h(ctx)}
