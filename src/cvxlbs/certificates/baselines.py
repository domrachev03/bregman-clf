"""Baseline certificates: per-constraint exponential CBF rows and a RES-CLF tracking row."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import casadi as ca
import numpy as np
import scipy.linalg

from ..context import Context
from ..tasks import Task
from .base import Certificate, Row

#: Stacked constraint functions `h_i(q) >= 0`, one entry per row.
HList = Callable[[Context], ca.SX]


def _col(v: float | Sequence[float] | np.ndarray, dim: int) -> np.ndarray:
    arr = np.asarray(v, dtype=float)
    if arr.ndim == 0:
        return np.full(dim, float(arr))
    arr = arr.reshape(-1)
    if arr.size != dim:
        raise ValueError(f"expected a scalar or {dim} entries, got {arr.size}")
    return arr


class PerConstraintECBF(Certificate):
    """One exponential CBF row per constraint `h_i(q) >= 0` (Nguyen and Sreenath, ACC 2016):

        h_ddot_i + k1 h_dot_i + k0 h_i >= 0,    k1 = 2 alpha, k0 = alpha^2,

    affine in the joint acceleration. `h_list(ctx)` returns the stacked constraint values.
    """

    def __init__(self, h_list: HList, *, alpha: float = 1.0, name: str | None = None) -> None:
        self.h_list = h_list
        self.alpha = float(alpha)
        lam = self.alpha
        self.k1 = float(2.0 * lam)
        self.k0 = float(lam * lam)
        self.name = name or "per_constraint_ecbf"

    def _h(self, ctx: Context) -> ca.SX:
        h = self.h_list(ctx)
        return ca.reshape(h, h.numel(), 1)

    def rows(self, ctx: Context) -> list[Row]:
        h = self._h(ctx)
        hdot = ctx.D(h)
        expr = -(ctx.D(hdot) + self.k1 * hdot + self.k0 * h)
        return [Row(expr=expr, ub=0.0, name=self.name)]

    def monitors(self, ctx: Context) -> dict[str, ca.SX]:
        h = self._h(ctx)
        return {"h_min": ca.mmin(h), "hdot_min": ca.mmin(ctx.D(h))}

    @classmethod
    def joint_limits(cls, model, **kw: object) -> PerConstraintECBF:
        """`h = [q - q_min; q_max - q]`."""
        q_min = ca.SX(np.asarray(model.q_min, dtype=float).reshape(-1, 1))
        q_max = ca.SX(np.asarray(model.q_max, dtype=float).reshape(-1, 1))

        def h_list(ctx: Context) -> ca.SX:
            return ca.vertcat(ctx.q - q_min, q_max - ctx.q)

        kw.setdefault("name", "ecbf_joint_limits")
        return cls(h_list, **kw)  # type: ignore[arg-type]

    @classmethod
    def workspace_box(
        cls, task: Task, lo: float | Sequence[float], hi: float | Sequence[float], **kw: object
    ) -> PerConstraintECBF:
        """`h = [x - lo; hi - x]` on `x = task.value(ctx)`: one row per face of the box."""
        lo_sx = ca.SX(_col(lo, task.dim).reshape(-1, 1))
        hi_sx = ca.SX(_col(hi, task.dim).reshape(-1, 1))

        def h_list(ctx: Context) -> ca.SX:
            x = task.value(ctx)
            return ca.vertcat(x - lo_sx, hi_sx - x)

        kw.setdefault("name", f"ecbf_workspace_box_{task.name}")
        return cls(h_list, **kw)  # type: ignore[arg-type]


class RESCLF(Certificate):
    """Rapidly exponentially stabilizing CLF (Ames et al., TAC 2014) for a relative-degree-two
    task: `eta = [e; e_dot]`, `e = x - x_d`, `V = eta^T P_eps eta` with
    `P_eps = diag(I/eps, I) P diag(I/eps, I)` and `P` the solution of the Riccati equation of the
    double integrator (`Q = I`, `R = I`). Row: `D(V) + (gamma / eps) V <= 0`,
    `gamma = lambda_min(Q) / lambda_max(P)`."""

    def __init__(self, task: Task, target: str, *, eps: float = 0.1) -> None:
        m = task.dim
        self.task = task
        self.target = target
        self.eps = float(eps)
        q_np = np.eye(2 * m)
        f_np = np.block([[np.zeros((m, m)), np.eye(m)], [np.zeros((m, m)), np.zeros((m, m))]])
        g_np = np.vstack([np.zeros((m, m)), np.eye(m)])
        p_np = scipy.linalg.solve_continuous_are(f_np, g_np, q_np, np.eye(m))
        eig_p = np.linalg.eigvalsh(p_np)
        eig_q = np.linalg.eigvalsh(q_np)
        self.gamma_rate = float(np.min(eig_q) / np.max(eig_p))
        scale = np.diag(np.concatenate([np.full(m, 1.0 / self.eps), np.ones(m)]))
        self.P_eps = ca.SX(scale @ p_np @ scale)
        # LQR gain K = G^T P = [K1, K2], rescaled by eps
        self.K1 = p_np[m:, :m] / self.eps**2
        self.K2 = p_np[m:, m:] / self.eps
        self.name = f"resclf_{task.name}_{target}"

    def _v(self, ctx: Context, x: ca.SX, x_d: ca.SX) -> ca.SX:
        e = x - x_d
        edot = ctx.D(e)
        eta = ca.vertcat(e, edot)
        return ca.mtimes([eta.T, self.P_eps, eta])

    def rows(self, ctx: Context) -> list[Row]:
        x = self.task.value(ctx)
        x_d = ctx.P(self.target)
        val = self._v(ctx, x, x_d)
        expr = ctx.D(val) + (self.gamma_rate / self.eps) * val
        return [Row(expr=expr, ub=0.0, name=self.name)]

    def monitors(self, ctx: Context) -> dict[str, ca.SX]:
        x = self.task.value(ctx)
        x_d = ctx.P(self.target)
        return {"V": self._v(ctx, x, x_d)}

    def nominal(self, ctx: Context) -> ca.SX:
        """The CLF's own stabilizing law `x_ddot_d - K1 e - K2 e_dot` (for `costs.CertificateNominal`)."""
        x = self.task.value(ctx)
        e = x - ctx.P(self.target)
        edot = ctx.D(x) - ctx.Pd(self.target)
        return ctx.Pdd(self.target) - ca.mtimes(ca.DM(self.K1), e) - ca.mtimes(ca.DM(self.K2), edot)
