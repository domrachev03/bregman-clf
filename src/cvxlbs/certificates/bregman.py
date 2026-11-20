"""Bregman control Lyapunov function with backstepping (relative degree two)."""

from __future__ import annotations

from typing import Literal

import casadi as ca

from ..context import Context
from ..psi import Psi, bregman
from ..tasks import Task
from .base import Certificate, Row

Metric = Literal["constant", "hessian", "target"]
_METRICS: tuple[Metric, ...] = ("constant", "hessian", "target")


class BregmanCLF(Certificate):
    """Rate constraint `D(V2) + gamma V2 <= 0` of the backstepping Lyapunov function

        V2 = D_psi(x_d, x) + 1/2 s^T Gamma s,    s = x_dot - x_dot_r,
        x_dot_r = H(x)^-1 (-k (grad psi(x) - grad psi(x_d)) + H(x_d) x_dot_d),

    with `x = task.value(ctx)`, `x_d = ctx.P(target)`, `H = Hess psi` and `gamma = min(k, 2 ks)`.
    The row is affine in the joint acceleration.

    `metric` selects the backstepping metric `Gamma`:

    - `"hessian"`: `Gamma = H(x)`. With the law `nominal`, `D(V2) = -k (D_psi(x_d, x) +
      D_psi(x, x_d)) - ks s^T H s` exactly, and for a self-concordant barrier psi the demanded
      acceleration stays bounded on sublevel sets of `V2` up to the boundary of dom psi;
    - `"constant"`: `Gamma = I`;
    - `"target"`: `Gamma = H(x_d)`.

    For the constant metrics the cross term `grad_x D_psi(x_d, x) . s = (x - x_d)^T H(x) s` must be
    cancelled by an acceleration `Gamma^-1 H(x) (x - x_d)`, which is unbounded near the boundary.

    `rate_scale` multiplies `gamma` in the row (0 gives the monotonicity row `D(V2) <= 0`).
    `monitor_demand` adds the monitor `a_nom = |nominal|`.
    """

    def __init__(
        self,
        task: Task,
        psi: Psi,
        target: str,
        *,
        k: float = 1.0,
        ks: float | None = None,
        metric: Metric = "hessian",
        monitor_demand: bool = False,
        rate_scale: float = 1.0,
        name: str | None = None,
    ) -> None:
        if metric not in _METRICS:
            raise ValueError(f"BregmanCLF: metric must be one of {_METRICS}, got {metric!r}")
        if psi.dim != task.dim:
            raise ValueError(f"BregmanCLF: psi.dim={psi.dim} != task.dim={task.dim}")
        self.task = task
        self.psi = psi
        self.target = target
        self.k = float(k)
        self.ks = float(k if ks is None else ks)
        self.metric = metric
        self.monitor_demand = monitor_demand
        self.rate_scale = float(rate_scale)
        self.name = name or f"bregman_clf_{task.name}_{target}"

    def _xdot_r(self, ctx: Context, x: ca.SX, x_d: ca.SX) -> ca.SX:
        """Reference velocity: the mirror-descent flow toward `x_d` plus the target feedforward."""
        H = self.psi.hess(x)
        rhs = -self.k * (self.psi.grad(x) - self.psi.grad(x_d))
        rhs = rhs + ca.mtimes(self.psi.hess(x_d), ctx.Pd(self.target))
        return ca.solve(H, rhs)

    def _gamma(self, x: ca.SX, x_d: ca.SX) -> ca.SX:
        if self.metric == "constant":
            return ca.SX.eye(self.task.dim)
        if self.metric == "target":
            return self.psi.hess(x_d)
        return self.psi.hess(x)

    def _v2(self, ctx: Context) -> tuple[ca.SX, ca.SX, ca.SX]:
        """`(W, V2, s)` with `W = D_psi(x_d, x)`."""
        x = self.task.value(ctx)
        x_d = ctx.P(self.target)
        W = bregman(self.psi, x_d, x)
        xdot_r = self._xdot_r(ctx, x, x_d)
        Gamma = self._gamma(x, x_d)
        s = ctx.D(x) - xdot_r
        V2 = W + 0.5 * ca.mtimes([s.T, Gamma, s])
        return W, V2, s

    def rows(self, ctx: Context) -> list[Row]:
        c = self.rate_scale * min(self.k, 2.0 * self.ks)
        _, V2, _ = self._v2(ctx)
        expr = ctx.D(V2) + c * V2
        return [Row(expr=expr, ub=0.0, name=self.name)]

    def nominal(self, ctx: Context) -> ca.SX:
        """Task acceleration of the backstepping law, a state expression:

            x_ddot = D(x_dot_r) - Gamma^-1 grad_x D_psi(x_d, x) - ks s - 1/2 Gamma^-1 Gamma_dot s.

        It satisfies the rate constraint; for `metric="hessian"` the middle terms reduce to
        `-(x - x_d) - ks s - 1/2 H^-1 H_dot s`.
        """
        x = self.task.value(ctx)
        x_d = ctx.P(self.target)
        xdot_r = self._xdot_r(ctx, x, x_d)
        Gamma = self._gamma(x, x_d)
        s = ctx.D(x) - xdot_r
        # differentiate with respect to a placeholder: `x` is an expression of q, not a symbol
        xs = ca.SX.sym("_bregman_nominal_x", self.task.dim)
        grad_x_w = ca.substitute(ca.gradient(bregman(self.psi, x_d, xs), xs), xs, x)
        gamma_dot = ctx.D(Gamma)
        sdot_star = -ca.solve(Gamma, grad_x_w) - self.ks * s - 0.5 * ca.solve(Gamma, ca.mtimes(gamma_dot, s))
        return ctx.D(xdot_r) + sdot_star

    def monitors(self, ctx: Context) -> dict[str, ca.SX]:
        W, V2, s = self._v2(ctx)
        out = {"W": W, "V2": V2, "s_norm": ca.sqrt(ca.dot(s, s) + 1e-30)}
        if self.monitor_demand:
            a_nom = self.nominal(ctx)
            out["a_nom"] = ca.sqrt(ca.dot(a_nom, a_nom) + 1e-30)
        return out

    def domain(self, ctx: Context) -> list[ca.SX]:
        x = self.task.value(ctx)
        x_d = ctx.P(self.target)
        return self.psi.domain_margins(x) + self.psi.domain_margins(x_d)
