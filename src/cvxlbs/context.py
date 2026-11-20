"""Symbolic context: state, parameter and decision symbols, and the total time derivative `D`.

The decision vector is the joint acceleration, `xi = vdot`. Along the closed loop `q` advances at
`v`, `v` at `vdot`, and the parameters at their rates `pd`, `pdd`. `D(expr)` is the exact
derivative of a state expression along that flow, hence affine in `xi`.
"""

from __future__ import annotations

from typing import ClassVar

import casadi as ca
import numpy as np

from .params import Params


class Context:
    input_names: ClassVar[list[str]] = ["q", "v", "t", "p", "pd", "pdd"]

    def __init__(self, model, params: Params) -> None:
        self.model = model
        self.params = params
        self.q = ca.SX.sym("q", model.nq)
        self.v = ca.SX.sym("v", model.nv)
        self.t = ca.SX.sym("t", 1)
        self.p = ca.SX.sym("p", params.size)
        self.pd = ca.SX.sym("pd", params.size)
        self.pdd = ca.SX.sym("pdd", params.size)
        self.xi = ca.SX.sym("xi", model.nv)

        # 0/1 masks: static parameters do not move, order-1 parameters have no second derivative
        mask1 = np.zeros((params.size, 1))
        mask2 = np.zeros((params.size, 1))
        for name in params.names:
            sl = params.slice(name)
            order = params.order(name)
            if order >= 1:
                mask1[sl, 0] = 1.0
            if order >= 2:
                mask2[sl, 0] = 1.0
        self._pd_mask = mask1
        self._pdd_mask = mask2

    def P(self, name: str) -> ca.SX:
        """Value of parameter `name`."""
        return self.p[self.params.slice(name)]

    def Pd(self, name: str) -> ca.SX:
        """First time derivative of parameter `name`."""
        return self.pd[self.params.slice(name)]

    def Pdd(self, name: str) -> ca.SX:
        """Second time derivative of parameter `name`."""
        return self.pdd[self.params.slice(name)]

    @property
    def vdot(self) -> ca.SX:
        """Joint acceleration (the decision vector)."""
        return self.xi

    @property
    def tau(self) -> ca.SX:
        """Joint torque `B(q)^-1 (M(q) vdot + h(q, v))`, affine in `vdot` (fully actuated model)."""
        mass = self.model.mass(self.q)
        bias = self.model.bias(self.q, self.v)
        actuation = self.model.actuation(self.q)
        rhs = ca.mtimes(mass, self.vdot) + bias
        return ca.solve(actuation, rhs)

    def inputs(self) -> list[ca.SX]:
        """`[q, v, t, p, pd, pdd]`, the input order of every compiled function."""
        return [self.q, self.v, self.t, self.p, self.pd, self.pdd]

    def D(self, expr: ca.SX) -> ca.SX:
        """Total time derivative of a state expression `expr(q, v, t, p, pd)` along the closed
        loop, by forward-mode differentiation with tangent `(v, vdot, 1, pd, pdd)`."""
        if ca.depends_on(expr, self.xi):
            raise ValueError("Context.D: expr must be a state expression (it depends on the decision vector xi)")
        pd_tan = self.pd * self._pd_mask
        pdd_tan = self.pdd * self._pdd_mask
        z = ca.vertcat(self.q, self.v, self.t, self.p, self.pd)
        zdot = ca.vertcat(self.v, self.vdot, ca.SX(1), pd_tan, pdd_tan)
        return ca.jtimes(expr, z, zdot)
