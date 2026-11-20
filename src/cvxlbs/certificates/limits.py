"""One-step joint velocity and position bounds for the inverse-dynamics QP."""

from __future__ import annotations

import casadi as ca
import numpy as np

from ..context import Context
from .base import Certificate, Row


class JointAccelerationLimits(Certificate):
    """Joint-velocity row `|v + dt vdot| <= v_max`. With `recovery` [rad/s^2], a speed that already
    exceeds its limit is only required to decrease at that rate, so the row stays feasible."""

    def __init__(self, model, dt: float, recovery: float | None = None, name: str | None = None) -> None:
        self.model = model
        self.dt = float(dt)
        self.recovery = None if recovery is None else float(recovery)
        self.name = name or "joint_acceleration_limits"

    def rows(self, ctx: Context) -> list[Row]:
        expr = ctx.v + self.dt * ctx.vdot
        bound = np.asarray(self.model.v_max, dtype=float)
        if self.recovery is None:
            return [Row(expr=expr, lb=-bound, ub=bound, name=self.name)]
        b = ca.DM(bound)
        step = self.dt * self.recovery
        ub = ca.fmax(b, ctx.v - step)
        lb = ca.fmin(-b, ctx.v + step)
        return [Row(expr=expr, lb=lb, ub=ub, name=self.name)]


class JointPositionLimitsID(Certificate):
    """Naive joint-position row `q_min <= q + dt v + dt^2/2 vdot <= q_max`."""

    def __init__(self, model, dt: float, name: str | None = None) -> None:
        self.model = model
        self.h = float(dt)
        self.name = name or "joint_position_limits_id"

    def rows(self, ctx: Context) -> list[Row]:
        expr = ctx.q + self.h * ctx.v + 0.5 * self.h**2 * ctx.vdot
        lb = np.asarray(self.model.q_min, dtype=float)
        ub = np.asarray(self.model.q_max, dtype=float)
        return [Row(expr=expr, lb=lb, ub=ub, name=self.name)]
