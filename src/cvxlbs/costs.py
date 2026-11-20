"""Cost terms of the QP: regularization, task-space and posture PD tracking, and tracking of a
certificate's own nominal law."""

from __future__ import annotations

from typing import Protocol

import casadi as ca
import numpy as np

from .certificates.base import Certificate, CostTerm
from .context import Context
from .tasks import Task

Gain = float | np.ndarray | ca.SX


class _NominalCertificate(Protocol):
    task: Task
    name: str

    def nominal(self, ctx: Context) -> ca.SX: ...


def _tracking_residual(ctx: Context, x: ca.SX, target: str, kp: Gain, kd: Gain) -> ca.SX:
    """`x_ddot - (x_ddot_d + kd (x_dot_d - x_dot) + kp (x_d - x))`, affine in `vdot`."""
    x_d = ctx.P(target)
    xdot_d = ctx.Pd(target)
    xdot = ctx.D(x)
    xddot_d = ctx.Pdd(target)
    return ctx.D(xdot) - (xddot_d + kd * (xdot_d - xdot) + kp * (x_d - x))


class Regularization(Certificate):
    """`weight/2 |vdot|^2`."""

    def __init__(self, weight: Gain):
        self.weight = weight
        self.name = "regularization_vdot"

    def costs(self, ctx: Context) -> list[CostTerm]:
        return [CostTerm(residual=ctx.vdot, weight=self.weight, name=self.name)]


class TaskTracking(Certificate):
    """PD tracking cost on a task: `weight/2 |x_ddot - (x_ddot_d + kd e_dot + kp e)|^2`."""

    def __init__(self, task: Task, target: str, kp: Gain, kd: Gain, *, weight: Gain):
        self.task = task
        self.target = target
        self.kp = kp
        self.kd = kd
        self.weight = weight
        self.name = f"tracking_{task.name}_{target}"

    def costs(self, ctx: Context) -> list[CostTerm]:
        residual = _tracking_residual(ctx, self.task.value(ctx), self.target, self.kp, self.kd)
        return [CostTerm(residual=residual, weight=self.weight, name=self.name)]


class CertificateNominal(Certificate):
    """`weight/2 |x_ddot - a_nom|^2`, with `a_nom = clf.nominal(ctx)` the task acceleration of the
    certificate's own stabilizing law. A single scalar CLF row constrains one direction of the
    task acceleration; this cost sets the others without changing the feasible set."""

    def __init__(self, clf: _NominalCertificate, weight: Gain):
        self.clf = clf
        self.weight = weight
        self.name = f"certificate_nominal_{clf.name}"

    def costs(self, ctx: Context) -> list[CostTerm]:
        x = self.clf.task.value(ctx)
        nominal = self.clf.nominal(ctx)
        residual = ctx.D(ctx.D(x)) - nominal
        return [CostTerm(residual=residual, weight=self.weight, name=self.name)]


class Posture(Certificate):
    """PD posture cost toward the joint configuration held by parameter `q_nom`; it fixes the
    null-space acceleration of a redundant arm."""

    def __init__(self, q_nom: str, kp: Gain, kd: Gain, weight: Gain):
        self.q_nom = q_nom
        self.kp = kp
        self.kd = kd
        self.weight = weight
        self.name = f"posture_{q_nom}"

    def costs(self, ctx: Context) -> list[CostTerm]:
        residual = _tracking_residual(ctx, ctx.q, self.q_nom, self.kp, self.kd)
        return [CostTerm(residual=residual, weight=self.weight, name=self.name)]
