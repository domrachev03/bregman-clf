"""Base types: a `Certificate` contributes inequality rows and cost terms to the QP."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass

import casadi as ca
import numpy as np
from numpy.typing import ArrayLike

from ..context import Context


@dataclass
class Row:
    """Inequality row `lb <= expr <= ub`; `expr` is `(m, 1)` and affine in `ctx.xi`."""

    expr: ca.SX
    lb: ArrayLike | ca.SX = -np.inf
    ub: ArrayLike | ca.SX = 0.0
    name: str = ""


@dataclass
class CostTerm:
    """Cost term `1/2 sum_j w_j r_j^2`; `residual` is `(m, 1)` and affine in `ctx.xi`."""

    residual: ca.SX
    weight: ArrayLike | ca.SX = 1.0
    name: str = ""


class Certificate(ABC):
    """Subclasses override whichever of `rows`, `costs`, `monitors`, `domain` apply."""

    name: str = ""

    def rows(self, ctx: Context) -> list[Row]:
        return []

    def costs(self, ctx: Context) -> list[CostTerm]:
        return []

    def monitors(self, ctx: Context) -> dict[str, ca.SX]:
        """Scalar state expressions to log at every step (e.g. the Lyapunov function)."""
        return {}

    def domain(self, ctx: Context) -> list[ca.SX]:
        """State expressions that must be positive (domain margins of psi)."""
        return []
