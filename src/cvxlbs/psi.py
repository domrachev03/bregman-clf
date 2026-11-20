"""Generating functions psi and the Bregman divergence.

`Psi` wraps a convex function of Legendre type as a CasADi expression builder. The library below
provides a quadratic, logarithmic barriers of a box, a polytope and an ellipsoid, and the negative
entropy; numeric arguments may also be CasADi SX expressions.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import casadi as ca
import numpy as np

Scalar = float | int | ca.SX
DomainFn = Callable[[ca.SX], Sequence[ca.SX]]


class Psi:
    """`psi(x)` as a CasADi expression builder, with gradient, Hessian and domain margins.

    `fn(x)` returns the scalar `psi(x)`; `domain(x)` returns expressions that are positive exactly
    in the interior of dom psi (none if the domain is the whole space). Sums and positive multiples
    of generating functions are generating functions (`+`, `*`).
    """

    def __init__(self, fn: Callable[[ca.SX], ca.SX], dim: int, *, name: str, domain: DomainFn | None = None) -> None:
        self.fn = fn
        self.dim = int(dim)
        self.name = name
        self._domain = domain

    def __call__(self, x: ca.SX) -> ca.SX:
        return self.fn(x)

    def grad(self, x: ca.SX) -> ca.SX:
        """`grad psi(x)`, shape `(dim, 1)`; `x` may be any expression."""
        xs = ca.SX.sym("_psi_x", self.dim)
        return ca.substitute(ca.gradient(self(xs), xs), xs, ca.SX(x))

    def hess(self, x: ca.SX) -> ca.SX:
        """`Hess psi(x)`, shape `(dim, dim)`; `x` may be any expression."""
        xs = ca.SX.sym("_psi_x", self.dim)
        h, _ = ca.hessian(self(xs), xs)
        return ca.substitute(h, xs, ca.SX(x))

    def domain_margins(self, x: ca.SX) -> list[ca.SX]:
        if self._domain is None:
            return []
        return list(self._domain(x))

    def __add__(self, other: Psi) -> Psi:
        if not isinstance(other, Psi):
            return NotImplemented
        if self.dim != other.dim:
            raise ValueError(f"Psi.__add__: dim mismatch ({self.name}.dim={self.dim} != {other.name}.dim={other.dim})")
        a, b = self, other

        def fn(x: ca.SX) -> ca.SX:
            return a(x) + b(x)

        domain: DomainFn | None
        if a._domain is None and b._domain is None:
            domain = None
        else:
            domain = lambda x: a.domain_margins(x) + b.domain_margins(x)

        return Psi(fn, self.dim, name=f"({a.name}+{b.name})", domain=domain)

    def __mul__(self, c: Scalar) -> Psi:
        if isinstance(c, ca.SX):
            pass  # symbolic weight: the caller is responsible for positivity
        elif isinstance(c, (int, float)):
            if not c > 0:
                raise ValueError(f"Psi.__mul__: scalar multiplier must be positive, got {c}")
        else:
            return NotImplemented
        base = self

        def fn(x: ca.SX) -> ca.SX:
            return c * base(x)

        return Psi(fn, self.dim, name=f"({c}*{base.name})", domain=base._domain)

    __rmul__ = __mul__


def bregman(psi: Psi, y: ca.SX, x: ca.SX) -> ca.SX:
    """`D_psi(y, x) = psi(y) - psi(x) - grad psi(x)^T (y - x)`."""
    return psi(y) - psi(x) - ca.dot(psi.grad(x), y - x)


def _col(v, dim: int) -> ca.SX:
    """`v` as a `(dim, 1)` SX; a scalar is broadcast."""
    if isinstance(v, ca.SX):
        m = v
    elif np.isscalar(v):
        return ca.SX(float(v)) * ca.SX.ones(dim, 1)
    else:
        m = ca.SX(np.asarray(v, dtype=float))
    return ca.reshape(m, dim, 1)


def _mat(v, rows: int, cols: int) -> ca.SX:
    m = v if isinstance(v, ca.SX) else ca.SX(np.asarray(v, dtype=float))
    return ca.reshape(m, rows, cols)


def _vec_len(v) -> int:
    if isinstance(v, ca.SX):
        return v.numel()
    return len(v)


def _shape(v) -> tuple[int, int]:
    if isinstance(v, ca.SX):
        return v.shape[0], v.shape[1]
    arr = np.asarray(v, dtype=float)
    if arr.ndim == 1:
        return 1, arr.shape[0]
    return arr.shape[0], arr.shape[1]


def quadratic(dim: int, P=None, center=None) -> Psi:
    """`psi(x) = 1/2 (x-c)^T P (x-c)`; default `P = I`, `c = 0`."""
    dim = int(dim)
    p_sx = ca.SX.eye(dim) if P is None else _mat(P, dim, dim)
    c_sx = ca.SX.zeros(dim, 1) if center is None else _col(center, dim)

    def fn(x: ca.SX) -> ca.SX:
        e = x - c_sx
        return 0.5 * ca.mtimes([e.T, p_sx, e])

    return Psi(fn, dim, name="quadratic")


def box_log_barrier(lo, hi, mu: Scalar = 1.0) -> Psi:
    """`psi(x) = -mu sum_i [log(x_i - lo_i) + log(hi_i - x_i)]` on `{lo < x < hi}`."""
    dim = _vec_len(lo)
    lo_sx = _col(lo, dim)
    hi_sx = _col(hi, dim)

    def fn(x: ca.SX) -> ca.SX:
        return -mu * ca.sum1(ca.log(x - lo_sx) + ca.log(hi_sx - x))

    def domain(x: ca.SX) -> list[ca.SX]:
        margins = [x[i] - lo_sx[i] for i in range(dim)]
        margins += [hi_sx[i] - x[i] for i in range(dim)]
        return margins

    return Psi(fn, dim, name="box_log_barrier", domain=domain)


def polytope_log_barrier(A, b, mu: Scalar = 1.0) -> Psi:
    """`psi(x) = -mu sum_i log(b_i - a_i^T x)` on `{A x < b}`."""
    n_rows, dim = _shape(A)
    a_sx = _mat(A, n_rows, dim)
    b_sx = _col(b, n_rows)

    def _margin(x: ca.SX) -> ca.SX:
        return b_sx - ca.mtimes(a_sx, x)

    def fn(x: ca.SX) -> ca.SX:
        return -mu * ca.sum1(ca.log(_margin(x)))

    def domain(x: ca.SX) -> list[ca.SX]:
        margin = _margin(x)
        return [margin[i] for i in range(n_rows)]

    return Psi(fn, dim, name="polytope_log_barrier", domain=domain)


def ellipsoid_barrier(c, S, mu: Scalar = 1.0) -> Psi:
    """`psi(x) = -mu log(1 - (x-c)^T S^-1 (x-c))` on `{(x-c)^T S^-1 (x-c) < 1}`."""
    dim = _vec_len(c)
    c_sx = _col(c, dim)
    s_sx = _mat(S, dim, dim)

    def _margin(x: ca.SX) -> ca.SX:
        e = x - c_sx
        return 1 - ca.mtimes(e.T, ca.solve(s_sx, e))

    def fn(x: ca.SX) -> ca.SX:
        return -mu * ca.log(_margin(x))

    def domain(x: ca.SX) -> list[ca.SX]:
        return [_margin(x)]

    return Psi(fn, dim, name="ellipsoid_barrier", domain=domain)


def neg_entropy(dim: int, shift: Scalar = 0.0) -> Psi:
    """`psi(x) = sum_i [z_i log z_i - z_i]`, `z = x + shift`, on `{z > 0}`."""
    dim = int(dim)

    def fn(x: ca.SX) -> ca.SX:
        z = x + shift
        return ca.sum1(z * ca.log(z) - z)

    def domain(x: ca.SX) -> list[ca.SX]:
        z = x + shift
        return [z[i] for i in range(dim)]

    return Psi(fn, dim, name="neg_entropy", domain=domain)
