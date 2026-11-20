"""DAQP backend for the dense QP `min 1/2 x^T H x + g^T x  s.t.  l <= C x <= u`."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import daqp
import numpy as np

Status = Literal["solved", "infeasible", "max_iter", "nonfinite", "error"]

#: Finite stand-in for infinite bounds.
_BOUND_INF = 1e20

#: Added to the diagonal of `H`: the cost is only weakly convex in some directions, where DAQP's
#: internal convexity and feasibility checks can otherwise misfire.
_DAQP_HESSIAN_FLOOR = 1e-6

#: Tolerance of the post-solve check that a "solved" result satisfies `l <= C x <= u`.
_DAQP_RESIDUAL_TOL = 1e-6


@dataclass
class SolveResult:
    x: np.ndarray
    status: Status
    iters: int
    t_solve: float
    obj: float


def _has_nonfinite(qp: dict[str, np.ndarray]) -> bool:
    """Whether any entry is NaN, or infinite outside the bounds `l`, `u`."""
    for key, val in qp.items():
        arr = np.asarray(val, dtype=float)
        if arr.size == 0:
            continue
        if key in ("l", "u"):
            if np.isnan(arr).any():
                return True
        elif not np.all(np.isfinite(arr)):
            return True
    return False


def _daqp_status(exitflag: int) -> Status:
    if exitflag in (1, 2):
        return "solved"
    if exitflag == -4:
        return "max_iter"
    if exitflag in (-1, -6):
        return "infeasible"
    return "error"


class DAQPSolver:
    """`daqp.Model` set up once and then updated and solved at every call, so that the active set
    and the factorization are reused between control steps.

    A warm-started solve can fail on a feasible problem when the data moved far from the previous
    active set, either with an error flag or with a "solved" flag whose solution violates the new
    bounds. Every "solved" result is therefore checked against the bounds, and a failed warm solve
    is repeated once from a cold `setup()`.
    """

    def __init__(self, n: int, n_in: int, *, eps: float = 1e-9) -> None:
        self.n, self.n_in = n, n_in
        self._model = daqp.Model()
        self._model.settings = {"primal_tol": eps, "dual_tol": eps}
        self._sense = np.zeros(n_in, dtype=np.int32)
        self._initialized = False

    def _status(self, exitflag: int, x: np.ndarray | None, C: np.ndarray, u: np.ndarray, l: np.ndarray) -> Status:
        status = _daqp_status(exitflag)
        if status == "solved" and x is not None:
            resid = C @ np.asarray(x, dtype=float)
            if np.any(resid > u + _DAQP_RESIDUAL_TOL) or np.any(resid < l - _DAQP_RESIDUAL_TOL):
                status = "infeasible"
        return status

    def _cold_setup_and_solve(self, H, g, C, u, l) -> tuple[int, np.ndarray | None, float | None, dict | None]:
        setup_flag, _setup_time = self._model.setup(H, g, C, u, l, self._sense)
        if setup_flag < 0:
            self._initialized = False
            return setup_flag, None, None, None
        self._initialized = True
        x, fval, exitflag, info = self._model.solve()
        return exitflag, x, fval, info

    def solve(self, qp: dict[str, np.ndarray]) -> SolveResult:
        if _has_nonfinite(qp):
            return SolveResult(np.full(self.n, np.nan), "nonfinite", 0, 0.0, float("nan"))
        H = np.asarray(qp["H"], dtype=float) + _DAQP_HESSIAN_FLOOR * np.eye(self.n)
        g = np.asarray(qp["g"], dtype=float).reshape(-1)
        C = np.asarray(qp["C"], dtype=float).reshape(-1, self.n)
        l = np.clip(np.asarray(qp["l"], dtype=float).reshape(-1), -_BOUND_INF, _BOUND_INF)
        u = np.clip(np.asarray(qp["u"], dtype=float).reshape(-1), -_BOUND_INF, _BOUND_INF)

        t0 = time.perf_counter()
        if not self._initialized:
            exitflag, x, fval, info = self._cold_setup_and_solve(H, g, C, u, l)
        else:
            self._model.update(H=H, f=g, A=C, bupper=u, blower=l)
            x, fval, exitflag, info = self._model.solve()
            if self._status(exitflag, x, C, u, l) != "solved":
                exitflag, x, fval, info = self._cold_setup_and_solve(H, g, C, u, l)
        t_solve = time.perf_counter() - t0
        iters = int(info.get("iterations", 0)) if isinstance(info, dict) else 0

        status = self._status(exitflag, x, C, u, l)
        if status == "solved":
            obj = float(fval) if fval is not None else float("nan")
            return SolveResult(np.asarray(x, dtype=float), status, iters, t_solve, obj)
        return SolveResult(np.full(self.n, np.nan), status, iters, t_solve, float("nan"))
