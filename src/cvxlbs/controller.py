"""`Controller`: builds a `Problem`, solves it at every `step`, and falls back when it fails."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from .certificates.base import Certificate
from .params import Params
from .problem import Problem, QPData
from .solver import DAQPSolver


@dataclass
class StepInfo:
    status: str  # QP solver status
    fallback: bool  # the fallback law was applied
    reason: str | None
    t_eval: float  # time to evaluate the QP data and the monitors
    t_solve: float  # time in the QP solver
    t_total: float
    monitors: dict[str, float] = field(default_factory=dict)
    xi: np.ndarray = field(default_factory=lambda: np.zeros(0))  # joint acceleration
    slack: np.ndarray = field(default_factory=lambda: np.zeros(0))
    domain_min: float = float("inf")  # smallest domain margin of the certificates


class Controller:
    """Inverse-dynamics QP controller. `add` certificates, `build`, then call `step` at every control
    period; it returns the joint torque.

    If the QP is not solved, or the state is outside the domain of a certificate
    (`domain_min <= 0`), `step` returns the fallback law `tau = g(q) - fallback_kd W v`, clipped to
    the torque limits, with `W = I` (`fallback_metric="identity"`) or `W = M(q)` (`"mass"`).
    """

    def __init__(
        self,
        model,
        params: Params,
        *,
        dt: float,
        fallback_kd: float = 5.0,
        fallback_metric: Literal["identity", "mass"] = "identity",
        **problem_kw,
    ) -> None:
        self.model = model
        self.params = params
        self.dt = dt
        self.fallback_kd = fallback_kd
        self.fallback_metric = fallback_metric
        self.problem = Problem(model, params, **problem_kw)
        self._qp_data: QPData | None = None
        self._solver: DAQPSolver | None = None

    def add(self, item: Certificate, *, soft_weight: float | None = None) -> Controller:
        self.problem.add(item, soft_weight=soft_weight)
        return self

    def build(self, *, jit: bool = False) -> Controller:
        """Compile the QP; `jit=True` compiles the generated C code with gcc (cached on disk)."""
        self._qp_data = self.problem.build(jit=jit)
        self._solver = DAQPSolver(self._qp_data.n, self._qp_data.n_in)
        return self

    def step(
        self,
        q: ArrayLike,
        v: ArrayLike,
        t: float,
        p: Mapping[str, ArrayLike] | None = None,
        pd: Mapping[str, ArrayLike] | None = None,
        pdd: Mapping[str, ArrayLike] | None = None,
    ) -> tuple[np.ndarray, StepInfo]:
        """`p`, `pd`, `pdd`: parameter values and their time derivatives by name (missing values
        default to the registered defaults, missing derivatives to zero)."""
        if self._qp_data is None or self._solver is None:
            raise RuntimeError("Controller.step: call .build() first")
        t0 = time.perf_counter()
        q = np.asarray(q, dtype=float).reshape(-1)
        v = np.asarray(v, dtype=float).reshape(-1)
        p_vec = self.params.pack(p, fill_default=True)
        pd_vec = self.params.pack(pd, fill_default=False)
        pdd_vec = self.params.pack(pdd, fill_default=False)

        t_eval0 = time.perf_counter()
        qp = self._qp_data(q, v, t, p_vec, pd_vec, pdd_vec)
        mon_out = self._qp_data.monitor_fn(q, v, t, p_vec, pd_vec, pdd_vec)
        monitors_vec = np.asarray(mon_out[0].full(), dtype=float).reshape(-1)
        domain_min = float(mon_out[1])
        t_eval = time.perf_counter() - t_eval0

        n_core = self._qp_data.n_core
        n_soft = self._qp_data.n - n_core
        monitors = dict(zip(self._qp_data.monitor_names, monitors_vec))

        result = self._solver.solve(qp)
        xi = np.full(n_core, np.nan)
        slack = np.full(n_soft, np.nan)
        fallback = False
        reason: str | None = None
        if result.status != "solved":
            fallback = True
            reason = f"solver status {result.status!r}"
        else:
            xi = result.x[:n_core]
            slack = result.x[n_core:]
            if domain_min <= 0:
                fallback = True
                reason = f"domain_min={domain_min:.6g} <= 0"

        if fallback:
            cmd = self._fallback_command(q, v)
        else:
            cmd = np.asarray(self._qp_data.tau_fn(q, v, xi).full(), dtype=float).reshape(-1)
            if not np.all(np.isfinite(cmd)):
                fallback = True
                reason = "non-finite torque"
                cmd = self._fallback_command(q, v)

        info = StepInfo(
            status=result.status,
            fallback=fallback,
            reason=reason,
            t_eval=t_eval,
            t_solve=result.t_solve,
            t_total=time.perf_counter() - t0,
            monitors=monitors,
            xi=xi,
            slack=slack,
            domain_min=domain_min,
        )
        return cmd, info

    def _fallback_command(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        g_q = self.model.numeric("gravity", q)
        if self.fallback_metric == "mass":
            tau = g_q - self.fallback_kd * (self.model.numeric("mass", q) @ v)
        else:
            tau = g_q - self.fallback_kd * v
        return np.clip(tau, -self.model.tau_max, self.model.tau_max)


IDController = Controller
