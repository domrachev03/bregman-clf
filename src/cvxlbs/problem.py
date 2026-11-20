"""QP assembly: `Problem` collects rows and costs from `Certificate`s and compiles them into `QPData`.

The decision vector is `x = (vdot, delta)`: the joint acceleration and one slack per scalar soft
row. The torque is eliminated, `tau = B^-1 (M vdot + h)`, which requires a fully actuated model.
Every row and cost residual must be affine in the decision vector; the QP is

    min 1/2 x^T H x + g^T x   s.t.   l <= C x <= u,

with `H, g, C, l, u` functions of `(q, v, t, p, pd, pdd)` compiled into one `casadi.Function`.
Besides the certificates' own rows and costs, the problem adds the torque limits
`|tau| <= tau_max` and the regularization `reg/2 |vdot|^2`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import casadi as ca
import numpy as np

from .certificates.base import Certificate, CostTerm, Row
from .codegen import build_function
from .context import Context
from .costs import Regularization

_INF = float("inf")


def _to_sx_column(value: object, m: int) -> ca.SX:
    """Broadcast a bound or weight (scalar or length `m`, numeric or SX) to an `(m, 1)` SX."""
    if isinstance(value, (ca.SX, ca.DM)):
        sx = ca.SX(value)
    else:
        arr = np.asarray(value, dtype=float)
        sx = ca.SX(float(arr)) if arr.ndim == 0 else ca.SX(arr.reshape(-1, 1))
    if sx.numel() == 1 and m > 1:
        sx = ca.repmat(sx, m, 1)
    elif sx.numel() != m:
        raise ValueError(f"Problem.build: expected 1 or {m} entries, got {sx.numel()}")
    return sx


def _is_all_neg_inf(value: object) -> bool:
    if isinstance(value, (ca.SX, ca.DM)):
        return False
    arr = np.asarray(value, dtype=float).reshape(-1)
    return bool(np.all(arr == -_INF))


def _find_offending_row(names: list[str], exprs: list[ca.SX], xi_full: ca.SX) -> str:
    for name, expr in zip(names, exprs):
        if ca.depends_on(ca.jacobian(expr, xi_full), xi_full):
            return name
    return "<unknown>"


@dataclass
class QPData:
    """Compiled QP: `fn` maps `(q, v, t, p, pd, pdd)` to `H, g, C, l, u`; `monitor_fn` to the
    certificates' monitors and the smallest domain margin; `tau_fn` maps `(q, v, vdot)` to the torque."""

    fn: ca.Function
    monitor_fn: ca.Function
    tau_fn: ca.Function
    n: int  # decision variables, slacks included
    n_core: int  # joint accelerations
    n_in: int  # inequality rows
    row_names: list[str] = field(default_factory=list)
    monitor_names: list[str] = field(default_factory=list)

    def __call__(self, q, v, t, p, pd, pdd) -> dict[str, np.ndarray]:
        h, g, c, l, u = self.fn(q, v, t, p, pd, pdd)
        return {
            "H": np.asarray(h.full(), dtype=float).reshape(self.n, self.n),
            "g": np.asarray(g.full(), dtype=float).reshape(self.n),
            "C": np.asarray(c.full(), dtype=float).reshape(self.n_in, self.n),
            "l": np.asarray(l.full(), dtype=float).reshape(self.n_in),
            "u": np.asarray(u.full(), dtype=float).reshape(self.n_in),
        }


class Problem:
    def __init__(self, model, params, *, reg: float = 1e-6, normalize_rows: bool = True) -> None:
        b = np.asarray(model.B, dtype=float)
        if not (model.nu == model.nv and b.shape == (model.nv, model.nv) and np.linalg.matrix_rank(b) == model.nv):
            raise ValueError("Problem: the model must be fully actuated (square, full-rank actuation matrix)")
        self.model = model
        self.params = params
        self.reg = reg
        self.normalize_rows = normalize_rows
        self.ctx = Context(model, params)
        self._items: list[tuple[Certificate, float | None]] = []

    def add(self, item: Certificate, *, soft_weight: float | None = None) -> None:
        """Register a certificate. With `soft_weight`, each of its rows `expr <= ub` is relaxed to
        `expr - delta <= ub`, `delta >= 0`, at the cost `soft_weight/2 delta^2`."""
        self._items.append((item, soft_weight))

    def build(self, *, jit: bool = False) -> QPData:
        ctx = self.ctx

        ineq_sources: list[tuple[Row, float | None]] = []
        cost_terms: list[CostTerm] = []
        monitors: dict[str, ca.SX] = {}
        domain_exprs: list[ca.SX] = []

        for item, soft_weight in self._items:
            ineq_sources.extend((r, soft_weight) for r in item.rows(ctx))
            cost_terms.extend(item.costs(ctx))
            prefix = f"{item.name}/" if item.name else ""
            for key, expr in item.monitors(ctx).items():
                monitors[f"{prefix}{key}"] = expr
            domain_exprs.extend(item.domain(ctx))

        tau = ctx.tau
        for i, tmax in enumerate(np.asarray(self.model.tau_max, dtype=float)):
            if np.isfinite(tmax):
                ineq_sources.append((Row(expr=tau[i], lb=-float(tmax), ub=float(tmax), name=f"torque_limit[{i}]"), None))
        cost_terms.extend(Regularization(self.reg).costs(ctx))

        # one slack per scalar soft row
        n_soft = 0
        for row, soft_weight in ineq_sources:
            if soft_weight is None:
                continue
            if not _is_all_neg_inf(row.lb):
                raise ValueError(f"Problem.build: soft row {row.name!r} must have lb=-inf (got {row.lb!r})")
            n_soft += row.expr.numel()
        delta = ca.SX.sym("delta", n_soft)
        xi_full = ca.vertcat(ctx.xi, delta)
        n_core = ctx.xi.numel()
        n_full = n_core + n_soft

        row_names: list[str] = []
        row_exprs: list[ca.SX] = []
        row_lb: list[ca.SX] = []
        row_ub: list[ca.SX] = []
        nonneg_names: list[str] = []
        nonneg_exprs: list[ca.SX] = []

        slack_idx = 0
        for row, soft_weight in ineq_sources:
            expr_vec = ca.reshape(row.expr, row.expr.numel(), 1)
            m = expr_vec.numel()
            lb_col = _to_sx_column(row.lb, m)
            ub_col = _to_sx_column(row.ub, m)
            is_soft = soft_weight is not None
            for k in range(m):
                name_k = f"{row.name}[{k}]" if m > 1 else row.name
                expr_k = expr_vec[k]
                if is_soft:
                    d_k = delta[slack_idx]
                    slack_idx += 1
                    cost_terms.append(CostTerm(residual=d_k, weight=soft_weight, name=f"{name_k}_slack"))
                    nonneg_names.append(f"{name_k}_slack_nonneg")
                    nonneg_exprs.append(d_k)
                    row_names.append(name_k)
                    row_exprs.append(expr_k - d_k)
                    row_lb.append(ca.SX(-_INF))
                    row_ub.append(ub_col[k])
                else:
                    row_names.append(name_k)
                    row_exprs.append(expr_k)
                    row_lb.append(lb_col[k])
                    row_ub.append(ub_col[k])
        row_names.extend(nonneg_names)
        row_exprs.extend(nonneg_exprs)
        row_lb.extend([ca.SX(0.0)] * len(nonneg_exprs))
        row_ub.extend([ca.SX(_INF)] * len(nonneg_exprs))

        # rows: C = d(expr)/d(x), bounds shifted by the constant part; rows scaled to unit norm
        n_in = len(row_names)
        exprs_col = ca.vertcat(*row_exprs)
        lb_col_all = ca.vertcat(*row_lb)
        ub_col_all = ca.vertcat(*row_ub)
        C = ca.jacobian(exprs_col, xi_full)
        if ca.depends_on(C, xi_full):
            bad = _find_offending_row(row_names, row_exprs, xi_full)
            raise ValueError(f"Problem.build: row {bad!r} is not affine in the decision vector")
        offsets = ca.substitute(exprs_col, xi_full, ca.SX.zeros(n_full, 1))
        l = lb_col_all - offsets
        u = ub_col_all - offsets
        if self.normalize_rows:
            row_norms = ca.fmax(ca.sqrt(ca.sum2(C * C)), 1e-9)
            C = C / row_norms
            l = l / row_norms
            u = u / row_norms

        # cost: sum of 1/2 w r^2 over affine residuals r, hence exactly quadratic
        cost_expr = ca.SX(0)
        for term in cost_terms:
            r = ca.reshape(term.residual, term.residual.numel(), 1)
            m = r.numel()
            w = _to_sx_column(term.weight, m)
            cost_expr = cost_expr + 0.5 * ca.dot(w, r * r)
        H, grad = ca.hessian(cost_expr, xi_full)
        if ca.depends_on(H, xi_full):
            raise ValueError("Problem.build: a cost residual is not affine in the decision vector")
        g = ca.substitute(grad, xi_full, ca.SX.zeros(n_full, 1))
        H = 0.5 * (H + H.T) + 1e-12 * ca.SX.eye(n_full)

        monitor_names = list(monitors.keys())
        monitors_vec = ca.vertcat(*monitors.values()) if monitor_names else ca.SX.zeros(0, 1)
        if domain_exprs:
            domain_min = domain_exprs[0]
            for e in domain_exprs[1:]:
                domain_min = ca.fmin(domain_min, e)
        else:
            domain_min = ca.SX(_INF)

        fn = ca.Function("qp_data", ctx.inputs(), [H, g, C, l, u], ctx.input_names, ["H", "g", "C", "l", "u"])
        monitor_fn = ca.Function(
            "qp_monitors", ctx.inputs(), [monitors_vec, domain_min], ctx.input_names, ["monitors", "domain_min"]
        )
        tau_fn = ca.Function("qp_tau", [ctx.q, ctx.v, ctx.xi], [tau], ["q", "v", "xi"], ["tau"])
        return QPData(
            fn=build_function(fn, jit=jit),
            monitor_fn=build_function(monitor_fn, jit=jit),
            tau_fn=build_function(tau_fn, jit=jit),
            n=n_full,
            n_core=n_core,
            n_in=n_in,
            row_names=row_names,
            monitor_names=monitor_names,
        )
