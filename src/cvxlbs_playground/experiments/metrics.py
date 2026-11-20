"""Episode metrics. Violations, errors and deadlock are computed from simulator ground truth
(`log.q`, `log.v`, `log.ground_truth`), never from controller-side monitors.

`compute_metrics` builds one row of a sweep's `summary.parquet`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cvxlbs_playground.scenarios import keep_in_margin, keep_in_margin_physical

from .logs import STATUS_CODES, EpisodeLog

_SOLVED_CODE = STATUS_CODES["solved"]

#: Tolerances of the `*_violation_rate_tol/_solved/_fb` and `*_violated_trial` metrics:
#: joint position [rad], joint velocity [fraction of v_max], keep-in [m].
VIOLATION_TOL = {"qpos": 1e-3, "qvel": 1e-2, "keep_in": 1e-3}

#: Steps within this window [s] after a fallback step count as fallback-affected.
FALLBACK_WINDOW = 0.2


def violation_rate_depth(margin: np.ndarray) -> tuple[float, float]:
    """`(fraction of steps with margin < 0, largest -margin over those steps)`."""
    margin = np.asarray(margin, dtype=float)
    if margin.size == 0:
        return 0.0, 0.0
    violated = margin < 0.0
    rate = float(np.mean(violated))
    depth = float(np.max(-margin[violated])) if np.any(violated) else 0.0
    return rate, depth


def fallback_affected(fallback: np.ndarray, dt: float, window: float = FALLBACK_WINDOW) -> np.ndarray:
    """`True` on fallback steps and on the `window` seconds after each one."""
    fallback = np.asarray(fallback, dtype=bool)
    n = int(round(window / dt)) if dt > 0 else 0
    if n <= 0 or not np.any(fallback):
        return fallback.copy()
    counts = np.convolve(fallback.astype(float), np.ones(n + 1), mode="full")[: fallback.size]
    return counts > 0


def split_violations(prefix: str, margin: np.ndarray, affected: np.ndarray, tol: float) -> dict[str, float]:
    """Violation rates beyond `tol`: over all steps (`_tol`), on steps outside the fallback window
    (`_solved`), inside it (`_fb`), and whether the trial has any violation (`_violated_trial`)."""
    margin = np.asarray(margin, dtype=float)
    if margin.size == 0:
        return {}
    bad = margin < -tol
    affected = np.asarray(affected, dtype=bool)[: margin.size]
    return {
        f"{prefix}_violation_rate_tol": float(np.mean(bad)),
        f"{prefix}_violation_rate_solved": float(np.mean(bad & ~affected)),
        f"{prefix}_violation_rate_fb": float(np.mean(bad & affected)),
        f"{prefix}_violated_trial": float(np.any(bad)),
    }


def joint_position_margin(q: np.ndarray, q_min: np.ndarray, q_max: np.ndarray) -> np.ndarray:
    """`min_j min(q_j - q_min_j, q_max_j - q_j)` per step."""
    q = np.asarray(q, dtype=float)
    q_min = np.asarray(q_min, dtype=float)
    q_max = np.asarray(q_max, dtype=float)
    return np.min(np.minimum(q - q_min, q_max - q), axis=-1)


def joint_velocity_margin(v: np.ndarray, v_max: np.ndarray) -> np.ndarray:
    """`min_j (v_max_j - |v_j|)` per step."""
    v = np.asarray(v, dtype=float)
    v_max = np.asarray(v_max, dtype=float)
    return np.min(v_max - np.abs(v), axis=-1)


def infeasibility_rate(status: np.ndarray, solved_code: int = _SOLVED_CODE) -> float:
    """Fraction of steps whose QP status is not "solved"."""
    status = np.asarray(status)
    if status.size == 0:
        return 0.0
    return float(np.mean(status != solved_code))


def fallback_rate(fallback: np.ndarray) -> float:
    """Fraction of steps on which the fallback law was applied."""
    fallback = np.asarray(fallback, dtype=bool)
    if fallback.size == 0:
        return 0.0
    return float(np.mean(fallback))


def error_to_optimum(pos: np.ndarray, x_star: np.ndarray) -> np.ndarray:
    """`||pos[k] - x_star||` per step."""
    pos = np.asarray(pos, dtype=float)
    x_star = np.asarray(x_star, dtype=float)
    return np.linalg.norm(pos - x_star, axis=-1)


def convergence_time(t: np.ndarray, error: np.ndarray, tol: float) -> float | None:
    """First time from which `error` stays `<= tol` until the end of the episode."""
    t = np.asarray(t, dtype=float)
    error = np.asarray(error, dtype=float)
    below = error <= tol
    if not np.any(below):
        return None
    idx = None
    for i in range(len(below) - 1, -1, -1):
        if below[i]:
            idx = i
        else:
            break
    return float(t[idx]) if idx is not None else None


def final_error(error: np.ndarray) -> float:
    error = np.asarray(error, dtype=float)
    if error.size == 0:
        return float("nan")
    return float(np.mean(error[-1:]))


def _speed(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return np.abs(v) if v.ndim == 1 else np.linalg.norm(v, axis=-1)


def deadlock_intervals(
    t: np.ndarray, v: np.ndarray, error: np.ndarray, *, eps_v: float, eps: float
) -> list[tuple[float, float]]:
    """`[(t_start, t_end), ...]` of the maximal runs of steps with `|v| < eps_v` and `error > eps`."""
    t = np.asarray(t, dtype=float)
    stuck = (_speed(v) < eps_v) & (np.asarray(error, dtype=float) > eps)
    intervals: list[tuple[float, float]] = []
    start: int | None = None
    for i, flag in enumerate(stuck):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            intervals.append((float(t[start]), float(t[i - 1])))
            start = None
    if start is not None:
        intervals.append((float(t[start]), float(t[-1])))
    return intervals


def has_deadlock(t: np.ndarray, v: np.ndarray, error: np.ndarray, *, eps_v: float, eps: float, T_dl: float) -> bool:
    """Whether some deadlock interval lasts longer than `T_dl`."""
    return any((t1 - t0) > T_dl for t0, t1 in deadlock_intervals(t, v, error, eps_v=eps_v, eps=eps))


def timing_stats(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return {"mean": float("nan"), "p50": float("nan"), "p99": float("nan")}
    return {"mean": float(np.mean(x)), "p50": float(np.percentile(x, 50)), "p99": float(np.percentile(x, 99))}


def torque_total_variation(tau: np.ndarray, dt: float, *, solved: np.ndarray | None = None) -> float:
    """`mean_k ||tau_{k+1} - tau_k||_1 / dt` [N m/s]. With `solved` (per-step bool), only transitions
    whose two endpoints are both QP-solved steps are averaged, which excludes the torque jumps at
    switches between the QP law and the fallback law."""
    tau = np.asarray(tau, dtype=float)
    if tau.shape[0] < 2:
        return 0.0
    diffs = np.sum(np.abs(np.diff(tau, axis=0)), axis=-1)
    if solved is not None:
        solved = np.asarray(solved, dtype=bool)
        diffs = diffs[solved[:-1] & solved[1:]]
    if diffs.size == 0:
        return 0.0
    return float(np.mean(diffs) / dt)


def torque_switch_events(fallback: np.ndarray) -> int:
    """Number of switches between the QP law and the fallback law."""
    fallback = np.asarray(fallback, dtype=bool)
    if fallback.size < 2:
        return 0
    return int(np.sum(np.diff(fallback.astype(np.int8)) != 0))


def torque_gap(tau_req: np.ndarray, tau_applied: np.ndarray) -> dict[str, float]:
    """`mean`/`max` of `||tau_req[k] - tau_applied[k]||` (actuator clipping)."""
    tau_req = np.asarray(tau_req, dtype=float)
    tau_applied = np.asarray(tau_applied, dtype=float)
    if tau_req.size == 0:
        return {"mean": 0.0, "max": 0.0}
    diff = np.linalg.norm(tau_req - tau_applied, axis=-1) if tau_req.ndim > 1 else np.abs(tau_req - tau_applied)
    return {"mean": float(np.mean(diff)), "max": float(np.max(diff))}


def _dt_from_t(t: np.ndarray) -> float:
    t = np.asarray(t, dtype=float)
    if t.size < 2:
        return float("nan")
    return float(np.median(np.diff(t)))


def compute_metrics(
    log: EpisodeLog,
    sets: dict[str, Any],
    target: np.ndarray,
    *,
    eps_v: float = 1e-3,
    eps: float = 1e-3,
    T_dl: float = 0.5,
    conv_tol: float = 1e-3,
) -> dict[str, float]:
    """Metrics of one episode. `sets` is `Scenario.sets()`, `target` the end-effector target.
    `eps_v`, `eps`, `T_dl` define a deadlock (speed below `eps_v` and error above `eps` for longer
    than `T_dl`); `conv_tol` is the error tolerance of the convergence time."""
    out: dict[str, float] = {}
    affected = fallback_affected(log.fallback, _dt_from_t(log.t)) if log.t.size > 1 else np.asarray(log.fallback)

    margin = joint_position_margin(log.q, sets["q_min"], sets["q_max"])
    rate, depth = violation_rate_depth(margin)
    out["qpos_violation_rate"], out["qpos_violation_depth"] = rate, depth
    out.update(split_violations("qpos", margin, affected, VIOLATION_TOL["qpos"]))

    v_max = np.asarray(sets["v_max"], dtype=float)
    rate, depth = violation_rate_depth(joint_velocity_margin(log.v, v_max))
    out["qvel_violation_rate"], out["qvel_violation_depth"] = rate, depth
    rel = np.min(1.0 - np.abs(log.v) / v_max, axis=-1)
    out.update(split_violations("qvel", rel, affected, VIOLATION_TOL["qvel"]))
    out["qvel_peak_ratio"] = float(np.max(np.abs(log.v) / v_max)) if log.v.size else 0.0

    pos = log.ground_truth["ee_pos"]
    rate, depth = violation_rate_depth(keep_in_margin(pos, sets["keep_in"]))
    out["keep_in_violation_rate"], out["keep_in_violation_depth"] = rate, depth
    margin = keep_in_margin_physical(pos, sets["keep_in"])
    out.update(split_violations("keep_in", margin, affected, VIOLATION_TOL["keep_in"]))
    out["keep_in_margin_min"] = float(np.min(margin)) if margin.size else float("nan")

    out["infeasibility_rate"] = infeasibility_rate(log.status)
    out["fallback_rate"] = fallback_rate(log.fallback)

    # `final_error` and `final_error_proj` coincide: the target lies inside the keep-in set, so it
    # is its own projection (both columns are kept for compatibility with the stored summaries)
    err = error_to_optimum(pos, target)
    ct = convergence_time(log.t, err, conv_tol)
    out["final_error"] = final_error(err)
    out["convergence_time"] = ct if ct is not None else float("nan")
    out["deadlock"] = float(has_deadlock(log.t, log.v, err, eps_v=eps_v, eps=eps, T_dl=T_dl))
    out["final_error_proj"] = out["final_error"]
    out["convergence_time_proj"] = out["convergence_time"]

    for name, arr in (("t_eval", log.t_eval), ("t_solve", log.t_solve), ("t_total", log.t_total)):
        for stat, value in timing_stats(arr).items():
            out[f"{name}_{stat}"] = value

    if log.tau_applied.size:
        dt = _dt_from_t(log.t)
        out["torque_total_variation"] = torque_total_variation(log.tau_applied, dt)
        solved_mask = ~np.asarray(log.fallback, dtype=bool)
        out["torque_total_variation_solved"] = torque_total_variation(log.tau_applied, dt, solved=solved_mask)
        out["torque_switch_events"] = float(torque_switch_events(log.fallback))

    for stat, value in torque_gap(log.tau_req, log.tau_applied).items():
        out[f"torque_gap_{stat}"] = value

    return out
