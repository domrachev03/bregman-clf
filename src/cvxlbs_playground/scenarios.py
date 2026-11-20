"""Reaching scenarios for the KUKA iiwa 14: keep the end-effector position inside a convex set while
moving it to a target in that set. `make_scenario(name, **options)` builds one from a sweep config.

A keep-in set is a dict:
`{"type": "box" | "box_entropy", "lo": (d,), "hi": (d,)}`, `{"type": "polytope", "A": (m, d), "b": (m,)}`
(`A x <= b`), or `{"type": "ellipsoid", "center": (d,), "shape": (d, d)}` (`(x-c)^T S^-1 (x-c) <= 1`).
`box_entropy` is the same region as `box`; the controller builds an entropic barrier for it instead
of a logarithmic one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from cvxlbs.params import Params
from cvxlbs.tasks import FramePosition
from scipy.spatial import ConvexHull

from cvxlbs_playground.robots import RobotEntry, load_robot

KeepIn = dict[str, Any]

# ---------------------------------------------------------------------------
# Keep-in set geometry
# ---------------------------------------------------------------------------


def keep_in_margin(pos: np.ndarray, keep_in: KeepIn) -> np.ndarray:
    """Signed margin (>= 0 inside) of `pos` (`(..., d)`). Box: `min_i min(x_i - lo_i, hi_i - x_i)`;
    polytope: `min_i (b_i - A_i x)`; ellipsoid: `1 - (x-c)^T S^-1 (x-c)` (dimensionless; see
    `keep_in_margin_physical`)."""
    pos = np.asarray(pos, dtype=float)
    kind = keep_in["type"]
    if kind in ("box", "box_entropy"):
        lo = np.asarray(keep_in["lo"], dtype=float)
        hi = np.asarray(keep_in["hi"], dtype=float)
        return np.min(np.minimum(pos - lo, hi - pos), axis=-1)
    if kind == "polytope":
        a = np.asarray(keep_in["A"], dtype=float)
        b = np.asarray(keep_in["b"], dtype=float)
        return np.min(b - pos @ a.T, axis=-1)
    if kind == "ellipsoid":
        c = np.asarray(keep_in["center"], dtype=float)
        shape = np.asarray(keep_in["shape"], dtype=float)
        d = pos - c
        value = np.einsum("...i,ij,...j->...", d, np.linalg.inv(shape), d)
        return 1.0 - value
    raise ValueError(f"unknown keep_in type {kind!r}")


def keep_in_margin_physical(pos: np.ndarray, keep_in: KeepIn) -> np.ndarray:
    """Signed margin in the length units of `pos`. Box and polytope: as `keep_in_margin`.
    Ellipsoid: the radial distance to the boundary along the ray from the center through `pos`,
    `||e|| (1/sqrt(e^T S^-1 e) - 1)` (the smallest semi-axis at the center itself)."""
    pos = np.asarray(pos, dtype=float)
    kind = keep_in["type"]
    if kind in ("box", "box_entropy", "polytope"):
        return keep_in_margin(pos, keep_in)
    if kind == "ellipsoid":
        c = np.asarray(keep_in["center"], dtype=float)
        shape = np.asarray(keep_in["shape"], dtype=float)
        e = pos - c
        value = np.einsum("...i,ij,...j->...", e, np.linalg.inv(shape), e)
        r = np.linalg.norm(e, axis=-1)
        scale = np.sqrt(np.maximum(value, 0.0))
        min_semi_axis = float(np.sqrt(np.min(np.linalg.eigvalsh(shape))))
        with np.errstate(invalid="ignore", divide="ignore"):
            margin = np.where(scale > 1e-12, r * (1.0 / np.where(scale > 1e-12, scale, 1.0) - 1.0), min_semi_axis)
        return margin
    raise ValueError(f"keep_in_margin_physical: unknown keep_in type {kind!r}")


def sample_in_box(rng: np.random.Generator, lo: np.ndarray, hi: np.ndarray, margin: float = 0.0) -> np.ndarray:
    lo_m = np.asarray(lo, dtype=float) + margin
    hi_m = np.asarray(hi, dtype=float) - margin
    if np.any(lo_m > hi_m):
        raise ValueError(f"sample_in_box: margin {margin} leaves an empty box ({lo_m} > {hi_m})")
    return rng.uniform(lo_m, hi_m)


def sample_in_polytope(
    rng: np.random.Generator,
    A: np.ndarray,
    b: np.ndarray,
    *,
    bbox_lo: np.ndarray,
    bbox_hi: np.ndarray,
    margin: float = 0.0,
    max_tries: int = 10_000,
) -> np.ndarray:
    """Rejection-sample uniformly in the bounding box until `A x <= b - margin` (Euclidean margin)."""
    row_norms = np.linalg.norm(A, axis=1)
    b_shrunk = b - margin * row_norms
    for _ in range(max_tries):
        x = rng.uniform(bbox_lo, bbox_hi)
        if np.all(A @ x <= b_shrunk):
            return x
    raise RuntimeError("sample_in_polytope: rejection sampling failed (polytope too small vs. bounding box/margin)")


def sample_in_ellipsoid(
    rng: np.random.Generator, center: np.ndarray, shape: np.ndarray, margin: float = 0.0
) -> np.ndarray:
    """Uniform sample in `{x : (x-c)^T S^-1 (x-c) <= (1-margin)^2}` (`margin` in [0, 1))."""
    center = np.asarray(center, dtype=float)
    dim = center.shape[0]
    L = np.linalg.cholesky(np.asarray(shape, dtype=float))
    u = rng.normal(size=dim)
    u /= np.linalg.norm(u)
    r = (1.0 - margin) * rng.uniform(0.0, 1.0) ** (1.0 / dim)
    return center + r * (L @ u)


def _random_convex_polytope_3d(
    rng: np.random.Generator, center: np.ndarray, radius: float, n_points: int = 26, radius_min: float = 0.55
) -> tuple[np.ndarray, np.ndarray]:
    """Half-space form `A x <= b` (unit-norm rows) of the convex hull of `n_points` random points at
    distance `U(radius_min, 1) * radius` from `center`."""
    directions = rng.normal(size=(n_points, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = radius * rng.uniform(radius_min, 1.0, size=n_points)
    pts = center + radii[:, None] * directions
    hull = ConvexHull(pts)
    eq = hull.equations  # (m, 4): a1 x1 + a2 x2 + a3 x3 + offset <= 0
    A = eq[:, :-1]
    b = -eq[:, -1]
    return A, b


def _keep_in_center(keep_in: KeepIn) -> np.ndarray:
    if keep_in["type"] in ("box", "box_entropy"):
        return 0.5 * (np.asarray(keep_in["lo"], dtype=float) + np.asarray(keep_in["hi"], dtype=float))
    if keep_in["type"] == "ellipsoid":
        return np.asarray(keep_in["center"], dtype=float)
    return 0.5 * (keep_in["_bbox_lo"] + keep_in["_bbox_hi"])


def _ray_exit(keep_in: KeepIn, center: np.ndarray, u: np.ndarray) -> float:
    """Distance along unit `u` from `center` to the keep-in boundary."""
    if keep_in["type"] in ("box", "box_entropy"):
        half = 0.5 * (np.asarray(keep_in["hi"], dtype=float) - np.asarray(keep_in["lo"], dtype=float))
        with np.errstate(divide="ignore"):
            return float(np.min(half / np.abs(u)))
    if keep_in["type"] == "ellipsoid":
        return float(1.0 / np.sqrt(u @ np.linalg.solve(keep_in["shape"], u)))
    A, b = keep_in["A"], keep_in["b"]
    denom = A @ u
    slack = b - A @ center
    return float(np.min(slack[denom > 1e-12] / denom[denom > 1e-12]))


# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------

#: `psi` option -> keep-in set type
_PSI_TO_KEEPIN_KIND = {"polytope": "polytope", "ellipsoid": "ellipsoid", "entropy": "box_entropy"}


@dataclass
class ScenarioSpec:
    name: str
    robot: str
    T: float
    dt: float = 1e-3
    options: dict[str, Any] = field(default_factory=dict)


class Scenario:
    """End-effector (`attachment_site`) position of the iiwa 14 inside a box, polytope or ball.

    Options (all optional):

    - `keep_in`: `"box"` (default), `"polytope"` or `"ellipsoid"`; or `psi`: `"polytope"`,
      `"ellipsoid"` or `"entropy"` (a box with an entropic barrier). The set is centered at
      `center` (default: the end-effector position at the home pose) with half-width / radius
      `half_width` (0.2 m). A polytope is the convex hull of `n_faces_seed_points` (26) random
      points drawn with `geometry_seed` (0) at radius `U(polytope_radius_min, 1) * half_width`.
    - `margin` (0.02 m): distance of the sampled start and target from the boundary.
    - `target_boundary_dist: [lo, hi]`: move the target along the ray from the center until it is
      `U(lo, hi)` from the boundary.
    - `v0_frac: [lo, hi]`: start with joint speeds `U(lo, hi) * v_max` (random signs), oriented so
      that the end effector moves away from the center of the set. Default: start at rest.
    - `tau_scale` in (0, 1]: scale the torque limits of both the controller model and the simulator.
    - `joint_margin` (0.1 rad): distance of the sampled start from the joint limits.
    - `joint_limit_stress`: sample the start and the posture reference with `stress_n_joints`
      (default half of them) joints within `stress_frac` (0.08) of their range from a limit.

    `reset(rng)` samples one episode (start state and target) and is a pure function of `rng`.
    """

    target = "x_d"  # parameter holding the task target

    def __init__(self, spec: ScenarioSpec) -> None:
        self.spec = spec
        self._entry: RobotEntry | None = None
        self._params: Params | None = None
        self._task_obj: FramePosition | None = None
        self._x_d: np.ndarray | None = None
        self._keep_in_cache: KeepIn | None = None
        self._q_home_episode: np.ndarray | None = None
        self._tau_scale_applied = False

    @property
    def entry(self) -> RobotEntry:
        """The robot, loaded on first access. Pinocchio does not read torque and velocity limits
        from the MJCF, so they are filled in from the robot's metadata."""
        if self._entry is None:
            entry = load_robot(self.spec.robot)
            for attr, key in (("tau_max", "torque_limits"), ("v_max", "velocity_limits")):
                value = np.asarray(getattr(entry.model, attr), dtype=float)
                if not np.all(np.isfinite(value)):
                    setattr(entry.model, attr, np.where(np.isfinite(value), value, np.asarray(entry.meta[key], dtype=float)))
            self._entry = entry
        return self._entry

    @property
    def params(self) -> Params:
        if self._params is None:
            params = Params()
            params.add("x_d", 3, default=self._home_ee(), order=1)
            params.add("q_home", self.entry.model.nq, default=np.asarray(self.entry.meta["home"], dtype=float))
            self._params = params
        return self._params

    @property
    def task_obj(self) -> FramePosition:
        if self._task_obj is None:
            self._task_obj = FramePosition(self.entry.model, self.entry.meta["ee_frame"], axes=(0, 1, 2))
        return self._task_obj

    def _home_ee(self) -> np.ndarray:
        model = self.entry.model
        home = np.asarray(self.entry.meta["home"], dtype=float)
        return model.numeric("frame_position", self.entry.meta["ee_frame"], home)

    def _keep_in(self) -> KeepIn:
        if self._keep_in_cache is not None:
            return self._keep_in_cache
        opts = self.spec.options
        center = np.asarray(opts.get("center", self._home_ee()), dtype=float)
        half_width = float(opts.get("half_width", 0.2))
        psi = opts.get("psi")
        if psi is not None:
            if psi not in _PSI_TO_KEEPIN_KIND:
                raise ValueError(f"unknown psi option {psi!r}; known: {sorted(_PSI_TO_KEEPIN_KIND)}")
            kind = _PSI_TO_KEEPIN_KIND[psi]
        else:
            kind = opts.get("keep_in", "box")
        if kind in ("box", "box_entropy"):
            keep_in: KeepIn = {"type": kind, "lo": center - half_width, "hi": center + half_width}
        elif kind == "polytope":
            rng = np.random.default_rng(int(opts.get("geometry_seed", 0)))
            A, b = _random_convex_polytope_3d(
                rng,
                center,
                half_width,
                n_points=int(opts.get("n_faces_seed_points", 26)),
                radius_min=float(opts.get("polytope_radius_min", 0.55)),
            )
            # `_bbox_*`: a bounding box of the polytope, used to sample targets by rejection
            keep_in = {
                "type": "polytope",
                "A": A,
                "b": b,
                "_bbox_lo": center - half_width,
                "_bbox_hi": center + half_width,
            }
        elif kind == "ellipsoid":
            keep_in = {"type": "ellipsoid", "center": center, "shape": np.eye(3) * half_width**2}
        else:
            raise ValueError(f"unknown keep_in option {kind!r}")
        self._keep_in_cache = keep_in
        return keep_in

    def _apply_tau_scale(self) -> None:
        """Scale `model.tau_max` (read by the QP) and MuJoCo's actuator control range (what the
        simulator clips to) by `tau_scale`, once."""
        tau_scale = float(self.spec.options.get("tau_scale", 1.0))
        if tau_scale == 1.0 or self._tau_scale_applied:
            return
        if not (0.0 < tau_scale <= 1.0):
            raise ValueError(f"tau_scale must be in (0, 1], got {tau_scale}")
        model = self.entry.model
        model.tau_max = np.asarray(model.tau_max, dtype=float) * tau_scale
        mj_model = self.entry.mj_model
        mj_model.actuator_ctrlrange[:] = np.asarray(mj_model.actuator_ctrlrange, dtype=float) * tau_scale
        self._tau_scale_applied = True

    def _sample_near_limits(self, rng: np.random.Generator, q_lo: np.ndarray, q_hi: np.ndarray) -> np.ndarray:
        """`joint_limit_stress` proposal: `stress_n_joints` random joints within `stress_frac` of
        their range from one of their limits, the others uniform."""
        frac = float(self.spec.options.get("stress_frac", 0.08))
        n = q_lo.shape[0]
        n_stress = int(self.spec.options.get("stress_n_joints", max(1, n // 2)))
        n_stress = min(max(n_stress, 0), n)
        span = q_hi - q_lo
        q = q_lo + span * rng.uniform(0.0, 1.0, size=n)
        stressed = rng.choice(n, size=n_stress, replace=False) if n_stress > 0 else np.array([], dtype=int)
        for j in stressed:
            near_lo = q_lo[j] + frac * span[j] * rng.uniform(0.05, 1.0)
            near_hi = q_hi[j] - frac * span[j] * rng.uniform(0.05, 1.0)
            q[j] = near_lo if rng.integers(0, 2) == 0 else near_hi
        return q

    @staticmethod
    def _ee_strictly_inside(ee: np.ndarray, keep_in: KeepIn, margin: float) -> bool:
        kind = keep_in["type"]
        if kind in ("box", "box_entropy"):
            return bool(np.all(ee >= keep_in["lo"] + margin) and np.all(ee <= keep_in["hi"] - margin))
        if kind == "polytope":
            A, b = keep_in["A"], keep_in["b"]
            row_norms = np.linalg.norm(A, axis=1)
            return bool(np.all(A @ ee <= b - margin * row_norms))
        e = ee - keep_in["center"]
        return bool(e @ np.linalg.solve(keep_in["shape"], e) <= (1.0 - margin) ** 2)

    def reset(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """`(q0, v0)` with the end effector strictly inside the keep-in set; also samples the target."""
        self._apply_tau_scale()
        margin = float(self.spec.options.get("margin", 0.02))
        model = self.entry.model
        keep_in = self._keep_in()
        stress = bool(self.spec.options.get("joint_limit_stress", False))
        q_lo, q_hi = model.q_min, model.q_max
        jm = float(self.spec.options.get("joint_margin", 0.1))
        for _ in range(10_000):
            q0 = self._sample_near_limits(rng, q_lo, q_hi) if stress else sample_in_box(rng, q_lo, q_hi, jm)
            ee = model.numeric("frame_position", self.entry.meta["ee_frame"], q0)
            inside = self._ee_strictly_inside(ee, keep_in, margin)
            if inside:
                break
        else:
            raise RuntimeError("reset: could not sample q0 with the end effector strictly inside the keep-in set")

        self._x_d = self._sample_target(rng, keep_in, margin)
        near = self.spec.options.get("target_boundary_dist")
        if near is not None:
            self._x_d = self._push_to_boundary(rng, keep_in, self._x_d, near)

        # under joint-limit stress the posture reference is also near the limits, otherwise the
        # posture cost pulls the arm straight back to the interior
        self._q_home_episode = self._sample_near_limits(rng, q_lo, q_hi) if stress else None

        v0 = np.zeros(model.nv)
        v0_frac = self.spec.options.get("v0_frac")
        if v0_frac is not None:
            v0 = self._sample_v0_toward_boundary(rng, q0, keep_in, v0_frac)
        return q0, v0

    def _sample_target(self, rng: np.random.Generator, keep_in: KeepIn, margin: float) -> np.ndarray:
        kind = keep_in["type"]
        if kind == "polytope":
            return sample_in_polytope(
                rng, keep_in["A"], keep_in["b"], bbox_lo=keep_in["_bbox_lo"], bbox_hi=keep_in["_bbox_hi"], margin=margin
            )
        if kind == "ellipsoid":
            return sample_in_ellipsoid(rng, keep_in["center"], keep_in["shape"], margin)
        return sample_in_box(rng, keep_in["lo"], keep_in["hi"], margin)

    def _push_to_boundary(self, rng: np.random.Generator, keep_in: KeepIn, x: np.ndarray, dist) -> np.ndarray:
        d = float(rng.uniform(float(dist[0]), float(dist[1])))
        center = _keep_in_center(keep_in)
        u = x - center
        norm = float(np.linalg.norm(u))
        u = u / norm if norm > 1e-9 else np.eye(x.size)[0]
        t_exit = _ray_exit(keep_in, center, u)
        for _ in range(60):  # back off until the Euclidean distance to the boundary is >= d
            y = center + max(t_exit - d, 0.0) * u
            if float(keep_in_margin_physical(y, keep_in)) >= d - 1e-6 or t_exit <= d:
                return y
            t_exit -= 0.25 * d
        return y

    def _sample_v0_toward_boundary(self, rng: np.random.Generator, q0: np.ndarray, keep_in: KeepIn, frac) -> np.ndarray:
        model = self.entry.model
        v_max = np.asarray(model.v_max, dtype=float)
        v0 = rng.uniform(float(frac[0]), float(frac[1]), size=model.nv) * v_max * rng.choice([-1.0, 1.0], size=model.nv)
        ee = model.numeric("frame_position", self.entry.meta["ee_frame"], q0)
        J = model.numeric("frame_jacobian", self.entry.meta["ee_frame"], q0)[:3, :]
        if float((J @ v0) @ (ee - _keep_in_center(keep_in))) < 0.0:
            v0 = -v0
        return v0

    def reference(self, t: float) -> tuple[dict, dict, dict]:
        """`(p, pd, pdd)` parameter dicts for `Controller.step` (a fixed target)."""
        x_d = self._x_d if self._x_d is not None else self._home_ee()
        p: dict[str, np.ndarray] = {"x_d": x_d}
        if self._q_home_episode is not None:
            p["q_home"] = self._q_home_episode
        return p, {"x_d": np.zeros(3)}, {}

    def sets(self) -> dict[str, Any]:
        model = self.entry.model
        return {
            "q_min": model.q_min,
            "q_max": model.q_max,
            "v_max": model.v_max,
            "tau_max": model.tau_max,
            "keep_in": self._keep_in(),
        }

    def target_position(self) -> np.ndarray | None:
        """The sampled target (inside the keep-in set), the reference of the error metrics."""
        return None if self._x_d is None else self._x_d.copy()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

#: Option presets of the comparison benchmark; a config's own options take precedence.
_PRESETS: dict[str, dict[str, Any]] = {
    "iiwa_constraints_joint_stress": {"joint_limit_stress": True, "keep_in": "box"},
    "iiwa_constraints_keep_in_box": {"keep_in": "box"},
    "iiwa_constraints_keep_in_ellipsoid": {"keep_in": "ellipsoid"},
    "iiwa_constraints_fast_motions": {"keep_in": "box", "v0_frac": [0.4, 0.7]},
}


def make_scenario(name: str, **options: object) -> Scenario:
    """`T` (default 1.0) and `dt` (default 1e-3) are taken from `options`; the rest are the options
    documented in `Scenario`. The name selects a preset, if any, and names the episode's output
    directory in a sweep; the metric-study (`iiwa_e1_*`) and stress-test (`iiwa_e8_*`) scenarios
    are defined entirely by the options in their configs."""
    if not (name in _PRESETS or name.startswith(("iiwa_e1_", "iiwa_e8_"))):
        raise ValueError(f"unknown scenario {name!r}; known: {sorted(_PRESETS)}, iiwa_e1_*, iiwa_e8_*")
    opts = dict(options)
    T = float(opts.pop("T", 1.0))
    dt = float(opts.pop("dt", 1e-3))
    for key, value in _PRESETS.get(name, {}).items():
        opts.setdefault(key, value)
    return Scenario(ScenarioSpec(name=name, robot="iiwa14", T=T, dt=dt, options=opts))
