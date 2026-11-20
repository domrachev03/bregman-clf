"""`PinocchioModel`: symbolic (CasADi SX, via `pinocchio.casadi`) and numeric rigid-body model of a
fixed-base, fully actuated robot with `nq == nv`."""

from __future__ import annotations

from pathlib import Path

import casadi as ca
import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin

#: Position limits at or beyond this magnitude mean "unbounded".
_HUGE_LIMIT = 1e19

#: `numeric()` methods that take a frame name first, and those returning a vector (flattened to 1-D).
_FRAME_METHODS = ("frame_position", "frame_jacobian")
_VECTOR_METHODS = ("bias", "gravity", "frame_position")


def _clip_position_limits(lower: np.ndarray, upper: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo = np.where(np.abs(lower) >= _HUGE_LIMIT, -np.inf, lower)
    hi = np.where(np.abs(upper) >= _HUGE_LIMIT, np.inf, upper)
    return lo, hi


def _clip_nonneg_limit(limit: np.ndarray) -> np.ndarray:
    """Velocity/effort limits: Pinocchio reports 0 (or a huge value) where the source file gives
    none; both are mapped to +inf."""
    arr = np.asarray(limit, dtype=float)
    return np.where((arr <= 0.0) | (arr >= _HUGE_LIMIT), np.inf, arr)


class PinocchioModel:
    """Attributes: `nq`, `nv`, `nu`, `joint_names`, `q_min`, `q_max`, `v_max`, `tau_max` (limits
    as arrays, +-inf where unknown) and the actuation matrix `B`. The symbolic methods take and
    return `casadi.SX`; `numeric(method, ...)` evaluates them on arrays."""

    def __init__(self, pin_model: pin.Model) -> None:
        if pin_model.nq != pin_model.nv:
            raise ValueError(
                f"PinocchioModel requires nq == nv (bounded revolute/prismatic joints, fixed base); "
                f"got nq={pin_model.nq}, nv={pin_model.nv}"
            )
        self._fn_cache: dict[tuple, ca.Function] = {}
        self.pin_model = pin_model
        self.cmodel = cpin.Model(pin_model)
        self.cdata = self.cmodel.createData()

        self.name = pin_model.name or "pinocchio_model"
        self.nq = pin_model.nq
        self.nv = pin_model.nv
        self.nu = pin_model.nv
        self.joint_names = [pin_model.names[i] for i in range(1, pin_model.njoints)]
        self.q_min, self.q_max = _clip_position_limits(
            np.asarray(pin_model.lowerPositionLimit, dtype=float),
            np.asarray(pin_model.upperPositionLimit, dtype=float),
        )
        self.v_max = _clip_nonneg_limit(pin_model.velocityLimit)
        self.tau_max = _clip_nonneg_limit(pin_model.effortLimit)
        self.B = np.eye(self.nv, self.nu)

    @classmethod
    def from_mjcf(cls, path: str | Path) -> PinocchioModel:
        return cls(pin.buildModelFromMJCF(str(path)))

    def mass(self, q: ca.SX) -> ca.SX:
        """`M(q)`, symmetric, including the joint armature."""
        cpin.crba(self.cmodel, self.cdata, q)
        m = ca.triu(self.cdata.M)
        return m + m.T - ca.diag(ca.diag(m))

    def bias(self, q: ca.SX, v: ca.SX) -> ca.SX:
        """`h(q, v) = C(q, v) v + g(q)`."""
        return cpin.nonLinearEffects(self.cmodel, self.cdata, q, v)

    def gravity(self, q: ca.SX) -> ca.SX:
        return cpin.computeGeneralizedGravity(self.cmodel, self.cdata, q)

    def actuation(self, q: ca.SX) -> ca.SX:
        """Generalized-force map of the joint torques, `(nv, nu)`."""
        return ca.DM(self.B)

    def frame_position(self, frame: str, q: ca.SX) -> ca.SX:
        """World position of the origin of `frame` (a body or an MJCF site), `(3, 1)`."""
        fid = self.cmodel.getFrameId(frame)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, q)
        return self.cdata.oMf[fid].translation

    def frame_jacobian(self, frame: str, q: ca.SX) -> ca.SX:
        """Translational Jacobian `d(frame_position)/dq`, `(3, nv)`."""
        return ca.jacobian(self.frame_position(frame, q), q)

    def numeric(self, method: str, *args: np.ndarray | str) -> np.ndarray:
        """Evaluate `mass`, `bias`, `gravity`, `frame_position` or `frame_jacobian` numerically,
        e.g. `model.numeric("frame_position", "ee", q)`. The compiled function is cached."""
        if method in _FRAME_METHODS:
            frame, num_args = args[0], args[1:]
        else:
            frame, num_args = None, args
        out = self._get_or_build_fn(method, frame)(*num_args)
        arr = np.asarray(out.full() if hasattr(out, "full") else out, dtype=float)
        if method in _VECTOR_METHODS:
            arr = arr.reshape(-1)
        return arr

    def _get_or_build_fn(self, method: str, frame: str | None) -> ca.Function:
        key = (method, frame)
        fn = self._fn_cache.get(key)
        if fn is not None:
            return fn
        q = ca.SX.sym("q", self.nq)
        if method == "mass":
            fn = ca.Function("mass", [q], [self.mass(q)])
        elif method == "gravity":
            fn = ca.Function("gravity", [q], [self.gravity(q)])
        elif method == "bias":
            v = ca.SX.sym("v", self.nv)
            fn = ca.Function("bias", [q, v], [self.bias(q, v)])
        elif method == "frame_position":
            fn = ca.Function(f"frame_position_{frame}", [q], [self.frame_position(frame, q)])
        elif method == "frame_jacobian":
            fn = ca.Function(f"frame_jacobian_{frame}", [q], [self.frame_jacobian(frame, q)])
        else:
            raise ValueError(f"PinocchioModel.numeric: unknown method {method!r}")
        self._fn_cache[key] = fn
        return fn
