"""MuJoCo torque-mode simulation of a `RobotEntry`."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np


class MujocoSim:
    """Steps `entry.mj_model` with joint torques, clipped to the actuator control ranges."""

    def __init__(self, entry: Any, *, dt: float = 1e-3) -> None:
        self.dt = float(dt)
        self.model: mujoco.MjModel = entry.mj_model
        self.model.opt.timestep = self.dt
        self.data = mujoco.MjData(self.model)
        self.nq, self.nv, self.nu = self.model.nq, self.model.nv, self.model.nu
        self._ctrl_lo, self._ctrl_hi = self._actuator_ctrl_range()
        self.last_requested_torque = np.zeros(self.nu)
        self.last_applied_torque = np.zeros(self.nu)
        self.reset(np.zeros(self.nq))

    def _actuator_ctrl_range(self) -> tuple[np.ndarray, np.ndarray]:
        lo = np.full(self.nu, -np.inf)
        hi = np.full(self.nu, np.inf)
        limited = self.model.actuator_ctrllimited.astype(bool)
        rng = self.model.actuator_ctrlrange
        lo[limited] = rng[limited, 0]
        hi[limited] = rng[limited, 1]
        return lo, hi

    def reset(self, q0: np.ndarray, v0: np.ndarray | None = None) -> None:
        q0 = np.asarray(q0, dtype=float)
        if q0.shape != (self.nq,):
            raise ValueError(f"q0 shape {q0.shape} != ({self.nq},)")
        v0 = np.zeros(self.nv) if v0 is None else np.asarray(v0, dtype=float)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = q0
        self.data.qvel[:] = v0
        mujoco.mj_forward(self.model, self.data)
        self.last_requested_torque = np.zeros(self.nu)
        self.last_applied_torque = np.zeros(self.nu)

    def step(self, tau: np.ndarray) -> None:
        tau = np.asarray(tau, dtype=float)
        if tau.shape != (self.nu,):
            raise ValueError(f"tau shape {tau.shape} != ({self.nu},)")
        self.last_requested_torque = tau.copy()
        applied = np.clip(tau.copy(), self._ctrl_lo, self._ctrl_hi)
        self.last_applied_torque = applied
        self.data.ctrl[:] = applied
        mujoco.mj_step(self.model, self.data)

    @property
    def q(self) -> np.ndarray:
        return self.data.qpos.copy()

    @property
    def v(self) -> np.ndarray:
        return self.data.qvel.copy()

    @property
    def t(self) -> float:
        return float(self.data.time)
