"""`EpisodeLog`: one episode's closed-loop trajectory, saved and loaded as `.npz`.

All arrays are indexed by control step `k`; `t[k]`, `q[k]`, `v[k]` are the state the controller
observed before producing `u[k]`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

#: QP status string -> int code stored in the log (unknown strings map to -1).
STATUS_CODES: dict[str, int] = {"solved": 0, "infeasible": 1, "max_iter": 2, "nonfinite": 3, "error": 4}
UNKNOWN_STATUS_CODE = -1

_MONITOR_PREFIX = "monitors__"
_GT_PREFIX = "ground_truth__"
_META_KEY = "__meta__"
_FIELDS = ("t", "q", "v", "u", "tau_req", "tau_applied", "status", "fallback", "t_eval", "t_solve", "t_total")


def status_to_code(status: str) -> int:
    return STATUS_CODES.get(status, UNKNOWN_STATUS_CODE)


@dataclass
class EpisodeLog:
    t: np.ndarray
    q: np.ndarray
    v: np.ndarray
    u: np.ndarray  # commanded torque
    tau_req: np.ndarray
    tau_applied: np.ndarray  # after clipping to the actuator range
    status: np.ndarray  # int codes, see STATUS_CODES
    fallback: np.ndarray  # bool: the fallback law was applied at this step
    t_eval: np.ndarray
    t_solve: np.ndarray
    t_total: np.ndarray
    monitors: dict[str, np.ndarray] = field(default_factory=dict)  # controller-side quantities (e.g. V2)
    ground_truth: dict[str, np.ndarray] = field(default_factory=dict)  # read from the simulator
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.t = np.asarray(self.t, dtype=float)
        self.q = np.asarray(self.q, dtype=float)
        self.v = np.asarray(self.v, dtype=float)
        self.u = np.asarray(self.u, dtype=float)
        self.tau_req = np.asarray(self.tau_req, dtype=float)
        self.tau_applied = np.asarray(self.tau_applied, dtype=float)
        self.status = np.asarray(self.status, dtype=np.int64)
        self.fallback = np.asarray(self.fallback, dtype=bool)
        self.t_eval = np.asarray(self.t_eval, dtype=float)
        self.t_solve = np.asarray(self.t_solve, dtype=float)
        self.t_total = np.asarray(self.t_total, dtype=float)
        self.monitors = {k: np.asarray(v, dtype=float) for k, v in self.monitors.items()}
        self.ground_truth = {k: np.asarray(v, dtype=float) for k, v in self.ground_truth.items()}

    def save(self, path: str | Path) -> None:
        """`monitors`/`ground_truth` are stored under `monitors__<name>`/`ground_truth__<name>`;
        `meta` as one JSON string, so the file loads without pickle."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {name: getattr(self, name) for name in _FIELDS}
        for name, arr in self.monitors.items():
            arrays[f"{_MONITOR_PREFIX}{name}"] = arr
        for name, arr in self.ground_truth.items():
            arrays[f"{_GT_PREFIX}{name}"] = arr
        arrays[_META_KEY] = np.array(json.dumps(self.meta, default=str))
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> EpisodeLog:
        with np.load(path, allow_pickle=False) as npz:
            monitors: dict[str, np.ndarray] = {}
            ground_truth: dict[str, np.ndarray] = {}
            for key in npz.files:
                if key.startswith(_MONITOR_PREFIX):
                    monitors[key[len(_MONITOR_PREFIX) :]] = npz[key]
                elif key.startswith(_GT_PREFIX):
                    ground_truth[key[len(_GT_PREFIX) :]] = npz[key]
            return cls(
                **{name: npz[name] for name in _FIELDS},
                monitors=monitors,
                ground_truth=ground_truth,
                meta=json.loads(str(npz[_META_KEY])),
            )
