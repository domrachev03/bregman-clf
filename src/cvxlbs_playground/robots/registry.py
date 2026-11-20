"""`load_robot(name) -> RobotEntry`: one converted, torque-mode MJCF shared by MuJoCo and Pinocchio.

The only robot is `iiwa14` (KUKA iiwa 14 from MuJoCo Menagerie, fetched by `robot_descriptions`).
The source MJCF is converted once per process by `convert.to_torque_mode`; the converted file is
cached under `.cvxlbs_cache/robots/` at the repository root and loaded by both engines, so the
simulator and the controller model never see different XML.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import yaml
from cvxlbs.model import PinocchioModel

from .convert import to_torque_mode

_HERE = Path(__file__).resolve().parent
_META_DIR = _HERE / "meta"
_KNOWN_ROBOTS = ("iiwa14",)


@dataclass(frozen=True)
class RobotEntry:
    name: str
    mj_model: mujoco.MjModel
    model: PinocchioModel
    meta: dict[str, Any]
    mjcf_path: str  # converted torque-mode MJCF loaded by both engines


def _mjcf_source(name: str) -> str | Path:
    """Path of the original (unconverted) MJCF."""
    if name == "iiwa14":
        from robot_descriptions import iiwa14_mj_description as d

        return d.MJCF_PATH
    raise ValueError(f"unknown robot {name!r}; known: {_KNOWN_ROBOTS}")


def _load_meta(name: str) -> dict[str, Any]:
    with (_META_DIR / f"{name}.yaml").open() as f:
        return yaml.safe_load(f)


def _repo_cache_dir() -> Path:
    """`<repository root>/.cvxlbs_cache/robots` (next to this file if no `pyproject.toml` is found)."""
    for parent in _HERE.parents:
        if (parent / "pyproject.toml").exists():
            return parent / ".cvxlbs_cache" / "robots"
    return _HERE / ".cvxlbs_cache" / "robots"


@functools.lru_cache
def _converted(name: str) -> tuple[Path, list[str]]:
    """`(converted MJCF path, joint names)`; the conversion runs once per process."""
    meta = _load_meta(name)
    return to_torque_mode(_mjcf_source(name), meta["torque_limits"], _repo_cache_dir() / name)


def load_robot(name: str) -> RobotEntry:
    """Fresh `mujoco.MjModel` and `PinocchioModel` on every call (callers mutate them, e.g. the
    simulator sets the time step and a scenario may scale the torque limits)."""
    if name not in _KNOWN_ROBOTS:
        raise ValueError(f"unknown robot {name!r}; known: {_KNOWN_ROBOTS}")
    path, joint_names = _converted(name)
    model = PinocchioModel.from_mjcf(path)
    mj_model = mujoco.MjModel.from_xml_path(str(path))
    meta = dict(_load_meta(name))
    meta["joint_names"] = list(joint_names)
    meta["position_limits"] = {"lower": model.q_min.tolist(), "upper": model.q_max.tolist()}
    return RobotEntry(name=name, mj_model=mj_model, model=model, meta=meta, mjcf_path=str(path))
