"""Tasks: state expressions `x(q)` whose derivatives enter the QP rows through `Context.D`."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import casadi as ca

from .context import Context


class Task(ABC):
    name: str = ""
    dim: int = 0

    @abstractmethod
    def value(self, ctx: Context) -> ca.SX:
        """Task value, shape `(dim, 1)`."""


class FramePosition(Task):
    """World position of the origin of `frame`; `axes` selects world axes."""

    def __init__(self, model, frame: str, axes: Sequence[int] = (0, 1, 2)):
        self.model = model
        self.frame = frame
        self.axes = list(axes)
        self.dim = len(self.axes)
        self.name = f"frame_position_{frame}"

    def value(self, ctx: Context) -> ca.SX:
        pos = self.model.frame_position(self.frame, ctx.q)
        return pos[self.axes]
