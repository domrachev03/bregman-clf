"""Parameter registry: named blocks packed into the flat vectors `p`, `pd`, `pdd`.

A parameter of `order` 0 is static, of order 1 has a first time derivative (in `pd`), of order 2
also a second one (in `pdd`). `Context.D` gives zero tangents to the orders a parameter does not
declare.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class _Entry:
    slice: slice
    dim: int
    order: int
    default: np.ndarray


class Params:
    def __init__(self) -> None:
        self._entries: dict[str, _Entry] = {}
        self._size = 0

    def add(self, name: str, dim: int, default: ArrayLike | None = None, order: int = 0) -> Params:
        """Register a parameter block; returns `self`."""
        if name in self._entries:
            raise ValueError(f"Params.add: duplicate parameter name {name!r}")
        if dim <= 0:
            raise ValueError(f"Params.add({name!r}): dim must be positive, got {dim}")
        if order not in (0, 1, 2):
            raise ValueError(f"Params.add({name!r}): order must be 0, 1 or 2, got {order}")
        if default is None:
            default_vec = np.zeros(dim, dtype=float)
        else:
            default_vec = np.asarray(default, dtype=float).reshape(-1)
            if default_vec.shape != (dim,):
                raise ValueError(f"Params.add({name!r}): default has shape {default_vec.shape}, expected ({dim},)")
        start = self._size
        self._entries[name] = _Entry(slice(start, start + dim), dim, order, default_vec)
        self._size += dim
        return self

    @property
    def size(self) -> int:
        """Length of `p` (and of `pd`, `pdd`)."""
        return self._size

    @property
    def names(self) -> list[str]:
        return list(self._entries.keys())

    def _entry(self, name: str) -> _Entry:
        try:
            return self._entries[name]
        except KeyError as exc:
            raise KeyError(f"Params: unknown parameter {name!r} (known: {self.names})") from exc

    def slice(self, name: str) -> slice:
        return self._entry(name).slice

    def order(self, name: str) -> int:
        return self._entry(name).order

    def pack(self, values: Mapping[str, ArrayLike] | None = None, *, fill_default: bool = True) -> np.ndarray:
        """Pack named values into a flat vector. Missing names get their default (`fill_default`,
        used for `p`) or zeros (used for `pd`, `pdd`)."""
        values = values or {}
        unknown = set(values) - set(self._entries)
        if unknown:
            raise ValueError(f"Params.pack: unknown parameter name(s) {sorted(unknown)}")
        out = np.zeros(self._size, dtype=float)
        for name, entry in self._entries.items():
            if name in values:
                v = np.asarray(values[name], dtype=float).reshape(-1)
                if v.shape != (entry.dim,):
                    raise ValueError(f"Params.pack({name!r}): got shape {v.shape}, expected ({entry.dim},)")
                out[entry.slice] = v
            elif fill_default:
                out[entry.slice] = entry.default
        return out
