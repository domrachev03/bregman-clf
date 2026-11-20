"""Optional compilation of a `casadi.Function` through generated C and `gcc`, cached on disk.

The cache key is `sha256(generated C source + casadi version + compiler flags)`; a repeated build of
the same function loads the cached shared library.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

import casadi as ca

#: Cache directory, overridable by the `CVXLBS_CACHE` environment variable.
_DEFAULT_CACHE_DIR = "~/.cache/cvxlbs"

_GCC_FLAGS: tuple[str, ...] = ("-O1", "-fPIC", "-shared")


def _resolve_cache_dir(cache_dir: str | os.PathLike[str] | None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir).expanduser()
    env = os.environ.get("CVXLBS_CACHE")
    return Path(env).expanduser() if env else Path(_DEFAULT_CACHE_DIR).expanduser()


def _generate_c_source(fn: ca.Function, directory: Path) -> Path:
    """Code-generate `fn` to `<directory>/<fn.name()>.c`; return that path.

    `ca.Function.generate` requires a bare filename (no path separators), so
    we `chdir` into `directory` for the call.
    """
    c_name = f"{fn.name()}.c"
    cwd = os.getcwd()
    os.chdir(directory)
    try:
        fn.generate(c_name)
    finally:
        os.chdir(cwd)
    return directory / c_name


def _cache_key(source: str, gcc_flags: tuple[str, ...]) -> str:
    payload = source + "\x00" + ca.__version__ + "\x00" + "\x00".join(gcc_flags)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_function(
    fn: ca.Function,
    jit: bool,
    cache_dir: str | os.PathLike[str] | None = None,
    *,
    gcc_flags: tuple[str, ...] = _GCC_FLAGS,
) -> ca.Function:
    """Return `fn` unchanged if not `jit`; else an equivalent function loaded from a compiled, cached
    shared library (compiled on the first build only)."""
    if not jit:
        return fn
    cache_root = _resolve_cache_dir(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    name = fn.name()
    with tempfile.TemporaryDirectory(prefix="cvxlbs_codegen_") as tmp:
        c_path = _generate_c_source(fn, Path(tmp))
        source = c_path.read_text()
        digest = _cache_key(source, gcc_flags)
        so_path = cache_root / f"{name}_{digest}.so"
        if not so_path.exists():
            tmp_so = so_path.with_suffix(f".so.tmp{os.getpid()}")
            result = subprocess.run(
                ["gcc", *gcc_flags, str(c_path), "-o", str(tmp_so)],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(f"codegen.build_function: gcc failed to compile {fn.name()!r}:\n{result.stderr}")
            os.replace(tmp_so, so_path)  # atomic within cache_root
    return ca.external(name, str(so_path))
