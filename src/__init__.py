from __future__ import annotations

import os
from pathlib import Path


def _configure_runtime_cache() -> None:
    """
    Configure writable cache directories for common scientific Python libs.

    In some environments (sandboxed runs, CI, containers), $HOME may not be writable.
    Matplotlib/lightgbm may import Matplotlib internally; setting these early avoids
    noisy warnings and repeated cache builds.
    """
    repo_root = Path(__file__).resolve().parents[1]
    cache_root = repo_root / ".cache"
    mpl_cache = cache_root / "matplotlib"
    try:
        mpl_cache.mkdir(parents=True, exist_ok=True)
    except Exception:
        return

    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))


_configure_runtime_cache()
