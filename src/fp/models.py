from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None

try:
    from catboost import CatBoostRegressor
except Exception:  # pragma: no cover
    CatBoostRegressor = None


class ModelError(ValueError):
    pass


def get_model(name: str, params: Optional[Dict[str, Any]] = None):
    params = params or {}
    name = name.lower()
    if name == "rf" or name == "random_forest":
        return RandomForestRegressor(**params)
    if name == "lightgbm" or name == "lgbm":
        if lgb is None:
            raise ImportError("lightgbm is not installed.")
        return lgb.LGBMRegressor(**params)
    if name == "catboost":
        if CatBoostRegressor is None:
            raise ImportError("catboost is not installed.")
        # silent defaults
        if "verbose" not in params:
            params["verbose"] = False
        return CatBoostRegressor(**params)
    if name == "gpr":
        # Kernel can be configured later; provide a decent default
        kernel = params.pop("kernel", None)
        if kernel is None:
            kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))
        return GaussianProcessRegressor(kernel=kernel, **params)
    raise ModelError(f"Unknown model name: {name}")
