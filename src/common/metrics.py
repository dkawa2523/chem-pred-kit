from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _normalize_metric_array(values: Any, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim == 2:
        return arr
    raise ValueError(f"{name} must be 1D or 2D, got shape={arr.shape}")


def uncertainty_metrics(
    y_true: Any,
    y_pred: Any,
    y_std: Any,
    mask: Any = None,
    target_names: Optional[list[str]] = None,
    eps: float = 1.0e-6,
) -> Dict[str, Any]:
    if y_std is None:
        return {}
    yt = _normalize_metric_array(y_true, "y_true")
    yp = _normalize_metric_array(y_pred, "y_pred")
    ys = _normalize_metric_array(y_std, "y_std")

    if yt.shape != yp.shape:
        raise ValueError(f"uncertainty_metrics shape mismatch: y_true={yt.shape} y_pred={yp.shape}")
    if ys.shape != yp.shape:
        raise ValueError(f"uncertainty_metrics shape mismatch: y_std={ys.shape} y_pred={yp.shape}")

    if mask is not None:
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.shape != yt.shape:
            try:
                mask_arr = np.broadcast_to(mask_arr, yt.shape)
            except ValueError as exc:
                raise ValueError("mask shape mismatch for uncertainty metrics") from exc
        mask_arr = mask_arr & np.isfinite(yt)
    else:
        mask_arr = np.isfinite(yt)

    names = target_names or [f"t{i}" for i in range(yt.shape[1])]
    per_target: Dict[str, Dict[str, float]] = {}
    for idx in range(yt.shape[1]):
        if idx >= len(names):
            break
        valid = mask_arr[:, idx] & np.isfinite(ys[:, idx])
        if not np.any(valid):
            continue
        sigma = np.maximum(ys[valid, idx], eps)
        err = yt[valid, idx] - yp[valid, idx]
        nll_vals = 0.5 * np.log(2.0 * np.pi * (sigma**2)) + 0.5 * (err**2) / (sigma**2)
        per_target[str(names[idx])] = {
            "nll": float(np.mean(nll_vals)),
            "y_std_mean": float(np.mean(sigma)),
        }

    if not per_target:
        return {}

    metric_keys = sorted({k for values in per_target.values() for k in values.keys()})
    mean_metrics: Dict[str, float] = {}
    for key in metric_keys:
        vals = [values[key] for values in per_target.values() if key in values]
        if vals:
            mean_metrics[key] = float(np.mean(vals))

    out: Dict[str, Any] = dict(mean_metrics)
    out["by_target"] = per_target
    return out


def merge_metrics(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in extra.items():
        if key == "by_target" and isinstance(value, dict):
            merged = out.get("by_target")
            if not isinstance(merged, dict):
                merged = {}
            for target, metrics in value.items():
                if not isinstance(metrics, dict):
                    continue
                current = merged.get(target)
                if not isinstance(current, dict):
                    current = {}
                current = dict(current)
                current.update(metrics)
                merged[target] = current
            out["by_target"] = merged
        else:
            out[key] = value
    return out


def build_metrics_by_target(by_split: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, float]]]:
    by_target: Dict[str, Dict[str, Dict[str, float]]] = {}
    if not isinstance(by_split, dict):
        return by_target
    for split_name, split_metrics in by_split.items():
        if not isinstance(split_metrics, dict):
            continue
        split_by_target = split_metrics.get("by_target")
        if not isinstance(split_by_target, dict):
            continue
        for target_name, target_metrics in split_by_target.items():
            if not isinstance(target_metrics, dict):
                continue
            filtered: Dict[str, float] = {}
            for key, value in target_metrics.items():
                if isinstance(value, (int, float)):
                    filtered[str(key)] = float(value)
            if not filtered:
                continue
            by_target.setdefault(str(target_name), {})[str(split_name)] = filtered
    return by_target


def build_overall_by_split(by_split: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    overall: Dict[str, Dict[str, float]] = {}
    if not isinstance(by_split, dict):
        return overall
    for split_name, split_metrics in by_split.items():
        if not isinstance(split_metrics, dict):
            continue
        filtered: Dict[str, float] = {}
        for key, value in split_metrics.items():
            if key == "by_target":
                continue
            if isinstance(value, (int, float)):
                filtered[str(key)] = float(value)
        if filtered:
            overall[str(split_name)] = filtered
    return overall
