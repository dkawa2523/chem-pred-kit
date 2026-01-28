from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from src.common.utils import save_json
from src.targets.registry import load_target_registry


class TargetTransform:
    def __init__(self, steps: Optional[List[Dict[str, Any]]] = None) -> None:
        self.steps = steps or []

    @staticmethod
    def _normalize_step(step: Any) -> Dict[str, Any]:
        if isinstance(step, str):
            return {"name": step, "params": {}, "state": {}}
        if isinstance(step, dict):
            if "name" in step:
                params = dict(step.get("params") or {})
                for key, value in step.items():
                    if key in {"name", "params", "state"}:
                        continue
                    params.setdefault(key, value)
                state = dict(step.get("state") or {})
                return {"name": str(step["name"]), "params": params, "state": state}
            if len(step) == 1:
                name, params = next(iter(step.items()))
                if params is None:
                    params = {}
                elif not isinstance(params, dict):
                    params = {"value": params}
                return {"name": str(name), "params": dict(params), "state": {}}
        raise ValueError(f"Unsupported transform step: {step!r}")

    @classmethod
    def from_config(cls, cfg: Any) -> "TargetTransform":
        if cfg is None or cfg == {}:
            return cls([])
        if isinstance(cfg, str):
            return cls([cls._normalize_step(cfg)])
        if isinstance(cfg, list):
            return cls([cls._normalize_step(step) for step in cfg])
        if isinstance(cfg, dict):
            if "steps" in cfg:
                steps_cfg = cfg.get("steps") or []
                return cls([cls._normalize_step(step) for step in steps_cfg])
            if "name" in cfg or len(cfg) == 1:
                return cls([cls._normalize_step(cfg)])
        raise ValueError(f"Unsupported target transform config: {cfg!r}")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TargetTransform":
        steps_cfg = data.get("steps") if isinstance(data, dict) else None
        if steps_cfg is None:
            return cls.from_config(None)
        return cls([cls._normalize_step(step) for step in steps_cfg])

    def to_dict(self, include_state: bool = True) -> Dict[str, Any]:
        steps = []
        for step in self.steps:
            payload = {"name": step["name"], "params": deepcopy(step.get("params", {}))}
            if include_state:
                payload["state"] = deepcopy(step.get("state", {}))
            steps.append(payload)
        return {"steps": steps}

    def fit(self, y: Iterable[float]) -> None:
        y_arr = np.asarray(list(y), dtype=float)
        for step in self.steps:
            name = step["name"].lower()
            if name in {"zscore", "standardize"}:
                mean = float(np.mean(y_arr))
                std = float(np.std(y_arr))
                if std == 0.0 or not np.isfinite(std):
                    std = 1.0
                step["state"]["mean"] = mean
                step["state"]["std"] = std
            elif name == "log":
                params = step.get("params", {}) or {}
                offset = float(params.get("offset", params.get("eps", 0.0) or 0.0))
                step["state"]["offset"] = offset
                if np.any((y_arr + offset) <= 0):
                    raise ValueError("log transform requires positive values. Use log1p or set offset/eps.")
            elif name == "clip":
                min_val = step.get("params", {}).get("min")
                max_val = step.get("params", {}).get("max")
                if min_val is None:
                    min_val = float(np.min(y_arr))
                if max_val is None:
                    max_val = float(np.max(y_arr))
                step["state"]["min"] = float(min_val)
                step["state"]["max"] = float(max_val)
            elif name == "winsorize":
                params = step.get("params", {}) or {}
                lower_q = params.get("lower", params.get("p_low", 0.01))
                upper_q = params.get("upper", params.get("p_high", 0.99))
                lower_q = float(lower_q)
                upper_q = float(upper_q)
                if not (0.0 <= lower_q < upper_q <= 1.0):
                    raise ValueError("winsorize requires 0 <= lower < upper <= 1")
                step["state"]["lower_q"] = lower_q
                step["state"]["upper_q"] = upper_q
                step["state"]["min"] = float(np.quantile(y_arr, lower_q))
                step["state"]["max"] = float(np.quantile(y_arr, upper_q))
            y_arr = self._apply_forward_step(y_arr, step)

    def transform(self, y: Iterable[float]) -> np.ndarray:
        y_arr = np.asarray(list(y), dtype=float)
        for step in self.steps:
            y_arr = self._apply_forward_step(y_arr, step)
        self._ensure_finite(y_arr, context="transform")
        return y_arr

    def inverse_transform(self, y: Iterable[float]) -> np.ndarray:
        y_arr = np.asarray(list(y), dtype=float)
        for step in reversed(self.steps):
            y_arr = self._apply_inverse_step(y_arr, step)
        self._ensure_finite(y_arr, context="inverse_transform")
        return y_arr

    def _apply_forward_step(self, y: np.ndarray, step: Dict[str, Any]) -> np.ndarray:
        name = step["name"].lower()
        if name in {"none", "identity"}:
            return y
        if name == "log":
            offset = step.get("state", {}).get("offset", step.get("params", {}).get("offset", step.get("params", {}).get("eps", 0.0)))
            offset = float(offset or 0.0)
            y_shift = y + offset
            if np.any(y_shift <= 0):
                raise ValueError("log transform requires positive values. Use log1p or set offset/eps.")
            return np.log(y_shift)
        if name == "log1p":
            return np.log1p(y)
        if name in {"zscore", "standardize"}:
            mean = step.get("state", {}).get("mean")
            std = step.get("state", {}).get("std")
            if mean is None or std is None:
                raise ValueError("zscore transform requires fitted mean/std")
            return (y - float(mean)) / float(std)
        if name in {"clip", "winsorize"}:
            min_val = step.get("state", {}).get("min", step.get("params", {}).get("min"))
            max_val = step.get("state", {}).get("max", step.get("params", {}).get("max"))
            if min_val is None and max_val is None:
                return y
            lo = float(min_val) if min_val is not None else -np.inf
            hi = float(max_val) if max_val is not None else np.inf
            return np.clip(y, lo, hi)
        raise ValueError(f"Unknown target transform step: {name}")

    def _apply_inverse_step(self, y: np.ndarray, step: Dict[str, Any]) -> np.ndarray:
        name = step["name"].lower()
        if name in {"none", "identity"}:
            return y
        if name == "log":
            offset = step.get("state", {}).get("offset", step.get("params", {}).get("offset", step.get("params", {}).get("eps", 0.0)))
            offset = float(offset or 0.0)
            return np.exp(y) - offset
        if name == "log1p":
            return np.expm1(y)
        if name in {"zscore", "standardize"}:
            mean = step.get("state", {}).get("mean")
            std = step.get("state", {}).get("std")
            if mean is None or std is None:
                raise ValueError("zscore transform requires fitted mean/std")
            return y * float(std) + float(mean)
        if name in {"clip", "winsorize"}:
            min_val = step.get("state", {}).get("min", step.get("params", {}).get("min"))
            max_val = step.get("state", {}).get("max", step.get("params", {}).get("max"))
            if min_val is None and max_val is None:
                return y
            lo = float(min_val) if min_val is not None else -np.inf
            hi = float(max_val) if max_val is not None else np.inf
            return np.clip(y, lo, hi)
        raise ValueError(f"Unknown target transform step: {name}")

    @staticmethod
    def _ensure_finite(y: np.ndarray, context: str) -> None:
        if not np.all(np.isfinite(y)):
            raise ValueError(f"Target transform produced non-finite values during {context}")


def _resolve_task_transform_map(cfg: Dict[str, Any]) -> Dict[str, Any]:
    task_cfg = cfg.get("task", {}) or {}
    raw = (
        task_cfg.get("transforms_per_target")
        or task_cfg.get("transforms")
        or task_cfg.get("target_transforms")
    )
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("task.transforms must be a mapping when provided.")
    return dict(raw)


def _normalize_transform_map(raw: Dict[str, Any], registry) -> Dict[str, Any]:
    if not raw:
        return {}
    out: Dict[str, Any] = {}
    for key, cfg in raw.items():
        name = str(key).strip()
        if not name:
            continue
        if name.startswith("target."):
            name = name[len("target.") :]
        if registry is not None:
            name = registry.resolve_alias(name) or name
        out[name] = cfg
    return out


def build_target_transforms(cfg: Dict[str, Any], target_names: List[str]) -> Dict[str, TargetTransform]:
    registry = load_target_registry(cfg)
    task_map = _normalize_transform_map(_resolve_task_transform_map(cfg), registry)
    transforms: Dict[str, TargetTransform] = {}
    for name in target_names:
        transform_cfg = None
        if name in task_map:
            transform_cfg = task_map[name]
        elif registry:
            transform_cfg = registry.transform_config_for(name)
            if transform_cfg is None:
                transform_cfg = registry.recommended_transform_for(name)
        transforms[name] = TargetTransform.from_config(transform_cfg)
    return transforms


def summarize_target_transforms(transforms: Dict[str, TargetTransform], include_state: bool = False) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for name, transform in transforms.items():
        summary[name] = transform.to_dict(include_state=include_state)
    return summary


def save_target_transforms(path: str, transforms: Dict[str, TargetTransform]) -> None:
    payload = {"version": 1, "targets": summarize_target_transforms(transforms, include_state=True)}
    save_json(path, payload)


def load_target_transforms(path: str) -> Dict[str, TargetTransform]:
    import json
    from pathlib import Path

    path_obj = Path(path)
    if not path_obj.exists():
        return {}
    with path_obj.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    targets_cfg = payload.get("targets", {}) if isinstance(payload, dict) else {}
    transforms: Dict[str, TargetTransform] = {}
    for name, cfg in targets_cfg.items():
        if not isinstance(cfg, dict):
            continue
        transforms[name] = TargetTransform.from_dict(cfg)
    return transforms


def fit_target_transforms(
    transforms: Dict[str, TargetTransform],
    y: np.ndarray,
    target_names: List[str],
    mask: Optional[np.ndarray] = None,
) -> None:
    y_arr = np.asarray(y, dtype=float)
    if y_arr.ndim == 1:
        y_arr = y_arr.reshape(-1, 1)
    if mask is None:
        mask_arr = np.isfinite(y_arr)
    else:
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.shape != y_arr.shape:
            try:
                mask_arr = np.broadcast_to(mask_arr, y_arr.shape)
            except ValueError as exc:
                raise ValueError("mask shape mismatch for target transforms") from exc
        mask_arr = mask_arr & np.isfinite(y_arr)

    for idx, name in enumerate(target_names):
        if idx >= y_arr.shape[1]:
            break
        valid = mask_arr[:, idx]
        if not np.any(valid):
            continue
        transforms[name].fit(y_arr[valid, idx])


def apply_target_transforms(
    y: np.ndarray,
    transforms: Dict[str, TargetTransform],
    target_names: List[str],
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    y_arr = np.asarray(y, dtype=float)
    y_was_1d = y_arr.ndim == 1
    if y_was_1d:
        y_arr = y_arr.reshape(-1, 1)
    if mask is None:
        mask_arr = np.isfinite(y_arr)
    else:
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.shape != y_arr.shape:
            try:
                mask_arr = np.broadcast_to(mask_arr, y_arr.shape)
            except ValueError as exc:
                raise ValueError("mask shape mismatch for target transforms") from exc
        mask_arr = mask_arr & np.isfinite(y_arr)

    out = np.zeros_like(y_arr, dtype=float)
    for idx, name in enumerate(target_names):
        if idx >= y_arr.shape[1]:
            break
        valid = mask_arr[:, idx]
        if not np.any(valid):
            continue
        out[valid, idx] = transforms[name].transform(y_arr[valid, idx])
    if y_was_1d:
        return out.reshape(-1)
    return out


def inverse_target_transforms(
    y: np.ndarray,
    transforms: Dict[str, TargetTransform],
    target_names: List[str],
) -> np.ndarray:
    y_arr = np.asarray(y, dtype=float)
    y_was_1d = y_arr.ndim == 1
    if y_was_1d:
        y_arr = y_arr.reshape(-1, 1)
    out = np.zeros_like(y_arr, dtype=float)
    for idx, name in enumerate(target_names):
        if idx >= y_arr.shape[1]:
            break
        transform = transforms.get(name, TargetTransform.from_config(None))
        out[:, idx] = transform.inverse_transform(y_arr[:, idx])
    if y_was_1d:
        return out.reshape(-1)
    return out
