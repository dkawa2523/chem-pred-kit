from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.targets.registry import load_target_registry, resolve_target_names, target_column_name

from src.common.metrics import regression_metrics

MetricsFn = Callable[..., Dict[str, float]]


def _empty_metrics(*_: Any, **__: Any) -> Dict[str, float]:
    return {}


def _normalize_columns(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        out: List[str] = []
        for item in value:
            if item is None:
                continue
            item_str = str(item).strip()
            if item_str:
                out.append(item_str)
        return out
    item_str = str(value).strip()
    return [item_str] if item_str else []


def _resolve_target_names(names: List[str], cfg: Dict[str, Any]) -> List[str]:
    registry = load_target_registry(cfg)
    out: List[str] = []
    for name in names:
        if name.startswith("target."):
            out.append(name)
            continue
        if registry is None:
            out.append(target_column_name(name))
            continue
        canonical = registry.resolve_alias(name) or name
        out.append(target_column_name(canonical))
    return out


def _canonicalize_target_names(names: List[str], cfg: Dict[str, Any]) -> List[str]:
    registry = load_target_registry(cfg)
    out: List[str] = []
    for name in names:
        name = str(name).strip()
        if not name:
            continue
        if registry is None:
            out.append(name)
            continue
        canonical = registry.resolve_alias(name) or name
        out.append(canonical)
    return out


def resolve_target_columns(cfg: Dict[str, Any]) -> List[str]:
    task_cfg = cfg.get("task", {}) or {}
    data_cfg = cfg.get("data", {}) or {}

    for key in ("target_names", "targets"):
        names = _normalize_columns(task_cfg.get(key))
        if names:
            return _resolve_target_names(names, cfg)

    for key in ("target_columns", "target_cols"):
        cols = _normalize_columns(task_cfg.get(key))
        if cols:
            return cols
    cols = _normalize_columns(task_cfg.get("target_col"))
    if cols:
        return cols

    for key in ("target_names", "targets"):
        names = _normalize_columns(data_cfg.get(key))
        if names:
            return _resolve_target_names(names, cfg)

    for key in ("target_columns", "target_cols"):
        cols = _normalize_columns(data_cfg.get(key))
        if cols:
            return cols
    return _normalize_columns(data_cfg.get("target_col"))


def _filter_metrics(
    metrics: Dict[str, Any],
    allow_overall: Optional[Sequence[str]],
    allow_per_target: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    if not allow_overall:
        if not allow_per_target:
            return metrics
        out = dict(metrics)
        if "by_target" in metrics and isinstance(metrics["by_target"], dict):
            out["by_target"] = {}
            per_target_allow = {str(item) for item in allow_per_target}
            for name, values in metrics["by_target"].items():
                if not isinstance(values, dict):
                    continue
                out["by_target"][name] = {k: v for k, v in values.items() if k in per_target_allow}
        return out
    allow_set = {str(item) for item in allow_overall}
    out: Dict[str, Any] = {k: v for k, v in metrics.items() if k in allow_set}
    if "by_target" in metrics and isinstance(metrics["by_target"], dict):
        out["by_target"] = {}
        per_target_allow = {str(item) for item in (allow_per_target or allow_overall)}
        for name, values in metrics["by_target"].items():
            if not isinstance(values, dict):
                continue
            out["by_target"][name] = {k: v for k, v in values.items() if k in per_target_allow}
    return out


def _resolve_metrics_fn(metrics_cfg: Any, task_type: str, target_names: Optional[List[str]] = None) -> MetricsFn:
    task_type = str(task_type).lower()
    base_fn = regression_metrics if task_type == "regression" else _empty_metrics

    allow: Optional[List[str]] = None
    allow_per_target: Optional[List[str]] = None

    if metrics_cfg is None:
        allow = None
    elif isinstance(metrics_cfg, str):
        key = metrics_cfg.lower()
        if key in {"regression", "default"}:
            allow = None
        elif key in {"none", "null", "off"}:
            return _empty_metrics
        else:
            raise ValueError(f"Unknown task.metrics: {metrics_cfg}")
    else:
        if isinstance(metrics_cfg, (list, tuple)):
            allow = [str(item) for item in metrics_cfg]
        elif isinstance(metrics_cfg, dict):
            overall = metrics_cfg.get("overall") or metrics_cfg.get("default") or metrics_cfg.get("metrics")
            per_target = metrics_cfg.get("per_target") or metrics_cfg.get("per_target_metrics") or metrics_cfg.get("metrics_per_target")
            if overall is None:
                allow = None
            elif isinstance(overall, str):
                key = overall.lower()
                if key in {"regression", "default"}:
                    allow = None
                elif key in {"none", "null", "off"}:
                    return _empty_metrics
                else:
                    raise ValueError(f"Unknown task.metrics.overall: {overall}")
            elif isinstance(overall, (list, tuple)):
                allow = [str(item) for item in overall]
            else:
                raise ValueError(f"Unsupported task.metrics.overall type: {type(overall).__name__}")
            if per_target is not None:
                if isinstance(per_target, str):
                    allow_per_target = [per_target]
                elif isinstance(per_target, (list, tuple)):
                    allow_per_target = [str(item) for item in per_target]
                else:
                    raise ValueError(f"Unsupported task.metrics.per_target type: {type(per_target).__name__}")
        else:
            raise ValueError(f"Unsupported task.metrics type: {type(metrics_cfg).__name__}")

    def _compute_metrics(
        y_true: Sequence[float],
        y_pred: Sequence[float],
        mask: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        if base_fn is _empty_metrics:
            return {}
        yt = np.asarray(y_true, dtype=float)
        yp = np.asarray(y_pred, dtype=float)
        if yt.ndim == 1 and yp.ndim == 2 and yp.shape[1] == 1:
            yt = yt.reshape(-1, 1)
        if yp.ndim == 1 and yt.ndim == 2 and yt.shape[1] == 1:
            yp = yp.reshape(-1, 1)
        if yt.ndim == 2 and yp.ndim == 2 and yt.shape[1] == 1 and yp.shape[1] == 1:
            yt = yt.reshape(-1)
            yp = yp.reshape(-1)
        if yt.ndim == 1 or yp.ndim == 1:
            yt = yt.reshape(-1)
            yp = yp.reshape(-1)
            if mask is not None:
                m = np.asarray(mask, dtype=bool).reshape(-1)
                m = m & np.isfinite(yt)
            else:
                m = np.isfinite(yt)
            if not np.any(m):
                return {}
            return _filter_metrics(base_fn(yt[m], yp[m]), allow, allow_per_target)

        if yt.ndim != 2 or yp.ndim != 2:
            raise ValueError("metrics_fn expects 1D or 2D arrays for y_true/y_pred.")
        if yt.shape != yp.shape:
            raise ValueError(f"metrics_fn shape mismatch: y_true={yt.shape} y_pred={yp.shape}")

        if mask is not None:
            m = np.asarray(mask, dtype=bool)
            if m.shape != yt.shape:
                try:
                    m = np.broadcast_to(m, yt.shape)
                except ValueError as exc:
                    raise ValueError("mask shape mismatch for multitask metrics") from exc
            m = m & np.isfinite(yt)
        else:
            m = np.isfinite(yt)

        names = target_names or [f"t{i}" for i in range(yt.shape[1])]
        per_target: Dict[str, Dict[str, float]] = {}
        for idx in range(yt.shape[1]):
            if idx >= len(names):
                break
            mask_idx = m[:, idx]
            if not np.any(mask_idx):
                continue
            per_target[str(names[idx])] = base_fn(yt[mask_idx, idx], yp[mask_idx, idx])

        if not per_target:
            return {}
        mean_metrics: Dict[str, float] = {}
        metric_keys = sorted({k for values in per_target.values() for k in values.keys()})
        for key in metric_keys:
            vals = [values[key] for values in per_target.values() if key in values]
            if vals:
                mean_metrics[key] = float(np.mean(vals))

        out: Dict[str, Any] = dict(mean_metrics)
        out["by_target"] = per_target
        return _filter_metrics(out, allow, allow_per_target)

    return _compute_metrics


def _resolve_loss_name(task_cfg: Dict[str, Any], train_cfg: Dict[str, Any], task_type: str) -> str:
    train_loss = train_cfg.get("loss")
    if train_loss is not None:
        return str(train_loss).lower()
    task_loss = task_cfg.get("loss")
    if task_loss is not None and not isinstance(task_loss, dict):
        return str(task_loss).lower()
    if str(task_type).lower() == "regression":
        return "mse"
    return "mse"


def _resolve_task_lists(task_cfg: Dict[str, Any], key: str) -> List[str]:
    raw = task_cfg.get(key)
    return _normalize_columns(raw)


def _resolve_task_target_map(task_cfg: Dict[str, Any], key: str) -> Dict[str, Any]:
    raw = task_cfg.get(key)
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    raise ValueError(f"task.{key} must be a mapping when provided.")


@dataclass(frozen=True)
class TaskSpec:
    name: str
    task_type: str
    target_columns: List[str]
    target_names: List[str]
    metrics_fn: MetricsFn
    loss_name: str
    targets: List[str] = field(default_factory=list)
    aux_targets: List[str] = field(default_factory=list)
    derived_targets: List[Dict[str, Any]] = field(default_factory=list)
    target_types: Dict[str, str] = field(default_factory=dict)
    loss_by_target: Dict[str, Any] = field(default_factory=dict)
    metrics_by_target: Dict[str, Any] = field(default_factory=dict)
    transforms_by_target: Dict[str, Any] = field(default_factory=dict)
    loss_per_target: Dict[str, Any] = field(default_factory=dict)
    metrics_per_target: Dict[str, Any] = field(default_factory=dict)
    transforms_per_target: Dict[str, Any] = field(default_factory=dict)

    def primary_target(self) -> Optional[str]:
        if not self.target_columns:
            return None
        return self.target_columns[0]


def resolve_task(cfg: Dict[str, Any]) -> TaskSpec:
    task_cfg = cfg.get("task", {}) or {}
    train_cfg = cfg.get("train", {}) or {}

    name = str(task_cfg.get("name") or cfg.get("task_name") or "task")
    task_type = str(task_cfg.get("type", "regression")).lower()
    target_columns = resolve_target_columns(cfg)
    target_names = resolve_target_names(cfg, target_columns)
    metrics_fn = _resolve_metrics_fn(task_cfg.get("metrics"), task_type, target_names=target_names)
    loss_name = _resolve_loss_name(task_cfg, train_cfg, task_type)
    targets = _canonicalize_target_names(_resolve_task_lists(task_cfg, "targets") or _resolve_task_lists(task_cfg, "target_names"), cfg)
    aux_targets = _canonicalize_target_names(_resolve_task_lists(task_cfg, "aux_targets"), cfg)
    derived_targets = task_cfg.get("derived_targets") or []
    if isinstance(derived_targets, dict):
        derived_targets = [derived_targets]
    if not isinstance(derived_targets, list):
        raise ValueError("task.derived_targets must be a list or mapping.")
    target_types = _resolve_task_target_map(task_cfg, "target_types")
    loss_by_target = task_cfg.get("loss") if isinstance(task_cfg.get("loss"), dict) else {}
    metrics_by_target = task_cfg.get("metrics") if isinstance(task_cfg.get("metrics"), dict) else {}
    transforms_by_target = _resolve_task_target_map(task_cfg, "transforms") if "transforms" in task_cfg else {}
    loss_per_target = _resolve_task_target_map(task_cfg, "loss_per_target") if "loss_per_target" in task_cfg else {}
    metrics_per_target = _resolve_task_target_map(task_cfg, "metrics_per_target") if "metrics_per_target" in task_cfg else {}
    transforms_per_target = _resolve_task_target_map(task_cfg, "transforms_per_target") if "transforms_per_target" in task_cfg else {}
    if not loss_per_target:
        loss_per_target = loss_by_target
    if not metrics_per_target:
        metrics_per_target = metrics_by_target
    if not transforms_per_target:
        transforms_per_target = transforms_by_target

    return TaskSpec(
        name=name,
        task_type=task_type,
        target_columns=target_columns,
        target_names=target_names,
        metrics_fn=metrics_fn,
        loss_name=loss_name,
        targets=targets,
        aux_targets=aux_targets,
        derived_targets=derived_targets,
        target_types=target_types,
        loss_by_target=loss_by_target,
        metrics_by_target=metrics_by_target,
        transforms_by_target=transforms_by_target,
        loss_per_target=loss_per_target,
        metrics_per_target=metrics_per_target,
        transforms_per_target=transforms_per_target,
    )
