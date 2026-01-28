from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
import inspect
import math

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from src.common.lj import ATM_IN_PA
from src.targets.registry import load_target_registry, resolve_target_metadata


class DerivedTargetError(ValueError):
    pass


ArrayLike = Any


@dataclass(frozen=True)
class DerivedTargetSpec:
    name: str
    inputs: List[str]
    outputs: List[str]
    mode: str = "consistency"
    loss_weight: float = 1.0
    params: Dict[str, Any] | None = None


_DERIVED_REGISTRY: Dict[str, Any] = {}


def register_derived_target(name: str, fn) -> None:
    key = str(name).strip().lower()
    if not key:
        raise DerivedTargetError("Derived target name must be non-empty.")
    if key in _DERIVED_REGISTRY:
        raise DerivedTargetError(f"Derived target already registered: {key}")
    _DERIVED_REGISTRY[key] = fn


def get_derived_target(name: str):
    key = str(name).strip().lower()
    fn = _DERIVED_REGISTRY.get(key)
    if fn is None:
        available = ", ".join(sorted(_DERIVED_REGISTRY.keys())) or "<none>"
        raise DerivedTargetError(f"Unknown derived target: {name}. Available: {available}")
    return fn


def _is_torch(x: Any) -> bool:
    return torch is not None and isinstance(x, torch.Tensor)


def _canonicalize_names(names: Iterable[str], cfg: Dict[str, Any]) -> List[str]:
    registry = load_target_registry(cfg)
    out: List[str] = []
    for name in names:
        name = str(name).strip()
        if not name:
            continue
        if name.startswith("target."):
            name = name[len("target.") :]
        if registry is not None:
            name = registry.resolve_alias(name) or name
        out.append(name)
    return out


def resolve_derived_targets(task_spec, cfg: Dict[str, Any], target_names: List[str]) -> List[DerivedTargetSpec]:
    specs = []
    raw = getattr(task_spec, "derived_targets", []) or []
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        raise DerivedTargetError("task.derived_targets must be a list or mapping.")
    for item in raw:
        if not isinstance(item, dict):
            raise DerivedTargetError("derived_targets entries must be mappings.")
        name = str(item.get("name") or item.get("method") or "").strip()
        if not name:
            raise DerivedTargetError("derived_targets entries must include name/method.")
        inputs = _canonicalize_names(item.get("inputs") or [], cfg)
        outputs = _canonicalize_names(item.get("outputs") or [], cfg)
        mode = str(item.get("mode", "consistency")).lower()
        if mode not in {"derive_only", "residual", "consistency"}:
            raise DerivedTargetError(f"Unknown derived_targets.mode: {mode}")
        loss_weight = float(item.get("loss_weight", item.get("weight", 1.0)))
        params = dict(item.get("params") or {})
        specs.append(
            DerivedTargetSpec(
                name=name,
                inputs=inputs,
                outputs=outputs,
                mode=mode,
                loss_weight=loss_weight,
                params=params,
            )
        )
    return specs


def build_derived_context(cfg: Dict[str, Any], target_names: List[str]) -> Dict[str, Any]:
    return {
        "target_meta": resolve_target_metadata(cfg, target_names),
        "task": cfg.get("task", {}) if isinstance(cfg, dict) else {},
        "dataset": cfg.get("dataset", {}) if isinstance(cfg, dict) else {},
    }


def _call_derived(fn, inputs: Dict[str, Any], params: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    try:
        sig = inspect.signature(fn)
        if len(sig.parameters) >= 3:
            return fn(inputs, params, context or {})
    except Exception:
        pass
    return fn(inputs, params)


def _as_array(x: ArrayLike) -> Tuple[Any, Any]:
    if _is_torch(x):
        return x, torch
    return np.asarray(x, dtype=float), np


def _lj_from_critical(inputs: Dict[str, ArrayLike], params: Dict[str, Any]) -> Dict[str, ArrayLike]:
    Tc = inputs.get("Tc") or inputs.get("tc") or inputs.get("Tc_K") or inputs.get("Tc [K]")
    Pc = inputs.get("Pc") or inputs.get("pc") or inputs.get("Pc_Pa") or inputs.get("Pc [Pa]")
    Tb = inputs.get("Tb") or inputs.get("tb") or inputs.get("Tb_K") or inputs.get("Tb [K]")
    if Tc is None or Pc is None:
        raise DerivedTargetError("lj_from_critical requires Tc and Pc inputs.")
    Tc_arr, lib = _as_array(Tc)
    Pc_arr, _ = _as_array(Pc)
    Tb_arr = None
    if Tb is not None:
        Tb_arr, _ = _as_array(Tb)

    eps_method = str(params.get("epsilon_method", "bird_critical")).lower()
    sig_method = str(params.get("sigma_method", "bird_critical")).lower()

    if eps_method == "bird_critical":
        eps = 0.77 * Tc_arr
    elif eps_method == "bird_boiling":
        if Tb_arr is None:
            raise DerivedTargetError("bird_boiling requires Tb input.")
        eps = 1.15 * Tb_arr
    elif eps_method == "flynn":
        eps = 1.77 * (Tc_arr ** (5.0 / 6.0))
    elif eps_method == "tee_gotoh_steward_1":
        eps = 0.7740 * Tc_arr
    else:
        raise DerivedTargetError(f"Unknown epsilon_method: {eps_method}")

    if sig_method == "bird_critical":
        Pc_atm = Pc_arr / float(ATM_IN_PA)
        if lib is np:
            invalid = (Pc_atm <= 0) | (Tc_arr <= 0)
            Pc_atm = np.where(invalid, np.nan, Pc_atm)
            sig = 2.44 * ((Tc_arr / Pc_atm) ** (1.0 / 3.0))
        else:
            invalid = (Pc_atm <= 0) | (Tc_arr <= 0)
            safe_pc = torch.where(invalid, torch.full_like(Pc_atm, float("nan")), Pc_atm)
            sig = 2.44 * ((Tc_arr / safe_pc) ** (1.0 / 3.0))
    else:
        raise DerivedTargetError(f"Unknown sigma_method: {sig_method}")

    return {
        "lj_epsilon_over_k_K": eps,
        "lj_sigma_A": sig,
    }


register_derived_target("lj_from_critical", _lj_from_critical)


def _build_pred_map(y_pred_raw: ArrayLike, target_names: List[str]) -> Dict[str, ArrayLike]:
    if _is_torch(y_pred_raw):
        arr = y_pred_raw
        if arr.ndim == 1:
            arr = arr.view(-1, 1)
        return {name: arr[:, idx] for idx, name in enumerate(target_names)}
    arr = np.asarray(y_pred_raw, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return {name: arr[:, idx] for idx, name in enumerate(target_names)}


def _stack_pred_map(pred_map: Dict[str, ArrayLike], target_names: List[str]) -> ArrayLike:
    first = pred_map[target_names[0]]
    if _is_torch(first):
        cols = [pred_map[name].view(-1, 1) for name in target_names]
        return torch.cat(cols, dim=1)
    cols = [np.asarray(pred_map[name], dtype=float).reshape(-1, 1) for name in target_names]
    return np.concatenate(cols, axis=1)


def _valid_mask_for_inputs(pred_map: Dict[str, ArrayLike], inputs: List[str]) -> ArrayLike:
    if not inputs:
        return None
    first = pred_map[inputs[0]]
    if _is_torch(first):
        mask = torch.isfinite(first)
        for name in inputs[1:]:
            mask = mask & torch.isfinite(pred_map[name])
        return mask
    mask = np.isfinite(pred_map[inputs[0]])
    for name in inputs[1:]:
        mask = mask & np.isfinite(pred_map[name])
    return mask


def apply_derived_numpy(
    y_pred_raw: np.ndarray,
    target_names: List[str],
    specs: List[DerivedTargetSpec],
    context: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    if not specs:
        return y_pred_raw, {}
    pred_map = _build_pred_map(y_pred_raw, target_names)
    derived_outputs: Dict[str, np.ndarray] = {}
    for spec in specs:
        fn = get_derived_target(spec.name)
        inputs = {name: pred_map.get(name) for name in spec.inputs}
        if any(v is None for v in inputs.values()):
            continue
        out_map = _call_derived(fn, inputs, spec.params or {}, context)
        for out_name, out_val in out_map.items():
            if spec.outputs and out_name not in spec.outputs:
                continue
            if out_name not in pred_map:
                derived_outputs[out_name] = np.asarray(out_val, dtype=float)
                continue
            derived_outputs[out_name] = np.asarray(out_val, dtype=float)
            valid = _valid_mask_for_inputs(pred_map, spec.inputs)
            if valid is None:
                continue
            valid = valid & np.isfinite(derived_outputs[out_name])
            if spec.mode == "derive_only":
                pred_map[out_name] = np.where(valid, derived_outputs[out_name], pred_map[out_name])
            elif spec.mode == "residual":
                pred_map[out_name] = np.where(valid, derived_outputs[out_name] + pred_map[out_name], pred_map[out_name])
            else:
                # consistency: keep direct prediction
                pass
    return _stack_pred_map(pred_map, target_names), derived_outputs


def _apply_step_torch(y: torch.Tensor, step: Dict[str, Any], inverse: bool = False) -> torch.Tensor:
    name = str(step.get("name", "")).lower()
    params = step.get("params", {}) or {}
    state = step.get("state", {}) or {}
    if name in {"none", "identity"}:
        return y
    if name in {"log", "log1p"}:
        if inverse:
            if name == "log1p":
                max_log = params.get("max_log")
                if max_log is None:
                    max_log = math.log(torch.finfo(y.dtype).max)
                y_clip = torch.clamp(y, max=float(max_log))
                return torch.expm1(y_clip)
            offset = float(state.get("offset", params.get("offset", params.get("eps", 0.0)) or 0.0))
            max_log = params.get("max_log")
            if max_log is None:
                max_log = math.log(torch.finfo(y.dtype).max)
            y_clip = torch.clamp(y, max=float(max_log))
            return torch.exp(y_clip) - offset
        if name == "log1p":
            invalid = y <= -1
            if invalid.any():
                y = torch.where(invalid, torch.full_like(y, float("nan")), y)
            return torch.log1p(y)
        offset = float(state.get("offset", params.get("offset", params.get("eps", 0.0)) or 0.0))
        y_shift = y + offset
        invalid = y_shift <= 0
        if invalid.any():
            y_shift = torch.where(invalid, torch.full_like(y_shift, float("nan")), y_shift)
        return torch.log(y_shift)
    if name in {"zscore", "standardize"}:
        mean = float(state.get("mean", 0.0))
        std = float(state.get("std", 1.0))
        if inverse:
            return y * std + mean
        return (y - mean) / std
    if name in {"clip", "winsorize"}:
        lo = state.get("min", params.get("min", None))
        hi = state.get("max", params.get("max", None))
        lo = -float("inf") if lo is None else float(lo)
        hi = float("inf") if hi is None else float(hi)
        return torch.clamp(y, min=lo, max=hi)
    raise DerivedTargetError(f"Unknown transform step for torch: {name}")


def _inverse_transform_torch(y: torch.Tensor, transform) -> torch.Tensor:
    out = y
    for step in reversed(transform.steps):
        out = _apply_step_torch(out, step, inverse=True)
    return out


def _forward_transform_torch(y: torch.Tensor, transform) -> torch.Tensor:
    out = y
    for step in transform.steps:
        out = _apply_step_torch(out, step, inverse=False)
    return out


def apply_derived_torch(
    y_pred_trans: torch.Tensor,
    transform_map: Dict[str, Any],
    target_names: List[str],
    specs: List[DerivedTargetSpec],
    mask: Optional[torch.Tensor] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if not specs:
        return y_pred_trans, y_pred_trans, None
    if y_pred_trans.dim() == 1:
        y_pred_trans = y_pred_trans.view(-1, 1)
    pred_raw_cols = []
    for idx, name in enumerate(target_names):
        transform = transform_map.get(name)
        col = y_pred_trans[:, idx]
        if transform is not None:
            col = _inverse_transform_torch(col, transform)
        pred_raw_cols.append(col)
    pred_raw = torch.stack(pred_raw_cols, dim=1)
    pred_map = _build_pred_map(pred_raw, target_names)
    derived_outputs: Dict[str, torch.Tensor] = {}
    consistency_loss = None
    for spec in specs:
        fn = get_derived_target(spec.name)
        inputs = {name: pred_map.get(name) for name in spec.inputs}
        if any(v is None for v in inputs.values()):
            continue
        out_map = _call_derived(fn, inputs, spec.params or {}, context)
        for out_name, out_val in out_map.items():
            if spec.outputs and out_name not in spec.outputs:
                continue
            if out_name not in pred_map:
                continue
            derived_val = out_val if _is_torch(out_val) else torch.as_tensor(out_val, device=pred_raw.device, dtype=pred_raw.dtype)
            derived_outputs[out_name] = derived_val
            valid = _valid_mask_for_inputs(pred_map, spec.inputs)
            if valid is None:
                continue
            if mask is not None:
                idx = target_names.index(out_name)
                valid = valid & mask[:, idx].bool()
            valid = valid & torch.isfinite(derived_val)
            if spec.mode == "derive_only":
                pred_map[out_name] = torch.where(valid, derived_val, pred_map[out_name])
            elif spec.mode == "residual":
                pred_map[out_name] = torch.where(valid, derived_val + pred_map[out_name], pred_map[out_name])
            elif spec.mode == "consistency":
                if valid.any():
                    diff = pred_map[out_name] - derived_val
                    mse = (diff[valid] ** 2).mean()
                    weighted = mse * float(spec.loss_weight)
                    consistency_loss = weighted if consistency_loss is None else consistency_loss + weighted
    pred_raw_adj = _stack_pred_map(pred_map, target_names)
    pred_trans_cols = []
    for idx, name in enumerate(target_names):
        transform = transform_map.get(name)
        col = pred_raw_adj[:, idx]
        if transform is not None:
            col = _forward_transform_torch(col, transform)
        pred_trans_cols.append(col)
    pred_trans_adj = torch.stack(pred_trans_cols, dim=1)
    return pred_trans_adj, pred_raw_adj, consistency_loss
