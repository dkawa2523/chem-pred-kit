from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    F = None


def _reduce_masked_loss(
    loss: "torch.Tensor",
    mask: Optional["torch.Tensor"],
    weights: Optional["torch.Tensor | Sequence[float]"],
) -> "torch.Tensor":
    if torch is None:
        raise ImportError("PyTorch is required for masked loss reduction.")
    if mask is None and weights is None:
        return loss.mean()

    weights_t = None
    if weights is not None:
        if torch.is_tensor(weights):
            weights_t = weights
        else:
            weights_t = torch.tensor(weights, dtype=loss.dtype, device=loss.device)
        if weights_t.dim() == 0:
            weights_t = weights_t.view(1)
        if weights_t.dim() != 1:
            weights_t = weights_t.view(-1)
        weights_t = weights_t.to(device=loss.device, dtype=loss.dtype)
        if weights_t.numel() != loss.shape[1]:
            raise ValueError(f"loss weights length mismatch: expected {loss.shape[1]}, got {weights_t.numel()}")

    if mask is None:
        per_task = loss.mean(dim=0)
        denom = weights_t.sum()
        if denom.item() == 0:
            return per_task.sum() * 0.0
        return (per_task * weights_t).sum() / denom

    mask_t = mask.float()
    if mask_t.dim() == 1:
        mask_t = mask_t.view(-1, 1)
    mask_t = mask_t.to(loss.device)

    if weights_t is None:
        denom = mask_t.sum()
        if denom.item() == 0:
            return loss.sum() * 0.0
        return (loss * mask_t).sum() / denom

    denom_per = mask_t.sum(dim=0)
    per_task = torch.zeros_like(weights_t)
    valid = denom_per > 0
    if valid.any():
        per_task[valid] = (loss * mask_t).sum(dim=0)[valid] / denom_per[valid]
    weights_t = weights_t * valid.float()
    denom = weights_t.sum()
    if denom.item() == 0:
        return per_task.sum() * 0.0
    return (per_task * weights_t).sum() / denom


def masked_regression_loss(
    pred,
    target,
    mask: Optional[torch.Tensor] = None,
    loss_name: str = "mse",
    weights: Optional["torch.Tensor | Sequence[float]"] = None,
) -> "torch.Tensor":
    if torch is None or F is None:
        raise ImportError("PyTorch is required for masked_regression_loss.")

    loss_name = str(loss_name).lower()
    pred_t = pred
    target_t = target
    if pred_t.dim() == 1:
        pred_t = pred_t.view(-1, 1)
    if target_t.dim() == 1:
        target_t = target_t.view(-1, 1)

    if loss_name == "mse":
        loss = (pred_t - target_t) ** 2
    elif loss_name == "huber":
        loss = F.smooth_l1_loss(pred_t, target_t, reduction="none")
    else:
        raise ValueError(f"Unknown loss: {loss_name}")

    return _reduce_masked_loss(loss, mask, weights)


def masked_multitask_loss(
    pred,
    target,
    target_names: Sequence[str],
    mask: Optional[torch.Tensor] = None,
    loss_name: str = "mse",
    loss_per_target: Optional[Sequence[str]] = None,
    weights: Optional["torch.Tensor | Sequence[float]"] = None,
) -> "torch.Tensor":
    if torch is None or F is None:
        raise ImportError("PyTorch is required for masked_multitask_loss.")
    pred_t = pred
    target_t = target
    if pred_t.dim() == 1:
        pred_t = pred_t.view(-1, 1)
    if target_t.dim() == 1:
        target_t = target_t.view(-1, 1)
    if loss_per_target is None:
        return masked_regression_loss(pred_t, target_t, mask=mask, loss_name=loss_name, weights=weights)
    if len(loss_per_target) != pred_t.shape[1]:
        raise ValueError("loss_per_target length mismatch with prediction width.")

    losses = []
    for idx, loss_name_i in enumerate(loss_per_target):
        lname = str(loss_name_i or loss_name).lower()
        if lname == "mse":
            col_loss = (pred_t[:, idx] - target_t[:, idx]) ** 2
        elif lname == "huber":
            col_loss = F.smooth_l1_loss(pred_t[:, idx], target_t[:, idx], reduction="none")
        else:
            raise ValueError(f"Unknown loss: {lname}")
        losses.append(col_loss.view(-1, 1))
    loss = torch.cat(losses, dim=1)
    return _reduce_masked_loss(loss, mask, weights)


def resolve_loss_per_target(
    loss_cfg: Optional[Dict[str, Any]],
    target_names: Sequence[str],
    default_loss: str,
    logger: Optional[Any] = None,
) -> Optional[List[str]]:
    if not target_names:
        return None
    if not loss_cfg:
        return None
    loss_per_target = loss_cfg.get("loss_per_target") if isinstance(loss_cfg, dict) else None
    if loss_per_target is None:
        return None
    if isinstance(loss_per_target, (list, tuple)):
        names = [str(v).lower() for v in loss_per_target]
        if len(names) != len(target_names):
            raise ValueError("loss_per_target length mismatch with target_names.")
        return names
    if isinstance(loss_per_target, dict):
        normalized: Dict[str, str] = {}
        for key, value in loss_per_target.items():
            key_str = str(key).strip()
            if key_str.startswith("target."):
                key_str = key_str[len("target.") :]
            normalized[key_str] = str(value).lower()
        out: List[str] = []
        missing: List[str] = []
        for name in target_names:
            if name in normalized:
                out.append(normalized[name])
            else:
                out.append(str(default_loss).lower())
                missing.append(name)
        if missing and logger is not None:
            logger.warning("loss_per_target missing %s; defaulting to %s.", missing, default_loss)
        extra = [k for k in normalized.keys() if k not in target_names]
        if extra and logger is not None:
            logger.warning("loss_per_target provided unused targets %s; ignoring.", extra)
        return out
    raise ValueError("loss_per_target must be a list or dict when provided.")


def resolve_loss_weights(
    cfg: Dict[str, Any],
    target_names: Sequence[str],
    logger: Optional[Any] = None,
) -> Optional[List[float]]:
    if not target_names:
        return None
    loss_cfg = cfg.get("loss", {}) if isinstance(cfg, dict) else {}
    if not isinstance(loss_cfg, dict):
        return None
    weights_cfg = loss_cfg.get("weights")
    if weights_cfg is None:
        return None

    if isinstance(weights_cfg, (list, tuple)):
        weights = [float(w) for w in weights_cfg]
        if len(weights) != len(target_names):
            raise ValueError(
                f"loss.weights length mismatch: expected {len(target_names)} for targets {list(target_names)}, got {len(weights)}"
            )
    elif isinstance(weights_cfg, dict):
        normalized: Dict[str, float] = {}
        for key, value in weights_cfg.items():
            key_str = str(key).strip()
            if key_str.startswith("target."):
                key_str = key_str[len("target.") :]
            normalized[key_str] = float(value)

        weights: List[float] = []
        missing: List[str] = []
        for name in target_names:
            if name in normalized:
                weights.append(float(normalized[name]))
            else:
                missing.append(name)
                weights.append(1.0)
        extra = [key for key in normalized.keys() if key not in target_names]
        if missing and logger is not None:
            logger.warning("loss.weights missing targets %s; defaulting to 1.0 for them.", missing)
        if extra and logger is not None:
            logger.warning("loss.weights provided unused targets %s; ignoring them.", extra)
    else:
        raise ValueError("loss.weights must be a list or dict of per-target weights.")

    for name, weight in zip(target_names, weights):
        if not math.isfinite(weight):
            raise ValueError(f"loss.weights for target {name!r} must be finite (got {weight}).")
        if weight < 0:
            raise ValueError(f"loss.weights for target {name!r} must be >= 0 (got {weight}).")

    return weights
