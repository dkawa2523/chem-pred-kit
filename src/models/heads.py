from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


class HeadConfigError(ValueError):
    pass


def _normalize_head_cfg(model_cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(model_cfg, dict):
        return None
    head_cfg = model_cfg.get("head")
    if head_cfg is None:
        return None
    if isinstance(head_cfg, str):
        return {"type": head_cfg}
    if isinstance(head_cfg, dict):
        return dict(head_cfg)
    raise HeadConfigError("model.head must be a string or mapping.")


def _get_activation(name: str):
    if nn is None:
        raise HeadConfigError("PyTorch is required for model.head configuration.")
    key = str(name or "relu").lower()
    if key in {"relu", "relu6"}:
        return nn.ReLU()
    if key in {"gelu"}:
        return nn.GELU()
    if key in {"silu", "swish"}:
        return nn.SiLU()
    if key in {"tanh"}:
        return nn.Tanh()
    if key in {"leaky_relu", "lrelu"}:
        return nn.LeakyReLU()
    raise HeadConfigError(f"Unknown head activation: {name}")


def _infer_head_input_dim(module: nn.Module) -> int:
    if hasattr(module, "head"):
        module = getattr(module, "head")
    if isinstance(module, nn.Linear):
        return int(module.in_features)
    if isinstance(module, nn.Sequential):
        for layer in module:
            if isinstance(layer, nn.Linear):
                return int(layer.in_features)
    raise HeadConfigError("Cannot infer head input dim (no Linear layer found).")


def _build_mlp_head(
    input_dim: int,
    out_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    activation: str,
) -> nn.Module:
    if nn is None:
        raise HeadConfigError("PyTorch is required for model.head configuration.")
    layers: List[nn.Module] = []
    dim = input_dim
    act = _get_activation(activation)
    for _ in range(max(int(num_layers), 1)):
        layers.append(nn.Linear(dim, int(hidden_dim)))
        layers.append(act)
        if dropout and float(dropout) > 0:
            layers.append(nn.Dropout(float(dropout)))
        dim = int(hidden_dim)
    layers.append(nn.Linear(dim, int(out_dim)))
    return nn.Sequential(*layers)


def _build_linear_head(input_dim: int, out_dim: int) -> nn.Module:
    if nn is None:
        raise HeadConfigError("PyTorch is required for model.head configuration.")
    return nn.Linear(int(input_dim), int(out_dim))


class MultiHead(nn.Module):
    def __init__(self, heads: Dict[str, nn.Module], order: List[str]):
        super().__init__()
        self.heads = nn.ModuleDict(heads)
        self.order = list(order)

    def forward(self, x):
        outs = [self.heads[name](x) for name in self.order]
        return torch.cat(outs, dim=-1)


def build_head(
    head_cfg: Dict[str, Any],
    input_dim: int,
    out_dim: int,
    target_names: Optional[List[str]] = None,
) -> nn.Module:
    if nn is None:
        raise HeadConfigError("PyTorch is required for model.head configuration.")
    cfg = dict(head_cfg or {})
    head_type = str(cfg.get("type", "linear")).lower()
    if head_type in {"linear", "shared"}:
        return _build_linear_head(input_dim, out_dim)
    if head_type in {"mlp", "regression"}:
        hidden_dim = int(cfg.get("hidden_dim", cfg.get("head_hidden_dim", input_dim)))
        num_layers = int(cfg.get("num_layers", 1))
        dropout = float(cfg.get("dropout", 0.0))
        activation = str(cfg.get("activation", "relu"))
        return _build_mlp_head(input_dim, out_dim, hidden_dim, num_layers, dropout, activation)
    if head_type in {"multi", "multi_head", "multitask"}:
        names = target_names or [f"t{i}" for i in range(out_dim)]
        if len(names) != out_dim:
            raise HeadConfigError("multi_head requires target_names length to match out_dim.")
        sub_type = str(cfg.get("sub_type", cfg.get("subtype", "linear"))).lower()
        sub_cfg = dict(cfg)
        sub_cfg["type"] = sub_type
        heads: Dict[str, nn.Module] = {}
        for name in names:
            heads[name] = build_head(sub_cfg, input_dim, 1, target_names=[name])
        return MultiHead(heads, names)
    if head_type in {"classification", "classifier"}:
        hidden_dim = int(cfg.get("hidden_dim", cfg.get("head_hidden_dim", input_dim)))
        num_layers = int(cfg.get("num_layers", 1))
        dropout = float(cfg.get("dropout", 0.0))
        activation = str(cfg.get("activation", "relu"))
        return _build_mlp_head(input_dim, out_dim, hidden_dim, num_layers, dropout, activation)
    raise HeadConfigError(f"Unknown model.head type: {head_type}")


def apply_head_override(
    model: Any,
    model_cfg: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
) -> Any:
    head_cfg = _normalize_head_cfg(model_cfg)
    if head_cfg is None:
        return model
    if nn is None or not isinstance(model, nn.Module):
        raise HeadConfigError("model.head requires a torch.nn.Module model.")
    if not hasattr(model, "encode"):
        raise HeadConfigError("model.head override requires model.encode().")
    input_dim = _infer_head_input_dim(model)
    out_dim = int((context or {}).get("out_dim", 1))
    target_names = (context or {}).get("target_names")
    head = build_head(head_cfg, input_dim=input_dim, out_dim=out_dim, target_names=target_names)
    setattr(model, "head", head)
    return model
