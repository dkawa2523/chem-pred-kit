from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    import torch
    import torch.nn.functional as F
    from torch_geometric.data import Data
except Exception:  # pragma: no cover
    torch = None
    F = None
    Data = None


@dataclass
class GraphCLAugmentConfig:
    edge_drop: float = 0.2
    node_feature_mask: float = 0.2


def _require_torch() -> None:
    if torch is None or F is None:
        raise ImportError("PyTorch is required for GraphCL utilities.")


def _require_pyg() -> None:
    if Data is None:
        raise ImportError("torch_geometric is required for GraphCL augmentations.")


def _validate_prob(value: float, name: str) -> float:
    try:
        p = float(value)
    except Exception as exc:
        raise ValueError(f"{name} must be a float") from exc
    if p < 0.0 or p > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return p


def _validate_positive(value: float, name: str) -> float:
    try:
        v = float(value)
    except Exception as exc:
        raise ValueError(f"{name} must be a float") from exc
    if v <= 0.0:
        raise ValueError(f"{name} must be > 0")
    return v


def _clone_data(data: "Data") -> "Data":
    _require_pyg()
    if hasattr(data, "clone"):
        return data.clone()
    import copy

    return copy.deepcopy(data)


def _drop_edges(
    data: "Data",
    drop_p: float,
    generator: Optional["torch.Generator"] = None,
) -> None:
    _require_torch()
    drop_p = _validate_prob(drop_p, "edge_drop")
    if drop_p <= 0.0:
        return
    edge_index = getattr(data, "edge_index", None)
    if edge_index is None or edge_index.numel() == 0:
        return
    num_edges = int(edge_index.shape[1])
    keep = torch.rand(num_edges, device=edge_index.device, generator=generator) >= drop_p
    if not torch.any(keep):
        keep_idx = torch.randint(num_edges, (1,), device=edge_index.device, generator=generator)
        keep[keep_idx] = True
    data.edge_index = edge_index[:, keep]
    if hasattr(data, "edge_attr") and data.edge_attr is not None:
        data.edge_attr = data.edge_attr[keep]


def _mask_node_features(
    data: "Data",
    mask_p: float,
    generator: Optional["torch.Generator"] = None,
) -> None:
    _require_torch()
    mask_p = _validate_prob(mask_p, "node_feature_mask")
    if mask_p <= 0.0:
        return
    x = getattr(data, "x", None)
    if x is None:
        return
    num_nodes = int(x.shape[0])
    mask = torch.rand(num_nodes, device=x.device, generator=generator) < mask_p
    if torch.any(mask):
        x = x.clone()
        x[mask] = 0.0
        data.x = x


def apply_graphcl_augmentation(
    data: "Data",
    cfg: GraphCLAugmentConfig,
    generator: Optional["torch.Generator"] = None,
) -> "Data":
    _require_torch()
    _require_pyg()
    augmented = _clone_data(data)
    _drop_edges(augmented, cfg.edge_drop, generator=generator)
    _mask_node_features(augmented, cfg.node_feature_mask, generator=generator)
    return augmented


def graphcl_loss(
    z1: "torch.Tensor",
    z2: "torch.Tensor",
    temperature: float = 0.2,
) -> "torch.Tensor":
    _require_torch()
    if z1 is None or z2 is None:
        raise ValueError("graphcl_loss requires non-empty tensors.")
    if z1.dim() == 1:
        z1 = z1.view(-1, 1)
    if z2.dim() == 1:
        z2 = z2.view(-1, 1)
    if z1.shape != z2.shape:
        raise ValueError(f"z1/z2 shape mismatch: {z1.shape} vs {z2.shape}")
    if z1.numel() == 0:
        return torch.tensor(0.0, device=z1.device)
    temp = _validate_positive(temperature, "temperature")
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.T) / temp
    n = z1.size(0)
    mask = torch.eye(2 * n, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -1e9)
    labels = torch.arange(n, device=z.device)
    labels = torch.cat([labels + n, labels], dim=0)
    return F.cross_entropy(sim, labels)
