from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from src.common.descriptors import calc_descriptors, descriptors_to_array

try:
    import torch
    from torch_geometric.data import Data
except Exception:  # pragma: no cover
    torch = None
    Data = None

try:
    from rdkit import Chem
    from rdkit.Chem import rdPartialCharges
except Exception:  # pragma: no cover
    Chem = None
    rdPartialCharges = None


class GraphFeaturizerError(ValueError):
    pass


@dataclass
class GaussianExpansionConfig:
    enabled: bool = False
    num_centers: int = 16
    min: float = 0.0
    max: float = 5.0
    sigma: Optional[float] = None

    def resolved_sigma(self) -> float:
        if self.sigma is not None:
            return float(self.sigma)
        if self.num_centers <= 1:
            return 1.0
        step = (float(self.max) - float(self.min)) / float(self.num_centers - 1)
        return step if step > 0 else 1.0


@dataclass
class GraphFeaturizerConfig:
    node_features: List[str]
    edge_features: List[str]
    use_3d_pos: bool = True
    angle_features: List[str] = field(default_factory=list)
    add_global_descriptors: Optional[List[str]] = None
    edge_mode: str = "bond"
    radius_cutoff: float = 5.0
    max_neighbors: int = 32
    gaussian_expansion: GaussianExpansionConfig = field(default_factory=GaussianExpansionConfig)
    atomic_num_max: int = 118
    atomic_num_vocab: Optional[List[int]] = None

    def __post_init__(self) -> None:
        self.edge_mode = str(self.edge_mode).lower()
        if self.gaussian_expansion is None:
            self.gaussian_expansion = GaussianExpansionConfig()
        elif isinstance(self.gaussian_expansion, dict):
            self.gaussian_expansion = GaussianExpansionConfig(**self.gaussian_expansion)
        if self.angle_features is None:
            self.angle_features = []
        else:
            self.angle_features = [str(v) for v in self.angle_features]
        if self.atomic_num_vocab is not None:
            self.atomic_num_vocab = [int(v) for v in self.atomic_num_vocab]
        self.atomic_num_max = int(self.atomic_num_max)

    def __setstate__(self, state) -> None:
        self.__dict__.update(state)
        if not hasattr(self, "edge_mode"):
            self.edge_mode = "bond"
        if not hasattr(self, "radius_cutoff"):
            self.radius_cutoff = 5.0
        if not hasattr(self, "max_neighbors"):
            self.max_neighbors = 32
        if not hasattr(self, "gaussian_expansion") or self.gaussian_expansion is None:
            self.gaussian_expansion = GaussianExpansionConfig()
        elif isinstance(self.gaussian_expansion, dict):
            self.gaussian_expansion = GaussianExpansionConfig(**self.gaussian_expansion)
        if not hasattr(self, "angle_features") or self.angle_features is None:
            self.angle_features = []
        else:
            self.angle_features = [str(v) for v in self.angle_features]
        if not hasattr(self, "atomic_num_max") or self.atomic_num_max is None:
            self.atomic_num_max = 118
        if hasattr(self, "atomic_num_vocab") and self.atomic_num_vocab is not None:
            self.atomic_num_vocab = [int(v) for v in self.atomic_num_vocab]
        self.atomic_num_max = int(self.atomic_num_max)
        self.edge_mode = str(self.edge_mode).lower()


def _require_pyg():
    if Data is None or torch is None:
        raise ImportError(
            "torch_geometric is required for GNN pipeline. "
            "Please install PyTorch Geometric (matching your torch/CUDA)."
        )


def _compute_gasteiger_charges(mol) -> Tuple[List[float], List[float]]:
    n_atoms = mol.GetNumAtoms()
    charges = [0.0] * n_atoms
    mask = [0.0] * n_atoms
    if rdPartialCharges is None:
        return charges, mask
    try:
        rdPartialCharges.ComputeGasteigerCharges(mol)
    except Exception:
        return charges, mask
    for idx, atom in enumerate(mol.GetAtoms()):
        value = None
        if atom.HasProp("_GasteigerCharge"):
            try:
                value = float(atom.GetProp("_GasteigerCharge"))
            except Exception:
                value = None
        if value is None or not np.isfinite(value):
            charges[idx] = 0.0
            mask[idx] = 0.0
        else:
            charges[idx] = float(value)
            mask[idx] = 1.0
    return charges, mask


_BOND_FEATURES = {"bond_type", "conjugated", "aromatic"}
_DIST_FEATURES = {"distance", "gaussian"}
_ALLOWED_EDGE_FEATURES = _BOND_FEATURES | _DIST_FEATURES
_ANGLE_FEATURES = {"angle", "cos_angle"}


def _one_hot(index: Optional[int], size: int) -> List[float]:
    out = [0.0] * size
    if index is None:
        return out
    if 0 <= index < size:
        out[index] = 1.0
    return out


def _atomic_num_onehot(z: int, cfg: GraphFeaturizerConfig) -> List[float]:
    if cfg.atomic_num_vocab:
        try:
            idx = cfg.atomic_num_vocab.index(int(z))
        except ValueError:
            idx = None
        return _one_hot(idx, len(cfg.atomic_num_vocab))
    size = int(cfg.atomic_num_max)
    if size <= 0:
        size = 118
    idx = int(z) - 1
    return _one_hot(idx, size)


def _hybridization_onehot(atom) -> List[float]:
    if Chem is None or atom is None:
        return [0.0] * 7
    hyb = atom.GetHybridization()
    order = [
        Chem.HybridizationType.SP,
        Chem.HybridizationType.SP2,
        Chem.HybridizationType.SP3,
        Chem.HybridizationType.SP3D,
        Chem.HybridizationType.SP3D2,
        Chem.HybridizationType.S,
    ]
    idx = None
    for i, item in enumerate(order):
        if hyb == item:
            idx = i
            break
    return _one_hot(idx, len(order) + 1)


def _chirality_onehot(atom) -> List[float]:
    if Chem is None or atom is None:
        return [0.0] * 4
    tag = atom.GetChiralTag()
    order = [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    ]
    idx = None
    for i, item in enumerate(order):
        if tag == item:
            idx = i
            break
    return _one_hot(idx, len(order) + 1)


def _hybridization_index(atom) -> int:
    if Chem is None or atom is None:
        return 0
    hyb = atom.GetHybridization()
    order = [
        Chem.HybridizationType.SP,
        Chem.HybridizationType.SP2,
        Chem.HybridizationType.SP3,
        Chem.HybridizationType.SP3D,
        Chem.HybridizationType.SP3D2,
        Chem.HybridizationType.S,
    ]
    for i, item in enumerate(order, start=1):
        if hyb == item:
            return i
    return 0


def _chirality_index(atom) -> int:
    if Chem is None or atom is None:
        return 0
    tag = atom.GetChiralTag()
    order = [
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER,
    ]
    for i, item in enumerate(order, start=1):
        if tag == item:
            return i
    return 0


def _min_ring_size(atom) -> float:
    if atom is None:
        return 0.0
    for size in (3, 4, 5, 6, 7, 8):
        if atom.IsInRingSize(size):
            return float(size)
    return 0.0


def requires_3d_pos(cfg: GraphFeaturizerConfig) -> bool:
    edge_mode = str(getattr(cfg, "edge_mode", "bond")).lower()
    edge_features = list(getattr(cfg, "edge_features", []) or [])
    uses_distance = any(name in _DIST_FEATURES for name in edge_features)
    angle_features = list(getattr(cfg, "angle_features", []) or [])
    uses_angles = bool(angle_features)
    return edge_mode == "radius" or uses_distance or uses_angles


def gaussian_expansion(distances: np.ndarray, cfg: GaussianExpansionConfig) -> np.ndarray:
    if not cfg.enabled:
        raise GraphFeaturizerError("Gaussian expansion is disabled but was requested.")
    num_centers = int(cfg.num_centers)
    if num_centers <= 0:
        raise GraphFeaturizerError("gaussian_expansion.num_centers must be > 0")
    min_v = float(cfg.min)
    max_v = float(cfg.max)
    if max_v < min_v:
        raise GraphFeaturizerError("gaussian_expansion.max must be >= min")
    centers = np.linspace(min_v, max_v, num_centers, dtype=np.float32)
    sigma = cfg.resolved_sigma()
    if sigma <= 0:
        raise GraphFeaturizerError("gaussian_expansion.sigma must be > 0")
    gamma = 1.0 / (2.0 * sigma * sigma)
    distances = np.asarray(distances, dtype=np.float32).reshape(-1, 1)
    diff = distances - centers.reshape(1, -1)
    return np.exp(-gamma * diff * diff).astype(np.float32)


def build_radius_graph(pos: np.ndarray, cutoff: float, max_neighbors: int) -> Tuple[np.ndarray, np.ndarray]:
    pos = np.asarray(pos, dtype=np.float32)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise GraphFeaturizerError(f"pos must have shape (N, 3); got {pos.shape}")
    cutoff = float(cutoff)
    if cutoff <= 0:
        raise GraphFeaturizerError("radius_cutoff must be > 0")
    max_neighbors = int(max_neighbors)
    if max_neighbors <= 0:
        max_neighbors = 0

    n = int(pos.shape[0])
    edge_index = []
    distances = []
    for i in range(n):
        diffs = pos - pos[i]
        dist = np.linalg.norm(diffs, axis=1)
        neighbors = np.where((dist > 0) & (dist <= cutoff))[0]
        if max_neighbors > 0 and neighbors.size > max_neighbors:
            order = np.argsort(dist[neighbors])[:max_neighbors]
            neighbors = neighbors[order]
        for j in neighbors.tolist():
            edge_index.append([i, int(j)])
            distances.append(float(dist[j]))

    if not edge_index:
        return np.empty((2, 0), dtype=np.int64), np.array([], dtype=np.float32)
    return np.array(edge_index, dtype=np.int64).T, np.array(distances, dtype=np.float32)


def _edge_feature_dim(cfg: GraphFeaturizerConfig) -> int:
    dim = 0
    for name in cfg.edge_features:
        if name == "bond_type":
            dim += 4  # single/double/triple/aromatic
        elif name in {"conjugated", "aromatic"}:
            dim += 1
        elif name == "distance":
            dim += 1
        elif name == "gaussian":
            if not cfg.gaussian_expansion.enabled:
                raise GraphFeaturizerError("Gaussian expansion requested but gaussian_expansion.enabled is False.")
            dim += int(cfg.gaussian_expansion.num_centers)
        else:
            raise GraphFeaturizerError(f"Unknown edge feature: {name}")
    return dim


def _prepare_gaussian(cfg: GaussianExpansionConfig) -> Tuple[np.ndarray, float]:
    if not cfg.enabled:
        raise GraphFeaturizerError("Gaussian expansion requested but disabled.")
    num_centers = int(cfg.num_centers)
    if num_centers <= 0:
        raise GraphFeaturizerError("gaussian_expansion.num_centers must be > 0")
    min_v = float(cfg.min)
    max_v = float(cfg.max)
    if max_v < min_v:
        raise GraphFeaturizerError("gaussian_expansion.max must be >= min")
    centers = np.linspace(min_v, max_v, num_centers, dtype=np.float32)
    sigma = cfg.resolved_sigma()
    if sigma <= 0:
        raise GraphFeaturizerError("gaussian_expansion.sigma must be > 0")
    gamma = 1.0 / (2.0 * sigma * sigma)
    return centers, gamma


def _angle_feature_dim(angle_features: Sequence[str]) -> int:
    dim = 0
    for name in angle_features:
        if name in _ANGLE_FEATURES:
            dim += 1
        else:
            raise GraphFeaturizerError(f"Unknown angle feature: {name}")
    return dim


def _compute_angle_features(
    pos: np.ndarray, edge_index: np.ndarray, angle_features: Sequence[str]
) -> Tuple[np.ndarray, np.ndarray]:
    num_nodes = int(pos.shape[0])
    neighbors: List[List[int]] = [[] for _ in range(num_nodes)]
    if edge_index.size > 0:
        for idx in range(edge_index.shape[1]):
            src = int(edge_index[0, idx])
            dst = int(edge_index[1, idx])
            neighbors[dst].append(src)

    angle_index: List[List[int]] = []
    angle_attr: List[List[float]] = []
    for center, srcs in enumerate(neighbors):
        if len(srcs) < 2:
            continue
        center_pos = pos[center]
        vecs = pos[np.array(srcs, dtype=np.int64)] - center_pos
        norms = np.linalg.norm(vecs, axis=1)
        for i_idx, src_i in enumerate(srcs):
            norm_i = float(norms[i_idx])
            if norm_i == 0.0:
                continue
            v_i = vecs[i_idx]
            for k_idx, src_k in enumerate(srcs):
                if i_idx == k_idx:
                    continue
                norm_k = float(norms[k_idx])
                if norm_k == 0.0:
                    continue
                v_k = vecs[k_idx]
                cos_val = float(np.dot(v_i, v_k) / (norm_i * norm_k))
                cos_val = float(np.clip(cos_val, -1.0, 1.0))
                feat_vals: List[float] = []
                for name in angle_features:
                    if name == "angle":
                        feat_vals.append(float(np.arccos(cos_val)))
                    elif name == "cos_angle":
                        feat_vals.append(cos_val)
                    else:
                        raise GraphFeaturizerError(f"Unknown angle feature: {name}")
                angle_index.append([src_i, center, src_k])
                angle_attr.append(feat_vals)

    if not angle_index:
        return np.empty((3, 0), dtype=np.int64), np.empty((0, len(angle_features)), dtype=np.float32)
    return np.array(angle_index, dtype=np.int64).T, np.array(angle_attr, dtype=np.float32)


def _resolve_pos(mol, pos: Optional[np.ndarray], require_pos: bool) -> Optional[np.ndarray]:
    pos_arr = None
    if pos is not None:
        pos_arr = np.asarray(pos, dtype=np.float32)
        if pos_arr.shape != (mol.GetNumAtoms(), 3):
            raise GraphFeaturizerError(f"pos shape mismatch: {pos_arr.shape} vs ({mol.GetNumAtoms()}, 3)")
    else:
        conf = mol.GetConformer() if mol.GetNumConformers() > 0 else None
        if conf is not None and conf.Is3D():
            coords = []
            for k in range(mol.GetNumAtoms()):
                p = conf.GetAtomPosition(k)
                coords.append([p.x, p.y, p.z])
            pos_arr = np.array(coords, dtype=np.float32)
    if require_pos and pos_arr is None:
        raise GraphFeaturizerError("3D positions are required but missing for this molecule.")
    return pos_arr


def _edge_feature_vector(
    edge_features: Sequence[str],
    bond,
    distance: Optional[float],
    centers: Optional[np.ndarray],
    gamma: Optional[float],
) -> List[float]:
    feats: List[float] = []
    for name in edge_features:
        if name == "bond_type":
            if bond is None:
                raise GraphFeaturizerError("bond features requested but bond data is unavailable.")
            btype = bond.GetBondType()
            feats.extend(
                [
                    float(btype == Chem.rdchem.BondType.SINGLE),
                    float(btype == Chem.rdchem.BondType.DOUBLE),
                    float(btype == Chem.rdchem.BondType.TRIPLE),
                    float(btype == Chem.rdchem.BondType.AROMATIC),
                ]
            )
        elif name == "conjugated":
            if bond is None:
                raise GraphFeaturizerError("bond features requested but bond data is unavailable.")
            feats.append(float(bond.GetIsConjugated()))
        elif name == "aromatic":
            if bond is None:
                raise GraphFeaturizerError("bond features requested but bond data is unavailable.")
            feats.append(float(bond.GetIsAromatic()))
        elif name == "distance":
            if distance is None:
                raise GraphFeaturizerError("distance features requested but distance is unavailable.")
            feats.append(float(distance))
        elif name == "gaussian":
            if distance is None or centers is None or gamma is None:
                raise GraphFeaturizerError("gaussian features requested but distance is unavailable.")
            diff = float(distance) - centers
            feats.extend(np.exp(-gamma * diff * diff).astype(np.float32).tolist())
        else:
            raise GraphFeaturizerError(f"Unknown edge feature: {name}")
    return feats


def _to_1d_tensor(values, dtype) -> "torch.Tensor":
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    elif arr.ndim > 1:
        arr = arr.reshape(-1)
    return torch.tensor(arr, dtype=dtype)


def featurize_mol_to_pyg(
    mol,
    y: Optional[Sequence[float]],
    cfg: GraphFeaturizerConfig,
    pos: Optional[np.ndarray] = None,
    mask_y: Optional[Sequence[float]] = None,
) -> "Data":
    """Convert an RDKit Mol to PyG Data."""
    _require_pyg()
    if Chem is None:
        raise ImportError("RDKit is required for reading SDF and building graphs.")
    if mol is None:
        raise GraphFeaturizerError("mol is None")

    # Node features
    x_list: List[List[float]] = []
    z_list: List[int] = []
    needs_partial_charge = any(
        name in {"partial_charge", "partial_charge_mask"} for name in (cfg.node_features or [])
    )
    charge_values = None
    charge_mask = None
    if needs_partial_charge:
        charge_values, charge_mask = _compute_gasteiger_charges(mol)
    for atom_idx, atom in enumerate(mol.GetAtoms()):
        feats: List[float] = []
        z_list.append(int(atom.GetAtomicNum()))
        for name in cfg.node_features:
            if name == "atomic_num":
                feats.append(float(atom.GetAtomicNum()))
            elif name == "atomic_num_onehot":
                feats.extend(_atomic_num_onehot(int(atom.GetAtomicNum()), cfg))
            elif name == "degree":
                feats.append(float(atom.GetDegree()))
            elif name == "formal_charge":
                feats.append(float(atom.GetFormalCharge()))
            elif name == "aromatic":
                feats.append(float(atom.GetIsAromatic()))
            elif name == "hybridization":
                feats.append(float(_hybridization_index(atom)))
            elif name == "hybridization_onehot":
                feats.extend(_hybridization_onehot(atom))
            elif name == "implicit_valence":
                feats.append(float(atom.GetImplicitValence()))
            elif name == "chirality":
                feats.append(float(_chirality_index(atom)))
            elif name == "chirality_onehot":
                feats.extend(_chirality_onehot(atom))
            elif name == "num_h":
                feats.append(float(atom.GetTotalNumHs(includeNeighbors=True)))
            elif name == "in_ring":
                feats.append(float(atom.IsInRing()))
            elif name == "ring_size":
                feats.append(float(_min_ring_size(atom)))
            elif name == "partial_charge":
                if charge_values is None or atom_idx >= len(charge_values):
                    feats.append(0.0)
                else:
                    feats.append(float(charge_values[atom_idx]))
            elif name == "partial_charge_mask":
                if charge_mask is None or atom_idx >= len(charge_mask):
                    feats.append(0.0)
                else:
                    feats.append(float(charge_mask[atom_idx]))
            else:
                raise GraphFeaturizerError(f"Unknown node feature: {name}")
        x_list.append(feats)

    x = torch.tensor(np.array(x_list, dtype=np.float32), dtype=torch.float32)
    z = torch.tensor(np.array(z_list, dtype=np.int64), dtype=torch.long)

    edge_features = list(cfg.edge_features or [])
    unknown = [name for name in edge_features if name not in _ALLOWED_EDGE_FEATURES]
    if unknown:
        raise GraphFeaturizerError(f"Unknown edge features: {unknown}")
    angle_features = list(getattr(cfg, "angle_features", []) or [])
    unknown_angles = [name for name in angle_features if name not in _ANGLE_FEATURES]
    if unknown_angles:
        raise GraphFeaturizerError(f"Unknown angle features: {unknown_angles}")
    if angle_features and not cfg.use_3d_pos:
        raise GraphFeaturizerError("angle_features require use_3d_pos=True.")
    edge_mode = str(cfg.edge_mode).lower()
    if edge_mode not in {"bond", "radius"}:
        raise GraphFeaturizerError(f"Unknown edge_mode: {edge_mode}")
    if edge_mode == "radius" and any(name in _BOND_FEATURES for name in edge_features):
        raise GraphFeaturizerError("bond edge features are not supported with edge_mode=radius")

    require_pos = requires_3d_pos(cfg)
    pos_arr = _resolve_pos(mol, pos, require_pos=require_pos)
    centers = None
    gamma = None
    if "gaussian" in edge_features:
        centers, gamma = _prepare_gaussian(cfg.gaussian_expansion)

    # Edges
    edge_index = []
    edge_attr_list: List[List[float]] = []
    if edge_mode == "bond":
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            dist = None
            if any(name in _DIST_FEATURES for name in edge_features):
                if pos_arr is None:
                    raise GraphFeaturizerError("distance features require 3D positions.")
                dist = float(np.linalg.norm(pos_arr[i] - pos_arr[j]))
            # undirected => add both directions
            for (u, v) in [(i, j), (j, i)]:
                edge_index.append([u, v])
                edge_attr_list.append(_edge_feature_vector(edge_features, bond, dist, centers, gamma))
    else:
        if pos_arr is None:
            raise GraphFeaturizerError("edge_mode=radius requires 3D positions.")
        edge_idx, distances = build_radius_graph(pos_arr, cfg.radius_cutoff, cfg.max_neighbors)
        if edge_idx.shape[1] > 0:
            for idx in range(edge_idx.shape[1]):
                edge_index.append([int(edge_idx[0, idx]), int(edge_idx[1, idx])])
                dist = float(distances[idx]) if distances.size > 0 else None
                edge_attr_list.append(_edge_feature_vector(edge_features, None, dist, centers, gamma))

    edge_attr_dim = _edge_feature_dim(cfg)
    if len(edge_index) == 0:
        edge_index_arr = np.empty((2, 0), dtype=np.int64)
        edge_index_t = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, edge_attr_dim), dtype=torch.float32)
    else:
        edge_index_arr = np.array(edge_index, dtype=np.int64).T
        edge_index_t = torch.tensor(edge_index_arr, dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_attr_list, dtype=np.float32), dtype=torch.float32)

    # 3D positions
    pos_t = None
    if cfg.use_3d_pos and pos_arr is not None:
        pos_t = torch.tensor(pos_arr, dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index_t, edge_attr=edge_attr)
    data.z = z
    if pos_t is not None:
        data.pos = pos_t
    if y is not None:
        data.y = _to_1d_tensor(y, dtype=torch.float32)
        if mask_y is not None:
            mask_t = _to_1d_tensor(mask_y, dtype=torch.float32)
            if mask_t.shape != data.y.shape:
                raise GraphFeaturizerError("mask_y shape does not match y.")
            data.mask_y = mask_t

    # Global descriptors
    if cfg.add_global_descriptors:
        desc = calc_descriptors(mol, cfg.add_global_descriptors)
        u = descriptors_to_array(desc, cfg.add_global_descriptors).astype(np.float32)
        data.u = torch.tensor(u.reshape(1, -1), dtype=torch.float32)
        data.u_names = cfg.add_global_descriptors

    if angle_features:
        if pos_arr is None:
            raise GraphFeaturizerError("angle features require 3D positions.")
        angle_index, angle_attr = _compute_angle_features(pos_arr, edge_index_arr, angle_features)
        data.angle_index = torch.tensor(angle_index, dtype=torch.long)
        data.angles = torch.tensor(angle_attr, dtype=torch.float32)

    return data
