from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.common.descriptors import calc_descriptors, descriptors_to_array

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem, MACCSkeys, rdMolDescriptors
except Exception:  # pragma: no cover
    Chem = None
    DataStructs = None
    AllChem = None
    MACCSkeys = None
    rdMolDescriptors = None


class FeaturizerError(ValueError):
    pass


@dataclass
class FPConfig:
    fingerprint: str = "morgan"
    morgan_radius: int = 2
    n_bits: int = 2048
    use_counts: bool = False  # if True, use count-based Morgan
    add_descriptors: Optional[List[str]] = None


def featurize_mol(mol, cfg: FPConfig) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Return (feature_vector, meta) for a single RDKit Mol."""
    if Chem is None:
        raise ImportError("RDKit is required for fingerprint featurization.")
    if mol is None:
        raise FeaturizerError("mol is None")

    fp_name = cfg.fingerprint.lower()
    if fp_name == "morgan":
        if cfg.use_counts:
            fp = rdMolDescriptors.GetMorganFingerprint(mol, cfg.morgan_radius)
            # Convert SparseIntVect to fixed array by hashing into n_bits (simple)
            arr = np.zeros((cfg.n_bits,), dtype=float)
            for idx, v in fp.GetNonzeroElements().items():
                arr[idx % cfg.n_bits] += float(v)
        else:
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, cfg.morgan_radius, nBits=cfg.n_bits)
            arr = np.zeros((cfg.n_bits,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            arr = arr.astype(float)
    elif fp_name == "maccs":
        fp = MACCSkeys.GenMACCSKeys(mol)
        arr = np.zeros((fp.GetNumBits(),), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        arr = arr.astype(float)
    elif fp_name == "rdkit":
        fp = Chem.RDKFingerprint(mol)
        arr = np.zeros((fp.GetNumBits(),), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        arr = arr.astype(float)
    else:
        raise FeaturizerError(f"Unknown fingerprint type: {cfg.fingerprint}")

    meta: Dict[str, Any] = {"fp_type": fp_name, "fp_dim": int(arr.shape[0])}

    if cfg.add_descriptors:
        desc = calc_descriptors(mol, cfg.add_descriptors)
        desc_arr = descriptors_to_array(desc, cfg.add_descriptors)
        arr = np.concatenate([arr, desc_arr.astype(float)], axis=0)
        meta["desc_names"] = list(cfg.add_descriptors)

    return arr.astype(float), meta


def morgan_bitvect(mol, radius: int = 2, n_bits: int = 2048):
    """Return RDKit ExplicitBitVect for similarity (AD)."""
    if rdMolDescriptors is None:
        raise ImportError("RDKit is required.")
    return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
