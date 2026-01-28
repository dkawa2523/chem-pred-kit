from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
except Exception:  # pragma: no cover
    Chem = None
    Descriptors = None
    Crippen = None
    rdMolDescriptors = None


_SUPPORTED = {
    "MolWt": lambda m: Descriptors.MolWt(m),
    "HeavyAtomCount": lambda m: Descriptors.HeavyAtomCount(m),
    "TPSA": lambda m: rdMolDescriptors.CalcTPSA(m),
    "MolLogP": lambda m: Crippen.MolLogP(m),
    "HBD": lambda m: rdMolDescriptors.CalcNumHBD(m),
    "HBA": lambda m: rdMolDescriptors.CalcNumHBA(m),
    "NumRotatableBonds": lambda m: rdMolDescriptors.CalcNumRotatableBonds(m),
    "RingCount": lambda m: rdMolDescriptors.CalcNumRings(m),
    "AromaticRings": lambda m: rdMolDescriptors.CalcNumAromaticRings(m),
}


def supported_descriptors() -> List[str]:
    return sorted(_SUPPORTED.keys())


def calc_descriptors(mol, names: Sequence[str]) -> Dict[str, float]:
    if Chem is None:
        raise ImportError("RDKit is required for descriptor calculation.")
    out: Dict[str, float] = {}
    for name in names:
        if name not in _SUPPORTED:
            raise ValueError(f"Unsupported descriptor: {name}. Supported: {supported_descriptors()}")
        try:
            out[name] = float(_SUPPORTED[name](mol))
        except Exception:
            out[name] = float("nan")
    return out


def descriptors_to_array(desc: Dict[str, float], names: Sequence[str]) -> np.ndarray:
    return np.array([desc.get(n, float("nan")) for n in names], dtype=float)
