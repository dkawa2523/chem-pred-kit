from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

try:
    from rdkit import Chem
except Exception:  # pragma: no cover
    Chem = None


def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(path))


def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def sdf_path_from_cas(sdf_dir: str | Path, cas: str, suffix: str = ".sdf") -> Path:
    cas = str(cas).strip()
    return Path(sdf_dir) / f"{cas}{suffix}"


def load_sdf_mol(sdf_path: str | Path) -> Optional["Chem.Mol"]:
    """Load the first molecule from an SDF file."""
    if Chem is None:
        raise ImportError("RDKit is required to load SDF files. Please install rdkit.")
    sdf_path = Path(sdf_path)
    if not sdf_path.exists():
        return None
    try:
        suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
        if len(suppl) == 0:
            return None
        mol = suppl[0]
        return mol
    except Exception:
        return None
