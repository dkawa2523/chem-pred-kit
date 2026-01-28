from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

PCQM4MV2_TARGET_REGISTRY: List[Dict[str, Any]] = [
    {"name": "gap", "aliases": ["homo_lumo_gap"], "units": "eV"},
]


def build_pcqm4mv2_dataset_cfg(run_dir: Path, root: Path, n_rows: int = 256) -> Dict[str, Any]:
    dataset_dir = run_dir / "dataset"
    return {
        "process": {"name": "build_dataset"},
        "paths": {
            "raw_csv": str(run_dir / "pcqm4mv2_dummy.csv"),
            "sdf_dir": str(dataset_dir / "sdf"),
            "out_csv": str(dataset_dir / "dataset.csv"),
            "out_indices_dir": str(dataset_dir / "indices"),
        },
        "columns": {
            "sample_id": "sample_id",
            "cas": "sample_id",
            "smiles": "smiles",
        },
        "split": {"method": "ogb", "seed": 42},
        "pcqm4mv2": {
            "root": str(root),
            "allow_download": True,
            "add_hs": True,
            "compute_2d": True,
            "overwrite_sdf": False,
            "write_sdf": True,
            "canonicalize_smiles": True,
            "only_smiles": False,
            "target_name": "homo_lumo_gap",
        },
        "seed": 42,
        "dataset": {"name": "pcqm4mv2"},
        "target_registry": PCQM4MV2_TARGET_REGISTRY,
        "output": {"run_dir": str(run_dir)},
        "limit_rows": int(n_rows),
    }
