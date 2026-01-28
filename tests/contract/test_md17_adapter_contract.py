from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.common.io import load_sdf_mol, sdf_path_from_cas
from src.common.utils import get_logger
from src.data.adapters.registry import create_dataset_adapter
from src.utils.artifacts import load_meta, validate_dataset_artifacts


def _write_raw_csv(path: Path) -> None:
    rows = [
        {
            "sample_id": "md17_0",
            "time": 0,
            "energy": -10.5,
            "forces": [[0.1, 0.0, 0.0], [0.0, -0.1, 0.0], [0.0, 0.0, 0.1]],
            "atomic_numbers": [6, 1, 1],
            "positions": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        },
        {
            "sample_id": "md17_1",
            "time": 1,
            "energy": -10.4,
            "forces": [[0.0, 0.1, 0.0], [0.0, -0.1, 0.1], [0.1, 0.0, 0.0]],
            "atomic_numbers": [6, 1, 1],
            "positions": [[0.0, 0.1, 0.0], [1.1, 0.0, 0.0], [0.0, 1.1, 0.0]],
        },
        {
            "sample_id": "md17_2",
            "time": 2,
            "energy": -10.3,
            "forces": [[0.0, 0.0, 0.1], [0.1, 0.0, -0.1], [0.0, 0.1, 0.0]],
            "atomic_numbers": [6, 1, 1],
            "positions": [[0.0, 0.2, 0.0], [1.2, 0.0, 0.0], [0.0, 1.2, 0.0]],
        },
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def test_md17_adapter_builds_dataset_artifacts(tmp_path: Path) -> None:
    pytest.importorskip("rdkit")

    raw_csv = tmp_path / "md17.csv"
    _write_raw_csv(raw_csv)

    out_dir = tmp_path / "processed"
    out_csv = out_dir / "dataset.csv"
    out_indices_dir = out_dir / "indices"
    sdf_dir = tmp_path / "sdf"

    cfg = {
        "process": {"name": "build_dataset"},
        "target_registry": [
            {"name": "energy", "aliases": ["energy"], "units": "kcal/mol"},
            {"name": "forces", "aliases": ["forces"], "units": "kcal/(mol*angstrom)"},
        ],
        "dataset": {
            "name": "md17",
            "units": {"energy": "kcal/mol", "forces": "kcal/(mol*angstrom)"},
        },
        "paths": {
            "raw_csv": str(raw_csv),
            "sdf_dir": str(sdf_dir),
            "out_csv": str(out_csv),
            "out_indices_dir": str(out_indices_dir),
        },
        "columns": {
            "sample_id": "sample_id",
            "cas": "sample_id",
            "time": "time",
            "energy": "energy",
            "forces": "forces",
            "atomic_numbers": "atomic_numbers",
            "positions": "positions",
        },
        "seed": 123,
        "md17": {"write_sdf": True, "overwrite_sdf": True, "require_3d": True},
        "split": {"method": "time", "time_key": "time", "fractions": [0.6, 0.2, 0.2]},
    }

    logger = get_logger("test_md17_adapter_contract")
    adapter = create_dataset_adapter("md17")
    artifacts = adapter.build_processed(cfg, logger)

    assert artifacts.dataset_csv.exists()
    assert artifacts.indices_dir.exists()
    assert artifacts.dataset_index.exists()
    assert artifacts.meta_path.exists()

    df = pd.read_csv(artifacts.dataset_csv)
    assert "target.energy" in df.columns
    assert "target.forces" in df.columns

    mol = load_sdf_mol(sdf_path_from_cas(sdf_dir, "md17_0"))
    assert mol is not None
    assert mol.GetNumConformers() > 0
    assert mol.GetConformer().Is3D()

    validate_dataset_artifacts(artifacts.meta_path.parent)
    meta = load_meta(artifacts.meta_path.parent)
    assert meta["dataset_hash"] == artifacts.dataset_hash
    assert meta["target_names"] == ["energy", "forces"]
    assert meta["units"]["energy"] == "kcal/mol"
