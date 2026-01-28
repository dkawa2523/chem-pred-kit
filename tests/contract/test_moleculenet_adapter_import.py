from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from src.common.utils import get_logger
from src.data.adapters.registry import create_dataset_adapter


def test_moleculenet_adapter_requires_torch_geometric(tmp_path: Path) -> None:
    if importlib.util.find_spec("torch_geometric") is not None:
        pytest.skip("torch_geometric is installed; skip import error test.")

    cfg = {
        "process": {"name": "build_dataset"},
        "paths": {
            "raw_csv": str(tmp_path / "esol.csv"),
            "sdf_dir": str(tmp_path / "sdf"),
            "out_csv": str(tmp_path / "dataset.csv"),
            "out_indices_dir": str(tmp_path / "indices"),
        },
        "columns": {"sample_id": "sample_id", "cas": "sample_id", "smiles": "smiles"},
        "split": {"method": "random", "seed": 42, "fractions": [0.8, 0.1, 0.1]},
        "moleculenet": {"root": str(tmp_path / "moleculenet_root"), "allow_download": False},
        "dataset": {"name": "moleculenet", "dataset_name": "esol"},
    }

    adapter = create_dataset_adapter("moleculenet")
    logger = get_logger("test_moleculenet_adapter_import")
    with pytest.raises(ImportError):
        adapter.build_processed(cfg, logger)
