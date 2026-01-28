from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from src.common.utils import get_logger
from src.data.adapters.registry import create_dataset_adapter


def test_pcqm4mv2_adapter_requires_ogb(tmp_path: Path) -> None:
    if importlib.util.find_spec("ogb") is not None:
        pytest.skip("ogb is installed; skip import error test.")

    cfg = {
        "process": {"name": "build_dataset"},
        "paths": {
            "raw_csv": str(tmp_path / "pcqm4mv2.csv"),
            "sdf_dir": str(tmp_path / "sdf"),
            "out_csv": str(tmp_path / "dataset.csv"),
            "out_indices_dir": str(tmp_path / "indices"),
        },
        "columns": {"sample_id": "sample_id", "cas": "sample_id", "smiles": "smiles"},
        "split": {"method": "ogb"},
        "pcqm4mv2": {"root": str(tmp_path / "pcqm4mv2_root"), "allow_download": False},
        "dataset": {"name": "pcqm4mv2"},
    }

    adapter = create_dataset_adapter("pcqm4mv2")
    logger = get_logger("test_pcqm4mv2_adapter_import")
    with pytest.raises(ImportError):
        adapter.build_processed(cfg, logger)
