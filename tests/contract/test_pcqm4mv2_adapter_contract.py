from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.common.io import load_sdf_mol, sdf_path_from_cas
from src.common.utils import get_logger
from src.data.adapters import pcqm4mv2 as pcqm4mv2_adapter
from src.data.adapters.registry import create_dataset_adapter
from src.utils.artifacts import load_meta, validate_dataset_artifacts

from tests.fixtures.pcqm4mv2_dummy import DummyPCQM4Mv2Dataset
from tests.fixtures.pcqm4mv2_helpers import PCQM4MV2_TARGET_REGISTRY, build_pcqm4mv2_dataset_cfg


def test_pcqm4mv2_adapter_builds_dataset_artifacts(tmp_path: Path) -> None:
    pytest.importorskip("rdkit")

    root = tmp_path / "pcqm4mv2_root"
    root.mkdir(parents=True, exist_ok=True)
    dataset_run_dir = tmp_path / "pcqm4mv2_run"

    DummyPCQM4Mv2Dataset.n_rows = 64

    original_dataset = pcqm4mv2_adapter.PCQM4Mv2Dataset
    pcqm4mv2_adapter.PCQM4Mv2Dataset = DummyPCQM4Mv2Dataset
    try:
        cfg = build_pcqm4mv2_dataset_cfg(dataset_run_dir, root, n_rows=64)
        adapter = create_dataset_adapter("pcqm4mv2")
        logger = get_logger("test_pcqm4mv2_adapter")
        artifacts = adapter.build_processed(cfg, logger)
    finally:
        pcqm4mv2_adapter.PCQM4Mv2Dataset = original_dataset

    df = pd.read_csv(artifacts.dataset_csv)
    assert "target.gap" in df.columns

    sample_id = df["sample_id"].astype(str).iloc[0]
    mol = load_sdf_mol(sdf_path_from_cas(cfg["paths"]["sdf_dir"], sample_id))
    assert mol is not None

    validate_dataset_artifacts(artifacts.meta_path.parent)
    meta = load_meta(artifacts.meta_path.parent)
    assert meta["dataset_hash"] == artifacts.dataset_hash
    assert meta["dataset_name"] == "pcqm4mv2"
    assert meta["target_names"] == ["gap"]
    assert meta["units"]["gap"] == PCQM4MV2_TARGET_REGISTRY[0]["units"]
