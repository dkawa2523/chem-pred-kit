from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.common.io import load_sdf_mol, sdf_path_from_cas
from src.common.utils import get_logger
from src.data.adapters import moleculenet as moleculenet_adapter
from src.data.adapters.registry import create_dataset_adapter
from src.utils.artifacts import load_meta, validate_dataset_artifacts

from tests.fixtures.moleculenet_dummy import DummyMoleculeNetDataset
from tests.fixtures.moleculenet_helpers import (
    MOLECULENET_PYG_NAMES,
    MOLECULENET_TARGET_REGISTRY,
    build_moleculenet_dataset_cfg,
)


@pytest.mark.parametrize("dataset_key", ["esol", "freesolv", "lipo"])
def test_moleculenet_adapter_builds_dataset_artifacts(tmp_path: Path, dataset_key: str) -> None:
    pytest.importorskip("rdkit")

    root = tmp_path / "moleculenet_root"
    root.mkdir(parents=True, exist_ok=True)
    raw_dir = root / MOLECULENET_PYG_NAMES[dataset_key] / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    dataset_run_dir = tmp_path / f"moleculenet_{dataset_key}"

    DummyMoleculeNetDataset.n_rows = 64

    original_moleculenet = moleculenet_adapter.MoleculeNet
    moleculenet_adapter.MoleculeNet = DummyMoleculeNetDataset
    try:
        cfg = build_moleculenet_dataset_cfg(dataset_run_dir, root, dataset_key=dataset_key, n_rows=64)
        adapter = create_dataset_adapter("moleculenet")
        logger = get_logger(f"test_moleculenet_adapter_{dataset_key}")
        artifacts = adapter.build_processed(cfg, logger)
    finally:
        moleculenet_adapter.MoleculeNet = original_moleculenet

    df = pd.read_csv(artifacts.dataset_csv)
    assert f"target.{dataset_key}" in df.columns

    sample_id = df["sample_id"].astype(str).iloc[0]
    mol = load_sdf_mol(sdf_path_from_cas(cfg["paths"]["sdf_dir"], sample_id))
    assert mol is not None

    validate_dataset_artifacts(artifacts.meta_path.parent)
    meta = load_meta(artifacts.meta_path.parent)
    assert meta["dataset_hash"] == artifacts.dataset_hash
    assert meta["dataset_variant"] == dataset_key
    assert meta["target_names"] == [dataset_key]
    assert meta["units"][dataset_key] == MOLECULENET_TARGET_REGISTRY[dataset_key][0]["units"]
