from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest


@pytest.fixture(scope="session")
def moleculenet_fp_smoke_artifacts(tmp_path_factory: pytest.TempPathFactory) -> Dict[str, Any]:
    pytest.importorskip("rdkit")
    pytest.importorskip("sklearn")

    from tests.fixtures.moleculenet_dummy import DummyMoleculeNetDataset
    from tests.fixtures.moleculenet_helpers import (
        MOLECULENET_PYG_NAMES,
        build_moleculenet_dataset_cfg,
        build_moleculenet_eval_cfg,
        build_moleculenet_train_cfg,
    )

    from src.common.utils import get_logger
    from src.data.adapters import moleculenet as moleculenet_adapter
    from src.data.adapters.registry import create_dataset_adapter
    from src.fp import evaluate as fp_evaluate
    from src.fp import train as fp_train

    dataset_key = "esol"
    root = tmp_path_factory.mktemp("moleculenet_fp_smoke")
    raw_root = root / "moleculenet_root"
    (raw_root / MOLECULENET_PYG_NAMES[dataset_key] / "raw").mkdir(parents=True, exist_ok=True)

    DummyMoleculeNetDataset.n_rows = 128

    dataset_run_dir = root / "runs" / "build_dataset" / "moleculenet_fp_smoke"

    original_moleculenet = moleculenet_adapter.MoleculeNet
    moleculenet_adapter.MoleculeNet = DummyMoleculeNetDataset
    try:
        dataset_cfg = build_moleculenet_dataset_cfg(dataset_run_dir, raw_root, dataset_key=dataset_key, n_rows=128)
        adapter = create_dataset_adapter("moleculenet")
        logger = get_logger("moleculenet_fp_dataset", log_file=dataset_run_dir / "build_dataset.log")
        artifacts = adapter.build_processed(dataset_cfg, logger)

        dataset_csv = artifacts.dataset_csv
        indices_dir = artifacts.indices_dir
        sdf_dir = Path(dataset_cfg["paths"]["sdf_dir"])

        train_run_root = root / "runs" / "train" / "moleculenet_fp"
        train_cfg = build_moleculenet_train_cfg(train_run_root, dataset_csv, indices_dir, sdf_dir, dataset_key)
        train_run_dir = fp_train.run(train_cfg)

        eval_run_root = root / "runs" / "evaluate" / "moleculenet_fp"
        eval_cfg = build_moleculenet_eval_cfg(eval_run_root, train_run_dir, dataset_key)
        eval_run_dir = fp_evaluate.run(eval_cfg)
    finally:
        moleculenet_adapter.MoleculeNet = original_moleculenet

    return {
        "dataset_run_dir": dataset_run_dir,
        "train_run_dir": train_run_dir,
        "eval_run_dir": eval_run_dir,
    }


def test_moleculenet_fp_smoke_artifacts(moleculenet_fp_smoke_artifacts) -> None:
    from src.utils.artifacts import validate_evaluate_artifacts, validate_train_artifacts

    validate_train_artifacts(moleculenet_fp_smoke_artifacts["train_run_dir"])
    validate_evaluate_artifacts(moleculenet_fp_smoke_artifacts["eval_run_dir"])
