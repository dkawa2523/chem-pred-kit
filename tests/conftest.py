from __future__ import annotations

import sys
from typing import Any, Dict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="session")
def qm9_smoke_artifacts(tmp_path_factory: pytest.TempPathFactory) -> Dict[str, Any]:
    pytest.importorskip("rdkit")
    pytest.importorskip("sklearn")

    from tests.fixtures.qm9_dummy import DummyQM9Dataset
    from tests.fixtures.qm9_helpers import (
        build_qm9_dataset_cfg,
        build_qm9_eval_cfg,
        build_qm9_train_cfg,
        run_qm9_featurize,
    )

    from src.data.adapters import qm9 as qm9_adapter
    from src.data.adapters.registry import create_dataset_adapter
    from src.common.utils import get_logger
    from src.fp import evaluate as fp_evaluate
    from src.fp import train as fp_train
    from src.utils.artifacts import compute_dataset_hash

    root = tmp_path_factory.mktemp("qm9_smoke")

    DummyQM9Dataset.n_rows = 256

    qm9_root = root / "qm9_root"
    (qm9_root / "raw").mkdir(parents=True, exist_ok=True)
    (qm9_root / "raw" / "gdb9.sdf").write_text("", encoding="utf-8")

    dataset_run_dir = root / "runs" / "build_dataset" / "qm9_smoke"

    original_qm9 = qm9_adapter.QM9
    qm9_adapter.QM9 = DummyQM9Dataset
    try:
        dataset_cfg = build_qm9_dataset_cfg(dataset_run_dir, qm9_root)
        adapter = create_dataset_adapter("qm9")
        logger = get_logger("qm9_smoke_dataset", log_file=dataset_run_dir / "build_dataset.log")
        artifacts = adapter.build_processed(dataset_cfg, logger)

        dataset_csv = artifacts.dataset_csv
        indices_dir = artifacts.indices_dir
        sdf_dir = Path(dataset_cfg["paths"]["sdf_dir"])

        dataset_hash = compute_dataset_hash(dataset_csv, indices_dir)
        featurize_run_dir = root / "runs" / "featurize" / "qm9_smoke"
        features_dir = featurize_run_dir / "features"

        train_run_root = root / "runs" / "train" / "qm9"
        train_cfg = build_qm9_train_cfg(train_run_root, dataset_csv, indices_dir, sdf_dir, features_dir)
        cache_key = run_qm9_featurize(
            train_cfg,
            dataset_csv,
            sdf_dir,
            features_dir,
            featurize_run_dir,
            dataset_hash,
            dataset_run_dir,
        )

        train_run_dir = fp_train.run(train_cfg)

        eval_run_root = root / "runs" / "evaluate" / "qm9"
        eval_cfg = build_qm9_eval_cfg(eval_run_root, train_run_dir)
        eval_run_dir = fp_evaluate.run(eval_cfg)
    finally:
        qm9_adapter.QM9 = original_qm9

    return {
        "dataset_run_dir": dataset_run_dir,
        "train_run_dir": train_run_dir,
        "eval_run_dir": eval_run_dir,
        "features_dir": features_dir,
        "cache_key": cache_key,
        "dataset_csv": dataset_csv,
        "indices_dir": indices_dir,
        "sdf_dir": sdf_dir,
        "dataset_hash": dataset_hash,
    }
