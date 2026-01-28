from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest

pytestmark = pytest.mark.optional


@pytest.fixture(scope="session")
def qm9_schnet_smoke_artifacts(tmp_path_factory: pytest.TempPathFactory) -> Dict[str, Any]:
    pytest.importorskip("rdkit")
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    pytest.importorskip("torch_cluster")

    try:
        from torch_geometric.nn.models import SchNet as _SchNet
    except Exception:
        try:
            from torch_geometric.nn import SchNet as _SchNet
        except Exception:
            _SchNet = None
    if _SchNet is None:
        pytest.skip("SchNet is unavailable in torch_geometric.")

    from tests.fixtures.qm9_dummy import DummyQM9Dataset
    from tests.fixtures.qm9_helpers import (
        build_qm9_dataset_cfg,
        build_qm9_gnn_eval_cfg,
        build_qm9_gnn_train_cfg,
    )

    from src.common.utils import get_logger
    from src.data.adapters import qm9 as qm9_adapter
    from src.data.adapters.registry import create_dataset_adapter
    from src.gnn import evaluate as gnn_evaluate
    from src.gnn import train as gnn_train

    root = tmp_path_factory.mktemp("qm9_schnet_smoke")
    DummyQM9Dataset.n_rows = 32

    qm9_root = root / "qm9_root"
    (qm9_root / "raw").mkdir(parents=True, exist_ok=True)
    (qm9_root / "raw" / "gdb9.sdf").write_text("", encoding="utf-8")

    dataset_run_dir = root / "runs" / "build_dataset" / "qm9_schnet_smoke"

    original_qm9 = qm9_adapter.QM9
    qm9_adapter.QM9 = DummyQM9Dataset
    try:
        dataset_cfg = build_qm9_dataset_cfg(dataset_run_dir, qm9_root, n_rows=32)
        adapter = create_dataset_adapter("qm9")
        logger = get_logger("qm9_schnet_dataset", log_file=dataset_run_dir / "build_dataset.log")
        artifacts = adapter.build_processed(dataset_cfg, logger)

        dataset_csv = artifacts.dataset_csv
        indices_dir = artifacts.indices_dir
        sdf_dir = Path(dataset_cfg["paths"]["sdf_dir"])

        train_run_root = root / "runs" / "train" / "qm9_schnet"
        train_cfg = build_qm9_gnn_train_cfg(train_run_root, dataset_csv, indices_dir, sdf_dir)
        train_run_dir = gnn_train.run(train_cfg)

        eval_run_root = root / "runs" / "evaluate" / "qm9_schnet"
        eval_cfg = build_qm9_gnn_eval_cfg(eval_run_root, train_run_dir)
        eval_run_dir = gnn_evaluate.run(eval_cfg)
    finally:
        qm9_adapter.QM9 = original_qm9

    return {
        "dataset_run_dir": dataset_run_dir,
        "train_run_dir": train_run_dir,
        "eval_run_dir": eval_run_dir,
    }


def test_qm9_schnet_smoke_artifacts(qm9_schnet_smoke_artifacts) -> None:
    from src.utils.artifacts import validate_evaluate_artifacts, validate_train_artifacts

    validate_train_artifacts(qm9_schnet_smoke_artifacts["train_run_dir"])
    validate_evaluate_artifacts(qm9_schnet_smoke_artifacts["eval_run_dir"])
