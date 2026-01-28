from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.fixtures.qm9_dummy import DummyQM9Dataset
from tests.fixtures.qm9_helpers import build_qm9_dataset_cfg

from src.common.utils import get_logger
from src.data.adapters.registry import create_dataset_adapter
from src.utils.artifacts import (
    compute_dataset_hash,
    validate_dataset_artifacts,
    validate_evaluate_artifacts,
    validate_train_artifacts,
)


def test_qm9_artifacts_contract(qm9_smoke_artifacts) -> None:
    validate_dataset_artifacts(qm9_smoke_artifacts["dataset_run_dir"])
    validate_train_artifacts(qm9_smoke_artifacts["train_run_dir"])
    validate_evaluate_artifacts(qm9_smoke_artifacts["eval_run_dir"])


def test_qm9_split_reproducible(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("rdkit")

    from src.data.adapters import qm9 as qm9_adapter

    DummyQM9Dataset.n_rows = 256

    qm9_root = tmp_path / "qm9_root"
    (qm9_root / "raw").mkdir(parents=True, exist_ok=True)
    (qm9_root / "raw" / "gdb9.sdf").write_text("", encoding="utf-8")

    monkeypatch.setattr(qm9_adapter, "QM9", DummyQM9Dataset)

    def _build(run_dir: Path) -> tuple[str | None, dict]:
        cfg = build_qm9_dataset_cfg(run_dir, qm9_root)
        adapter = create_dataset_adapter("qm9")
        logger = get_logger(f"qm9_split_{run_dir.name}", log_file=run_dir / "build_dataset.log")
        artifacts = adapter.build_processed(cfg, logger)
        dataset_hash = compute_dataset_hash(artifacts.dataset_csv, artifacts.indices_dir)
        split_json = json.loads((artifacts.indices_dir / "split.json").read_text(encoding="utf-8"))
        return dataset_hash, split_json

    hash_a, split_a = _build(tmp_path / "run_a")
    hash_b, split_b = _build(tmp_path / "run_b")

    assert hash_a == hash_b
    assert split_a == split_b
