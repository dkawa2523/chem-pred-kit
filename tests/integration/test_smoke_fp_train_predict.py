from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from src.common.config import dump_yaml, load_config
from src.utils.artifacts import validate_predict_artifacts, validate_train_artifacts


def test_smoke_fp_train_predict(tmp_path: Path) -> None:
    pytest.importorskip("rdkit")
    pytest.importorskip("sklearn")

    repo_root = Path(__file__).resolve().parents[2]

    train_cfg = load_config(repo_root / "configs/fp/train_fixture.yaml")
    train_run_root = tmp_path / "runs" / "train" / "fp"
    train_cfg["output"]["run_dir"] = str(train_run_root)
    train_cfg["output"]["exp_name"] = "smoke_train"
    train_cfg["output"]["plots"] = False
    train_cfg.setdefault("data", {})["cache_dir"] = str(tmp_path / "cache")

    train_cfg_path = tmp_path / "train_smoke.yaml"
    dump_yaml(train_cfg_path, train_cfg)

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts/train.py"),
            "--config",
            str(train_cfg_path),
        ],
        check=True,
        cwd=repo_root,
    )

    train_run_dir = train_run_root / "smoke_train"
    validate_train_artifacts(train_run_dir)

    predict_cfg = load_config(repo_root / "configs/fp/predict_fixture.yaml")
    predict_cfg["model_artifact_dir"] = str(train_run_dir)
    predict_cfg["output"]["out_dir"] = str(tmp_path / "runs" / "predict")
    predict_cfg["output"]["exp_name"] = "smoke_predict"

    predict_cfg_path = tmp_path / "predict_smoke.yaml"
    dump_yaml(predict_cfg_path, predict_cfg)

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts/predict.py"),
            "--config",
            str(predict_cfg_path),
            "--query",
            "64-17-5",
        ],
        check=True,
        cwd=repo_root,
    )

    predict_run_dir = Path(predict_cfg["output"]["out_dir"]) / "smoke_predict"
    validate_predict_artifacts(predict_run_dir)
