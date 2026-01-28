from __future__ import annotations

from pathlib import Path

from src.utils.artifacts import (
    REQUIRED_META_KEYS,
    build_meta,
    compute_config_hash,
    compute_dataset_hash,
    save_meta,
    validate_evaluate_artifacts,
    validate_predict_artifacts,
    validate_train_artifacts,
)


def _write_indices(indices_dir: Path) -> None:
    indices_dir.mkdir(parents=True, exist_ok=True)
    (indices_dir / "train.txt").write_text("0\n", encoding="utf-8")
    (indices_dir / "val.txt").write_text("0\n", encoding="utf-8")
    (indices_dir / "test.txt").write_text("0\n", encoding="utf-8")


def _write_predictions(path: Path) -> None:
    path.write_text("sample_id,y_pred\nA,1.23\n", encoding="utf-8")


def test_artifact_contract_helpers(tmp_path: Path) -> None:
    dataset_csv = tmp_path / "dataset.csv"
    dataset_csv.write_text("CAS,lj_epsilon_over_k_K\nA,100.0\n", encoding="utf-8")
    indices_dir = tmp_path / "indices"
    _write_indices(indices_dir)

    cfg = {
        "data": {"dataset_csv": str(dataset_csv), "indices_dir": str(indices_dir)},
        "task": {"name": "lj_epsilon"},
        "model": {"name": "lightgbm"},
        "featurizer": {"fingerprint": "morgan", "add_descriptors": ["MolWt"]},
    }

    dataset_hash = compute_dataset_hash(dataset_csv, indices_dir)
    config_hash = compute_config_hash(cfg)
    meta = build_meta(process_name="train", cfg=cfg, dataset_hash=dataset_hash)

    assert dataset_hash is not None
    assert meta["dataset_hash"] == dataset_hash
    assert meta["config_hash"] == config_hash
    assert meta["task_name"] == "lj_epsilon"
    assert meta["model_name"] == "lightgbm"
    assert meta["featureset_name"] == "fp_morgan_desc"
    for key in REQUIRED_META_KEYS:
        assert key in meta

    train_dir = tmp_path / "train_run"
    train_dir.mkdir()
    (train_dir / "config.yaml").write_text("x: 1\n", encoding="utf-8")
    save_meta(train_dir, meta)
    model_dir = train_dir / "model"
    model_dir.mkdir()
    (model_dir / "model.ckpt").write_text("stub", encoding="utf-8")
    (train_dir / "metrics.json").write_text("{}", encoding="utf-8")
    validate_train_artifacts(train_dir)

    predict_dir = tmp_path / "predict_run"
    predict_dir.mkdir()
    (predict_dir / "config.yaml").write_text("x: 1\n", encoding="utf-8")
    save_meta(predict_dir, build_meta(process_name="predict", cfg=cfg, dataset_hash=dataset_hash))
    _write_predictions(predict_dir / "predictions.csv")
    validate_predict_artifacts(predict_dir)

    eval_dir = tmp_path / "evaluate_run"
    eval_dir.mkdir()
    (eval_dir / "config.yaml").write_text("x: 1\n", encoding="utf-8")
    save_meta(eval_dir, build_meta(process_name="evaluate", cfg=cfg, dataset_hash=dataset_hash))
    _write_predictions(eval_dir / "predictions.csv")
    (eval_dir / "metrics.json").write_text("{}", encoding="utf-8")
    validate_evaluate_artifacts(eval_dir)
