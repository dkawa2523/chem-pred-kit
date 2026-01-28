from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.common.config import dump_yaml, load_config
from src.common.metrics import build_metrics_by_target, build_overall_by_split, merge_metrics, uncertainty_metrics
from src.common.meta import build_meta, save_meta
from src.common.splitters import load_split_indices
from src.common.utils import ensure_dir, get_logger, save_json
from src.common.uncertainty import normalize_uncertainty_cfg, resolve_model_artifact_dirs
from src.fp.feature_utils import hash_cfg
from src.featuresets.base import TabularFeatureSet
from src.featuresets.registry import load_featureset
from src.tasks import resolve_task
from src.targets.registry import resolve_target_names, resolve_target_units
from src.targets.transform import build_target_transforms, load_target_transforms, summarize_target_transforms, TargetTransform
from src.targets.derived import apply_derived_numpy, build_derived_context, resolve_derived_targets
from src.utils.artifacts import compute_dataset_hash, load_meta, resolve_training_context
from src.utils.validate_config import validate_config


def run(cfg: Dict[str, Any]) -> Path:
    validate_config(cfg)

    model_dirs = resolve_model_artifact_dirs(cfg)
    if not model_dirs:
        raise ValueError("model_artifact_dir(s) missing in config.")
    model_artifact_dir = Path(model_dirs[0])
    artifacts_dir = model_artifact_dir / "artifacts"
    if not artifacts_dir.exists():
        raise FileNotFoundError(f"Artifacts dir not found: {artifacts_dir}")

    train_cfgs: List[Dict[str, Any]] = []
    for model_dir in model_dirs:
        train_cfg_path = Path(model_dir) / "config_snapshot.yaml"
        if not train_cfg_path.exists():
            raise FileNotFoundError(f"config_snapshot.yaml not found in model dir: {train_cfg_path}")
        train_cfgs.append(load_config(train_cfg_path))
    train_cfg = train_cfgs[0]

    data_cfg = train_cfg.get("data", {})
    data_override = cfg.get("data", {})
    dataset_csv = Path(data_override.get("dataset_csv", data_cfg.get("dataset_csv", "data/processed/dataset_with_lj.csv")))
    indices_dir = Path(data_override.get("indices_dir", data_cfg.get("indices_dir", "data/processed/indices")))
    sdf_dir = Path(data_override.get("sdf_dir", data_cfg.get("sdf_dir", "data/raw/sdf_files")))
    cas_col = str(data_override.get("cas_col", data_cfg.get("cas_col", "CAS")))

    task_spec = resolve_task(train_cfg)
    target_col = task_spec.primary_target()
    if target_col is None:
        raise ValueError("No target column resolved from task/data config.")
    target_names = resolve_target_names(train_cfg, [target_col])
    target_name = target_names[0]
    target_units = resolve_target_units(train_cfg, target_names)
    derived_specs = resolve_derived_targets(task_spec, train_cfg, target_names)
    derived_context = build_derived_context(train_cfg, target_names)

    out_cfg = cfg.get("output", {})
    run_dir_root = Path(out_cfg.get("run_dir", "runs/evaluate"))
    experiment_cfg = cfg.get("experiment", {})
    exp_name = str(out_cfg.get("exp_name", experiment_cfg.get("name", "fp_evaluate")))
    run_dir = ensure_dir(run_dir_root / exp_name)
    logger = get_logger("fp_evaluate", log_file=run_dir / "evaluate.log")

    dump_yaml(run_dir / "config.yaml", cfg)
    uncertainty_cfg = normalize_uncertainty_cfg(cfg)
    uncertainty_method = uncertainty_cfg["method"]
    if not uncertainty_method and len(model_dirs) > 1:
        uncertainty_method = "ensemble"
    if uncertainty_method == "mc_dropout":
        raise ValueError("mc_dropout uncertainty is not supported for fingerprint models.")
    if uncertainty_method == "ensemble" and len(model_dirs) < 2:
        logger.warning("ensemble uncertainty requested with <2 models; disabling uncertainty.")
        uncertainty_method = ""

    if not dataset_csv.exists():
        raise FileNotFoundError(f"dataset_csv not found: {dataset_csv}")
    if not indices_dir.exists():
        raise FileNotFoundError(f"indices_dir not found: {indices_dir}")
    if not sdf_dir.exists():
        raise FileNotFoundError(f"sdf_dir not found: {sdf_dir}")

    train_metas = [load_meta(Path(model_dir)) for model_dir in model_dirs]
    train_meta = train_metas[0]
    train_context = resolve_training_context(train_cfg, train_meta, model_artifact_dir)
    dataset_hash = train_context.get("dataset_hash") or compute_dataset_hash(dataset_csv, indices_dir)
    transforms_loaded = load_target_transforms(artifacts_dir / "target_transform.json")
    if not transforms_loaded:
        transforms_loaded = build_target_transforms(train_cfg, target_names)
    transform_summary = summarize_target_transforms(transforms_loaded, include_state=False)
    transform_state_summary = summarize_target_transforms(transforms_loaded, include_state=True)

    if uncertainty_method == "ensemble" and len(model_dirs) > 1:
        ref_model_name = train_context.get("model_name")
        ref_featureset_name = train_context.get("featureset_name")
        ref_featureset_hash = train_context.get("featureset_hash")
        ref_model_cfg = train_cfg.get("model", {}) or {}
        ref_task_name = train_context.get("task_name")
        for model_dir, other_cfg, other_meta in zip(model_dirs[1:], train_cfgs[1:], train_metas[1:]):
            other_context = resolve_training_context(other_cfg, other_meta, Path(model_dir))
            if ref_task_name and other_context.get("task_name") and other_context.get("task_name") != ref_task_name:
                raise ValueError("Ensemble task_name mismatch across model artifacts.")
            if ref_model_name and other_context.get("model_name") and other_context.get("model_name") != ref_model_name:
                raise ValueError("Ensemble model_name mismatch across model artifacts.")
            if ref_featureset_name and other_context.get("featureset_name") and other_context.get("featureset_name") != ref_featureset_name:
                raise ValueError("Ensemble featureset_name mismatch across model artifacts.")
            other_featureset_hash = other_context.get("featureset_hash")
            if ref_featureset_hash and other_featureset_hash and other_featureset_hash != ref_featureset_hash:
                raise ValueError("Ensemble featureset_hash mismatch across model artifacts.")
            other_task = resolve_task(other_cfg)
            other_target = other_task.primary_target()
            if other_target and other_target != target_col:
                raise ValueError("Ensemble target column mismatch across model artifacts.")
            other_model_cfg = other_cfg.get("model", {}) or {}
            if other_model_cfg != ref_model_cfg:
                raise ValueError("Ensemble model config mismatch across model artifacts.")
            other_transforms = load_target_transforms(Path(model_dir) / "artifacts" / "target_transform.json")
            if not other_transforms:
                other_transforms = build_target_transforms(other_cfg, target_names)
            other_state_summary = summarize_target_transforms(other_transforms, include_state=True)
            if other_state_summary != transform_state_summary:
                raise ValueError("Ensemble target_transform mismatch across model artifacts.")

    meta = build_meta(
        process_name=str(cfg.get("process", {}).get("name", "evaluate")),
        cfg=cfg,
        upstream_artifacts=[str(Path(model_dir)) for model_dir in model_dirs],
        dataset_hash=dataset_hash,
        model_version=train_context.get("model_version"),
        extra={
            "task_name": train_context.get("task_name"),
            "model_name": train_context.get("model_name"),
            "featureset_name": train_context.get("featureset_name"),
            "featureset_hash": train_context.get("featureset_hash"),
            "target_names": target_names,
            "units": target_units,
            "target_transform": transform_summary,
            "uncertainty_method": uncertainty_method or None,
            "ensemble_size": int(len(model_dirs)),
        },
    )
    save_meta(run_dir, meta)

    df = pd.read_csv(dataset_csv)
    indices = load_split_indices(indices_dir)

    featureset = load_featureset(artifacts_dir, train_cfg)
    if not isinstance(featureset, TabularFeatureSet):
        raise ValueError("FP evaluation requires a tabular FeatureSet.")
    feat_cfg = train_cfg.get("featurizer", {})

    cache_dir = data_cfg.get("cache_dir", None)
    cache_dir = Path(cache_dir) if cache_dir else None

    cache_key = hash_cfg({"featurizer": feat_cfg, "dataset": str(dataset_csv)})
    X_all, _, _, _ = featureset.build_features(
        df=df,
        sdf_dir=sdf_dir,
        cas_col=cas_col,
        cache_dir=cache_dir,
        cache_key=cache_key,
        logger=logger,
    )

    transform = transforms_loaded.get(target_name, TargetTransform.from_config(None))
    split_payloads: Dict[str, Dict[str, Any]] = {}
    for split_name in ["train", "val", "test"]:
        if split_name not in indices:
            continue
        split_idx = indices[split_name]
        split_df = df.loc[split_idx]
        idx = split_df.index.to_numpy()
        X = X_all[idx]
        y = split_df[target_col].astype(float).to_numpy()
        ids = split_df[cas_col].astype(str).tolist()

        valid_mask = (~np.isnan(X).all(axis=1)) & np.isfinite(y)
        X = X[valid_mask]
        y = y[valid_mask]
        ids = [cid for cid, ok in zip(ids, valid_mask.tolist()) if ok]

        if len(ids) == 0:
            split_payloads[split_name] = {"ids": [], "y": np.array([]), "X": np.zeros((0, X_all.shape[1]))}
            continue

        X = featureset.transform(X)
        split_payloads[split_name] = {"ids": ids, "y": y, "X": X}

    rows = []
    metrics_by_split: Dict[str, Dict[str, float]] = {}

    if uncertainty_method == "ensemble":
        preds_by_split: Dict[str, List[np.ndarray]] = {name: [] for name in split_payloads}
        for model_dir in model_dirs:
            with open(Path(model_dir) / "artifacts" / "model.pkl", "rb") as f:
                model = pickle.load(f)
            for split_name, payload in split_payloads.items():
                X = payload["X"]
                if X.size == 0:
                    preds_by_split[split_name].append(np.array([]))
                    continue
                preds_t = model.predict(X)
                preds = transform.inverse_transform(preds_t)
                preds_by_split[split_name].append(np.asarray(preds, dtype=float).reshape(-1))

        for split_name, payload in split_payloads.items():
            ids = payload["ids"]
            y = payload["y"]
            if len(ids) == 0:
                metrics_by_split[split_name] = {}
                continue
            pred_stack = np.stack(preds_by_split[split_name], axis=0)
            y_pred = np.mean(pred_stack, axis=0)
            y_std = np.std(pred_stack, axis=0)
            if derived_specs:
                y_pred, _ = apply_derived_numpy(y_pred, target_names, derived_specs, context=derived_context)
            metrics = task_spec.metrics_fn(y, y_pred)
            metrics = merge_metrics(
                metrics,
                uncertainty_metrics(
                    y,
                    y_pred,
                    y_std,
                    target_names=target_names,
                    eps=float(uncertainty_cfg["eps"]),
                ),
            )
            metrics_by_split[split_name] = metrics
            for cid, yt, yp, ys in zip(ids, y.tolist(), y_pred.tolist(), y_std.tolist()):
                rows.append(
                    {
                        "sample_id": cid,
                        "y_true": yt,
                        "y_pred": yp,
                        "y_std": ys,
                        "split": split_name,
                    }
                )
    else:
        with open(artifacts_dir / "model.pkl", "rb") as f:
            model = pickle.load(f)
        for split_name, payload in split_payloads.items():
            ids = payload["ids"]
            y = payload["y"]
            X = payload["X"]
            if len(ids) == 0:
                metrics_by_split[split_name] = {}
                continue
            preds_t = model.predict(X)
            preds = transform.inverse_transform(preds_t)
            if derived_specs:
                preds, _ = apply_derived_numpy(preds, target_names, derived_specs, context=derived_context)
            metrics_by_split[split_name] = task_spec.metrics_fn(y, preds)
            for cid, yt, yp in zip(ids, y.tolist(), preds.tolist()):
                rows.append({"sample_id": cid, "y_true": yt, "y_pred": yp, "split": split_name})

    pred_df = pd.DataFrame(rows)
    pred_df["model_name"] = train_context.get("model_name")
    pred_df["model_version"] = train_context.get("model_version")
    pred_df["dataset_hash"] = dataset_hash
    pred_df["run_id"] = meta["run_id"]
    pred_path = run_dir / "predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    logger.info(f"Saved predictions to {pred_path}")

    for split_name, metrics in metrics_by_split.items():
        if not metrics:
            continue
        save_json(run_dir / f"metrics_{split_name}.json", metrics)

    split_counts = pred_df["split"].value_counts().to_dict() if "split" in pred_df.columns else {}
    by_target = build_metrics_by_target(metrics_by_split)
    overall_by_split = build_overall_by_split(metrics_by_split)
    save_json(
        run_dir / "metrics.json",
        {
            "by_split": metrics_by_split,
            "by_target": by_target,
            "overall": overall_by_split,
            "per_target": by_target,
            "n_train": int(split_counts.get("train", 0)),
            "n_val": int(split_counts.get("val", 0)),
            "n_test": int(split_counts.get("test", 0)),
            "target_names": target_names,
            "units": target_units,
            "target_transform": transform_summary,
            "uncertainty_method": uncertainty_method or None,
            "ensemble_size": int(len(model_dirs)),
            "uncertainty_eps": float(uncertainty_cfg["eps"]) if uncertainty_method else None,
        },
    )

    logger.info("Done.")
    return run_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate fingerprint model and write predictions/metrics.")
    ap.add_argument("--config", required=True, help="Path to configs/fp/evaluate.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    run(cfg)
