from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.common.config import dump_yaml, load_config
from src.common.io import read_csv
from src.common.metrics import build_metrics_by_target, build_overall_by_split, merge_metrics, uncertainty_metrics
from src.common.meta import build_meta, save_meta
from src.common.splitters import load_split_indices
from src.common.utils import ensure_dir, get_logger, save_json
from src.common.uncertainty import normalize_uncertainty_cfg, resolve_model_artifact_dirs
from src.featuresets.base import SmilesFeatureSet
from src.featuresets.registry import load_featureset
from src.models import (
    create_model,
    feature_inputs_from_featureset,
    get_model_spec,
    resolve_model_name,
    validate_model_requirements,
)
from src.tasks import resolve_task
from src.targets.registry import resolve_target_names, resolve_target_units
from src.targets.transform import (
    build_target_transforms,
    inverse_target_transforms,
    load_target_transforms,
    summarize_target_transforms,
)
from src.targets.derived import apply_derived_numpy, build_derived_context, resolve_derived_targets
from src.utils.artifacts import compute_dataset_hash, load_meta, resolve_training_context
from src.utils.validate_config import validate_config

try:
    import torch
    from torch.utils.data import DataLoader
except Exception:  # pragma: no cover
    torch = None
    DataLoader = None


def _require_torch() -> None:
    if torch is None or DataLoader is None:
        raise ImportError("PyTorch is required for SMILES transformer evaluation.")


def _select_device(prefer: str) -> "torch.device":
    prefer = str(prefer).lower()
    if prefer in {"auto", ""}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        try:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
        except Exception:
            pass
        return torch.device("cpu")
    if prefer in {"cuda", "gpu"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if prefer == "mps":
        try:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
        except Exception:
            pass
        return torch.device("cpu")
    return torch.device("cpu")


def _resolve_columns(
    cfg: Dict[str, Any],
    data_cfg: Dict[str, Any],
    data_override: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Optional[str]]:
    data_override = data_override or {}
    cas_col = str(data_cfg.get("cas_col", "CAS"))
    columns_cfg = cfg.get("columns", {}) or {}
    sample_id_col = str(data_override.get("sample_id_col", data_cfg.get("sample_id_col", columns_cfg.get("sample_id", cas_col))))
    smiles_col = data_override.get("smiles_col", data_cfg.get("smiles_col", columns_cfg.get("smiles", None)))
    smiles_col = str(smiles_col) if smiles_col is not None else None
    return sample_id_col, smiles_col


def _build_dataset(
    df: pd.DataFrame,
    indices: Dict[str, List[int]],
    split_name: str,
    sample_id_col: str,
    smiles_col: Optional[str],
    target_cols: List[str],
    featureset: SmilesFeatureSet,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    split_idx = indices.get(split_name, [])
    split_df = df.loc[split_idx]
    sample_ids = split_df[sample_id_col].astype(str).tolist() if sample_id_col in split_df.columns else split_df.index.astype(str).tolist()
    if smiles_col and smiles_col in split_df.columns:
        smiles_values = split_df[smiles_col].astype(str).tolist()
    else:
        smiles_values = [""] * len(split_df)
    y_values = split_df[target_cols].to_numpy(dtype=float)
    mask_values = np.isfinite(y_values)

    data_list: List[Dict[str, Any]] = []
    kept_ids: List[str] = []
    for sample_id, smiles, y_row, mask_row in zip(sample_ids, smiles_values, y_values, mask_values):
        smiles = str(smiles).strip()
        if not smiles or not np.any(mask_row):
            continue
        try:
            tokens = featureset.tokenize_smiles(smiles)
        except Exception:
            continue
        item = {
            "input_ids": tokens.get("input_ids"),
            "attention_mask": tokens.get("attention_mask"),
            "token_type_ids": tokens.get("token_type_ids"),
            "y_raw": np.asarray(y_row, dtype=float),
            "mask_y": np.asarray(mask_row, dtype=bool),
            "sample_id": sample_id,
        }
        if item["input_ids"] is None:
            continue
        data_list.append(item)
        kept_ids.append(sample_id)
    return data_list, kept_ids


def _collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    def _stack(key: str, dtype: torch.dtype) -> Optional[torch.Tensor]:
        vals = [item.get(key) for item in batch]
        if any(v is None for v in vals):
            return None
        return torch.tensor(np.stack(vals), dtype=dtype)

    return {
        "input_ids": _stack("input_ids", torch.long),
        "attention_mask": _stack("attention_mask", torch.long),
        "token_type_ids": _stack("token_type_ids", torch.long),
        "y_raw": _stack("y_raw", torch.float32),
        "mask_y": _stack("mask_y", torch.float32),
        "sample_ids": [item["sample_id"] for item in batch],
    }


def run(cfg: Dict[str, Any]) -> Path:
    _require_torch()
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
    sample_id_col, smiles_col = _resolve_columns(train_cfg, data_cfg, data_override=data_override)

    task_spec = resolve_task(train_cfg)
    target_columns = task_spec.target_columns
    if not target_columns:
        raise ValueError("No target column resolved from training config.")
    target_names = task_spec.target_names or resolve_target_names(train_cfg, target_columns)
    target_units = resolve_target_units(train_cfg, target_names)
    derived_specs = resolve_derived_targets(task_spec, train_cfg, target_names)
    derived_context = build_derived_context(train_cfg, target_names)

    out_cfg = cfg.get("output", {})
    run_dir_root = Path(out_cfg.get("run_dir", "runs/evaluate"))
    experiment_cfg = cfg.get("experiment", {})
    exp_name = str(out_cfg.get("exp_name", experiment_cfg.get("name", "smiles_evaluate")))
    run_dir = ensure_dir(run_dir_root / exp_name)
    logger = get_logger("smiles_evaluate", log_file=run_dir / "evaluate.log")
    dump_yaml(run_dir / "config.yaml", cfg)
    uncertainty_cfg = normalize_uncertainty_cfg(cfg)
    uncertainty_method = uncertainty_cfg["method"]
    if not uncertainty_method and len(model_dirs) > 1:
        uncertainty_method = "ensemble"
    if uncertainty_method == "mc_dropout" and len(model_dirs) > 1:
        logger.warning("mc_dropout requested with multiple model dirs; using the first model only.")
        model_dirs = model_dirs[:1]
        train_cfgs = train_cfgs[:1]
    if uncertainty_method == "ensemble" and len(model_dirs) < 2:
        logger.warning("ensemble uncertainty requested with <2 models; disabling uncertainty.")
        uncertainty_method = ""

    if not dataset_csv.exists():
        raise FileNotFoundError(f"dataset_csv not found: {dataset_csv}")
    if not indices_dir.exists():
        raise FileNotFoundError(f"indices_dir not found: {indices_dir}")

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
            other_targets = other_task.target_columns
            if other_targets and other_targets != target_columns:
                raise ValueError("Ensemble target_columns mismatch across model artifacts.")
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
            "mc_dropout_samples": int(uncertainty_cfg["n_samples"]) if uncertainty_method == "mc_dropout" else None,
        },
    )
    save_meta(run_dir, meta)

    df = read_csv(dataset_csv)
    indices = load_split_indices(indices_dir)

    featureset = load_featureset(artifacts_dir, train_cfg)
    if not isinstance(featureset, SmilesFeatureSet):
        raise ValueError("SMILES evaluation requires a SMILES FeatureSet.")

    model_cfg = train_cfg.get("model", {}) or {}
    model_name = resolve_model_name(train_cfg, default="chemberta")
    model_spec = get_model_spec(model_name)
    feature_inputs = feature_inputs_from_featureset(featureset)
    validate_model_requirements(model_spec, task_spec.target_columns, feature_inputs)

    train_data, train_ids = _build_dataset(
        df, indices, "train", sample_id_col, smiles_col, target_columns, featureset
    )
    val_data, val_ids = _build_dataset(
        df, indices, "val", sample_id_col, smiles_col, target_columns, featureset
    )
    test_data, test_ids = _build_dataset(
        df, indices, "test", sample_id_col, smiles_col, target_columns, featureset
    )

    model_context = {"out_dim": len(target_columns), "target_names": target_names}
    eval_cfg = cfg.get("eval", {})
    prefer_device = eval_cfg.get("device", train_cfg.get("train", {}).get("device", "auto"))
    device = _select_device(prefer_device)

    batch_size = int(eval_cfg.get("batch_size", train_cfg.get("train", {}).get("batch_size", 16)))
    loader_kwargs = {"batch_size": batch_size, "shuffle": False, "collate_fn": _collate_batch}

    def eval_loader(model, data_list: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        if not data_list:
            empty = np.zeros((0, len(target_names)))
            return empty, empty, empty, []
        loader = DataLoader(data_list, **loader_kwargs)
        ys, ps, ms, out_ids = [], [], [], []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                token_type_ids = batch.get("token_type_ids")
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(device)
                pred_t = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                ).detach().cpu().numpy()
                pred = inverse_target_transforms(pred_t, transforms_loaded, target_names)
                if derived_specs:
                    pred, _ = apply_derived_numpy(pred, target_names, derived_specs, context=derived_context)
                y_raw = batch["y_raw"].detach().cpu().numpy()
                mask = batch["mask_y"].detach().cpu().numpy().astype(bool)
                ys.append(y_raw)
                ps.append(pred)
                ms.append(mask)
                out_ids.extend(batch.get("sample_ids", []))
        y_true = np.concatenate(ys, axis=0) if ys else np.zeros((0, len(target_names)))
        y_pred = np.concatenate(ps, axis=0) if ps else np.zeros((0, len(target_names)))
        mask_arr = np.concatenate(ms, axis=0) if ms else np.isfinite(y_true)
        return y_true, y_pred, mask_arr, out_ids

    def eval_loader_mc(
        model, data_list: List[Dict[str, Any]], n_samples: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        if not data_list:
            empty = np.zeros((0, len(target_names)))
            return empty, empty, empty, empty, []
        n_samples = max(1, int(n_samples))
        loader = DataLoader(data_list, **loader_kwargs)
        ys, ps_mean, ps_std, ms, out_ids = [], [], [], [], []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                token_type_ids = batch.get("token_type_ids")
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(device)
                preds = []
                for _ in range(n_samples):
                    pred_t = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                    ).detach().cpu().numpy()
                    pred = inverse_target_transforms(pred_t, transforms_loaded, target_names)
                    if derived_specs:
                        pred, _ = apply_derived_numpy(pred, target_names, derived_specs, context=derived_context)
                    preds.append(pred)
                pred_stack = np.stack(preds, axis=0)
                pred_mean = np.mean(pred_stack, axis=0)
                pred_std = np.std(pred_stack, axis=0)
                y_raw = batch["y_raw"].detach().cpu().numpy()
                mask = batch["mask_y"].detach().cpu().numpy().astype(bool)
                ys.append(y_raw)
                ps_mean.append(pred_mean)
                ps_std.append(pred_std)
                ms.append(mask)
                out_ids.extend(batch.get("sample_ids", []))
        y_true = np.concatenate(ys, axis=0) if ys else np.zeros((0, len(target_names)))
        y_pred = np.concatenate(ps_mean, axis=0) if ps_mean else np.zeros((0, len(target_names)))
        y_std = np.concatenate(ps_std, axis=0) if ps_std else np.zeros((0, len(target_names)))
        mask_arr = np.concatenate(ms, axis=0) if ms else np.isfinite(y_true)
        return y_true, y_pred, y_std, mask_arr, out_ids

    rows = []
    metrics_by_split: Dict[str, Dict[str, float]] = {}
    splits = [
        ("train", train_data, train_ids),
        ("val", val_data, val_ids),
        ("test", test_data, test_ids),
    ]

    def append_rows(ids: List[str], split_name: str, y_true, y_pred, mask, y_std=None) -> None:
        y_std_rows = y_std if y_std is not None else [None] * len(ids)
        for cid, yt_row, yp_row, mask_row, ys_row in zip(ids, y_true, y_pred, mask, y_std_rows):
            row = {"sample_id": cid, "split": split_name}
            row["y_true"] = float(yt_row[0]) if mask_row[0] else float("nan")
            row["y_pred"] = float(yp_row[0])
            if ys_row is not None:
                row["y_std"] = float(ys_row[0]) if mask_row[0] else float("nan")
            for name, yt, yp, m in zip(target_names, yt_row, yp_row, mask_row):
                row[f"y_true_{name}"] = float(yt) if m else float("nan")
                row[f"y_pred_{name}"] = float(yp)
            if ys_row is not None:
                for name, ys, m in zip(target_names, ys_row, mask_row):
                    row[f"y_std_{name}"] = float(ys) if m else float("nan")
            rows.append(row)

    if uncertainty_method == "ensemble":
        preds_by_split: Dict[str, List[np.ndarray]] = {name: [] for name, _, _ in splits}
        truth_by_split: Dict[str, np.ndarray] = {}
        mask_by_split: Dict[str, np.ndarray] = {}
        ids_by_split: Dict[str, List[str]] = {}
        for model_dir in model_dirs:
            model = create_model(model_name, model_cfg=model_cfg, context=model_context)
            model = model.to(device)
            state_path = Path(model_dir) / "artifacts" / "model_best.pt"
            model.load_state_dict(torch.load(state_path, map_location=device))
            model.eval()
            for split_name, data_list, _ids in splits:
                y_true, y_pred, mask, out_ids = eval_loader(model, data_list)
                if split_name not in truth_by_split:
                    truth_by_split[split_name] = y_true
                    mask_by_split[split_name] = mask
                    ids_by_split[split_name] = out_ids
                else:
                    if out_ids != ids_by_split[split_name]:
                        raise ValueError("Ensemble evaluation mismatch: sample_id order differs across models.")
                preds_by_split[split_name].append(y_pred)

        for split_name, _data_list, _ids in splits:
            y_true = truth_by_split.get(split_name, np.zeros((0, len(target_names))))
            mask = mask_by_split.get(split_name, np.zeros((0, len(target_names)), dtype=bool))
            out_ids = ids_by_split.get(split_name, [])
            pred_list = preds_by_split.get(split_name, [])
            if len(y_true) == 0 or not pred_list:
                metrics_by_split[split_name] = {}
                continue
            pred_stack = np.stack(pred_list, axis=0)
            y_pred = np.mean(pred_stack, axis=0)
            y_std = np.std(pred_stack, axis=0)
            metrics = task_spec.metrics_fn(y_true, y_pred, mask)
            metrics = merge_metrics(
                metrics,
                uncertainty_metrics(
                    y_true,
                    y_pred,
                    y_std,
                    mask=mask,
                    target_names=target_names,
                    eps=float(uncertainty_cfg["eps"]),
                ),
            )
            metrics_by_split[split_name] = metrics
            append_rows(out_ids, split_name, y_true, y_pred, mask, y_std=y_std)
    else:
        model = create_model(model_name, model_cfg=model_cfg, context=model_context)
        model = model.to(device)
        state_path = artifacts_dir / "model_best.pt"
        model.load_state_dict(torch.load(state_path, map_location=device))
        if uncertainty_method == "mc_dropout":
            model.train()
            for split_name, data_list, _ids in splits:
                y_true, y_pred, y_std, mask, out_ids = eval_loader_mc(
                    model, data_list, uncertainty_cfg["n_samples"]
                )
                if len(y_true) == 0:
                    metrics_by_split[split_name] = {}
                    continue
                metrics = task_spec.metrics_fn(y_true, y_pred, mask)
                metrics = merge_metrics(
                    metrics,
                    uncertainty_metrics(
                        y_true,
                        y_pred,
                        y_std,
                        mask=mask,
                        target_names=target_names,
                        eps=float(uncertainty_cfg["eps"]),
                    ),
                )
                metrics_by_split[split_name] = metrics
                append_rows(out_ids, split_name, y_true, y_pred, mask, y_std=y_std)
        else:
            model.eval()
            for split_name, data_list, _ids in splits:
                y_true, y_pred, mask, out_ids = eval_loader(model, data_list)
                if len(y_true) == 0:
                    metrics_by_split[split_name] = {}
                    continue
                metrics_by_split[split_name] = task_spec.metrics_fn(y_true, y_pred, mask)
                append_rows(out_ids, split_name, y_true, y_pred, mask)

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
            "mc_dropout_samples": int(uncertainty_cfg["n_samples"]) if uncertainty_method == "mc_dropout" else None,
            "uncertainty_eps": float(uncertainty_cfg["eps"]) if uncertainty_method else None,
        },
    )

    logger.info("Done.")
    return run_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate SMILES transformer model.")
    ap.add_argument("--config", required=True, help="Path to configs/smiles/evaluate.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    run(cfg)
