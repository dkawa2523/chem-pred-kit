from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.common.config import dump_yaml, load_config
from src.common.io import read_csv
from src.common.meta import build_meta, save_meta
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
except Exception:  # pragma: no cover
    torch = None


def _require_torch() -> None:
    if torch is None:
        raise ImportError("PyTorch is required for SMILES transformer prediction.")


def _resolve_columns(cfg: Dict[str, Any], data_cfg: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    cas_col = str(data_cfg.get("cas_col", "CAS"))
    columns_cfg = cfg.get("columns", {}) or {}
    sample_id_col = str(data_cfg.get("sample_id_col", columns_cfg.get("sample_id", cas_col)))
    smiles_col = data_cfg.get("smiles_col", columns_cfg.get("smiles", None))
    smiles_col = str(smiles_col) if smiles_col is not None else None
    return sample_id_col, smiles_col


def _resolve_sample(
    mode: str,
    query: str,
    dataset_csv: Optional[Path],
    cas_col: str,
    sample_id_col: str,
    smiles_col: Optional[str],
) -> Tuple[str, str, str, Dict[str, Any]]:
    meta: Dict[str, Any] = {"mode": mode, "query": query}
    mode = mode.lower()
    if mode == "smiles":
        smiles = query.strip()
        if not smiles:
            raise ValueError("Empty SMILES query.")
        return "", smiles, smiles, meta
    if dataset_csv is None or not dataset_csv.exists():
        raise FileNotFoundError(f"dataset_csv not found: {dataset_csv}")
    df = read_csv(dataset_csv)
    if mode == "cas":
        cas = query.strip()
        if cas_col in df.columns:
            matches = df[df[cas_col].astype(str) == cas]
            if len(matches) > 0:
                row = matches.iloc[0]
                sample_id = str(row[sample_id_col]) if sample_id_col in df.columns else cas
                smiles = str(row[smiles_col]) if smiles_col and smiles_col in df.columns and pd.notna(row[smiles_col]) else ""
                meta["resolved_sample_id"] = sample_id
                return cas, sample_id, smiles, meta
        return cas, cas, "", meta
    if mode == "formula":
        matches = df[df["MolecularFormula"].astype(str) == query.strip()]
        if len(matches) == 0:
            raise ValueError(f"No CAS found for formula={query} in {dataset_csv}")
        if len(matches) > 1:
            meta["warning"] = f"Multiple entries share the same formula. Using first match (n={len(matches)}). Consider using CAS instead."
        row = matches.iloc[0]
        cas = str(row[cas_col]) if cas_col in df.columns else str(row.get("CAS", ""))
        sample_id = str(row[sample_id_col]) if sample_id_col in df.columns else cas
        smiles = str(row[smiles_col]) if smiles_col and smiles_col in df.columns and pd.notna(row[smiles_col]) else ""
        meta["resolved_cas"] = cas
        meta["resolved_sample_id"] = sample_id
        return cas, sample_id, smiles, meta
    raise ValueError(f"Unknown input mode: {mode}. Use 'smiles', 'cas', or 'formula'.")


def run(cfg: Dict[str, Any], query: str) -> Path:
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
    dataset_csv = Path(
        cfg.get(
            "dataset_csv",
            data_override.get("dataset_csv", data_cfg.get("dataset_csv", "data/processed/dataset_with_lj.csv")),
        )
    )
    cas_col = str(data_cfg.get("cas_col", "CAS"))
    sample_id_col, smiles_col = _resolve_columns(train_cfg, data_cfg)

    output_cfg = cfg.get("output", {})
    experiment_cfg = cfg.get("experiment", {})
    exp_name = str(output_cfg.get("exp_name", experiment_cfg.get("name", "smiles_predict")))
    out_dir = ensure_dir(Path(output_cfg.get("out_dir", "runs/predict")) / exp_name)
    logger = get_logger("smiles_predict", log_file=out_dir / "predict.log")
    dump_yaml(out_dir / "config.yaml", cfg)
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

    train_metas = [load_meta(Path(model_dir)) for model_dir in model_dirs]
    train_meta = train_metas[0]
    train_context = resolve_training_context(train_cfg, train_meta, model_artifact_dir)
    dataset_hash = train_context.get("dataset_hash") or compute_dataset_hash(dataset_csv, None)
    task_spec = resolve_task(train_cfg)
    target_columns = task_spec.target_columns
    if not target_columns:
        raise ValueError("No target column resolved from training config.")
    target_names = task_spec.target_names or resolve_target_names(train_cfg, target_columns)
    target_units = resolve_target_units(train_cfg, target_names)
    derived_specs = resolve_derived_targets(task_spec, train_cfg, target_names)
    derived_context = build_derived_context(train_cfg, target_names)
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

    run_meta = build_meta(
        process_name=str(cfg.get("process", {}).get("name", "predict")),
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
    save_meta(out_dir, run_meta)

    mode = str(cfg.get("input", {}).get("mode", "smiles"))
    cas, sample_id, smiles, resolve_meta = _resolve_sample(
        mode=mode,
        query=query,
        dataset_csv=dataset_csv if dataset_csv.exists() else None,
        cas_col=cas_col,
        sample_id_col=sample_id_col,
        smiles_col=smiles_col,
    )
    if not smiles:
        raise ValueError(f"SMILES not found for query={query} (mode={mode}).")
    logger.info(f"Resolved sample_id={sample_id} from query={query} (mode={mode})")

    featureset = load_featureset(artifacts_dir, train_cfg)
    if not isinstance(featureset, SmilesFeatureSet):
        raise ValueError("SMILES prediction requires a SMILES FeatureSet.")
    model_cfg = train_cfg.get("model", {}) or {}
    model_name = resolve_model_name(train_cfg, default="chemberta")
    model_spec = get_model_spec(model_name)
    feature_inputs = feature_inputs_from_featureset(featureset)
    validate_model_requirements(model_spec, task_spec.target_columns, feature_inputs)

    tokens = featureset.tokenize_smiles(smiles)
    input_ids = torch.tensor(np.asarray(tokens.get("input_ids"))).long().unsqueeze(0)
    attention_mask = tokens.get("attention_mask")
    attention_mask_t = None
    if attention_mask is not None:
        attention_mask_t = torch.tensor(np.asarray(attention_mask)).long().unsqueeze(0)
    token_type_ids = tokens.get("token_type_ids")
    token_type_ids_t = None
    if token_type_ids is not None:
        token_type_ids_t = torch.tensor(np.asarray(token_type_ids)).long().unsqueeze(0)

    model_context = {"out_dim": len(target_columns), "target_names": target_names}
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    pred_std = None
    if uncertainty_method == "ensemble":
        preds = []
        for model_dir in model_dirs:
            model = create_model(model_name, model_cfg=model_cfg, context=model_context)
            model = model.to(device)
            state_path = Path(model_dir) / "artifacts" / "model_best.pt"
            model.load_state_dict(torch.load(state_path, map_location=device))
            model.eval()
            with torch.no_grad():
                pred_t = model(
                    input_ids=input_ids.to(device),
                    attention_mask=attention_mask_t.to(device) if attention_mask_t is not None else None,
                    token_type_ids=token_type_ids_t.to(device) if token_type_ids_t is not None else None,
                ).detach().cpu().numpy()
            if pred_t.ndim == 1:
                pred_t = pred_t.reshape(1, -1)
            pred = inverse_target_transforms(pred_t, transforms_loaded, target_names)
            if derived_specs:
                pred, _ = apply_derived_numpy(pred, target_names, derived_specs, context=derived_context)
            preds.append(pred.reshape(-1))
        pred_stack = np.stack(preds, axis=0)
        pred_vec = np.mean(pred_stack, axis=0)
        pred_std = np.std(pred_stack, axis=0)
    elif uncertainty_method == "mc_dropout":
        model = create_model(model_name, model_cfg=model_cfg, context=model_context)
        model = model.to(device)
        state_path = artifacts_dir / "model_best.pt"
        model.load_state_dict(torch.load(state_path, map_location=device))
        model.train()
        preds = []
        with torch.no_grad():
            for _ in range(max(1, int(uncertainty_cfg["n_samples"]))):
                pred_t = model(
                    input_ids=input_ids.to(device),
                    attention_mask=attention_mask_t.to(device) if attention_mask_t is not None else None,
                    token_type_ids=token_type_ids_t.to(device) if token_type_ids_t is not None else None,
                ).detach().cpu().numpy()
                if pred_t.ndim == 1:
                    pred_t = pred_t.reshape(1, -1)
                pred = inverse_target_transforms(pred_t, transforms_loaded, target_names)
                if derived_specs:
                    pred, _ = apply_derived_numpy(pred, target_names, derived_specs, context=derived_context)
                preds.append(pred.reshape(-1))
        pred_stack = np.stack(preds, axis=0)
        pred_vec = np.mean(pred_stack, axis=0)
        pred_std = np.std(pred_stack, axis=0)
    else:
        model = create_model(model_name, model_cfg=model_cfg, context=model_context)
        model = model.to(device)
        state_path = artifacts_dir / "model_best.pt"
        model.load_state_dict(torch.load(state_path, map_location=device))
        model.eval()
        with torch.no_grad():
            pred_t = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask_t.to(device) if attention_mask_t is not None else None,
                token_type_ids=token_type_ids_t.to(device) if token_type_ids_t is not None else None,
            ).detach().cpu().numpy()
        if pred_t.ndim == 1:
            pred_t = pred_t.reshape(1, -1)
        pred = inverse_target_transforms(pred_t, transforms_loaded, target_names)
        if derived_specs:
            pred, _ = apply_derived_numpy(pred, target_names, derived_specs, context=derived_context)
        pred_vec = pred.reshape(-1)

    predictions = {name: float(value) for name, value in zip(target_names, pred_vec)}
    std_predictions = None
    if pred_std is not None:
        std_predictions = {name: float(value) for name, value in zip(target_names, pred_std)}
    primary_name = target_names[0]
    primary_pred = predictions.get(primary_name, float("nan"))
    primary_std = std_predictions.get(primary_name) if std_predictions else None
    row = {
        "sample_id": sample_id,
        "y_pred": primary_pred,
        "model_name": train_context.get("model_name"),
        "model_version": train_context.get("model_version"),
        "dataset_hash": dataset_hash,
        "run_id": run_meta["run_id"],
    }
    if primary_std is not None:
        row["y_std"] = primary_std
    pred_df = pd.DataFrame([row])
    for name, value in predictions.items():
        pred_df[f"y_pred_{name}"] = value
    if std_predictions is not None:
        for name, value in std_predictions.items():
            pred_df[f"y_std_{name}"] = value
    pred_path = out_dir / "predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    logger.info(f"Saved predictions to {pred_path}")

    print("=" * 70)
    print("Prediction (SMILES transformer)")
    print(f"Query: {query} (sample_id: {sample_id})")
    for name in target_names:
        unit = target_units.get(name, "")
        unit_suffix = f" {unit}" if unit else ""
        value = predictions.get(name, float("nan"))
        if std_predictions and name in std_predictions:
            std_val = std_predictions[name]
            print(f"Predicted {name}: {value:.6g} +/- {std_val:.6g}{unit_suffix}")
        else:
            print(f"Predicted {name}: {value:.6g}{unit_suffix}")
    print("=" * 70)

    uncertainty_payload = None
    if std_predictions is not None:
        uncertainty_payload = {
            "method": uncertainty_method,
            "y_std": primary_std,
            "y_std_by_target": std_predictions,
            "ensemble_size": int(len(model_dirs)),
            "mc_dropout_samples": int(uncertainty_cfg["n_samples"]) if uncertainty_method == "mc_dropout" else None,
        }
    result = {
        "cas": cas,
        "query": query,
        "prediction": primary_pred,
        "predictions": predictions,
        "target_name": primary_name,
        "target_names": target_names,
        "units": target_units,
        "target_transform": transform_summary,
        "resolve_meta": resolve_meta,
        "uncertainty": uncertainty_payload,
    }
    result_path = out_dir / f"prediction_{sample_id}.json"
    save_json(result_path, result)
    logger.info(f"Saved prediction json to {result_path}")
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict with SMILES transformer model.")
    ap.add_argument("--config", required=True, help="Path to configs/smiles/predict.yaml")
    ap.add_argument("--query", required=True, help="SMILES/CAS/formula depending on config.input.mode")
    args = ap.parse_args()

    cfg = load_config(args.config)
    run(cfg, args.query)


if __name__ == "__main__":
    main()
