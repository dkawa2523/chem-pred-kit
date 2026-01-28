from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.common.ad import applicability_domain
from src.common.chemistry import get_elements_from_mol
from src.common.config import dump_yaml, load_config
from src.common.meta import build_meta, save_meta
from src.common.io import load_sdf_mol, read_csv, sdf_path_from_cas
from src.common.utils import ensure_dir, get_logger, save_json
from src.common.uncertainty import normalize_uncertainty_cfg, resolve_model_artifact_dirs
from src.fp.featurizer_fp import morgan_bitvect
from src.featuresets.base import TabularFeatureSet
from src.featuresets.registry import load_featureset
from src.tasks import resolve_task
from src.targets.registry import resolve_target_names, resolve_target_units
from src.targets.transform import build_target_transforms, load_target_transforms, summarize_target_transforms, TargetTransform
from src.targets.derived import apply_derived_numpy, build_derived_context, resolve_derived_targets
from src.utils.artifacts import compute_dataset_hash, load_meta, resolve_training_context
from src.utils.validate_config import validate_config


def _resolve_cas(mode: str, query: str, dataset_csv: Path) -> Tuple[str, Dict[str, Any]]:
    meta: Dict[str, Any] = {"mode": mode, "query": query}
    mode = mode.lower()
    if mode == "cas":
        return query.strip(), meta
    if mode == "formula":
        df = read_csv(dataset_csv)
        # exact match
        matches = df[df["MolecularFormula"].astype(str) == query.strip()]
        if len(matches) == 0:
            raise ValueError(f"No CAS found for formula={query} in {dataset_csv}")
        if len(matches) > 1:
            # Ambiguous (isomers). Use first but record.
            meta["warning"] = f"Multiple entries share the same formula. Using first match (n={len(matches)}). Consider using CAS instead."
        cas = str(matches.iloc[0]["CAS"])
        meta["resolved_cas"] = cas
        return cas, meta
    raise ValueError(f"Unknown input mode: {mode}. Use 'cas' or 'formula'.")


def run(cfg: Dict[str, Any], query: str) -> Path:
    validate_config(cfg)
    model_dirs = resolve_model_artifact_dirs(cfg)
    if not model_dirs:
        raise ValueError("model_artifact_dir(s) missing in config.")
    model_artifact_dir = Path(model_dirs[0])
    artifacts_dir = model_artifact_dir / "artifacts"
    if not artifacts_dir.exists():
        raise FileNotFoundError(f"Artifacts dir not found: {artifacts_dir}")

    # Load training snapshot to guarantee feature consistency
    train_cfgs: List[Dict[str, Any]] = []
    for model_dir in model_dirs:
        train_cfg_path = Path(model_dir) / "config_snapshot.yaml"
        if not train_cfg_path.exists():
            raise FileNotFoundError(f"config_snapshot.yaml not found in model dir: {train_cfg_path}")
        train_cfgs.append(load_config(train_cfg_path))
    train_cfg = train_cfgs[0]

    data_cfg = train_cfg.get("data", {})
    sdf_dir = Path(data_cfg.get("sdf_dir", "data/raw/sdf_files"))
    data_override = cfg.get("data", {})
    dataset_csv = Path(
        cfg.get(
            "dataset_csv",
            data_override.get("dataset_csv", data_cfg.get("dataset_csv", "data/processed/dataset_with_lj.csv")),
        )
    )

    output_cfg = cfg.get("output", {})
    experiment_cfg = cfg.get("experiment", {})
    exp_name = str(output_cfg.get("exp_name", experiment_cfg.get("name", "fp_predict")))
    out_dir = ensure_dir(Path(output_cfg.get("out_dir", "runs/predict")) / exp_name)
    logger = get_logger("fp_predict", log_file=out_dir / "predict.log")
    dump_yaml(out_dir / "config.yaml", cfg)
    uncertainty_cfg = normalize_uncertainty_cfg(cfg)
    uncertainty_method = uncertainty_cfg["method"]
    if not uncertainty_method and len(model_dirs) > 1:
        uncertainty_method = "ensemble"
    if uncertainty_method == "mc_dropout":
        raise ValueError("mc_dropout uncertainty is not supported for fingerprint models.")
    if uncertainty_method == "ensemble" and len(model_dirs) < 2:
        logger.warning("ensemble uncertainty requested with <2 models; disabling uncertainty.")
        uncertainty_method = ""
    train_metas = [load_meta(Path(model_dir)) for model_dir in model_dirs]
    train_meta = train_metas[0]
    train_context = resolve_training_context(train_cfg, train_meta, model_artifact_dir)
    dataset_hash = train_context.get("dataset_hash") or compute_dataset_hash(dataset_csv, None)
    task_spec = resolve_task(train_cfg)
    target_col = task_spec.primary_target()
    if target_col is None:
        raise ValueError("No target column resolved from training config.")
    target_names = resolve_target_names(train_cfg, [target_col])
    target_name = target_names[0]
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
        },
    )
    save_meta(out_dir, run_meta)

    mode = str(cfg.get("input", {}).get("mode", "formula"))
    cas, resolve_meta = _resolve_cas(mode=mode, query=query, dataset_csv=dataset_csv)
    logger.info(f"Resolved CAS={cas} from query={query} (mode={mode})")

    mol = load_sdf_mol(sdf_path_from_cas(sdf_dir, cas))
    if mol is None:
        raise FileNotFoundError(f"SDF not found or invalid for CAS={cas} in {sdf_dir}")

    featureset = load_featureset(artifacts_dir, train_cfg)
    if not isinstance(featureset, TabularFeatureSet):
        raise ValueError("FP prediction requires a tabular FeatureSet.")

    x, meta = featureset.transform_mol(mol)
    transform = transforms_loaded.get(target_name, TargetTransform.from_config(None))
    pred_std = None
    if uncertainty_method == "ensemble":
        preds = []
        for model_dir in model_dirs:
            with open(Path(model_dir) / "artifacts" / "model.pkl", "rb") as f:
                model = pickle.load(f)
            pred_t = float(model.predict(x.reshape(1, -1)).reshape(-1)[0])
            preds.append(float(transform.inverse_transform([pred_t])[0]))
        pred_stack = np.asarray(preds, dtype=float)
        pred = float(np.mean(pred_stack))
        pred_std = float(np.std(pred_stack))
    else:
        with open(artifacts_dir / "model.pkl", "rb") as f:
            model = pickle.load(f)
        pred_t = float(model.predict(x.reshape(1, -1)).reshape(-1)[0])
        pred = float(transform.inverse_transform([pred_t])[0])
    if derived_specs:
        pred_arr = np.asarray([pred], dtype=float).reshape(1, -1)
        pred_arr, _ = apply_derived_numpy(pred_arr, target_names, derived_specs, context=derived_context)
        pred = float(pred_arr.reshape(-1)[0])

    row = {
        "sample_id": cas,
        "y_pred": pred,
        "model_name": train_context.get("model_name"),
        "model_version": train_context.get("model_version"),
        "dataset_hash": dataset_hash,
        "run_id": run_meta["run_id"],
    }
    if pred_std is not None:
        row["y_std"] = pred_std
    pred_df = pd.DataFrame([row])
    pred_path = out_dir / "predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    logger.info(f"Saved predictions to {pred_path}")

    # AD
    with open(artifacts_dir / "ad.pkl", "rb") as f:
        ad_artifact = pickle.load(f)

    query_elements = sorted(get_elements_from_mol(mol).keys())
    query_fp = morgan_bitvect(mol, radius=ad_artifact["morgan_radius"], n_bits=ad_artifact["n_bits"])
    ad_res = applicability_domain(
        query_elements=query_elements,
        training_elements=ad_artifact["training_elements"],
        query_fp=query_fp,
        train_fps=ad_artifact["train_fps"],
        train_ids=ad_artifact["train_ids"],
        top_k=int(ad_artifact.get("top_k", 5)),
        tanimoto_warn_threshold=float(ad_artifact.get("tanimoto_warn_threshold", 0.5)),
    )

    # Print user-friendly summary
    print("=" * 70)
    print("LJ parameter prediction (Fingerprint model)")
    print(f"Query: {query} (resolved CAS: {cas})")
    unit = target_units.get(target_name, "")
    unit_suffix = f" {unit}" if unit else ""
    if pred_std is not None:
        print(f"Predicted target: {pred:.6g} +/- {pred_std:.6g}{unit_suffix}")
    else:
        print(f"Predicted target: {pred:.6g}{unit_suffix}")
    if ad_res.max_tanimoto is not None:
        print(f"Nearest-neighbor similarity (Tanimoto): {ad_res.max_tanimoto:.3f}")
    print(f"Trust score: {ad_res.trust_score}/100")
    if ad_res.warnings:
        print("\nWarnings:")
        for w in ad_res.warnings:
            print(f"  - {w}")
    if ad_res.top_k:
        print("\nTop neighbors (similarity, CAS):")
        for sim, cid in ad_res.top_k:
            print(f"  - {sim:.3f}  {cid}")
    print("=" * 70)

    uncertainty_payload = None
    if pred_std is not None:
        uncertainty_payload = {
            "method": uncertainty_method,
            "y_std": pred_std,
            "ensemble_size": int(len(model_dirs)),
        }
    result = {
        "cas": cas,
        "query": query,
        "prediction": pred,
        "target_name": target_name,
        "units": target_units,
        "target_transform": transform_summary,
        "resolve_meta": resolve_meta,
        "ad": ad_res.to_dict(),
        "feature_meta": meta,
        "uncertainty": uncertainty_payload,
    }
    save_json(out_dir / f"prediction_{cas}.json", result)
    logger.info(f"Saved prediction json to {out_dir / f'prediction_{cas}.json'}")
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict LJ parameter using fingerprint model with applicability-domain diagnostics.")
    ap.add_argument("--config", required=True, help="Path to configs/fp/predict.yaml")
    ap.add_argument("--query", required=True, help="CAS (e.g. 71-43-2) or MolecularFormula (Hill) depending on config.input.mode")
    args = ap.parse_args()

    cfg = load_config(args.config)
    run(cfg, args.query)


if __name__ == "__main__":
    main()
