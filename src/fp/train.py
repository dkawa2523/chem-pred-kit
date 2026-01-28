from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.common.ad import applicability_domain
from src.common.config import dump_yaml, load_config
from src.common.feature_pipeline import FingerprintFeaturePipeline
from src.common.meta import build_meta, save_meta
from src.common.metrics import build_metrics_by_target, build_overall_by_split
from src.common.io import load_sdf_mol, read_csv, sdf_path_from_cas
from src.common.plots import (
    save_abs_error_cdf,
    save_bland_altman_plot,
    save_parity_plot,
    save_residual_hist,
    save_residual_plot,
    save_hist,
)
from src.common.splitters import load_split_indices
from src.common.utils import ensure_dir, get_logger, save_json, set_seed
from src.fp.feature_utils import hash_cfg
from src.fp.featurizer_fp import morgan_bitvect
from src.featuresets.base import TabularFeatureSet
from src.featuresets.registry import create_featureset
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
    safe_inverse_transform,
    save_target_transforms,
    summarize_target_transforms,
)
from src.utils.artifacts import compute_dataset_hash
from src.utils.validate_config import validate_config


def run(cfg: Dict[str, Any]) -> Path:
    validate_config(cfg)
    seed = int(cfg.get("train", {}).get("seed", 42))
    set_seed(seed)

    data_cfg = cfg.get("data", {})
    dataset_csv = Path(data_cfg.get("dataset_csv", "data/processed/dataset_with_lj.csv"))
    indices_dir = Path(data_cfg.get("indices_dir", "data/processed/indices"))
    sdf_dir = Path(data_cfg.get("sdf_dir", "data/raw/sdf_files"))
    task_spec = resolve_task(cfg)
    target_col = task_spec.primary_target()
    if target_col is None:
        raise ValueError("No target column resolved from task/data config.")
    target_names = resolve_target_names(cfg, [target_col])
    target_name = target_names[0]
    target_units = resolve_target_units(cfg, target_names)
    transform_map = build_target_transforms(cfg, target_names)
    transform = transform_map[target_name]
    transform_summary = summarize_target_transforms(transform_map, include_state=False)
    cas_col = str(data_cfg.get("cas_col", "CAS"))
    cache_dir = data_cfg.get("cache_dir", None)
    cache_dir = Path(cache_dir) if cache_dir else None

    out_cfg = cfg.get("output", {})
    run_dir_root = Path(out_cfg.get("run_dir", "runs/train/fp"))
    experiment_cfg = cfg.get("experiment", {})
    exp_name = str(out_cfg.get("exp_name", experiment_cfg.get("name", "fp_experiment")))
    run_dir = ensure_dir(run_dir_root / exp_name)
    plots_dir = ensure_dir(run_dir / "plots")

    logger = get_logger("fp_train", log_file=run_dir / "train.log")

    if not dataset_csv.exists():
        raise FileNotFoundError(f"dataset_csv not found: {dataset_csv}")
    if not indices_dir.exists():
        raise FileNotFoundError(f"indices_dir not found: {indices_dir}")
    if not sdf_dir.exists():
        raise FileNotFoundError(f"sdf_dir not found: {sdf_dir}")

    dump_yaml(run_dir / "config.yaml", cfg)
    dataset_hash = compute_dataset_hash(dataset_csv, indices_dir)
    model_version = str(out_cfg.get("model_version", exp_name))
    meta = build_meta(
        process_name=str(cfg.get("process", {}).get("name", "train")),
        cfg=cfg,
        dataset_hash=dataset_hash,
        model_version=model_version,
        extra={
            "target_names": target_names,
            "units": target_units,
            "target_transform": transform_summary,
        },
    )
    save_meta(run_dir, meta)

    df = read_csv(dataset_csv)
    indices = load_split_indices(indices_dir)
    for k in ["train", "val", "test"]:
        if k not in indices:
            raise FileNotFoundError(f"Split indices missing: {indices_dir}/{k}.txt")

    df_train = df.loc[indices["train"]].copy()
    df_val = df.loc[indices["val"]].copy()
    df_test = df.loc[indices["test"]].copy()
    logger.info(f"Split sizes: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")

    feat_cfg = cfg.get("featurizer", {})
    featureset = create_featureset(cfg)
    if not isinstance(featureset, TabularFeatureSet):
        raise ValueError("FP training requires a tabular FeatureSet.")
    pipeline = featureset.pipeline

    model_cfg = cfg.get("model", {}) or {}
    model_name = resolve_model_name(cfg, default="lightgbm")
    model_spec = get_model_spec(model_name)
    feature_inputs = feature_inputs_from_featureset(featureset)
    validate_model_requirements(model_spec, task_spec.target_columns, feature_inputs)
    model_params = model_cfg.get("params", {}) or {}

    # Build features for all rows once (for caching + AD)
    cache_key = hash_cfg({"featurizer": feat_cfg, "dataset": str(dataset_csv)})
    X_all, _, _, _ = featureset.build_features(
        df=df,
        sdf_dir=sdf_dir,
        cas_col=cas_col,
        cache_dir=cache_dir,
        cache_key=cache_key,
        logger=logger,
    )

    # Build masks for valid feature rows
    missing_mask = np.isnan(X_all).all(axis=1)
    if np.any(missing_mask):
        logger.warning(f"Found {int(np.sum(missing_mask))} rows with missing/invalid SDF; they will be ignored in training/eval.")

    # Utility to pick split arrays
    def pick(split_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str], List[Any]]:
        idx = split_df.index.to_numpy()
        X = X_all[idx]
        y = split_df[target_col].astype(float).to_numpy()
        ids = split_df[cas_col].astype(str).tolist()
        elements = split_df.get("elements", pd.Series([""] * len(split_df))).astype(str).tolist()
        # filter invalid
        m = (~np.isnan(X).all(axis=1)) & np.isfinite(y)
        X = X[m]
        y = y[m]
        ids = [i for i, ok in zip(ids, m.tolist()) if ok]
        elements = [e for e, ok in zip(elements, m.tolist()) if ok]
        return X, y, elements, ids

    X_train, y_train, el_train, ids_train = pick(df_train)
    X_val, y_val, el_val, ids_val = pick(df_val)
    X_test, y_test, el_test, ids_test = pick(df_test)

    logger.info(f"After filtering invalid rows: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")
    if len(y_train) < 50:
        logger.warning("Training set is very small; consider loosening selectors or split strategy.")

    y_train_raw = y_train.copy()
    y_val_raw = y_val.copy()
    y_test_raw = y_test.copy()
    transform.fit(y_train_raw)
    transform_state_summary = summarize_target_transforms(transform_map, include_state=True)
    meta["target_transform_state"] = transform_state_summary
    save_meta(run_dir, meta)
    y_train = transform.transform(y_train_raw)
    y_val = transform.transform(y_val_raw)
    y_test = transform.transform(y_test_raw)

    # Impute NaNs (descriptors may yield NaN) + optional standardization
    featureset.fit(X_train)
    X_train = featureset.transform(X_train)
    X_val = featureset.transform(X_val)
    X_test = featureset.transform(X_test)

    model = create_model(model_name, model_cfg=model_cfg, context={})
    logger.info(f"Training model: {model_name} with params={model_params}")
    fit_kwargs = {}
    # Add eval_set if supported
    if model_name.lower() in {"lightgbm", "lgbm"}:
        fit_kwargs = {"eval_set": [(X_val, y_val)], "eval_metric": "rmse"}
    elif model_name.lower() == "catboost":
        fit_kwargs = {"eval_set": (X_val, y_val), "use_best_model": True}

    model.fit(X_train, y_train, **fit_kwargs)

    # Predictions
    pred_val_t = model.predict(X_val)
    pred_test_t = model.predict(X_test)
    pred_val = safe_inverse_transform(pred_val_t, transform)
    pred_test = safe_inverse_transform(pred_test_t, transform)

    metrics_val = task_spec.metrics_fn(y_val_raw, pred_val)
    metrics_test = task_spec.metrics_fn(y_test_raw, pred_test)
    logger.info(f"Val metrics: {metrics_val}")
    logger.info(f"Test metrics: {metrics_test}")

    # Plots
    if bool(out_cfg.get("plots", True)):
        save_parity_plot(y_val_raw, pred_val, plots_dir / "parity_val.png", title="Parity (val)", xlabel="true", ylabel="pred")
        save_residual_plot(y_val_raw, pred_val, plots_dir / "residual_val.png", title="Residual (val)")
        save_residual_hist(y_val_raw, pred_val, plots_dir / "residual_hist_val.png", title="Residual distribution (val)")
        save_abs_error_cdf(y_val_raw, pred_val, plots_dir / "abs_error_cdf_val.png", title="Absolute error CDF (val)")
        save_bland_altman_plot(y_val_raw, pred_val, plots_dir / "bland_altman_val.png", title="Bland-Altman (val)")
        save_parity_plot(y_test_raw, pred_test, plots_dir / "parity_test.png", title="Parity (test)", xlabel="true", ylabel="pred")
        save_residual_plot(y_test_raw, pred_test, plots_dir / "residual_test.png", title="Residual (test)")
        save_residual_hist(y_test_raw, pred_test, plots_dir / "residual_hist_test.png", title="Residual distribution (test)")
        save_abs_error_cdf(y_test_raw, pred_test, plots_dir / "abs_error_cdf_test.png", title="Absolute error CDF (test)")
        save_bland_altman_plot(y_test_raw, pred_test, plots_dir / "bland_altman_test.png", title="Bland-Altman (test)")
        target_label = target_name
        unit = target_units.get(target_name)
        if unit:
            target_label = f"{target_label} ({unit})"
        save_hist(y_train_raw, plots_dir / "y_train_hist.png", title="Target distribution (train)", xlabel=target_label)

        # Learning curves if available
        if model_name.lower() in {"lightgbm", "lgbm"} and hasattr(model, "evals_result_"):
            import matplotlib.pyplot as plt

            evals = model.evals_result_
            # keys can be like {'valid_0': {'rmse': [...]}}
            try:
                k0 = list(evals.keys())[0]
                metric_name = list(evals[k0].keys())[0]
                vals = evals[k0][metric_name]
                plt.figure()
                plt.plot(np.arange(1, len(vals) + 1), vals)
                plt.xlabel("iteration")
                plt.ylabel(metric_name)
                if np.all(np.asarray(vals, dtype=float) > 0):
                    plt.yscale("log")
                else:
                    plt.yscale("symlog", linthresh=1e-6)
                plt.title("LightGBM eval metric (val)")
                plt.tight_layout()
                plt.savefig(plots_dir / "learning_curve_val.png", dpi=200)
                plt.close()
            except Exception:
                pass
        if model_name.lower() == "catboost":
            try:
                import matplotlib.pyplot as plt

                evals = model.get_evals_result()
                # Typically {'learn': {'RMSE': [...]}, 'validation': {'RMSE': [...]}}
                if "validation" in evals:
                    metric_name = list(evals["validation"].keys())[0]
                    vals = evals["validation"][metric_name]
                    plt.figure()
                    plt.plot(np.arange(1, len(vals) + 1), vals)
                    plt.xlabel("iteration")
                    plt.ylabel(metric_name)
                    if np.all(np.asarray(vals, dtype=float) > 0):
                        plt.yscale("log")
                    else:
                        plt.yscale("symlog", linthresh=1e-6)
                    plt.title("CatBoost eval metric (val)")
                    plt.tight_layout()
                    plt.savefig(plots_dir / "learning_curve_val.png", dpi=200)
                    plt.close()
            except Exception:
                pass

    # Save artifacts
    artifacts_dir = ensure_dir(run_dir / "artifacts")
    featureset_hash = featureset.save(artifacts_dir)
    save_target_transforms(artifacts_dir / "target_transform.json", transform_map)
    model_path = artifacts_dir / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # AD artifacts: training fingerprints and training elements
    # Compute Morgan bitvect for all training mols in ORIGINAL dataset order to simplify indexing.
    # We'll store only training set for speed.
    ad_cfg = cfg.get("ad", {}) or {}
    if isinstance(pipeline, FingerprintFeaturePipeline):
        ad_radius = pipeline.fp_cfg.morgan_radius
        ad_n_bits = pipeline.fp_cfg.n_bits
    else:
        ad_radius = int(ad_cfg.get("morgan_radius", 2))
        ad_n_bits = int(ad_cfg.get("n_bits", 2048))

    train_mols = [load_sdf_mol(sdf_path_from_cas(sdf_dir, cas)) for cas in ids_train]
    train_fps = [
        morgan_bitvect(m, radius=ad_radius, n_bits=ad_n_bits) if m is not None else None for m in train_mols
    ]
    # Filter None
    train_pairs = [(fp, cas) for fp, cas in zip(train_fps, ids_train) if fp is not None]
    train_fps = [p[0] for p in train_pairs]
    train_ids_for_ad = [p[1] for p in train_pairs]

    training_elements = sorted({el for e_str in el_train for el in e_str.split(",") if el})
    heavy_atoms_train = df_train["n_heavy_atoms"].dropna().astype(int).tolist()
    heavy_min = int(min(heavy_atoms_train)) if heavy_atoms_train else 0
    heavy_max = int(max(heavy_atoms_train)) if heavy_atoms_train else 0

    ad_artifact = {
        "training_elements": training_elements,
        "heavy_atom_range": [heavy_min, heavy_max],
        "morgan_radius": ad_radius,
        "n_bits": ad_n_bits,
        "train_ids": train_ids_for_ad,
        "train_fps": train_fps,  # RDKit ExplicitBitVect list (pickleable)
        "tanimoto_warn_threshold": float(ad_cfg.get("tanimoto_warn_threshold", 0.5)),
        "top_k": int(ad_cfg.get("top_k", 5)),
    }
    with open(artifacts_dir / "ad.pkl", "wb") as f:
        pickle.dump(ad_artifact, f)

    # Save metrics + config snapshot
    save_json(run_dir / "metrics_val.json", metrics_val)
    save_json(run_dir / "metrics_test.json", metrics_test)
    split_metrics = {"val": metrics_val, "test": metrics_test}
    by_target = build_metrics_by_target(split_metrics)
    overall_by_split = build_overall_by_split(split_metrics)
    save_json(
        run_dir / "metrics.json",
        {
            "val": metrics_val,
            "test": metrics_test,
            "by_target": by_target,
            "overall": overall_by_split,
            "per_target": by_target,
            "n_train": int(len(y_train)),
            "n_val": int(len(y_val)),
            "n_test": int(len(y_test)),
            "seed": int(seed),
            "target_names": target_names,
            "units": target_units,
            "target_transform": transform_summary,
            "target_transform_state": transform_state_summary,
        },
    )
    model_dir = ensure_dir(run_dir / "model")
    with open(model_dir / "model.ckpt", "wb") as f:
        pickle.dump(model, f)
    with open(model_dir / "preprocess.pkl", "wb") as f:
        pickle.dump(
            {
                "imputer": pipeline.imputer,
                "scaler": pipeline.scaler,
                "standardize": pipeline.standardize,
                "impute_strategy": pipeline.impute_strategy,
            },
            f,
        )
    save_json(model_dir / "featurizer_state.json", pipeline.featurizer_state())
    dump_yaml(run_dir / "config_snapshot.yaml", cfg)

    meta["featureset_hash"] = featureset_hash
    save_meta(run_dir, meta)

    logger.info(f"Saved model to {model_path}")
    logger.info("Done.")
    return run_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Train fingerprint-based regression model for LJ parameter.")
    ap.add_argument("--config", required=True, help="Path to configs/fp/train.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
