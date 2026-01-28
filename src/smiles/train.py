from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.common.config import dump_yaml, load_config
from src.common.io import read_csv
from src.common.meta import build_meta, save_meta
from src.common.metrics import build_metrics_by_target, build_overall_by_split
from src.common.plots import (
    save_abs_error_cdf,
    save_bland_altman_plot,
    save_learning_curve,
    save_parity_plot,
    save_residual_hist,
    save_residual_plot,
    save_hist,
)
from src.common.splitters import load_split_indices
from src.common.utils import ensure_dir, get_logger, save_json, set_seed
from src.common.pretrain import (
    apply_pretrain_freeze,
    build_pretrain_metadata,
    collect_trainable_params,
    load_pretrained_weights,
)
from src.featuresets.base import SmilesFeatureSet
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
    apply_target_transforms,
    build_target_transforms,
    fit_target_transforms,
    inverse_target_transforms,
    save_target_transforms,
    summarize_target_transforms,
)
from src.targets.derived import (
    apply_derived_numpy,
    apply_derived_torch,
    build_derived_context,
    resolve_derived_targets,
)
from src.common.losses import masked_multitask_loss, masked_regression_loss, resolve_loss_per_target, resolve_loss_weights
from src.utils.artifacts import compute_dataset_hash
from src.utils.validate_config import validate_config

try:
    import torch
    from torch.optim import Adam
    from torch.utils.data import DataLoader
except Exception:  # pragma: no cover
    torch = None
    Adam = None
    DataLoader = None


def _require_torch() -> None:
    if torch is None or DataLoader is None:
        raise ImportError("PyTorch is required for SMILES transformer training.")


def _select_device(cfg_train: Dict[str, Any]) -> "torch.device":
    prefer = str(cfg_train.get("device", "auto")).lower()
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
    if prefer == "cpu":
        return torch.device("cpu")
    return torch.device("cpu")


def _resolve_columns(cfg: Dict[str, Any], data_cfg: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    cas_col = str(data_cfg.get("cas_col", "CAS"))
    columns_cfg = cfg.get("columns", {}) or {}
    sample_id_col = str(data_cfg.get("sample_id_col", columns_cfg.get("sample_id", cas_col)))
    smiles_col = data_cfg.get("smiles_col", columns_cfg.get("smiles", None))
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
    logger,
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

    logger.info(f"{split_name}: kept {len(data_list)} / {len(split_df)} rows after SMILES filtering.")
    return data_list, kept_ids


def _stack_targets(data_list: List[Dict[str, Any]], n_targets: int) -> Tuple[np.ndarray, np.ndarray]:
    if not data_list:
        return np.zeros((0, n_targets)), np.zeros((0, n_targets), dtype=bool)
    y_raw = np.stack([item["y_raw"] for item in data_list])
    mask = np.stack([item["mask_y"] for item in data_list]).astype(bool)
    return y_raw, mask


def _apply_transform(
    data_list: List[Dict[str, Any]],
    transform_map: Dict[str, Any],
    target_names: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    if not data_list:
        return np.zeros((0, len(target_names))), np.zeros((0, len(target_names)), dtype=bool)
    y_raw, mask = _stack_targets(data_list, len(target_names))
    y_trans = apply_target_transforms(y_raw, transform_map, target_names, mask=mask)
    for item, y_val in zip(data_list, y_trans):
        item["y"] = np.asarray(y_val, dtype=np.float32)
    return y_raw, mask


def _collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    def _stack(key: str, dtype: torch.dtype) -> Optional[torch.Tensor]:
        vals = [item.get(key) for item in batch]
        if any(v is None for v in vals):
            return None
        return torch.tensor(np.stack(vals), dtype=dtype)

    out: Dict[str, Any] = {
        "input_ids": _stack("input_ids", torch.long),
        "attention_mask": _stack("attention_mask", torch.long),
        "token_type_ids": _stack("token_type_ids", torch.long),
        "y": _stack("y", torch.float32),
        "y_raw": _stack("y_raw", torch.float32),
        "mask_y": _stack("mask_y", torch.float32),
        "sample_ids": [item["sample_id"] for item in batch],
    }
    return out


def run(cfg: Dict[str, Any]) -> Path:
    _require_torch()
    validate_config(cfg)
    seed = int(cfg.get("train", {}).get("seed", 42))
    set_seed(seed)

    data_cfg = cfg.get("data", {})
    dataset_csv = Path(data_cfg.get("dataset_csv", "data/processed/dataset_with_lj.csv"))
    indices_dir = Path(data_cfg.get("indices_dir", "data/processed/indices"))
    task_spec = resolve_task(cfg)
    target_columns = task_spec.target_columns
    if not target_columns:
        raise ValueError("No target column resolved from task/data config.")
    target_names = task_spec.target_names or resolve_target_names(cfg, target_columns)
    target_units = resolve_target_units(cfg, target_names)
    derived_specs = resolve_derived_targets(task_spec, cfg, target_names)
    derived_context = build_derived_context(cfg, target_names)
    transform_map = build_target_transforms(cfg, target_names)
    transform_summary = summarize_target_transforms(transform_map, include_state=False)
    sample_id_col, smiles_col = _resolve_columns(cfg, data_cfg)

    out_cfg = cfg.get("output", {})
    run_dir_root = Path(out_cfg.get("run_dir", "runs/train/smiles"))
    experiment_cfg = cfg.get("experiment", {})
    exp_name = str(out_cfg.get("exp_name", experiment_cfg.get("name", "smiles_experiment")))
    run_dir = ensure_dir(run_dir_root / exp_name)
    plots_dir = ensure_dir(run_dir / "plots")
    artifacts_dir = ensure_dir(run_dir / "artifacts")
    logger = get_logger("smiles_train", log_file=run_dir / "train.log")

    if not dataset_csv.exists():
        raise FileNotFoundError(f"dataset_csv not found: {dataset_csv}")
    if not indices_dir.exists():
        raise FileNotFoundError(f"indices_dir not found: {indices_dir}")

    dump_yaml(run_dir / "config.yaml", cfg)

    loss_weights = resolve_loss_weights(cfg, target_names, logger=logger)
    loss_weights_by_target = dict(zip(target_names, loss_weights)) if loss_weights else None
    if loss_weights_by_target:
        logger.info("Using loss weights: %s", loss_weights_by_target)

    pretrain_cfg = cfg.get("pretrain", {}) or {}
    pretrain_meta = build_pretrain_metadata(pretrain_cfg, base_dir=Path.cwd())
    upstream_artifacts: List[str] = []
    pretrain_fields: Dict[str, Any] = {}
    if pretrain_meta:
        pretrain_fields = pretrain_meta.meta_fields()
        if pretrain_meta.upstream_artifact_dir:
            upstream_artifacts.append(str(pretrain_meta.upstream_artifact_dir))
        elif pretrain_meta.upstream_run_id:
            upstream_artifacts.append(str(pretrain_meta.upstream_run_id))
        else:
            upstream_artifacts.append(str(pretrain_meta.ckpt_path))

    df = read_csv(dataset_csv)
    indices = load_split_indices(indices_dir)

    dataset_hash = compute_dataset_hash(dataset_csv, indices_dir)
    model_version = str(out_cfg.get("model_version", exp_name))
    meta_extra = {
        "target_names": target_names,
        "units": target_units,
        "target_transform": transform_summary,
    }
    if loss_weights_by_target:
        meta_extra["loss_weights"] = loss_weights_by_target
    if pretrain_fields:
        meta_extra.update(pretrain_fields)
    meta = build_meta(
        process_name=str(cfg.get("process", {}).get("name", "train")),
        cfg=cfg,
        upstream_artifacts=upstream_artifacts,
        dataset_hash=dataset_hash,
        model_version=model_version,
        extra=meta_extra,
    )
    save_meta(run_dir, meta)

    featureset = create_featureset(cfg)
    if not isinstance(featureset, SmilesFeatureSet):
        raise ValueError("SMILES training requires a SMILES FeatureSet.")

    model_cfg = cfg.get("model", {}) or {}
    model_name = resolve_model_name(cfg, default="chemberta")
    model_spec = get_model_spec(model_name)
    feature_inputs = feature_inputs_from_featureset(featureset)
    validate_model_requirements(model_spec, task_spec.target_columns, feature_inputs)

    train_data, _ = _build_dataset(
        df, indices, "train", sample_id_col, smiles_col, target_columns, featureset, logger
    )
    val_data, _ = _build_dataset(
        df, indices, "val", sample_id_col, smiles_col, target_columns, featureset, logger
    )
    test_data, _ = _build_dataset(
        df, indices, "test", sample_id_col, smiles_col, target_columns, featureset, logger
    )

    if len(train_data) == 0:
        raise RuntimeError("No valid SMILES rows were created for training.")

    y_train_raw, mask_train = _stack_targets(train_data, len(target_names))
    fit_target_transforms(transform_map, y_train_raw, target_names, mask=mask_train)
    transform_state_summary = summarize_target_transforms(transform_map, include_state=True)
    meta["target_transform_state"] = transform_state_summary
    save_meta(run_dir, meta)
    y_train_raw, mask_train = _apply_transform(train_data, transform_map, target_names)
    _apply_transform(val_data, transform_map, target_names)
    _apply_transform(test_data, transform_map, target_names)

    model = create_model(
        model_name,
        model_cfg=model_cfg,
        context={"out_dim": len(target_columns), "target_names": target_names},
    )

    train_cfg = cfg.get("train", {})
    device = _select_device(train_cfg)
    num_threads = train_cfg.get("num_threads", None)
    if num_threads is not None:
        try:
            torch.set_num_threads(int(num_threads))
        except Exception:
            pass
    pretrain_load_report = None
    pretrain_freeze_report = None
    if pretrain_meta:
        pretrain_load_report = load_pretrained_weights(model, pretrain_meta, pretrain_cfg=pretrain_cfg, logger=logger)
        pretrain_freeze_report = apply_pretrain_freeze(model, pretrain_cfg, logger=logger)
        meta["pretrain_load"] = pretrain_load_report.to_meta()
        meta["pretrain_freeze"] = pretrain_freeze_report.to_meta()
        save_meta(run_dir, meta)

    total_params = int(sum(p.numel() for p in model.parameters()))
    trainable_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    logger.info(
        "Model params: total=%.2fM trainable=%.2fM (~%.1f MB fp32, excluding optimizer state)",
        total_params / 1e6,
        trainable_params / 1e6,
        (trainable_params * 4) / 1e6,
    )
    logger.info(f"torch={torch.__version__} device={device} num_threads={torch.get_num_threads()}")
    model = model.to(device)

    epochs = int(train_cfg.get("epochs", 5))
    batch_size = int(train_cfg.get("batch_size", 16))
    lr = float(train_cfg.get("lr", 1e-5))
    weight_decay = float(train_cfg.get("weight_decay", 1e-5))
    loss_name = task_spec.loss_name
    loss_per_target = resolve_loss_per_target(cfg.get("task", {}), target_names, loss_name, logger=logger)
    if loss_name not in {"mse", "huber"}:
        raise ValueError(f"Unknown loss: {loss_name}")
    loss_weights_tensor = None
    if loss_weights is not None:
        loss_weights_tensor = torch.tensor(loss_weights, dtype=torch.float32, device=device)

    def _build_optimizer() -> Adam:
        trainable = collect_trainable_params(model)
        if not trainable:
            raise ValueError("No trainable parameters remain after pretrain freeze.")
        return Adam(trainable, lr=lr, weight_decay=weight_decay)

    freeze_encoder_epochs = int(train_cfg.get("freeze_encoder_epochs", 0) or 0)
    frozen_encoder_params: List[torch.nn.Parameter] = []
    if freeze_encoder_epochs > 0:
        if hasattr(model, "encoder"):
            for param in model.encoder.parameters():
                if param.requires_grad:
                    param.requires_grad = False
                    frozen_encoder_params.append(param)
            if frozen_encoder_params:
                logger.info(
                    "Froze %d encoder params for %d epochs.",
                    len(frozen_encoder_params),
                    freeze_encoder_epochs,
                )
            else:
                logger.info(
                    "freeze_encoder_epochs=%d but encoder had no trainable params.",
                    freeze_encoder_epochs,
                )
        else:
            logger.warning("freeze_encoder_epochs set but model has no encoder; ignoring.")

    opt = _build_optimizer()

    num_workers = int(train_cfg.get("num_workers", 0))
    pin_memory = bool(train_cfg.get("pin_memory", device.type == "cuda"))
    persistent_workers = bool(train_cfg.get("persistent_workers", False)) if num_workers > 0 else False
    loader_kwargs: Dict[str, Any] = {"num_workers": num_workers, "pin_memory": pin_memory, "persistent_workers": persistent_workers}
    if num_workers > 0 and "prefetch_factor" in train_cfg:
        loader_kwargs["prefetch_factor"] = int(train_cfg["prefetch_factor"])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=_collate_batch, **loader_kwargs)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=_collate_batch, **loader_kwargs)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=_collate_batch, **loader_kwargs)

    logger.info(f"DataLoader: batches(train)={len(train_loader)} batch_size={batch_size} num_workers={num_workers} pin_memory={pin_memory}")

    best_val_rmse = float("inf")
    best_path = artifacts_dir / "model_best.pt"
    history_train, history_val = [], []

    patience = int(train_cfg.get("early_stopping", {}).get("patience", 10))
    bad_epochs = 0

    def eval_loader(loader) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
        model.eval()
        ys, ps, ms = [], [], []
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
                y_raw = batch["y_raw"].detach().cpu().numpy()
                mask = batch["mask_y"].detach().cpu().numpy().astype(bool)
                pred = inverse_target_transforms(pred_t, transform_map, target_names)
                ys.append(y_raw)
                ps.append(pred)
                ms.append(mask)
        if not ys:
            empty = np.zeros((0, len(target_names)))
            return {}, empty, empty, empty
        y_true = np.concatenate(ys, axis=0)
        y_pred = np.concatenate(ps, axis=0)
        mask_arr = np.concatenate(ms, axis=0) if ms else np.isfinite(y_true)
        if derived_specs:
            y_pred, _ = apply_derived_numpy(y_pred, target_names, derived_specs, context=derived_context)
        return task_spec.metrics_fn(y_true, y_pred, mask_arr), y_true, y_pred, mask_arr

    log_interval_sec = float(train_cfg.get("log_interval_sec", 30.0))
    show_pbar = bool(train_cfg.get("progress_bar", True))
    max_batches_per_epoch = train_cfg.get("max_batches_per_epoch", None)
    max_batches_per_epoch = int(max_batches_per_epoch) if max_batches_per_epoch is not None else None

    for epoch in range(1, epochs + 1):
        if frozen_encoder_params and epoch == freeze_encoder_epochs + 1:
            for param in frozen_encoder_params:
                param.requires_grad = True
            frozen_encoder_params.clear()
            opt = _build_optimizer()
            logger.info("Unfroze encoder params at epoch %d and rebuilt optimizer.", epoch)
        model.train()
        losses = []
        logger.info(f"Epoch {epoch:04d} start")
        epoch_iter = train_loader
        if show_pbar:
            epoch_iter = tqdm(train_loader, total=len(train_loader), desc=f"train epoch {epoch}", leave=False)
        import time

        start_t = time.monotonic()
        last_log_t = start_t
        n_batches = len(train_loader)

        for step, batch in enumerate(epoch_iter, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            y = batch["y"].to(device)
            mask = batch.get("mask_y")
            if mask is not None:
                mask = mask.to(device)

            opt.zero_grad()
            pred = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            if derived_specs:
                pred_adj, _, consistency_loss = apply_derived_torch(
                    pred,
                    transform_map,
                    target_names,
                    derived_specs,
                    mask=mask,
                    context=derived_context,
                )
                loss = masked_multitask_loss(
                    pred_adj,
                    y,
                    target_names=target_names,
                    mask=mask,
                    loss_name=loss_name,
                    loss_per_target=loss_per_target,
                    weights=loss_weights_tensor,
                )
                if consistency_loss is not None:
                    loss = loss + consistency_loss
            else:
                loss = masked_multitask_loss(
                    pred,
                    y,
                    target_names=target_names,
                    mask=mask,
                    loss_name=loss_name,
                    loss_per_target=loss_per_target,
                    weights=loss_weights_tensor,
                )
            loss.backward()
            opt.step()
            losses.append(loss.item())

            if max_batches_per_epoch is not None and step >= max_batches_per_epoch:
                break

            now = time.monotonic()
            if log_interval_sec > 0 and (now - last_log_t) >= log_interval_sec:
                elapsed = now - start_t
                avg_loss = float(np.mean(losses[-10:])) if losses else float("nan")
                eta = (elapsed / step) * (n_batches - step) if step > 0 else float("nan")
                logger.info(
                    f"Epoch {epoch:04d} step {step}/{n_batches} "
                    f"avg_loss(last10)={avg_loss:.6g} elapsed={elapsed:.1f}s eta~{eta:.1f}s"
                )
                last_log_t = now

        train_loss = float(np.mean(losses)) if losses else float("nan")
        val_metrics, _, _, _ = eval_loader(val_loader)
        val_rmse = val_metrics.get("rmse", float("inf"))

        history_train.append(train_loss)
        history_val.append(val_rmse)
        logger.info(f"Epoch {epoch:04d}: train_loss={train_loss:.6g} val_rmse={val_rmse:.6g}")

        if (val_rmse < best_val_rmse) or (not best_path.exists()):
            best_val_rmse = val_rmse
            torch.save(model.state_dict(), best_path)
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                logger.info(f"Early stopping at epoch {epoch} (patience={patience})")
                break

    model.load_state_dict(torch.load(best_path, map_location=device))

    val_metrics, yv, pv, mv = eval_loader(val_loader)
    test_metrics, yt, pt, mt = eval_loader(test_loader)
    logger.info(f"Best val metrics: {val_metrics}")
    logger.info(f"Test metrics: {test_metrics}")

    if bool(out_cfg.get("plots", True)):
        save_learning_curve(history_train, history_val, plots_dir / "learning_curve.png", ylabel="loss / rmse", yscale="log")
        if len(yv) > 0:
            yv_primary = yv[:, 0]
            pv_primary = pv[:, 0]
            mask_primary = mv[:, 0] if mv.size else np.isfinite(yv_primary)
            yv_plot = yv_primary[mask_primary]
            pv_plot = pv_primary[mask_primary]
            if len(yv_plot) > 0:
                save_parity_plot(yv_plot, pv_plot, plots_dir / "parity_val.png", title="Parity (val)", xlabel="true", ylabel="pred")
                save_residual_plot(yv_plot, pv_plot, plots_dir / "residual_val.png", title="Residual (val)")
                save_residual_hist(yv_plot, pv_plot, plots_dir / "residual_hist_val.png", title="Residual distribution (val)")
                save_abs_error_cdf(yv_plot, pv_plot, plots_dir / "abs_error_cdf_val.png", title="Absolute error CDF (val)")
                save_bland_altman_plot(yv_plot, pv_plot, plots_dir / "bland_altman_val.png", title="Bland-Altman (val)")
        if len(yt) > 0:
            yt_primary = yt[:, 0]
            pt_primary = pt[:, 0]
            mask_primary = mt[:, 0] if mt.size else np.isfinite(yt_primary)
            yt_plot = yt_primary[mask_primary]
            pt_plot = pt_primary[mask_primary]
            if len(yt_plot) > 0:
                save_parity_plot(yt_plot, pt_plot, plots_dir / "parity_test.png", title="Parity (test)", xlabel="true", ylabel="pred")
                save_residual_plot(yt_plot, pt_plot, plots_dir / "residual_test.png", title="Residual (test)")
                save_residual_hist(yt_plot, pt_plot, plots_dir / "residual_hist_test.png", title="Residual distribution (test)")
                save_abs_error_cdf(yt_plot, pt_plot, plots_dir / "abs_error_cdf_test.png", title="Absolute error CDF (test)")
                save_bland_altman_plot(yt_plot, pt_plot, plots_dir / "bland_altman_test.png", title="Bland-Altman (test)")
        primary_name = target_names[0] if target_names else "target"
        unit = target_units.get(primary_name)
        target_label = f"{primary_name} ({unit})" if unit else primary_name
        y_train_primary = y_train_raw[:, 0][mask_train[:, 0]] if y_train_raw.size else []
        if len(y_train_primary) > 0:
            save_hist(y_train_primary.tolist(), plots_dir / "y_train_hist.png", title="Target distribution (train)", xlabel=target_label)

    featureset_hash = featureset.save(artifacts_dir)
    save_target_transforms(artifacts_dir / "target_transform.json", transform_map)
    dump_yaml(run_dir / "config_snapshot.yaml", cfg)
    save_json(run_dir / "metrics_val.json", val_metrics)
    save_json(run_dir / "metrics_test.json", test_metrics)
    split_metrics = {"val": val_metrics, "test": test_metrics}
    by_target = build_metrics_by_target(split_metrics)
    overall_by_split = build_overall_by_split(split_metrics)
    save_json(
        run_dir / "metrics.json",
        {
            "val": val_metrics,
            "test": test_metrics,
            "by_target": by_target,
            "overall": overall_by_split,
            "per_target": by_target,
            "n_train": int(len(train_data)),
            "n_val": int(len(val_data)),
            "n_test": int(len(test_data)),
            "seed": int(seed),
            "target_names": target_names,
            "units": target_units,
            "target_transform": transform_summary,
            "target_transform_state": transform_state_summary,
            "loss_weights": loss_weights_by_target,
        },
    )
    model_dir = ensure_dir(run_dir / "model")
    torch.save(model.state_dict(), model_dir / "model.ckpt")
    save_json(model_dir / "featurizer_state.json", featureset.pipeline.tokenizer_state())

    meta["featureset_hash"] = featureset_hash
    save_meta(run_dir, meta)
    logger.info(f"Saved best model to {best_path}")
    logger.info("Done.")
    return run_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Train SMILES transformer model.")
    ap.add_argument("--config", required=True, help="Path to configs/smiles/train.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
