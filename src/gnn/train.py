from __future__ import annotations

import argparse
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.common.config import dump_yaml, load_config
from src.common.conformer import (
    ConformerCache,
    ConformerService,
    conformer_config_from_cfg,
    conformer_config_payload,
    mol_from_smiles,
)
from src.common.meta import build_meta, save_meta
from src.common.io import load_sdf_mol, read_csv, sdf_path_from_cas
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
from src.gnn.featurizer_graph import GraphFeaturizerError, requires_3d_pos
from src.featuresets.base import GraphFeatureSet
from src.featuresets.registry import create_featureset
from src.models import (
    create_model,
    feature_inputs_from_featureset,
    get_model_spec,
    resolve_model_name,
    validate_model_requirements,
)
from src.models.capabilities import apply_model_capabilities
from src.tasks import resolve_task
from src.targets.registry import resolve_target_names, resolve_target_units
from src.targets.derived import (
    apply_derived_numpy,
    apply_derived_torch,
    build_derived_context,
    resolve_derived_targets,
)
from src.targets.transform import (
    apply_target_transforms,
    build_target_transforms,
    fit_target_transforms,
    inverse_target_transforms,
    save_target_transforms,
    summarize_target_transforms,
)
from src.common.losses import masked_multitask_loss, masked_regression_loss, resolve_loss_per_target, resolve_loss_weights
from src.utils.artifacts import compute_dataset_hash
from src.utils.validate_config import validate_config

try:
    import torch
    import torch.nn as nn
    from torch.optim import Adam
    from torch_geometric.loader import DataLoader
except Exception:  # pragma: no cover
    torch = None
    nn = None
    Adam = None
    DataLoader = None

try:
    from torch_geometric.typing import WITH_TORCH_SCATTER, WITH_TORCH_SPARSE
except Exception:  # pragma: no cover
    WITH_TORCH_SCATTER = False
    WITH_TORCH_SPARSE = False


def _require_pyg():
    if torch is None or DataLoader is None:
        raise ImportError(
            "PyTorch and PyTorch Geometric are required for GNN training. "
            "Install torch and torch_geometric (matching your environment)."
        )


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


def _warn_if_pyg_extensions_missing(logger) -> None:
    if WITH_TORCH_SCATTER and WITH_TORCH_SPARSE:
        return
    missing = []
    if not WITH_TORCH_SCATTER:
        missing.append("torch_scatter")
    if not WITH_TORCH_SPARSE:
        missing.append("torch_sparse")
    logger.warning(
        "PyG optional extensions are missing (%s). Training can be extremely slow and may look like a hang. "
        "Install matching wheels from https://data.pyg.org/whl/ or follow the official PyG install guide.",
        ", ".join(missing),
    )


def _set_mps_memory_env(cfg_train: Dict[str, Any], logger) -> None:
    """
    Optionally set MPS memory watermark env var.

    PYTORCH_MPS_HIGH_WATERMARK_RATIO must be set before the first MPS allocation to be effective.
    Leaving it unset is safest; lowering it can avoid OOMs but may reduce usable memory.
    Setting it to 0.0 removes the limit and may destabilize the system.
    """
    if torch is None:
        return
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_built()):
        return
    ratio = cfg_train.get("mps_high_watermark_ratio", None)
    if ratio is None:
        return
    try:
        ratio_f = float(ratio)
    except Exception:
        logger.warning("Invalid train.mps_high_watermark_ratio=%r (expected float). Ignoring.", ratio)
        return
    import os

    if "PYTORCH_MPS_HIGH_WATERMARK_RATIO" in os.environ:
        logger.info("PYTORCH_MPS_HIGH_WATERMARK_RATIO is already set; leaving as-is.")
        return
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = str(ratio_f)
    logger.warning("Set PYTORCH_MPS_HIGH_WATERMARK_RATIO=%s (config).", ratio_f)


def _is_oom_error(e: BaseException) -> bool:
    s = str(e).lower()
    return "out of memory" in s or "mps backend out of memory" in s


def run(cfg: Dict[str, Any]) -> Path:
    validate_config(cfg)
    seed = int(cfg.get("train", {}).get("seed", 42))
    set_seed(seed)

    data_cfg = cfg.get("data", {})
    dataset_csv = Path(data_cfg.get("dataset_csv", "data/processed/dataset_with_lj.csv"))
    indices_dir = Path(data_cfg.get("indices_dir", "data/processed/indices"))
    sdf_dir = Path(data_cfg.get("sdf_dir", "data/raw/sdf_files"))
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
    primary_name = target_names[0]
    cas_col = str(data_cfg.get("cas_col", "CAS"))
    columns_cfg = cfg.get("columns", {}) or {}
    sample_id_col = str(data_cfg.get("sample_id_col", columns_cfg.get("sample_id", cas_col)))
    smiles_col = data_cfg.get("smiles_col", columns_cfg.get("smiles", None))
    smiles_col = str(smiles_col) if smiles_col is not None else None

    out_cfg = cfg.get("output", {})
    run_dir_root = Path(out_cfg.get("run_dir", "runs/train/gnn"))
    experiment_cfg = cfg.get("experiment", {})
    exp_name = str(out_cfg.get("exp_name", experiment_cfg.get("name", "gnn_experiment")))
    run_dir = ensure_dir(run_dir_root / exp_name)
    plots_dir = ensure_dir(run_dir / "plots")
    artifacts_dir = ensure_dir(run_dir / "artifacts")

    logger = get_logger("gnn_train", log_file=run_dir / "train.log")

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

    try:
        _require_pyg()
    except Exception as e:
        logger.error(str(e))
        raise
    _warn_if_pyg_extensions_missing(logger)
    _set_mps_memory_env(cfg.get("train", {}) or {}, logger)

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

    # Featurizer config
    featureset = create_featureset(cfg)
    if not isinstance(featureset, GraphFeatureSet):
        raise ValueError("GNN training requires a graph FeatureSet.")
    pipeline = featureset.pipeline
    model_cfg = cfg.get("model", {}) or {}
    model_name = resolve_model_name(cfg, default="mpnn")
    model_spec = get_model_spec(model_name)
    gcfg = pipeline.graph_cfg
    conformer_cfg = conformer_config_from_cfg(cfg)
    apply_model_capabilities(cfg, model_spec, gcfg, conformer_cfg, logger)
    feature_inputs = feature_inputs_from_featureset(featureset)
    validate_model_requirements(model_spec, task_spec.target_columns, feature_inputs)
    requires_pos = requires_3d_pos(gcfg)
    conformer_cache = None
    conformer_service = None
    if (gcfg.use_3d_pos or requires_pos) and conformer_cfg.enabled:
        conformer_cache = ConformerCache(artifacts_dir / "conformer_cache.json", config=conformer_config_payload(conformer_cfg))
        conformer_service = ConformerService(conformer_cfg, cache=conformer_cache)

    def build_dataset(split_name: str) -> List[Any]:
        split_df = df.loc[indices[split_name]]
        data_list = []
        cas_values = split_df[cas_col].astype(str).tolist()
        sample_ids = split_df[sample_id_col].astype(str).tolist() if sample_id_col in split_df.columns else cas_values
        smiles_values = None
        if smiles_col and smiles_col in split_df.columns:
            smiles_values = split_df[smiles_col].astype(str).tolist()
        else:
            smiles_values = [""] * len(split_df)
        y_values = split_df[target_columns].to_numpy(dtype=float)
        mask_values = np.isfinite(y_values)

        for cas, sample_id, smiles, y_row, mask_row in tqdm(
            zip(cas_values, sample_ids, smiles_values, y_values, mask_values),
            total=len(split_df),
            desc=f"featurize {split_name}",
        ):
            mask_row = np.asarray(mask_row, dtype=bool)
            y_row = np.asarray(y_row, dtype=float)
            mol = load_sdf_mol(sdf_path_from_cas(sdf_dir, cas))
            if mol is None and smiles:
                mol = mol_from_smiles(smiles)
            if mol is None or not np.any(mask_row):
                continue
            pos = None
            if conformer_service is not None:
                pos, meta = conformer_service.get_pos(sample_id=sample_id, mol=mol, smiles=smiles, allow_generate=True)
                if pos is None and meta.get("status") in {"failed", "cache_miss"}:
                    if requires_pos:
                        raise ValueError(
                            f"3D positions required but missing for sample_id={sample_id} (status={meta.get('status')})."
                        )
                    continue
            if requires_pos and pos is None:
                raise ValueError(f"3D positions required but missing for sample_id={sample_id}.")
            try:
                y_filled = np.where(mask_row, y_row, 0.0).astype(float)
                data = pipeline.featurize_mol(mol, y=y_filled, pos=pos, mask_y=mask_row.astype(float))
                data.y = torch.tensor(y_filled, dtype=torch.float32).view(1, -1)
                data.y_raw = torch.tensor(y_row, dtype=torch.float32).view(1, -1)
                data.mask_y = torch.tensor(mask_row.astype(np.float32)).view(1, -1)
                data_list.append(data)
            except Exception as exc:
                if requires_pos and isinstance(exc, GraphFeaturizerError):
                    raise
                continue
        return data_list

    train_data = build_dataset("train")
    val_data = build_dataset("val")
    test_data = build_dataset("test")

    if conformer_cache is not None and conformer_cfg.cache_enabled:
        conformer_cache.save()

    logger.info(f"Data sizes: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
    if len(train_data) == 0:
        raise RuntimeError("No valid training graphs were created. Check sdf_dir, target values, and featurizer settings.")

    def stack_targets(data_list: List[Any]) -> tuple[np.ndarray, np.ndarray]:
        if not data_list:
            return np.zeros((0, len(target_names))), np.zeros((0, len(target_names)), dtype=bool)
        y_raw = np.stack([data.y_raw.detach().cpu().numpy() for data in data_list])
        if y_raw.ndim == 3 and y_raw.shape[1] == 1:
            y_raw = y_raw[:, 0, :]
        if hasattr(data_list[0], "mask_y"):
            mask = np.stack([data.mask_y.detach().cpu().numpy() for data in data_list]).astype(bool)
            if mask.ndim == 3 and mask.shape[1] == 1:
                mask = mask[:, 0, :]
        else:
            mask = np.isfinite(y_raw)
        return y_raw, mask

    y_train_raw, mask_train = stack_targets(train_data)
    fit_target_transforms(transform_map, y_train_raw, target_names, mask=mask_train)
    transform_state_summary = summarize_target_transforms(transform_map, include_state=True)
    meta["target_transform_state"] = transform_state_summary
    save_meta(run_dir, meta)

    def apply_transform(data_list: List[Any]) -> np.ndarray:
        if not data_list:
            return np.zeros((0, len(target_names)))
        raw_vals, mask_vals = stack_targets(data_list)
        trans_vals = apply_target_transforms(raw_vals, transform_map, target_names, mask=mask_vals)
        for data, y_trans in zip(data_list, trans_vals):
            data.y = torch.tensor(y_trans, dtype=torch.float32).view(1, -1)
        return raw_vals

    y_train_raw = apply_transform(train_data)
    apply_transform(val_data)
    apply_transform(test_data)

    # Model config
    in_dim = train_data[0].x.shape[1]
    edge_dim = train_data[0].edge_attr.shape[1] if hasattr(train_data[0], "edge_attr") else 0
    global_dim = int(train_data[0].u.shape[1]) if hasattr(train_data[0], "u") else 0
    out_dim = len(target_columns)
    model = create_model(
        model_name,
        model_cfg=model_cfg,
        context={
            "in_dim": in_dim,
            "edge_dim": edge_dim,
            "global_dim": global_dim,
            "out_dim": out_dim,
            "target_names": target_names,
        },
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

    epochs = int(train_cfg.get("epochs", 200))
    batch_size = int(train_cfg.get("batch_size", 64))
    lr = float(train_cfg.get("lr", 1e-3))
    base_lr = lr
    weight_decay = float(train_cfg.get("weight_decay", 1e-5))
    warmup_steps = train_cfg.get("warmup_steps", None)
    warmup_epochs = train_cfg.get("warmup_epochs", None)
    grad_clip_norm = train_cfg.get("grad_clip_norm", train_cfg.get("clip_grad_norm", train_cfg.get("grad_clip", None)))
    nan_diagnostics = bool(train_cfg.get("nan_diagnostics", False))
    loss_name = task_spec.loss_name
    loss_per_target = resolve_loss_per_target(cfg.get("task", {}), target_names, loss_name, logger=logger)
    if loss_name not in {"mse", "huber"}:
        raise ValueError(f"Unknown loss: {loss_name}")
    loss_weights_tensor = None
    if loss_weights is not None:
        loss_weights_tensor = torch.tensor(loss_weights, dtype=torch.float32, device=device)

    trainable = collect_trainable_params(model)
    if not trainable:
        raise ValueError("No trainable parameters remain after pretrain freeze.")
    opt = Adam(trainable, lr=lr, weight_decay=weight_decay)

    num_workers = int(train_cfg.get("num_workers", 0))
    pin_memory = bool(train_cfg.get("pin_memory", device.type == "cuda"))
    persistent_workers = bool(train_cfg.get("persistent_workers", False)) if num_workers > 0 else False
    loader_kwargs: Dict[str, Any] = {"num_workers": num_workers, "pin_memory": pin_memory, "persistent_workers": persistent_workers}
    if num_workers > 0 and "prefetch_factor" in train_cfg:
        loader_kwargs["prefetch_factor"] = int(train_cfg["prefetch_factor"])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, **loader_kwargs)
    logger.info(f"DataLoader: batches(train)={len(train_loader)} batch_size={batch_size} num_workers={num_workers} pin_memory={pin_memory}")
    if warmup_steps is None and warmup_epochs is not None:
        warmup_steps = int(warmup_epochs) * max(1, len(train_loader))

    best_val_rmse = float("inf")
    best_path = artifacts_dir / "model_best.pt"
    history_train, history_val = [], []

    patience = int(train_cfg.get("early_stopping", {}).get("patience", 20))
    bad_epochs = 0

    def eval_loader(loader):
        model.eval()
        ys, ps, ms = [], [], []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                pred_t = model(batch).detach().cpu().numpy()
                if pred_t.ndim == 1:
                    pred_t = pred_t.reshape(-1, 1)
                if hasattr(batch, "y_raw"):
                    y = batch.y_raw.detach().cpu().numpy()
                else:
                    y = batch.y.detach().cpu().numpy()
                if y.ndim == 1:
                    y = y.reshape(-1, 1)
                mask = None
                if hasattr(batch, "mask_y"):
                    mask = batch.mask_y.detach().cpu().numpy()
                    if mask.ndim == 1:
                        mask = mask.reshape(-1, 1)
                    mask = mask.astype(bool)
                pred = inverse_target_transforms(pred_t, transform_map, target_names)
                if derived_specs:
                    pred, _ = apply_derived_numpy(pred, target_names, derived_specs, context=derived_context)
                ys.append(y)
                ps.append(pred)
                if mask is not None:
                    ms.append(mask)
        if not ys:
            empty = np.zeros((0, len(target_names)))
            return {}, empty, empty, empty
        y_true = np.concatenate(ys, axis=0)
        y_pred = np.concatenate(ps, axis=0)
        if ms:
            mask_arr = np.concatenate(ms, axis=0)
        else:
            mask_arr = np.isfinite(y_true)
        return task_spec.metrics_fn(y_true, y_pred, mask_arr), y_true, y_pred, mask_arr

    log_interval_sec = float(train_cfg.get("log_interval_sec", 30.0))
    show_pbar = bool(train_cfg.get("progress_bar", True))
    max_batches_per_epoch = train_cfg.get("max_batches_per_epoch", None)
    max_batches_per_epoch = int(max_batches_per_epoch) if max_batches_per_epoch is not None else None
    global_step = 0

    for epoch in range(1, epochs + 1):
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
        last_pred_nonfinite = None
        last_loss_finite = None

        for step, batch in enumerate(epoch_iter, start=1):
            try:
                batch = batch.to(device)
                opt.zero_grad()
                pred = model(batch)
                y = batch.y
                mask = getattr(batch, "mask_y", None)
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
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(trainable, float(grad_clip_norm))
                if warmup_steps:
                    scale = min(1.0, float(global_step + 1) / float(warmup_steps))
                    scaled_lr = base_lr * scale
                    for group in opt.param_groups:
                        group["lr"] = scaled_lr
                opt.step()
                if nan_diagnostics:
                    last_pred_nonfinite = float((~torch.isfinite(pred)).float().mean().item())
                    last_loss_finite = bool(torch.isfinite(loss).item())
                losses.append(loss.item())
            except RuntimeError as e:
                if _is_oom_error(e):
                    try:
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        elif device.type == "mps" and hasattr(torch, "mps"):
                            torch.mps.empty_cache()
                    except Exception:
                        pass
                    logger.error(
                        "Out of memory on device=%s during training. "
                        "Try reducing train.batch_size / model.hidden_dim / model.num_layers, "
                        "or set train.device: cpu in the config.",
                        device,
                    )
                raise
            if max_batches_per_epoch is not None and step >= max_batches_per_epoch:
                break
            global_step += 1

            now = time.monotonic()
            if log_interval_sec > 0 and (now - last_log_t) >= log_interval_sec:
                elapsed = now - start_t
                avg_loss = float(np.mean(losses[-10:])) if losses else float("nan")
                eta = (elapsed / step) * (n_batches - step) if step > 0 else float("nan")
                logger.info(
                    f"Epoch {epoch:04d} step {step}/{n_batches} "
                    f"avg_loss(last10)={avg_loss:.6g} elapsed={elapsed:.1f}s eta~{eta:.1f}s"
                )
                if nan_diagnostics and last_pred_nonfinite is not None:
                    logger.info(
                        "Diagnostics: pred_nonfinite=%.4f loss_finite=%s lr=%.3g",
                        last_pred_nonfinite,
                        last_loss_finite,
                        opt.param_groups[0]["lr"],
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

    # Load best
    model.load_state_dict(torch.load(best_path, map_location=device))

    val_metrics, yv, pv, mv = eval_loader(val_loader)
    test_metrics, yt, pt, mt = eval_loader(test_loader)
    logger.info(f"Best val metrics: {val_metrics}")
    logger.info(f"Test metrics: {test_metrics}")

    # Plots
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
        target_label = primary_name
        unit = target_units.get(primary_name)
        if unit:
            target_label = f"{target_label} ({unit})"
        y_train_primary = y_train_raw[:, 0][mask_train[:, 0]] if y_train_raw.size else []
        if len(y_train_primary) > 0:
            save_hist(y_train_primary.tolist(), plots_dir / "y_train_hist.png", title="Target distribution (train)", xlabel=target_label)

    # Save artifacts
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
    save_json(model_dir / "featurizer_state.json", {"config": asdict(gcfg)})

    # AD artifacts (for inference-time applicability domain)
    try:
        from src.fp.featurizer_fp import morgan_bitvect

        train_df_for_ad = df.loc[indices["train"]].copy()
        train_ids_for_ad = train_df_for_ad[cas_col].astype(str).tolist()
        train_elements_str = train_df_for_ad.get("elements", pd.Series([""] * len(train_df_for_ad))).astype(str).tolist()
        training_elements = sorted({el for e_str in train_elements_str for el in e_str.split(",") if el})

        train_mols = [load_sdf_mol(sdf_path_from_cas(sdf_dir, cas)) for cas in train_ids_for_ad]
        train_fps = [morgan_bitvect(m, radius=2, n_bits=2048) if m is not None else None for m in train_mols]
        train_pairs = [(fp, cas) for fp, cas in zip(train_fps, train_ids_for_ad) if fp is not None]
        train_fps = [p[0] for p in train_pairs]
        train_ids_for_ad = [p[1] for p in train_pairs]

        heavy_atoms_train = train_df_for_ad["n_heavy_atoms"].dropna().astype(int).tolist()
        heavy_min = int(min(heavy_atoms_train)) if heavy_atoms_train else 0
        heavy_max = int(max(heavy_atoms_train)) if heavy_atoms_train else 0

        ad_artifact = {
            "training_elements": training_elements,
            "heavy_atom_range": [heavy_min, heavy_max],
            "morgan_radius": 2,
            "n_bits": 2048,
            "train_ids": train_ids_for_ad,
            "train_fps": train_fps,
            "tanimoto_warn_threshold": float(cfg.get("ad", {}).get("tanimoto_warn_threshold", 0.5)),
            "top_k": int(cfg.get("ad", {}).get("top_k", 5)),
        }
        with open(artifacts_dir / "ad.pkl", "wb") as f:
            pickle.dump(ad_artifact, f)
        logger.info(f"Saved AD artifact to {artifacts_dir / 'ad.pkl'}")
    except Exception as e:
        logger.warning(f"Failed to create AD artifact (will disable AD in predict): {e}")

    logger.info(f"Saved best model to {best_path}")
    logger.info("Done.")

    meta["featureset_hash"] = featureset_hash
    save_meta(run_dir, meta)
    return run_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Train GNN model (PyTorch Geometric) for LJ parameter regression.")
    ap.add_argument("--config", required=True, help="Path to configs/gnn/train.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
