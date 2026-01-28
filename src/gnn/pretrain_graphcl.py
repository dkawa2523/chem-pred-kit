from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm

from src.common.config import dump_yaml, load_config
from src.common.conformer import (
    ConformerCache,
    ConformerService,
    conformer_config_from_cfg,
    conformer_config_payload,
    mol_from_smiles,
)
from src.common.io import load_sdf_mol, read_csv, sdf_path_from_cas
from src.common.meta import build_meta, save_meta
from src.common.splitters import load_split_indices
from src.common.utils import ensure_dir, get_logger, save_json, set_seed
from src.common.pretrain import collect_trainable_params
from src.featuresets.base import GraphFeatureSet
from src.featuresets.registry import create_featureset
from src.gnn.featurizer_graph import GraphFeaturizerError, requires_3d_pos
from src.gnn.graphcl import GraphCLAugmentConfig, apply_graphcl_augmentation, graphcl_loss
from src.models import create_model, feature_inputs_from_featureset, get_model_spec, resolve_model_name, validate_model_requirements
from src.utils.artifacts import compute_dataset_hash
from src.utils.validate_config import validate_config

try:
    import torch
    import torch.nn as nn
    from torch.optim import Adam
    from torch_geometric.data import Batch
    from torch_geometric.loader import DataLoader
except Exception:  # pragma: no cover
    torch = None
    nn = None
    Adam = None
    Batch = None
    DataLoader = None

try:
    from torch_geometric.typing import WITH_TORCH_SCATTER, WITH_TORCH_SPARSE
except Exception:  # pragma: no cover
    WITH_TORCH_SCATTER = False
    WITH_TORCH_SPARSE = False


def _require_pyg():
    if torch is None or DataLoader is None or Batch is None:
        raise ImportError(
            "PyTorch and PyTorch Geometric are required for GraphCL pretraining. "
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
        "PyG optional extensions are missing (%s). Training can be extremely slow. "
        "Install matching wheels from https://data.pyg.org/whl/ or follow the official PyG install guide.",
        ", ".join(missing),
    )


def _set_mps_memory_env(cfg_train: Dict[str, Any], logger) -> None:
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


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        if in_dim <= 0 or hidden_dim <= 0 or out_dim <= 0:
            raise ValueError("ProjectionHead dims must be > 0")
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.net(x)


def _parse_splits(cfg: Dict[str, Any], available: List[str]) -> List[str]:
    raw = cfg.get("splits", ["train"])
    if isinstance(raw, str):
        raw = [raw]
    splits = [str(v).strip() for v in (raw or []) if str(v).strip()]
    if not splits:
        splits = ["train"]
    if "all" in {s.lower() for s in splits}:
        return available
    return [s for s in splits if s in available]


def _resolve_aug_cfg(selfsup_cfg: Dict[str, Any]) -> GraphCLAugmentConfig:
    aug_cfg = selfsup_cfg.get("augmentations", selfsup_cfg.get("augmentation", selfsup_cfg.get("aug", {})))
    if not isinstance(aug_cfg, dict):
        aug_cfg = {}
    edge_drop = aug_cfg.get("edge_drop", aug_cfg.get("drop_edge", aug_cfg.get("edge_dropout", 0.2)))
    node_mask = aug_cfg.get(
        "node_feature_mask",
        aug_cfg.get("node_mask", aug_cfg.get("mask_node", 0.2)),
    )
    return GraphCLAugmentConfig(edge_drop=float(edge_drop), node_feature_mask=float(node_mask))


def run(cfg: Dict[str, Any]) -> Path:
    validate_config(cfg)
    seed = int(cfg.get("train", {}).get("seed", 42))
    set_seed(seed)

    data_cfg = cfg.get("data", {})
    dataset_csv = Path(data_cfg.get("dataset_csv", "data/processed/dataset_with_lj.csv"))
    indices_dir = Path(data_cfg.get("indices_dir", "data/processed/indices"))
    sdf_dir = Path(data_cfg.get("sdf_dir", "data/raw/sdf_files"))
    cas_col = str(data_cfg.get("cas_col", "CAS"))
    columns_cfg = cfg.get("columns", {}) or {}
    sample_id_col = str(data_cfg.get("sample_id_col", columns_cfg.get("sample_id", cas_col)))
    smiles_col = data_cfg.get("smiles_col", columns_cfg.get("smiles", None))
    smiles_col = str(smiles_col) if smiles_col is not None else None

    out_cfg = cfg.get("output", {})
    run_dir_root = Path(out_cfg.get("run_dir", "runs/pretrain/gnn"))
    experiment_cfg = cfg.get("experiment", {})
    exp_name = str(out_cfg.get("exp_name", experiment_cfg.get("name", "graphcl_pretrain")))
    run_dir = ensure_dir(run_dir_root / exp_name)
    artifacts_dir = ensure_dir(run_dir / "artifacts")

    logger = get_logger("gnn_pretrain_graphcl", log_file=run_dir / "pretrain.log")
    dump_yaml(run_dir / "config.yaml", cfg)

    dataset_hash = compute_dataset_hash(dataset_csv, indices_dir)
    selfsup_cfg = cfg.get("self_supervised", {}) or {}
    method = str(selfsup_cfg.get("method", "graphcl")).lower()
    if method != "graphcl":
        raise ValueError(f"Unsupported self_supervised.method: {method}")

    meta_extra = {
        "self_supervised_method": method,
        "projection_dim": int(selfsup_cfg.get("projection_dim", 128)),
        "projection_hidden_dim": int(selfsup_cfg.get("projection_hidden_dim", 256)),
        "temperature": float(selfsup_cfg.get("temperature", 0.2)),
        "augmentations": asdict(_resolve_aug_cfg(selfsup_cfg)),
    }
    meta = build_meta(
        process_name=str(cfg.get("process", {}).get("name", "pretrain")),
        cfg=cfg,
        upstream_artifacts=[],
        dataset_hash=dataset_hash,
        extra=meta_extra,
    )
    save_meta(run_dir, meta)

    _require_pyg()
    _warn_if_pyg_extensions_missing(logger)
    _set_mps_memory_env(cfg.get("train", {}) or {}, logger)

    df = read_csv(dataset_csv)
    indices = load_split_indices(indices_dir)
    available_splits = [k for k in indices.keys() if indices.get(k)]
    split_names = _parse_splits(selfsup_cfg, available_splits or ["train"])

    featureset = create_featureset(cfg)
    if not isinstance(featureset, GraphFeatureSet):
        raise ValueError("GraphCL pretraining requires a graph FeatureSet.")
    pipeline = featureset.pipeline
    gcfg = pipeline.graph_cfg
    requires_pos = requires_3d_pos(gcfg)
    conformer_cfg = conformer_config_from_cfg(cfg)
    conformer_cache = None
    conformer_service = None
    if (gcfg.use_3d_pos or requires_pos) and conformer_cfg.enabled:
        conformer_cache = ConformerCache(
            artifacts_dir / "conformer_cache.json",
            config=conformer_config_payload(conformer_cfg),
        )
        conformer_service = ConformerService(conformer_cfg, cache=conformer_cache)

    def build_dataset(split_names: List[str]) -> List[Any]:
        data_list: List[Any] = []
        invalid = 0
        missing = 0
        for split_name in split_names:
            if split_name not in indices:
                continue
            split_df = df.loc[indices[split_name]]
            cas_values = split_df[cas_col].astype(str).tolist()
            sample_ids = split_df[sample_id_col].astype(str).tolist() if sample_id_col in split_df.columns else cas_values
            if smiles_col and smiles_col in split_df.columns:
                smiles_values = split_df[smiles_col].astype(str).tolist()
            else:
                smiles_values = [""] * len(split_df)
            for cas, sample_id, smiles in tqdm(
                zip(cas_values, sample_ids, smiles_values),
                total=len(split_df),
                desc=f"featurize {split_name}",
            ):
                mol = load_sdf_mol(sdf_path_from_cas(sdf_dir, cas))
                if mol is None and smiles:
                    mol = mol_from_smiles(smiles)
                if mol is None:
                    missing += 1
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
                    data = pipeline.featurize_mol(mol, y=None, pos=pos, mask_y=None)
                    data_list.append(data)
                except Exception as exc:
                    if requires_pos and isinstance(exc, GraphFeaturizerError):
                        raise
                    invalid += 1
                    continue
        if missing or invalid:
            logger.warning("Skipped graphs: missing_mol=%d invalid_graph=%d", missing, invalid)
        return data_list

    train_data = build_dataset(split_names)

    if conformer_cache is not None and conformer_cfg.cache_enabled:
        conformer_cache.save()

    if not train_data:
        raise RuntimeError("No valid graphs were created for pretraining.")

    max_samples = int(selfsup_cfg.get("max_samples", 0) or 0)
    if max_samples > 0 and len(train_data) > max_samples:
        rng = np.random.default_rng(seed)
        keep_idx = rng.permutation(len(train_data))[:max_samples]
        train_data = [train_data[i] for i in keep_idx.tolist()]
        logger.info("Downsampled pretrain data to %d graphs", len(train_data))

    model_cfg = cfg.get("model", {}) or {}
    model_name = resolve_model_name(cfg, default="mpnn")
    model_spec = get_model_spec(model_name)
    feature_inputs = feature_inputs_from_featureset(featureset)
    validate_model_requirements(model_spec, [], feature_inputs)
    in_dim = train_data[0].x.shape[1]
    edge_dim = train_data[0].edge_attr.shape[1] if hasattr(train_data[0], "edge_attr") else 0
    global_dim = int(train_data[0].u.shape[1]) if hasattr(train_data[0], "u") else 0
    out_dim = int(selfsup_cfg.get("dummy_out_dim", 1))
    model = create_model(
        model_name,
        model_cfg=model_cfg,
        context={"in_dim": in_dim, "edge_dim": edge_dim, "global_dim": global_dim, "out_dim": out_dim},
    )

    train_cfg = cfg.get("train", {})
    device = _select_device(train_cfg)
    num_threads = train_cfg.get("num_threads", None)
    if num_threads is not None:
        try:
            torch.set_num_threads(int(num_threads))
        except Exception:
            pass
    model = model.to(device)

    if not hasattr(model, "encode"):
        raise ValueError("GraphCL pretraining requires a model with an encode() method.")
    with torch.no_grad():
        probe = Batch.from_data_list([train_data[0]]).to(device)
        emb = model.encode(probe)
    if emb.dim() == 1:
        emb = emb.view(1, -1)
    embed_dim = int(emb.shape[1])
    proj_hidden = int(selfsup_cfg.get("projection_hidden_dim", 256))
    proj_dim = int(selfsup_cfg.get("projection_dim", 128))
    proj_dropout = float(selfsup_cfg.get("projection_dropout", 0.0))
    projector = ProjectionHead(embed_dim, proj_hidden, proj_dim, dropout=proj_dropout).to(device)

    total_params = int(sum(p.numel() for p in model.parameters()))
    proj_params = int(sum(p.numel() for p in projector.parameters()))
    logger.info(
        "Model params: total=%.2fM projector=%.2fM device=%s",
        total_params / 1e6,
        proj_params / 1e6,
        device,
    )

    epochs = int(train_cfg.get("epochs", 1))
    batch_size = int(train_cfg.get("batch_size", 256))
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 1e-5))

    trainable = collect_trainable_params(model) + collect_trainable_params(projector)
    opt = Adam(trainable, lr=lr, weight_decay=weight_decay)

    num_workers = int(train_cfg.get("num_workers", 0))
    pin_memory = bool(train_cfg.get("pin_memory", device.type == "cuda"))
    persistent_workers = bool(train_cfg.get("persistent_workers", False)) if num_workers > 0 else False
    loader_kwargs: Dict[str, Any] = {"num_workers": num_workers, "pin_memory": pin_memory, "persistent_workers": persistent_workers}
    if num_workers > 0 and "prefetch_factor" in train_cfg:
        loader_kwargs["prefetch_factor"] = int(train_cfg["prefetch_factor"])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, **loader_kwargs)
    logger.info(
        "DataLoader: batches(train)=%d batch_size=%d num_workers=%d pin_memory=%s",
        len(train_loader),
        batch_size,
        num_workers,
        pin_memory,
    )

    log_interval_sec = float(train_cfg.get("log_interval_sec", 30.0))
    show_pbar = bool(train_cfg.get("progress_bar", True))
    max_batches_per_epoch = train_cfg.get("max_batches_per_epoch", None)
    max_batches_per_epoch = int(max_batches_per_epoch) if max_batches_per_epoch is not None else None
    temperature = float(selfsup_cfg.get("temperature", 0.2))
    aug_cfg = _resolve_aug_cfg(selfsup_cfg)

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        projector.train()
        losses = []
        logger.info("Epoch %04d start", epoch)
        epoch_iter = train_loader
        if show_pbar:
            epoch_iter = tqdm(train_loader, total=len(train_loader), desc=f"pretrain epoch {epoch}", leave=False)
        start_t = time.monotonic()
        last_log_t = start_t
        n_batches = len(train_loader)

        for step, batch in enumerate(epoch_iter, start=1):
            try:
                data_list = batch.to_data_list()
                view1 = [apply_graphcl_augmentation(d, aug_cfg) for d in data_list]
                view2 = [apply_graphcl_augmentation(d, aug_cfg) for d in data_list]
                batch1 = Batch.from_data_list(view1).to(device)
                batch2 = Batch.from_data_list(view2).to(device)

                opt.zero_grad()
                h1 = model.encode(batch1)
                h2 = model.encode(batch2)
                z1 = projector(h1)
                z2 = projector(h2)
                loss = graphcl_loss(z1, z2, temperature=temperature)
                loss.backward()
                opt.step()
                losses.append(float(loss.item()))
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
                        "Out of memory during pretraining. "
                        "Reduce train.batch_size or model.hidden_dim, or set train.device: cpu.",
                    )
                raise
            if max_batches_per_epoch is not None and step >= max_batches_per_epoch:
                break
            now = time.monotonic()
            if log_interval_sec > 0 and (now - last_log_t) >= log_interval_sec:
                elapsed = now - start_t
                avg_loss = float(np.mean(losses[-10:])) if losses else float("nan")
                eta = (elapsed / step) * (n_batches - step) if step > 0 else float("nan")
                logger.info(
                    "Epoch %04d step %d/%d avg_loss(last10)=%.6g elapsed=%.1fs eta~%.1fs",
                    epoch,
                    step,
                    n_batches,
                    avg_loss,
                    elapsed,
                    eta,
                )
                last_log_t = now

        train_loss = float(np.mean(losses)) if losses else float("nan")
        history.append(train_loss)
        logger.info("Epoch %04d: train_loss=%.6g", epoch, train_loss)

    featureset_hash = featureset.save(artifacts_dir)
    dump_yaml(run_dir / "config_snapshot.yaml", cfg)
    save_json(
        run_dir / "metrics.json",
        {
            "train_loss": history[-1] if history else None,
            "loss_history": history,
            "n_train": int(len(train_data)),
            "seed": int(seed),
            "method": method,
        },
    )

    model_dir = ensure_dir(run_dir / "model")
    state_dict = model.state_dict()
    filtered = {k: v for k, v in state_dict.items() if not k.startswith("head.")}
    torch.save({"state_dict": filtered}, model_dir / "model.ckpt")
    save_json(model_dir / "featurizer_state.json", {"config": asdict(gcfg)})

    meta["featureset_hash"] = featureset_hash
    save_meta(run_dir, meta)
    logger.info("Saved pretrained encoder to %s", model_dir / "model.ckpt")
    logger.info("Done.")
    return run_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Self-supervised GraphCL pretraining (GNN).")
    ap.add_argument("--config", required=True, help="Path to a composed pretrain config.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
