from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.common.config import load_config
from src.common.conformer import (
    ConformerCache,
    ConformerService,
    conformer_config_from_cfg,
    conformer_config_payload,
    mol_from_smiles,
)
from src.common.io import load_sdf_mol, read_csv, sdf_path_from_cas
from src.common.plots import save_embedding_plot
from src.featuresets.base import GraphFeatureSet, SmilesFeatureSet, TabularFeatureSet
from src.featuresets.registry import load_featureset
from src.fp.feature_utils import hash_cfg
from src.gnn.featurizer_graph import GraphFeaturizerError, requires_3d_pos
from src.models import (
    create_model,
    feature_inputs_from_featureset,
    get_model_spec,
    resolve_model_name,
    validate_model_requirements,
)
from src.tasks import resolve_task
from src.visualize.utils import sanitize_filename

try:
    import torch
    from torch_geometric.loader import DataLoader as GeoDataLoader
except Exception:  # pragma: no cover
    torch = None
    GeoDataLoader = None


def _normalize_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if str(v)]
    return [str(value)]


def _select_device(prefer: str) -> "torch.device":
    if torch is None:
        raise ImportError("PyTorch is required for embedding extraction.")
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


def _resolve_eval_cfg(cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    input_cfg = cfg.get("input", {}) or {}
    eval_dir = input_cfg.get("evaluate_run_dir")
    if not eval_dir:
        return None
    eval_cfg_path = Path(eval_dir) / "config.yaml"
    if eval_cfg_path.exists():
        return load_config(eval_cfg_path)
    return None


def _resolve_model_artifact_dir(cfg: Dict[str, Any], eval_cfg: Optional[Dict[str, Any]]) -> Path:
    input_cfg = cfg.get("input", {}) or {}
    model_dir = input_cfg.get("model_artifact_dir")
    if not model_dir and eval_cfg:
        model_dir = eval_cfg.get("model_artifact_dir")
    if not model_dir:
        raise ValueError("embedding visualization requires input.model_artifact_dir or input.evaluate_run_dir.")
    return Path(model_dir)


def _resolve_data_context(
    train_cfg: Dict[str, Any],
    eval_cfg: Optional[Dict[str, Any]],
) -> Tuple[Path, Path, Path, str, str, Optional[str], Dict[str, Any]]:
    data_cfg = train_cfg.get("data", {}) or {}
    data_override = (eval_cfg or {}).get("data", {}) or {}
    dataset_csv = Path(data_override.get("dataset_csv", data_cfg.get("dataset_csv", "data/processed/dataset_with_lj.csv")))
    indices_dir = Path(data_override.get("indices_dir", data_cfg.get("indices_dir", "data/processed/indices")))
    sdf_dir = Path(data_override.get("sdf_dir", data_cfg.get("sdf_dir", "data/raw/sdf_files")))
    cas_col = str(data_override.get("cas_col", data_cfg.get("cas_col", "CAS")))
    columns_cfg = train_cfg.get("columns", {}) or {}
    sample_id_col = str(
        data_override.get("sample_id_col", data_cfg.get("sample_id_col", columns_cfg.get("sample_id", cas_col)))
    )
    smiles_col = data_override.get("smiles_col", data_cfg.get("smiles_col", columns_cfg.get("smiles", None)))
    smiles_col = str(smiles_col) if smiles_col is not None else None
    return dataset_csv, indices_dir, sdf_dir, cas_col, sample_id_col, smiles_col, data_cfg


def _filter_pred_df(
    pred_df: pd.DataFrame,
    embed_cfg: Dict[str, Any],
    logger,
) -> pd.DataFrame:
    if "sample_id" not in pred_df.columns:
        raise ValueError("predictions.csv missing sample_id for embedding visualization.")
    df = pred_df.copy()
    df["sample_id"] = df["sample_id"].astype(str)
    splits = _normalize_list(embed_cfg.get("splits") or embed_cfg.get("split"))
    if splits and "split" in df.columns:
        df = df[df["split"].astype(str).isin([str(s) for s in splits])]
    elif splits:
        logger.warning("embedding.splits requested but predictions.csv has no split column; using all rows.")

    max_samples = int(embed_cfg.get("max_samples", 0))
    if max_samples > 0 and len(df) > max_samples:
        random_state = int(embed_cfg.get("random_state", 42))
        strategy = str(embed_cfg.get("sample_strategy", "random")).lower()
        rng = np.random.default_rng(random_state)
        if strategy == "head":
            keep_idx = list(range(max_samples))
        else:
            perm = rng.permutation(len(df))[:max_samples]
            keep_idx = sorted(int(i) for i in perm)
        df = df.iloc[keep_idx].reset_index(drop=True)
    return df.reset_index(drop=True)


def _reindex_by_sample_id(df: pd.DataFrame, sample_ids: Sequence[str]) -> pd.DataFrame:
    if "sample_id" not in df.columns:
        return df
    lookup = df.set_index("sample_id")
    existing = [sid for sid in sample_ids if sid in lookup.index]
    if not existing:
        return df.iloc[0:0].copy()
    return lookup.loc[existing].reset_index()


def _resolve_element_count_column(df: pd.DataFrame) -> Optional[str]:
    for col in ["n_elements", "n_heavy_atoms"]:
        if col in df.columns:
            return col
    return None


def resolve_color_values(
    embed_df: pd.DataFrame,
    color_by: Optional[str],
    color_target: Optional[str],
    primary_target: Optional[str],
) -> Tuple[Optional[pd.Series], Optional[str], Optional[str]]:
    if not color_by:
        return None, None, None
    color_by = str(color_by).strip().lower()
    if not color_by or color_by in {"none", "off"}:
        return None, None, None

    if color_by == "split":
        if "split" in embed_df.columns:
            return embed_df["split"], "split", "split"
        return None, None, None

    if color_by in {"element_count", "n_elements", "n_heavy_atoms"}:
        col = _resolve_element_count_column(embed_df)
        if col:
            return embed_df[col], col, col
        return None, None, None

    if color_by == "elements":
        if "elements" in embed_df.columns:
            return embed_df["elements"], "elements", "elements"
        return None, None, None

    if color_by == "target":
        target = str(color_target or primary_target or "").strip()
        candidates = []
        if target:
            candidates.extend([f"y_true_{target}", f"y_pred_{target}"])
        candidates.extend(["y_true", "y_pred"])
        for col in candidates:
            if col in embed_df.columns:
                label = target or col.replace("y_true_", "").replace("y_pred_", "")
                return embed_df[col], label, col
        return None, None, None

    if color_by in embed_df.columns:
        return embed_df[color_by], color_by, color_by

    return None, None, None


def project_embeddings(
    embeddings: np.ndarray,
    method: str,
    random_state: int,
    params: Optional[Dict[str, Any]] = None,
    logger=None,
) -> np.ndarray:
    method = str(method or "tsne").lower()
    params = dict(params or {})
    n_samples = int(embeddings.shape[0])
    if n_samples < 2:
        raise ValueError("Need at least 2 samples for embedding projection.")

    if method in {"identity", "none"}:
        if embeddings.shape[1] < 2:
            raise ValueError("Embedding dimension < 2 for identity projection.")
        return embeddings[:, :2]

    if method == "pca":
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2, random_state=random_state)
        return pca.fit_transform(embeddings)

    if method in {"tsne", "t-sne"}:
        from sklearn.manifold import TSNE

        perplexity = float(params.pop("perplexity", 30.0))
        max_perplexity = max(1.0, (n_samples - 1) / 3.0)
        if perplexity > max_perplexity:
            if logger:
                logger.info("t-SNE perplexity adjusted from %s to %s", perplexity, max_perplexity)
            perplexity = max_perplexity
        tsne = TSNE(
            n_components=2,
            random_state=random_state,
            perplexity=perplexity,
            init=str(params.pop("init", "pca")),
            learning_rate=params.pop("learning_rate", "auto"),
            metric=str(params.pop("metric", "euclidean")),
            **params,
        )
        return tsne.fit_transform(embeddings)

    if method == "umap":
        try:
            import umap
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "UMAP is not installed. Install optional deps: pip install umap-learn"
            ) from exc
        reducer = umap.UMAP(
            n_components=2,
            random_state=random_state,
            n_neighbors=int(params.pop("n_neighbors", 15)),
            min_dist=float(params.pop("min_dist", 0.1)),
            metric=str(params.pop("metric", "euclidean")),
            **params,
        )
        return reducer.fit_transform(embeddings)

    raise ValueError(f"Unknown embedding projection method: {method}")


def _build_graph_data(
    df: pd.DataFrame,
    sample_ids: Sequence[str],
    sdf_dir: Path,
    cas_col: str,
    sample_id_col: str,
    smiles_col: Optional[str],
    pipeline,
    conformer_service: Optional[ConformerService],
    allow_generate: bool,
    requires_pos: bool,
    logger,
) -> Tuple[List[Any], List[str]]:
    id_col = sample_id_col if sample_id_col in df.columns else cas_col
    id_map = {str(v): int(i) for i, v in enumerate(df[id_col].astype(str).tolist())}
    missing = [sid for sid in sample_ids if sid not in id_map]
    if missing:
        logger.warning("embedding skipped %s samples missing from dataset.", len(missing))
    row_indices = [id_map[sid] for sid in sample_ids if sid in id_map]

    data_list: List[Any] = []
    kept_ids: List[str] = []
    for idx in row_indices:
        row = df.iloc[idx]
        cas = str(row[cas_col]) if cas_col in row else str(row[id_col])
        sample_id = str(row[sample_id_col]) if sample_id_col in row else cas
        smiles = ""
        if smiles_col and smiles_col in row and pd.notna(row[smiles_col]):
            smiles = str(row[smiles_col])
        mol = load_sdf_mol(sdf_path_from_cas(sdf_dir, cas))
        if mol is None and smiles:
            mol = mol_from_smiles(smiles)
        if mol is None:
            continue
        pos = None
        if conformer_service is not None:
            pos, meta = conformer_service.get_pos(
                sample_id=sample_id, mol=mol, smiles=smiles, allow_generate=allow_generate
            )
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
        except GraphFeaturizerError:
            if requires_pos:
                raise
            continue
        data_list.append(data)
        kept_ids.append(sample_id)
    return data_list, kept_ids


def _pool_graph_features(data_list: Sequence[Any], pooling: str = "mean") -> np.ndarray:
    pooled = []
    pooling = str(pooling).lower()
    for data in data_list:
        x = data.x.detach().cpu().numpy()
        if pooling == "sum":
            vec = x.sum(axis=0)
        else:
            vec = x.mean(axis=0)
        if hasattr(data, "u"):
            u = data.u.detach().cpu().numpy().reshape(-1)
            vec = np.concatenate([vec, u], axis=0)
        pooled.append(vec)
    if not pooled:
        return np.zeros((0, 0))
    return np.stack(pooled, axis=0)


def _compute_tabular_embeddings(
    featureset: TabularFeatureSet,
    df: pd.DataFrame,
    sample_ids: Sequence[str],
    sdf_dir: Path,
    cas_col: str,
    data_cfg: Dict[str, Any],
    train_cfg: Dict[str, Any],
    dataset_csv: Path,
    logger,
) -> Tuple[np.ndarray, List[str]]:
    df_subset = df[df[cas_col].astype(str).isin([str(sid) for sid in sample_ids])].reset_index(drop=True)
    if df_subset.empty:
        return np.zeros((0, 0)), []
    cache_dir = data_cfg.get("cache_dir", None)
    cache_dir = Path(cache_dir) if cache_dir else None
    feat_cfg = train_cfg.get("featurizer", {})
    cache_key = hash_cfg({"featurizer": feat_cfg, "dataset": str(dataset_csv)})
    X_all, ids, _, _ = featureset.build_features(
        df=df_subset,
        sdf_dir=sdf_dir,
        cas_col=cas_col,
        cache_dir=cache_dir,
        cache_key=cache_key,
        logger=logger,
    )
    X_all = featureset.transform(X_all)
    id_map = {str(cid): int(i) for i, cid in enumerate(ids)}
    selected_idx = []
    kept_ids: List[str] = []
    for sid in sample_ids:
        idx = id_map.get(str(sid))
        if idx is None:
            continue
        selected_idx.append(idx)
        kept_ids.append(str(sid))
    if not selected_idx:
        return np.zeros((0, 0)), []
    embeddings = X_all[np.array(selected_idx, dtype=int)]
    return embeddings, kept_ids


def _compute_gnn_embeddings(
    featureset: GraphFeatureSet,
    train_cfg: Dict[str, Any],
    artifacts_dir: Path,
    df: pd.DataFrame,
    sample_ids: Sequence[str],
    sdf_dir: Path,
    cas_col: str,
    sample_id_col: str,
    smiles_col: Optional[str],
    embed_cfg: Dict[str, Any],
    logger,
) -> Tuple[np.ndarray, List[str], str]:
    pipeline = featureset.pipeline
    gcfg = pipeline.graph_cfg
    requires_pos = requires_3d_pos(gcfg)
    conformer_service = None
    if gcfg.use_3d_pos or requires_pos:
        conformer_cfg = conformer_config_from_cfg(train_cfg)
        cache_path = artifacts_dir / "conformer_cache.json"
        if not cache_path.exists() and conformer_cfg.enabled:
            raise FileNotFoundError(f"conformer cache not found: {cache_path}")
        conformer_cache = (
            ConformerCache(cache_path, config=conformer_config_payload(conformer_cfg))
            if cache_path.exists()
            else None
        )
        conformer_service = ConformerService(conformer_cfg, cache=conformer_cache)

    allow_generate = bool(embed_cfg.get("allow_generate", False))
    data_list, kept_ids = _build_graph_data(
        df=df,
        sample_ids=sample_ids,
        sdf_dir=sdf_dir,
        cas_col=cas_col,
        sample_id_col=sample_id_col,
        smiles_col=smiles_col,
        pipeline=pipeline,
        conformer_service=conformer_service,
        allow_generate=allow_generate,
        requires_pos=requires_pos,
        logger=logger,
    )
    if not data_list:
        return np.zeros((0, 0)), [], "none"

    source = str(embed_cfg.get("source", "encoder")).lower()
    if source == "features":
        embeddings = _pool_graph_features(data_list, pooling=str(embed_cfg.get("feature_pooling", "mean")))
        return embeddings, kept_ids, "features"

    if torch is None or GeoDataLoader is None:
        raise ImportError("PyTorch Geometric is required for encoder embeddings.")

    task_spec = resolve_task(train_cfg)
    target_columns = task_spec.target_columns
    if not target_columns:
        raise ValueError("No target column resolved from training config.")
    model_cfg = train_cfg.get("model", {}) or {}
    model_name = resolve_model_name(train_cfg, default="mpnn")
    model_spec = get_model_spec(model_name)
    feature_inputs = feature_inputs_from_featureset(featureset)
    validate_model_requirements(model_spec, target_columns, feature_inputs)

    sample_data = data_list[0]
    in_dim = sample_data.x.shape[1]
    edge_dim = sample_data.edge_attr.shape[1] if hasattr(sample_data, "edge_attr") else 0
    global_dim = int(sample_data.u.shape[1]) if hasattr(sample_data, "u") else 0
    model = create_model(
        model_name,
        model_cfg=model_cfg,
        context={"in_dim": in_dim, "edge_dim": edge_dim, "global_dim": global_dim, "out_dim": len(target_columns)},
    )

    prefer_device = embed_cfg.get("device", train_cfg.get("train", {}).get("device", "auto"))
    device = _select_device(prefer_device)
    model = model.to(device)
    state_path = artifacts_dir / "model_best.pt"
    model.load_state_dict(torch.load(state_path, map_location=device))
    model.eval()

    if not hasattr(model, "encode"):
        if source == "encoder":
            embeddings = _pool_graph_features(data_list, pooling=str(embed_cfg.get("feature_pooling", "mean")))
            return embeddings, kept_ids, "features"
        raise ValueError("Model does not support encoder embeddings.")

    batch_size = int(embed_cfg.get("batch_size", train_cfg.get("train", {}).get("batch_size", 32)))
    loader = GeoDataLoader(data_list, batch_size=batch_size, shuffle=False)
    out = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            emb_t = model.encode(batch)
            out.append(emb_t.detach().cpu().numpy())
    embeddings = np.concatenate(out, axis=0) if out else np.zeros((0, 0))
    return embeddings, kept_ids, "encoder"


def _compute_smiles_embeddings(
    featureset: SmilesFeatureSet,
    train_cfg: Dict[str, Any],
    artifacts_dir: Path,
    df: pd.DataFrame,
    sample_ids: Sequence[str],
    sample_id_col: str,
    smiles_col: Optional[str],
    embed_cfg: Dict[str, Any],
    logger,
) -> Tuple[np.ndarray, List[str], str]:
    if torch is None:
        raise ImportError("PyTorch is required for SMILES embeddings.")
    if not smiles_col or smiles_col not in df.columns:
        raise ValueError("smiles column not found in dataset for SMILES embeddings.")

    id_col = sample_id_col if sample_id_col in df.columns else None
    if id_col is None:
        raise ValueError("sample_id column not found for SMILES embeddings.")

    id_map = {str(v): int(i) for i, v in enumerate(df[id_col].astype(str).tolist())}
    kept_ids: List[str] = []
    smiles_list: List[str] = []
    for sid in sample_ids:
        idx = id_map.get(str(sid))
        if idx is None:
            continue
        smiles = str(df.iloc[idx][smiles_col])
        if not smiles or smiles.lower() == "nan":
            continue
        kept_ids.append(str(sid))
        smiles_list.append(smiles)
    if not smiles_list:
        return np.zeros((0, 0)), [], "none"

    task_spec = resolve_task(train_cfg)
    target_columns = task_spec.target_columns
    if not target_columns:
        raise ValueError("No target column resolved from training config.")
    model_cfg = train_cfg.get("model", {}) or {}
    model_name = resolve_model_name(train_cfg, default="chemberta")
    model_spec = get_model_spec(model_name)
    feature_inputs = feature_inputs_from_featureset(featureset)
    validate_model_requirements(model_spec, target_columns, feature_inputs)

    model = create_model(model_name, model_cfg=model_cfg, context={"out_dim": len(target_columns)})
    prefer_device = embed_cfg.get("device", train_cfg.get("train", {}).get("device", "auto"))
    device = _select_device(prefer_device)
    model = model.to(device)
    state_path = artifacts_dir / "model_best.pt"
    model.load_state_dict(torch.load(state_path, map_location=device))
    model.eval()

    if not hasattr(model, "encode"):
        raise ValueError("SMILES model does not support encoder embeddings.")

    batch_size = int(embed_cfg.get("batch_size", train_cfg.get("train", {}).get("batch_size", 16)))
    out = []
    with torch.no_grad():
        for start in range(0, len(smiles_list), batch_size):
            batch_smiles = smiles_list[start : start + batch_size]
            tokens = featureset.tokenize_batch(batch_smiles)
            input_ids = torch.tensor(tokens.get("input_ids"), dtype=torch.long, device=device)
            attention_mask = tokens.get("attention_mask")
            if attention_mask is not None:
                attention_mask = torch.tensor(attention_mask, dtype=torch.long, device=device)
            token_type_ids = tokens.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = torch.tensor(token_type_ids, dtype=torch.long, device=device)
            emb_t = model.encode(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            out.append(emb_t.detach().cpu().numpy())
    embeddings = np.concatenate(out, axis=0) if out else np.zeros((0, 0))
    return embeddings, kept_ids, "encoder"


def run_embedding_visualization(
    cfg: Dict[str, Any],
    pred_df: pd.DataFrame,
    out_dir: Path,
    plots_dir: Path,
    primary_target: Optional[str],
    logger,
) -> Optional[Path]:
    embed_cfg = cfg.get("plots", {}).get("embedding", {}) or {}
    if not bool(embed_cfg.get("enabled", False)):
        return None
    on_fail = str(embed_cfg.get("on_fail", "skip")).lower()
    try:
        eval_cfg = _resolve_eval_cfg(cfg)
        model_artifact_dir = _resolve_model_artifact_dir(cfg, eval_cfg)
        train_cfg_path = model_artifact_dir / "config_snapshot.yaml"
        if not train_cfg_path.exists():
            raise FileNotFoundError(f"config_snapshot.yaml not found in model dir: {train_cfg_path}")
        train_cfg = load_config(train_cfg_path)
        artifacts_dir = model_artifact_dir / "artifacts"
        if not artifacts_dir.exists():
            raise FileNotFoundError(f"Artifacts dir not found: {artifacts_dir}")

        dataset_csv, _, sdf_dir, cas_col, sample_id_col, smiles_col, data_cfg = _resolve_data_context(
            train_cfg, eval_cfg
        )
        if not dataset_csv.exists():
            raise FileNotFoundError(f"dataset_csv not found: {dataset_csv}")

        pred_sel_df = _filter_pred_df(pred_df, embed_cfg, logger=logger)
        if pred_sel_df.empty:
            raise ValueError("No rows selected for embedding visualization.")
        sample_ids = pred_sel_df["sample_id"].astype(str).tolist()

        df = read_csv(dataset_csv).reset_index(drop=True)
        featureset = load_featureset(artifacts_dir, train_cfg)

        embeddings = np.zeros((0, 0))
        kept_ids: List[str] = []
        source = "none"
        if isinstance(featureset, TabularFeatureSet):
            embeddings, kept_ids = _compute_tabular_embeddings(
                featureset=featureset,
                df=df,
                sample_ids=sample_ids,
                sdf_dir=sdf_dir,
                cas_col=cas_col,
                data_cfg=data_cfg,
                train_cfg=train_cfg,
                dataset_csv=dataset_csv,
                logger=logger,
            )
            source = "features"
        elif isinstance(featureset, GraphFeatureSet):
            embeddings, kept_ids, source = _compute_gnn_embeddings(
                featureset=featureset,
                train_cfg=train_cfg,
                artifacts_dir=artifacts_dir,
                df=df,
                sample_ids=sample_ids,
                sdf_dir=sdf_dir,
                cas_col=cas_col,
                sample_id_col=sample_id_col,
                smiles_col=smiles_col,
                embed_cfg=embed_cfg,
                logger=logger,
            )
        elif isinstance(featureset, SmilesFeatureSet):
            embeddings, kept_ids, source = _compute_smiles_embeddings(
                featureset=featureset,
                train_cfg=train_cfg,
                artifacts_dir=artifacts_dir,
                df=df,
                sample_ids=sample_ids,
                sample_id_col=sample_id_col,
                smiles_col=smiles_col,
                embed_cfg=embed_cfg,
                logger=logger,
            )
        else:
            raise ValueError("Unsupported featureset for embedding visualization.")

        if embeddings.size == 0 or not kept_ids:
            raise ValueError("No embeddings produced for visualization.")

        finite_mask = np.isfinite(embeddings).all(axis=1)
        if not np.all(finite_mask):
            embeddings = embeddings[finite_mask]
            kept_ids = [sid for sid, ok in zip(kept_ids, finite_mask.tolist()) if ok]
        if embeddings.size == 0:
            raise ValueError("Embeddings are empty after filtering non-finite values.")

        pred_sel_df = _reindex_by_sample_id(pred_sel_df, kept_ids)
        if pred_sel_df.empty:
            raise ValueError("No predictions matched embedding samples.")

        embed_df = pd.DataFrame({"sample_id": kept_ids})
        embed_df = embed_df.merge(pred_sel_df, on="sample_id", how="left")

        color_by = embed_cfg.get("color_by", "target")
        color_target = embed_cfg.get("color_target")
        needs_dataset = str(color_by).lower() in {"element_count", "n_elements", "n_heavy_atoms", "elements"}
        if needs_dataset and not df.empty:
            id_col = sample_id_col if sample_id_col in df.columns else cas_col
            subset_cols = [id_col]
            element_col = _resolve_element_count_column(df) if str(color_by).lower() != "elements" else "elements"
            if element_col and element_col in df.columns:
                subset_cols.append(element_col)
            if "elements" in df.columns and "elements" not in subset_cols:
                subset_cols.append("elements")
            dataset_slice = df[subset_cols].copy()
            dataset_slice["sample_id"] = dataset_slice[id_col].astype(str)
            dataset_slice = dataset_slice.drop(columns=[id_col])
            embed_df = embed_df.merge(dataset_slice, on="sample_id", how="left")

        color_values, color_label, color_col = resolve_color_values(
            embed_df, color_by=color_by, color_target=color_target, primary_target=primary_target
        )

        method = str(embed_cfg.get("method", "tsne")).lower()
        params = embed_cfg.get("params", {}) or {}
        random_state = int(embed_cfg.get("random_state", 42))
        scale = bool(embed_cfg.get("scale", False))
        if scale and embeddings.shape[0] > 1:
            from sklearn.preprocessing import StandardScaler

            embeddings = StandardScaler().fit_transform(embeddings)

        coords = project_embeddings(
            embeddings=embeddings,
            method=method,
            random_state=random_state,
            params=params,
            logger=logger,
        )

        embed_df["proj_0"] = coords[:, 0]
        embed_df["proj_1"] = coords[:, 1]
        embed_df["embedding_source"] = source
        embed_df["embedding_method"] = method

        save_full = bool(embed_cfg.get("save_full", False))
        if save_full:
            for idx in range(embeddings.shape[1]):
                embed_df[f"emb_{idx}"] = embeddings[:, idx]

        out_csv = str(embed_cfg.get("output_csv", "embeddings.csv"))
        out_path = out_dir / out_csv
        embed_df.to_csv(out_path, index=False)

        plot_name = f"embedding_{sanitize_filename(method)}"
        if color_label:
            plot_name = f"{plot_name}_{sanitize_filename(color_label)}"
        plot_path = plots_dir / f"{plot_name}.png"
        save_embedding_plot(
            coords=coords,
            out_path=plot_path,
            color=color_values.to_numpy() if isinstance(color_values, pd.Series) else color_values,
            title=f"Embedding ({method})",
            color_label=color_label,
            max_categories=int(embed_cfg.get("max_categories", 12)),
        )

        logger.info("Saved embeddings to %s", out_path)
        logger.info("Saved embedding plot to %s", plot_path)
        return out_path
    except Exception as exc:
        if on_fail in {"skip", "warn"}:
            logger.warning("Embedding visualization skipped: %s", exc)
            return None
        raise
