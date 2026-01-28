from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.common.io import load_sdf_mol, sdf_path_from_cas
from src.common.utils import ensure_dir
from src.fp.featurizer_embedding import PretrainedEmbeddingConfig, featurize_mol as featurize_embedding_mol
from src.fp.featurizer_fp import FPConfig, featurize_mol


def hash_cfg(obj: Dict[str, Any]) -> str:
    s = repr(obj).encode("utf-8")
    return hashlib.sha256(s).hexdigest()[:12]


def build_features(
    df: pd.DataFrame,
    sdf_dir: Path,
    cas_col: str,
    fp_cfg: FPConfig,
    cache_dir: Optional[Path],
    cache_key: str,
    logger,
) -> Tuple[np.ndarray, List[Any], List[str], Dict[str, Any]]:
    """Return X (N,D), ids (CAS), elements_list, meta.

    Notes:
      - Rows with missing/invalid SDF are returned as all-NaN feature rows.
      - Descriptor NaNs are allowed; they will be imputed later.
    """
    ensure_dir(cache_dir) if cache_dir is not None else None
    cache_path = None
    meta_path = None
    if cache_dir is not None:
        cache_path = cache_dir / f"fp_features_{cache_key}.npz"
        meta_path = cache_dir / f"fp_features_{cache_key}_meta.pkl"
        if cache_path.exists() and meta_path.exists():
            logger.info(f"Loading cached features: {cache_path}")
            npz = np.load(cache_path, allow_pickle=True)
            X = npz["X"]
            ids = npz["ids"].tolist()
            elements = npz["elements"].tolist()
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            return X, ids, elements, meta

    ids = df[cas_col].astype(str).tolist()
    elements = df.get("elements", pd.Series([""] * len(df))).astype(str).tolist()
    X_list: List[np.ndarray] = []
    meta_last: Dict[str, Any] = {}

    logger.info("Featurizing molecules (fingerprint + optional descriptors) ...")
    for cas in tqdm(ids, total=len(ids)):
        mol = load_sdf_mol(sdf_path_from_cas(sdf_dir, cas))
        if mol is None:
            X_list.append(np.array([np.nan], dtype=float))
            continue
        x, meta = featurize_mol(mol, fp_cfg)
        meta_last = meta
        X_list.append(x)

    dims = [x.shape[0] for x in X_list if x.ndim == 1 and x.shape[0] > 1]
    if len(dims) == 0:
        raise RuntimeError("No valid molecules could be featurized. Check sdf_dir and SDF files.")
    dim = int(max(dims))
    X = np.full((len(X_list), dim), np.nan, dtype=float)
    for i, x in enumerate(X_list):
        if x.ndim != 1:
            continue
        if x.shape[0] == 1 and np.isnan(x[0]):
            continue
        if x.shape[0] != dim:
            x2 = np.full((dim,), np.nan, dtype=float)
            n = min(dim, x.shape[0])
            x2[:n] = x[:n]
            x = x2
        X[i] = x

    meta_last["feature_dim"] = dim

    if cache_path is not None:
        logger.info(f"Saving feature cache: {cache_path}")
        np.savez_compressed(
            cache_path,
            X=X,
            ids=np.array(ids, dtype=object),
            elements=np.array(elements, dtype=object),
        )
        with open(meta_path, "wb") as f:
            pickle.dump(meta_last, f)

    return X, ids, elements, meta_last


def build_embedding_features(
    df: pd.DataFrame,
    sdf_dir: Path,
    cas_col: str,
    emb_cfg: PretrainedEmbeddingConfig,
    cache_dir: Optional[Path],
    cache_key: str,
    logger,
) -> Tuple[np.ndarray, List[Any], List[str], Dict[str, Any]]:
    """Return X (N,D), ids (CAS), elements_list, meta for embedding features."""
    ensure_dir(cache_dir) if cache_dir is not None else None
    cache_path = None
    meta_path = None
    if cache_dir is not None:
        cache_path = cache_dir / f"emb_features_{cache_key}.npz"
        meta_path = cache_dir / f"emb_features_{cache_key}_meta.pkl"
        if cache_path.exists() and meta_path.exists():
            logger.info(f"Loading cached features: {cache_path}")
            npz = np.load(cache_path, allow_pickle=True)
            X = npz["X"]
            ids = npz["ids"].tolist()
            elements = npz["elements"].tolist()
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            return X, ids, elements, meta

    ids = df[cas_col].astype(str).tolist()
    elements = df.get("elements", pd.Series([""] * len(df))).astype(str).tolist()
    X_list: List[np.ndarray] = []
    meta_last: Dict[str, Any] = {}

    logger.info("Featurizing molecules (pretrained embedding stub) ...")
    for cas in tqdm(ids, total=len(ids)):
        mol = load_sdf_mol(sdf_path_from_cas(sdf_dir, cas))
        if mol is None:
            X_list.append(np.full((int(emb_cfg.embedding_dim),), np.nan, dtype=float))
            continue
        x, meta = featurize_embedding_mol(mol, emb_cfg)
        meta_last = meta
        X_list.append(x)

    X = np.vstack(X_list).astype(float)
    if np.isnan(X).all():
        raise RuntimeError("No valid molecules could be featurized. Check sdf_dir and SDF files.")
    meta_last["feature_dim"] = int(X.shape[1])

    if cache_path is not None:
        logger.info(f"Saving feature cache: {cache_path}")
        np.savez_compressed(
            cache_path,
            X=X,
            ids=np.array(ids, dtype=object),
            elements=np.array(elements, dtype=object),
        )
        with open(meta_path, "wb") as f:
            pickle.dump(meta_last, f)

    return X, ids, elements, meta_last
