from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.common.chemistry import elements_string, get_elements_from_mol, parse_formula
from src.common.config import load_config
from src.common.conformer import mol_from_smiles
from src.common.descriptors import calc_descriptors
from src.common.io import load_sdf_mol, read_csv, sdf_path_from_cas
from src.tasks import resolve_task
from src.visualize.utils import PlotTarget, resolve_plot_targets

DEFAULT_ELEMENT_SLICES = ["C", "H", "O", "N", "S", "P", "F", "Cl", "Br", "I"]
DEFAULT_HALOGENS = ["F", "Cl", "Br", "I"]
DEFAULT_SIZE_BINS = {
    "n_atoms": [0, 5, 10, 20, 30, 40, 60, 80, 120],
    "n_heavy_atoms": [0, 5, 10, 20, 30, 40, 60, 80],
}
DEFAULT_RING_BINS = [0, 1, 2, 3, 4, 6, 10]
DEFAULT_ROT_BONDS_BINS = [0, 1, 2, 3, 4, 6, 10, 20]


@dataclass(frozen=True)
class ErrorSlicingConfig:
    splits: List[str]
    min_samples: int
    target_mode: str
    element_slices: List[str]
    include_halogen: bool
    size_bins: Dict[str, List[float]]
    ring_bins: List[float]
    rot_bonds_bins: List[float]


def _normalize_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        out = [str(v).strip() for v in value if str(v).strip()]
        return out
    item = str(value).strip()
    return [item] if item else []


def _normalize_bins(value: Any) -> List[float]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        out: List[float] = []
        for item in value:
            try:
                out.append(float(item))
            except Exception:
                continue
        return out
    try:
        return [float(value)]
    except Exception:
        return []


def parse_error_slicing_config(plots_cfg: Dict[str, Any]) -> ErrorSlicingConfig:
    err_cfg = plots_cfg.get("error_slicing", {}) or {}
    splits = _normalize_list(err_cfg.get("splits") or plots_cfg.get("splits") or ["val", "test"])
    if not splits:
        splits = ["all"]

    target_mode = str(err_cfg.get("target_mode", plots_cfg.get("target_mode", "primary"))).lower()
    min_samples = int(err_cfg.get("min_samples", 10))

    if "element_slices" in err_cfg:
        element_slices = _normalize_list(err_cfg.get("element_slices"))
    elif "elements" in err_cfg:
        element_slices = _normalize_list(err_cfg.get("elements"))
    else:
        element_slices = list(DEFAULT_ELEMENT_SLICES)
    include_halogen = bool(err_cfg.get("include_halogen", True))

    if "size_bins" in err_cfg:
        size_bins_cfg = err_cfg.get("size_bins", {}) or {}
        size_bins = {}
        for key, bins in size_bins_cfg.items():
            norm = _normalize_bins(bins)
            if norm:
                size_bins[str(key)] = norm
    else:
        size_bins = {key: list(vals) for key, vals in DEFAULT_SIZE_BINS.items()}

    if "ring_bins" in err_cfg:
        ring_bins = _normalize_bins(err_cfg.get("ring_bins"))
    else:
        ring_bins = list(DEFAULT_RING_BINS)

    if "rot_bonds_bins" in err_cfg or "rotatable_bonds_bins" in err_cfg:
        rot_bonds_bins = _normalize_bins(
            err_cfg.get("rot_bonds_bins", err_cfg.get("rotatable_bonds_bins"))
        )
    else:
        rot_bonds_bins = list(DEFAULT_ROT_BONDS_BINS)

    return ErrorSlicingConfig(
        splits=splits,
        min_samples=min_samples,
        target_mode=target_mode,
        element_slices=element_slices,
        include_halogen=include_halogen,
        size_bins=size_bins,
        ring_bins=ring_bins,
        rot_bonds_bins=rot_bonds_bins,
    )


def _resolve_eval_cfg(cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    input_cfg = cfg.get("input", {}) or {}
    eval_dir = input_cfg.get("evaluate_run_dir")
    if not eval_dir:
        return None
    eval_cfg_path = Path(eval_dir) / "config.yaml"
    if eval_cfg_path.exists():
        return load_config(eval_cfg_path)
    return None


def _resolve_model_artifact_dir(
    cfg: Dict[str, Any],
    eval_cfg: Optional[Dict[str, Any]],
    err_cfg: Dict[str, Any],
) -> Optional[Path]:
    err_input = err_cfg.get("input", {}) or {}
    input_cfg = cfg.get("input", {}) or {}
    model_dir = err_input.get("model_artifact_dir") or input_cfg.get("model_artifact_dir")
    if not model_dir and eval_cfg:
        model_dir = eval_cfg.get("model_artifact_dir")
    return Path(model_dir) if model_dir else None


def _resolve_data_context(
    cfg: Dict[str, Any],
    train_cfg: Optional[Dict[str, Any]],
    eval_cfg: Optional[Dict[str, Any]],
    err_cfg: Dict[str, Any],
) -> Tuple[Optional[Path], Optional[Path], Optional[str], Optional[str], Optional[str], Optional[str], str]:
    err_input = err_cfg.get("input", {}) or {}
    base_cfg = train_cfg or cfg
    data_cfg = base_cfg.get("data", {}) or {}
    data_override = (eval_cfg or {}).get("data", {}) or {}
    columns_cfg = base_cfg.get("columns", {}) or {}

    dataset_csv = (
        err_input.get("dataset_csv")
        or data_override.get("dataset_csv")
        or data_cfg.get("dataset_csv")
    )
    sdf_dir = err_input.get("sdf_dir") or data_override.get("sdf_dir") or data_cfg.get("sdf_dir")
    cas_col = str(err_input.get("cas_col") or data_override.get("cas_col") or data_cfg.get("cas_col") or "CAS")
    sample_id_col = (
        err_input.get("sample_id_col")
        or data_override.get("sample_id_col")
        or data_cfg.get("sample_id_col")
        or columns_cfg.get("sample_id")
        or cas_col
    )
    smiles_col = (
        err_input.get("smiles_col")
        or data_override.get("smiles_col")
        or data_cfg.get("smiles_col")
        or columns_cfg.get("smiles")
    )
    formula_col = (
        err_input.get("formula_col")
        or data_override.get("formula_col")
        or data_cfg.get("formula_col")
        or columns_cfg.get("formula")
    )
    sdf_path_col = (
        err_input.get("sdf_path_col")
        or data_override.get("sdf_path_col")
        or data_cfg.get("sdf_path_col")
        or columns_cfg.get("sdf_path")
        or columns_cfg.get("sdf")
        or columns_cfg.get("sdf_file")
    )

    return (
        Path(dataset_csv) if dataset_csv else None,
        Path(sdf_dir) if sdf_dir else None,
        str(sample_id_col) if sample_id_col else None,
        str(smiles_col) if smiles_col else None,
        str(formula_col) if formula_col else None,
        str(sdf_path_col) if sdf_path_col else None,
        str(cas_col) if cas_col else "CAS",
    )


def _resolve_id_column(
    df: pd.DataFrame,
    sample_id_col: Optional[str],
    cas_col: Optional[str],
) -> Optional[str]:
    if sample_id_col and sample_id_col in df.columns:
        return sample_id_col
    if cas_col and cas_col in df.columns:
        return cas_col
    return None


def _elements_from_string(value: Any) -> Tuple[str, List[str]]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "", []
    raw = str(value).strip()
    if not raw:
        return "", []
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return ",".join(sorted(set(parts))), parts


def _ensure_bins(values: np.ndarray, bins: List[float]) -> Optional[List[float]]:
    if not bins:
        return None
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    edges = sorted({float(b) for b in bins})
    if arr.min() < edges[0]:
        edges = [float(arr.min())] + edges
    if arr.max() >= edges[-1]:
        pad = 1.0 if float(arr.max()).is_integer() else 1e-6
        edges = edges + [float(arr.max()) + pad]
    if len(edges) < 2:
        return None
    return edges


def _format_edge(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.3g}"


def _build_bin_labels(edges: Sequence[float]) -> List[str]:
    labels = []
    for left, right in zip(edges[:-1], edges[1:]):
        labels.append(f"{_format_edge(left)}-{_format_edge(right)}")
    return labels


def _bin_series(values: pd.Series, bins: List[float]) -> Optional[pd.Series]:
    edges = _ensure_bins(values.to_numpy(), bins)
    if not edges:
        return None
    labels = _build_bin_labels(edges)
    return pd.cut(values, bins=edges, labels=labels, include_lowest=True, right=False)


def _safe_float_series(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def _resolve_mol(
    row: pd.Series,
    sample_id: str,
    smiles_col: Optional[str],
    sdf_path_col: Optional[str],
    sdf_dir: Optional[Path],
) -> Optional[Any]:
    if smiles_col and smiles_col in row.index:
        try:
            mol = mol_from_smiles(row.get(smiles_col))
            if mol is not None:
                return mol
        except Exception:
            return None
    if sdf_path_col and sdf_path_col in row.index:
        raw_path = row.get(sdf_path_col)
        if raw_path is not None and str(raw_path).strip():
            try:
                mol = load_sdf_mol(str(raw_path))
                if mol is not None:
                    return mol
            except Exception:
                return None
    if sdf_dir:
        try:
            return load_sdf_mol(sdf_path_from_cas(sdf_dir, sample_id))
        except Exception:
            return None
    return None


def _compute_features_from_mols(
    df: pd.DataFrame,
    sample_ids: Sequence[str],
    needs: Dict[str, bool],
    smiles_col: Optional[str],
    sdf_path_col: Optional[str],
    sdf_dir: Optional[Path],
) -> Dict[str, List[float | str | List[str]]]:
    out: Dict[str, List[float | str | List[str]]] = {
        "n_atoms": [],
        "n_heavy_atoms": [],
        "ring_count": [],
        "rotatable_bonds": [],
        "elements": [],
        "elements_list": [],
    }
    for idx, sample_id in zip(df.index, sample_ids):
        row = df.loc[idx]
        mol = _resolve_mol(row, sample_id, smiles_col, sdf_path_col, sdf_dir)
        if mol is None:
            out["n_atoms"].append(np.nan)
            out["n_heavy_atoms"].append(np.nan)
            out["ring_count"].append(np.nan)
            out["rotatable_bonds"].append(np.nan)
            out["elements"].append("")
            out["elements_list"].append([])
            continue

        out["n_atoms"].append(float(mol.GetNumAtoms()) if needs.get("n_atoms") else np.nan)
        out["n_heavy_atoms"].append(float(mol.GetNumHeavyAtoms()) if needs.get("n_heavy_atoms") else np.nan)

        ring_val = np.nan
        rot_val = np.nan
        if needs.get("ring_count") or needs.get("rotatable_bonds"):
            desc = calc_descriptors(mol, ["RingCount", "NumRotatableBonds"])
            if needs.get("ring_count"):
                ring_val = float(desc.get("RingCount", np.nan))
            if needs.get("rotatable_bonds"):
                rot_val = float(desc.get("NumRotatableBonds", np.nan))
        out["ring_count"].append(ring_val)
        out["rotatable_bonds"].append(rot_val)

        if needs.get("elements"):
            counts = get_elements_from_mol(mol)
            elements = sorted(counts.keys())
            out["elements"].append(elements_string(counts))
            out["elements_list"].append(elements)
        else:
            out["elements"].append("")
            out["elements_list"].append([])

    return out


def build_error_slicing_features(
    df: pd.DataFrame,
    slice_cfg: ErrorSlicingConfig,
    sample_id_col: Optional[str],
    cas_col: Optional[str],
    smiles_col: Optional[str],
    formula_col: Optional[str],
    sdf_path_col: Optional[str],
    sdf_dir: Optional[Path],
    logger,
) -> pd.DataFrame:
    df = df.copy()
    id_col = _resolve_id_column(df, sample_id_col, cas_col)
    if id_col:
        df["sample_id"] = df[id_col].astype(str)
    else:
        df["sample_id"] = df.index.astype(str)
    sample_ids = df["sample_id"].astype(str).tolist()

    features = pd.DataFrame({"sample_id": sample_ids})

    if "n_atoms" in df.columns:
        features["n_atoms"] = _safe_float_series(df["n_atoms"])
    if "n_heavy_atoms" in df.columns:
        features["n_heavy_atoms"] = _safe_float_series(df["n_heavy_atoms"])

    if "ring_count" in df.columns:
        features["ring_count"] = _safe_float_series(df["ring_count"])
    if "RingCount" in df.columns and "ring_count" not in features.columns:
        features["ring_count"] = _safe_float_series(df["RingCount"])

    if "rotatable_bonds" in df.columns:
        features["rotatable_bonds"] = _safe_float_series(df["rotatable_bonds"])
    if "NumRotatableBonds" in df.columns and "rotatable_bonds" not in features.columns:
        features["rotatable_bonds"] = _safe_float_series(df["NumRotatableBonds"])

    elements_series = None
    elements_list: List[List[str]] = []
    if "elements" in df.columns:
        cleaned, elements_list = zip(*[_elements_from_string(v) for v in df["elements"].tolist()])
        elements_series = pd.Series(list(cleaned), index=df.index)
    elif formula_col and formula_col in df.columns:
        values: List[str] = []
        for item in df[formula_col].tolist():
            counts = parse_formula(item)
            values.append(elements_string(counts))
            elements_list.append(sorted(counts.keys()))
        elements_series = pd.Series(values, index=df.index)

    needs = {
        "n_atoms": "n_atoms" not in features.columns and "n_atoms" in slice_cfg.size_bins,
        "n_heavy_atoms": "n_heavy_atoms" not in features.columns and "n_heavy_atoms" in slice_cfg.size_bins,
        "ring_count": "ring_count" not in features.columns and bool(slice_cfg.ring_bins),
        "rotatable_bonds": "rotatable_bonds" not in features.columns and bool(slice_cfg.rot_bonds_bins),
        "elements": elements_series is None and bool(slice_cfg.element_slices or slice_cfg.include_halogen),
    }
    needs_rdkit = any(needs.values())
    if needs_rdkit:
        try:
            computed = _compute_features_from_mols(
                df,
                sample_ids,
                needs=needs,
                smiles_col=smiles_col,
                sdf_path_col=sdf_path_col,
                sdf_dir=sdf_dir,
            )
        except Exception as exc:
            logger.warning("RDKit-based error slicing features skipped: %s", exc)
            computed = {}

        if needs.get("n_atoms") and "n_atoms" in computed:
            features["n_atoms"] = pd.Series(computed["n_atoms"], index=df.index, dtype=float)
        if needs.get("n_heavy_atoms") and "n_heavy_atoms" in computed:
            features["n_heavy_atoms"] = pd.Series(computed["n_heavy_atoms"], index=df.index, dtype=float)
        if needs.get("ring_count") and "ring_count" in computed:
            features["ring_count"] = pd.Series(computed["ring_count"], index=df.index, dtype=float)
        if needs.get("rotatable_bonds") and "rotatable_bonds" in computed:
            features["rotatable_bonds"] = pd.Series(computed["rotatable_bonds"], index=df.index, dtype=float)

        if needs.get("elements") and "elements" in computed:
            elements_series = pd.Series(computed["elements"], index=df.index)
            elements_list = [list(items) for items in computed.get("elements_list", [])]

    if elements_series is not None:
        features["elements"] = elements_series
    features["_elements_list"] = elements_list or [[] for _ in range(len(features))]

    return features


def _iter_element_slices(
    df: pd.DataFrame,
    element_slices: Sequence[str],
    include_halogen: bool,
) -> Iterable[Tuple[str, str, str, pd.Series]]:
    if "_elements_list" not in df.columns:
        return []
    elements = df["_elements_list"].tolist()
    index = df.index
    for element in element_slices:
        mask = pd.Series([element in items for items in elements], index=index)
        if mask.any():
            yield "element", f"element_{element}", "present", mask
    if include_halogen:
        mask = pd.Series([any(e in DEFAULT_HALOGENS for e in items) for items in elements], index=index)
        if mask.any():
            yield "element", "element_halogen", "present", mask


def _iter_binned_slices(
    df: pd.DataFrame,
    column: str,
    bins: List[float],
    slice_type: str,
) -> Iterable[Tuple[str, str, str, pd.Series]]:
    if column not in df.columns:
        return []
    values = _safe_float_series(df[column])
    binned = _bin_series(values, bins)
    if binned is None:
        return []
    for label in binned.dropna().unique().tolist():
        mask = binned == label
        if mask.any():
            yield slice_type, column, str(label), mask


def _collect_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics_fn,
    min_samples: int,
) -> Tuple[Dict[str, float], int]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    n_valid = int(np.sum(mask))
    if n_valid < min_samples:
        return {}, n_valid
    metrics = metrics_fn(y_true[mask], y_pred[mask])
    row = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float)) and np.isfinite(v)}
    return row, n_valid


def build_error_slicing_table(
    df: pd.DataFrame,
    plot_targets: List[PlotTarget],
    metrics_fn,
    slice_cfg: ErrorSlicingConfig,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    has_split = "split" in df.columns
    splits = slice_cfg.splits if has_split else ["all"]

    rows: List[Dict[str, Any]] = []
    for split_name in splits:
        if has_split:
            split_df = df[df["split"].astype(str) == str(split_name)]
        else:
            split_df = df
        if split_df.empty:
            continue

        slice_defs = []
        for col, bins in slice_cfg.size_bins.items():
            slice_defs.extend(_iter_binned_slices(split_df, col, bins, "size"))
        if slice_cfg.ring_bins:
            slice_defs.extend(_iter_binned_slices(split_df, "ring_count", slice_cfg.ring_bins, "ring_count"))
        if slice_cfg.rot_bonds_bins:
            slice_defs.extend(
                _iter_binned_slices(split_df, "rotatable_bonds", slice_cfg.rot_bonds_bins, "rotatable_bonds")
            )
        slice_defs.extend(
            _iter_element_slices(split_df, slice_cfg.element_slices, slice_cfg.include_halogen)
        )

        for slice_type, slice_key, slice_value, mask in slice_defs:
            slice_rows = split_df[mask]
            if slice_rows.empty:
                continue
            for target in plot_targets:
                if target.y_true_col not in slice_rows.columns or target.y_pred_col not in slice_rows.columns:
                    continue
                y_true = slice_rows[target.y_true_col].to_numpy(dtype=float)
                y_pred = slice_rows[target.y_pred_col].to_numpy(dtype=float)
                metrics, n_valid = _collect_metrics(y_true, y_pred, metrics_fn, slice_cfg.min_samples)
                if not metrics:
                    continue
                row: Dict[str, Any] = {
                    "slice_type": slice_type,
                    "slice_key": slice_key,
                    "slice_value": slice_value,
                    "split": split_name,
                    "target": target.name,
                    "n_total": int(len(slice_rows)),
                    "n_samples": int(n_valid),
                }
                row.update(metrics)
                rows.append(row)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    sort_cols = [c for c in ["slice_type", "slice_key", "slice_value", "split", "target"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


def run_error_slicing(
    cfg: Dict[str, Any],
    pred_df: pd.DataFrame,
    out_dir: Path,
    plots_dir: Path,
    primary_target: Optional[str],
    logger,
) -> Optional[Path]:
    plots_cfg = cfg.get("plots", {}) or {}
    err_cfg = plots_cfg.get("error_slicing", {}) or {}
    if not bool(err_cfg.get("enabled", True)):
        return None

    on_fail = str(err_cfg.get("on_fail", "skip")).lower()
    try:
        if "sample_id" not in pred_df.columns:
            raise ValueError("predictions.csv missing sample_id for error slicing.")

        eval_cfg = _resolve_eval_cfg(cfg)
        train_cfg = None
        model_artifact_dir = _resolve_model_artifact_dir(cfg, eval_cfg, err_cfg)
        if model_artifact_dir:
            train_cfg_path = model_artifact_dir / "config_snapshot.yaml"
            if train_cfg_path.exists():
                train_cfg = load_config(train_cfg_path)

        dataset_csv, sdf_dir, sample_id_col, smiles_col, formula_col, sdf_path_col, cas_col = _resolve_data_context(
            cfg=cfg, train_cfg=train_cfg, eval_cfg=eval_cfg, err_cfg=err_cfg
        )
        if dataset_csv is None:
            raise ValueError("error slicing requires input.dataset_csv or evaluate_run_dir/model_artifact_dir.")
        if not dataset_csv.exists():
            raise FileNotFoundError(f"dataset_csv not found: {dataset_csv}")

        df = read_csv(dataset_csv)
        sample_ids = pred_df["sample_id"].astype(str).tolist()
        id_col = _resolve_id_column(df, sample_id_col, cas_col)
        if id_col:
            df = df[df[id_col].astype(str).isin(set(sample_ids))].copy()
        else:
            df = df[df.index.astype(str).isin(set(sample_ids))].copy()
        if df.empty:
            raise ValueError("No dataset rows matched predictions for error slicing.")

        slice_cfg = parse_error_slicing_config(plots_cfg)
        features = build_error_slicing_features(
            df=df,
            slice_cfg=slice_cfg,
            sample_id_col=sample_id_col,
            cas_col=cas_col,
            smiles_col=smiles_col,
            formula_col=formula_col,
            sdf_path_col=sdf_path_col,
            sdf_dir=sdf_dir,
            logger=logger,
        )

        merged = pred_df.merge(features, on="sample_id", how="left")
        plot_targets, _ = resolve_plot_targets(
            merged.columns, target_mode=slice_cfg.target_mode, primary_target=primary_target
        )

        metrics_cfg = train_cfg or eval_cfg or cfg
        metrics_fn = resolve_task(metrics_cfg).metrics_fn

        table = build_error_slicing_table(merged, plot_targets, metrics_fn, slice_cfg)
        if table.empty:
            logger.info("Error slicing produced no metrics.")
            return None
        out_path = plots_dir / "error_slicing.csv"
        table.to_csv(out_path, index=False)
        logger.info("Saved error slicing table to %s", out_path)
        return out_path
    except Exception as exc:
        if on_fail in {"skip", "warn"}:
            logger.warning("Error slicing skipped: %s", exc)
            return None
        raise
