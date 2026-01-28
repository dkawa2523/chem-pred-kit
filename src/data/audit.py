from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.common.io import load_sdf_mol
from src.common.splitters import load_split_indices
from src.utils.artifacts import compute_dataset_hash
from src.tasks import resolve_target_columns
from src.targets.registry import load_target_registry, resolve_target_names

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
except Exception:  # pragma: no cover
    Chem = None
    Descriptors = None


def _require_rdkit() -> None:
    if Chem is None or Descriptors is None:
        raise ImportError("RDKit is required for dataset audit. Please install rdkit.")


def _merge_columns(cfg: Dict[str, Any]) -> Dict[str, str]:
    base_cols = cfg.get("columns", {}) or {}
    audit_cols = cfg.get("audit", {}).get("columns", {}) or {}
    cols = dict(base_cols)
    cols.update(audit_cols)
    defaults = {
        "sample_id": "sample_id",
        "cas": "CAS",
        "smiles": "smiles",
        "formula": "MolecularFormula",
        "sdf_path": "sdf_path",
        "target": None,
    }
    for key, val in defaults.items():
        if key not in cols and val is not None:
            cols[key] = val
    return cols


def _normalize_target_columns(values: Any) -> List[str]:
    if values is None:
        return []
    if isinstance(values, (list, tuple)):
        return [str(v).strip() for v in values if v is not None and str(v).strip()]
    text = str(values).strip()
    return [text] if text else []


def _resolve_target_cols(cfg: Dict[str, Any], cols: Dict[str, str]) -> List[str]:
    audit_cfg = cfg.get("audit", {}) or {}
    for key in ("target_cols", "target_columns", "targets"):
        raw = audit_cfg.get(key)
        values = _normalize_target_columns(raw)
        if values:
            return values
    if audit_cfg.get("target_col"):
        return _normalize_target_columns(audit_cfg.get("target_col"))
    if cols.get("target"):
        return _normalize_target_columns(cols.get("target"))
    task_targets = resolve_target_columns(cfg)
    if task_targets:
        return [str(v) for v in task_targets if str(v).strip()]
    return []


def _normalize_valid_range(value: Any) -> Optional[Tuple[float, float]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return (float(value[0]), float(value[1]))
        except Exception:
            return None
    if isinstance(value, dict):
        for lo_key, hi_key in (("min", "max"), ("lower", "upper"), ("low", "high")):
            if lo_key in value or hi_key in value:
                lo = value.get(lo_key)
                hi = value.get(hi_key)
                try:
                    lo_val = float(lo) if lo is not None else None
                    hi_val = float(hi) if hi is not None else None
                except Exception:
                    return None
                if lo_val is None or hi_val is None:
                    return None
                return (lo_val, hi_val)
        if "range" in value:
            return _normalize_valid_range(value.get("range"))
    return None


def _resolve_valid_ranges(
    cfg: Dict[str, Any],
    target_cols: List[str],
    target_names: List[str],
) -> Dict[str, Tuple[float, float]]:
    out: Dict[str, Tuple[float, float]] = {}
    audit_cfg = cfg.get("audit", {}) or {}
    audit_ranges = audit_cfg.get("valid_ranges") if isinstance(audit_cfg.get("valid_ranges"), dict) else {}
    registry = load_target_registry(cfg)

    for col, name in zip(target_cols, target_names):
        if col in out:
            continue
        if isinstance(audit_ranges, dict):
            if col in audit_ranges:
                rng = _normalize_valid_range(audit_ranges.get(col))
                if rng:
                    out[col] = rng
                    continue
            if name in audit_ranges:
                rng = _normalize_valid_range(audit_ranges.get(name))
                if rng:
                    out[col] = rng
                    continue
            if name.startswith("target."):
                base = name[len("target.") :]
                rng = _normalize_valid_range(audit_ranges.get(base))
                if rng:
                    out[col] = rng
                    continue
        if registry is not None:
            spec = registry.get(name) if name else None
            if spec and spec.range_hint:
                rng = _normalize_valid_range(spec.range_hint)
                if rng:
                    out[col] = rng
                    continue
        lj_valid = cfg.get("lj", {}).get("valid_range")
        if isinstance(lj_valid, dict) and name:
            if name in lj_valid:
                rng = _normalize_valid_range(lj_valid.get(name))
                if rng:
                    out[col] = rng
                    continue
            if name.startswith("lj_"):
                short = name[len("lj_") :]
                rng = _normalize_valid_range(lj_valid.get(short))
                if rng:
                    out[col] = rng
                    continue
    return out


def _resolve_input(cfg: Dict[str, Any]) -> Tuple[Path, Optional[Path], Optional[Path]]:
    audit_cfg = cfg.get("audit", {}) or {}
    input_cfg = audit_cfg.get("input", {}) or {}
    source = str(input_cfg.get("source", "processed")).lower()

    if source == "raw":
        raw_csv = input_cfg.get("raw_csv") or cfg.get("paths", {}).get("raw_csv")
        sdf_dir = input_cfg.get("sdf_dir") or cfg.get("paths", {}).get("sdf_dir")
        indices_dir = input_cfg.get("indices_dir")
        if not raw_csv:
            raise ValueError("audit.input.source=raw requires paths.raw_csv or audit.input.raw_csv")
        return Path(raw_csv), Path(sdf_dir) if sdf_dir else None, Path(indices_dir) if indices_dir else None

    dataset_csv = input_cfg.get("dataset_csv") or cfg.get("data", {}).get("dataset_csv") or cfg.get("paths", {}).get("out_csv")
    sdf_dir = input_cfg.get("sdf_dir") or cfg.get("data", {}).get("sdf_dir") or cfg.get("paths", {}).get("sdf_dir")
    indices_dir = input_cfg.get("indices_dir") or cfg.get("data", {}).get("indices_dir")
    if not dataset_csv:
        raise ValueError("audit.input.source=processed requires data.dataset_csv or audit.input.dataset_csv")
    return Path(dataset_csv), Path(sdf_dir) if sdf_dir else None, Path(indices_dir) if indices_dir else None


def _series_values(df: pd.DataFrame, col: Optional[str]) -> Optional[pd.Series]:
    if col is None:
        return None
    if col not in df.columns:
        return None
    return df[col]


def _numeric_stats(values: Iterable[float], quantiles: Iterable[float]) -> Dict[str, Any]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    stats: Dict[str, Any] = {
        "count": int(arr.size),
        "min": float(np.min(arr)) if arr.size > 0 else None,
        "max": float(np.max(arr)) if arr.size > 0 else None,
        "mean": float(np.mean(arr)) if arr.size > 0 else None,
        "std": float(np.std(arr)) if arr.size > 0 else None,
    }
    if arr.size > 0:
        qs = np.quantile(arr, list(quantiles))
        stats["quantiles"] = {str(q): float(v) for q, v in zip(quantiles, qs)}
    else:
        stats["quantiles"] = {}
    return stats


def _collect_elements(mol) -> List[str]:
    return sorted({atom.GetSymbol() for atom in mol.GetAtoms()})


def _duplicate_groups(
    keys: Dict[str, List[int]],
    row_id_to_sample_id: Dict[int, str],
    max_groups: int,
    max_examples: int,
) -> List[Dict[str, Any]]:
    groups = []
    for key, row_ids in keys.items():
        if len(row_ids) <= 1:
            continue
        sample_ids = [row_id_to_sample_id[rid] for rid in row_ids]
        groups.append({"key": key, "count": len(sample_ids), "sample_ids": sample_ids[:max_examples]})
    groups.sort(key=lambda x: x["count"], reverse=True)
    return groups[:max_groups]


def _label_variance_by_groups(
    df: pd.DataFrame,
    target_cols: List[str],
    duplicate_groups: Dict[str, Dict[str, List[int]]],
    row_id_to_sample_id: Dict[int, str],
    max_groups: int,
    max_examples: int,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for method, groups in duplicate_groups.items():
        method_stats: Dict[str, Any] = {}
        for col in target_cols:
            stds: List[float] = []
            top_groups: List[Dict[str, Any]] = []
            for key, row_ids in groups.items():
                if len(row_ids) <= 1:
                    continue
                vals = pd.to_numeric(df.loc[row_ids, col], errors="coerce").dropna().to_numpy(dtype=float)
                if vals.size <= 1:
                    continue
                std = float(np.std(vals))
                stds.append(std)
                if len(top_groups) < max_groups:
                    sample_ids = [row_id_to_sample_id.get(rid, str(rid)) for rid in row_ids[:max_examples]]
                    top_groups.append(
                        {
                            "key": key,
                            "count": int(len(row_ids)),
                            "std": std,
                            "sample_ids": sample_ids,
                        }
                    )
            top_groups.sort(key=lambda x: x.get("std", 0.0), reverse=True)
            stats = _numeric_stats(stds, quantiles=[0.5, 0.9, 0.99]) if stds else {"count": 0}
            method_stats[col] = {
                "group_count": int(len(stds)),
                "std_stats": stats,
                "top_groups": top_groups[:max_groups],
            }
        out[method] = method_stats
    return out


def audit_dataset(cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], str, Dict[str, List[float]]]:
    _require_rdkit()

    dataset_csv, sdf_dir, indices_dir = _resolve_input(cfg)
    if not dataset_csv.exists():
        raise FileNotFoundError(f"dataset_csv not found: {dataset_csv}")
    if sdf_dir is not None and not sdf_dir.exists():
        raise FileNotFoundError(f"sdf_dir not found: {sdf_dir}")
    dataset_hash = compute_dataset_hash(dataset_csv, indices_dir)

    cols = _merge_columns(cfg)
    target_cols = _resolve_target_cols(cfg, cols)

    df = pd.read_csv(dataset_csv)
    total_rows = int(df.shape[0])
    row_index_values = df.index.tolist()

    sample_id_col = cols.get("sample_id")
    cas_col = cols.get("cas")
    smiles_col = cols.get("smiles")
    sdf_path_col = cols.get("sdf_path")

    sample_ids: List[str] = []
    cas_values = _series_values(df, cas_col)
    if sample_id_col and sample_id_col in df.columns:
        sample_ids = df[sample_id_col].astype(str).tolist()
    elif cas_values is not None:
        sample_ids = cas_values.astype(str).tolist()
    else:
        sample_ids = [str(i) for i in row_index_values]

    row_id_to_sample_id = {idx: sample_ids[pos] for pos, idx in enumerate(row_index_values)}

    invalid_samples: List[str] = []
    invalid_reason_counts: Counter[str] = Counter()
    mols = []

    smiles_values = _series_values(df, smiles_col)
    sdf_path_values = _series_values(df, sdf_path_col)

    for pos, idx in enumerate(row_index_values):
        row = df.loc[idx]
        mol = None
        if smiles_values is not None:
            raw_smiles = str(smiles_values.loc[idx]) if pd.notna(smiles_values.loc[idx]) else ""
            if raw_smiles:
                mol = Chem.MolFromSmiles(raw_smiles)
                if mol is None:
                    invalid_reason_counts["invalid_smiles"] += 1
        if mol is None and sdf_path_values is not None:
            raw_path = sdf_path_values.loc[idx]
            if pd.notna(raw_path) and str(raw_path).strip():
                mol = load_sdf_mol(str(raw_path))
                if mol is None:
                    invalid_reason_counts["invalid_sdf"] += 1
        if mol is None and sdf_dir is not None and cas_values is not None:
            cas = str(cas_values.loc[idx]).strip()
            if cas:
                mol = load_sdf_mol(sdf_dir / f"{cas}.sdf")
                if mol is None:
                    invalid_reason_counts["missing_sdf"] += 1
        if mol is None:
            invalid_samples.append(sample_ids[pos])
        mols.append(mol)

    duplicate_cfg = cfg.get("audit", {}).get("duplicate", {}) or {}
    methods = duplicate_cfg.get("methods", ["canonical_smiles", "inchikey"]) or []
    max_groups = int(duplicate_cfg.get("max_groups", 20))
    max_examples = int(duplicate_cfg.get("max_examples", 10))

    duplicates: Dict[str, Dict[str, List[int]]] = {m: defaultdict(list) for m in methods}
    element_counts: Counter[str] = Counter()
    mol_weights: List[float] = []

    for pos, mol in enumerate(mols):
        if mol is None:
            continue
        if "canonical_smiles" in methods:
            key = Chem.MolToSmiles(mol, canonical=True)
            duplicates["canonical_smiles"][key].append(row_index_values[pos])
        if "inchikey" in methods:
            try:
                key = Chem.inchi.MolToInchiKey(mol)
            except Exception:
                key = ""
            if key:
                duplicates["inchikey"][key].append(row_index_values[pos])
        for el in _collect_elements(mol):
            element_counts[el] += 1
        mol_weights.append(float(Descriptors.MolWt(mol)))

    duplicate_groups = {
        method: _duplicate_groups(keys, row_id_to_sample_id, max_groups=max_groups, max_examples=max_examples)
        for method, keys in duplicates.items()
    }

    duplicate_summary = {}
    for method, keys in duplicates.items():
        unique_count = len(keys)
        duplicate_row_count = sum(len(ids) for ids in keys.values() if len(ids) > 1)
        duplicate_group_count = sum(1 for ids in keys.values() if len(ids) > 1)
        duplicate_summary[method] = {
            "unique_keys": int(unique_count),
            "duplicate_group_count": int(duplicate_group_count),
            "duplicate_row_count": int(duplicate_row_count),
            "duplicate_row_rate": float(duplicate_row_count) / float(total_rows) if total_rows > 0 else None,
        }

    split_leakage: Dict[str, Any] = {"available": False}
    if indices_dir is not None and indices_dir.exists():
        indices = load_split_indices(indices_dir)
        split_map = {}
        for split_name, idxs in indices.items():
            for idx in idxs:
                split_map[idx] = split_name
        split_leakage = {"available": True, "by_method": {}}
        for method, keys in duplicates.items():
            leak_groups = []
            for key, row_ids in keys.items():
                if len(row_ids) <= 1:
                    continue
                splits = set()
                for row_id in row_ids:
                    if row_id in split_map:
                        splits.add(split_map[row_id])
                if len(splits) > 1:
                    sample_ids_for_group = [row_id_to_sample_id[rid] for rid in row_ids]
                    leak_groups.append(
                        {"key": key, "splits": sorted(splits), "sample_ids": sample_ids_for_group[:max_examples]}
                    )
            split_leakage["by_method"][method] = {
                "leakage_group_count": int(len(leak_groups)),
                "leakage_groups": leak_groups[:max_groups],
            }

    quantiles = cfg.get("audit", {}).get("stats", {}).get("quantiles", [0.01, 0.05, 0.5, 0.95, 0.99])

    target_cols = [col for col in target_cols if col in df.columns]
    target_names = resolve_target_names(cfg, target_cols) if target_cols else []
    valid_ranges = _resolve_valid_ranges(cfg, target_cols, target_names)

    target_stats: Dict[str, Any] = {"available": False, "columns": {}}
    target_values: List[float] = []
    target_values_by_col: Dict[str, List[float]] = {}
    if target_cols:
        target_stats["available"] = True
        for col in target_cols:
            vals = pd.to_numeric(df[col], errors="coerce")
            missing = int(vals.isna().sum())
            values = vals.dropna().tolist()
            stats = _numeric_stats(values, quantiles)
            median = None
            if stats.get("quantiles") and "0.5" in stats["quantiles"]:
                median = stats["quantiles"].get("0.5")
            range_info = {}
            if col in valid_ranges:
                lo, hi = valid_ranges[col]
                in_vals = vals.dropna()
                if len(in_vals) > 0:
                    out_count = int(((in_vals < lo) | (in_vals > hi)).sum())
                    range_info = {
                        "min": float(lo),
                        "max": float(hi),
                        "out_of_range_count": out_count,
                        "out_of_range_rate": float(out_count) / float(len(in_vals)),
                    }
            target_stats["columns"][col] = {
                "missing_count": missing,
                "missing_rate": float(missing) / float(total_rows) if total_rows > 0 else None,
                "median": median,
                **stats,
                "valid_range": range_info,
            }
            target_values_by_col[col] = [float(v) for v in values]
        if target_cols:
            target_values = target_values_by_col.get(target_cols[0], [])

    missingness: Dict[str, Any] = {}
    for col in df.columns:
        missing_count = int(df[col].isna().sum())
        missingness[col] = {
            "missing_count": missing_count,
            "missing_rate": float(missing_count) / float(total_rows) if total_rows > 0 else None,
        }

    label_variance: Dict[str, Any] = {}
    if target_cols:
        label_variance = _label_variance_by_groups(
            df=df,
            target_cols=target_cols,
            duplicate_groups=duplicates,
            row_id_to_sample_id=row_id_to_sample_id,
            max_groups=max_groups,
            max_examples=max_examples,
        )

    report = {
        "dataset_csv": str(dataset_csv),
        "dataset_hash": dataset_hash,
        "total_rows": total_rows,
        "invalid_mol_count": int(len(invalid_samples)),
        "invalid_reason_counts": dict(invalid_reason_counts),
        "duplicate_groups": duplicate_groups,
        "duplicate_summary": duplicate_summary,
        "target_stats": target_stats,
        "missingness": missingness,
        "label_variance": label_variance,
        "split_leakage": split_leakage,
        "element_counts": dict(element_counts),
        "molecular_weight_stats": _numeric_stats(mol_weights, quantiles),
    }

    max_invalid_examples = int(cfg.get("audit", {}).get("report", {}).get("max_invalid_examples", 20))
    report["invalid_samples"] = invalid_samples[:max_invalid_examples]

    report_md = _render_markdown(report, max_groups=max_groups)
    plot_data = {
        "target_values": [float(v) for v in target_values],
        "target_values_by_col": target_values_by_col,
        "mol_weights": [float(v) for v in mol_weights],
    }
    return report, report_md, plot_data


def _render_markdown(report: Dict[str, Any], max_groups: int) -> str:
    lines = ["# Dataset Audit Report", "", f"- Total rows: {report.get('total_rows')}"]
    if report.get("dataset_hash"):
        lines.append(f"- Dataset hash: {report.get('dataset_hash')}")
    lines.append(f"- Invalid molecules: {report.get('invalid_mol_count')}")

    target_stats = report.get("target_stats", {})
    if target_stats.get("available"):
        lines.append("\n## Target Stats")
        cols = target_stats.get("columns", {})
        for col, stats in cols.items():
            lines.append(f"### {col}")
            lines.append(f"- Missing: {stats.get('missing_count')}")
            lines.append(f"- Mean: {stats.get('mean')}")
            lines.append(f"- Std: {stats.get('std')}")
            lines.append(f"- Median: {stats.get('median')}")
            lines.append(f"- Min: {stats.get('min')}")
            lines.append(f"- Max: {stats.get('max')}")
            vr = stats.get("valid_range") or {}
            if vr:
                lines.append(
                    f"- Valid range: [{vr.get('min')}, {vr.get('max')}] out_of_range_rate={vr.get('out_of_range_rate')}"
                )

    lines.append("\n## Duplicate Summary")
    dup_summary = report.get("duplicate_summary", {})
    for method, summary in dup_summary.items():
        lines.append(f"- {method}: groups={summary.get('duplicate_group_count')}, rows={summary.get('duplicate_row_count')}")

    lines.append("\n## Split Leakage")
    split_leakage = report.get("split_leakage", {})
    if not split_leakage.get("available"):
        lines.append("- Split indices not provided.")
    else:
        for method, info in split_leakage.get("by_method", {}).items():
            lines.append(f"- {method}: leakage_groups={info.get('leakage_group_count')}")

    lines.append("\n## Duplicate Groups (sample)")
    for method, groups in report.get("duplicate_groups", {}).items():
        lines.append(f"### {method}")
        if not groups:
            lines.append("- None")
            continue
        for g in groups[:max_groups]:
            lines.append(f"- {g.get('key')}: count={g.get('count')} samples={g.get('sample_ids')}")

    return "\n".join(lines)
