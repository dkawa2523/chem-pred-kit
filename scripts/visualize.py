from __future__ import annotations

import argparse
import math
import pickle
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# Allow running as `python scripts/visualize.py ...` without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.config import dump_yaml, load_config
from src.common.io import load_sdf_mol, read_csv, sdf_path_from_cas
from src.common.meta import build_meta, save_meta
from src.common.plots import (
    save_abs_error_cdf,
    save_bland_altman_plot,
    save_hist,
    save_parity_plot,
    save_residual_hist,
    save_residual_plot,
)
from src.common.utils import ensure_dir, get_logger, load_json
from src.targets.registry import resolve_target_names, resolve_target_units
from src.tasks import resolve_target_columns
from src.utils.validate_config import validate_config
from src.visualize.error_slicing import run_error_slicing
from src.visualize.utils import build_metrics_tables, resolve_plot_targets

try:
    from rdkit import Chem
except Exception:  # pragma: no cover
    Chem = None


def _resolve_predictions_path(cfg) -> Path:
    input_cfg = cfg.get("input", {})
    pred_path = input_cfg.get("predictions_path")
    if pred_path:
        return Path(pred_path)
    eval_dir = input_cfg.get("evaluate_run_dir")
    if eval_dir:
        return Path(eval_dir) / "predictions.csv"
    raise ValueError("visualize requires input.predictions_path or input.evaluate_run_dir")


def _resolve_metrics_path(cfg) -> Path | None:
    input_cfg = cfg.get("input", {})
    metrics_path = input_cfg.get("metrics_path")
    if metrics_path:
        return Path(metrics_path)
    eval_dir = input_cfg.get("evaluate_run_dir")
    if eval_dir:
        return Path(eval_dir) / "metrics.json"
    return None


def _resolve_primary_target(cfg, plots_cfg) -> str | None:
    primary = plots_cfg.get("primary_target") or plots_cfg.get("target_name")
    if primary:
        return str(primary)
    target_cols = resolve_target_columns(cfg)
    if not target_cols:
        return None
    target_names = resolve_target_names(cfg, target_cols)
    return target_names[0] if target_names else None


def _format_label(name: str, units: dict) -> str:
    unit = units.get(name)
    if unit:
        return f"{name} ({unit})"
    return name


def _resolve_eval_cfg(cfg: dict) -> Optional[dict]:
    input_cfg = cfg.get("input", {})
    eval_dir = input_cfg.get("evaluate_run_dir")
    if not eval_dir:
        return None
    eval_cfg_path = Path(eval_dir) / "config.yaml"
    if not eval_cfg_path.exists():
        return None
    return load_config(eval_cfg_path)


def _resolve_dataset_context(cfg: dict, eval_cfg: Optional[dict]) -> Tuple[Optional[Path], Optional[Path], str, str, Optional[str], str, str]:
    data_cfg = cfg.get("data") or (eval_cfg.get("data") if eval_cfg else {}) or {}
    columns_cfg = cfg.get("columns") or (eval_cfg.get("columns") if eval_cfg else {}) or {}
    paths_cfg = cfg.get("paths") or (eval_cfg.get("paths") if eval_cfg else {}) or {}

    dataset_csv = data_cfg.get("dataset_csv") or paths_cfg.get("out_csv")
    sdf_dir = data_cfg.get("sdf_dir") or paths_cfg.get("sdf_dir")
    cas_col = str(data_cfg.get("cas_col") or columns_cfg.get("cas") or "CAS")
    sample_id_col = str(data_cfg.get("sample_id_col") or columns_cfg.get("sample_id") or cas_col)
    smiles_col = data_cfg.get("smiles_col") or columns_cfg.get("smiles")
    formula_col = str(columns_cfg.get("formula") or "MolecularFormula")
    name_col = str(columns_cfg.get("name") or "Name")
    return (
        Path(dataset_csv) if dataset_csv else None,
        Path(sdf_dir) if sdf_dir else None,
        cas_col,
        sample_id_col,
        smiles_col,
        formula_col,
        name_col,
    )


def _load_dataset_df(dataset_csv: Optional[Path], logger) -> Optional[pd.DataFrame]:
    if not dataset_csv:
        return None
    if not dataset_csv.exists():
        logger.warning("dataset_csv not found: %s", dataset_csv)
        return None
    try:
        return read_csv(dataset_csv)
    except Exception as e:
        logger.warning("Failed to read dataset_csv=%s: %s", dataset_csv, e)
        return None


def _build_meta_maps(
    dataset_df: Optional[pd.DataFrame],
    sample_id_col: str,
    cas_col: str,
    smiles_col: Optional[str],
    formula_col: str,
    name_col: str,
) -> Tuple[Dict[str, dict], Dict[str, List[str]]]:
    meta_map: Dict[str, dict] = {}
    elements_map: Dict[str, List[str]] = {}
    if dataset_df is None or dataset_df.empty:
        return meta_map, elements_map
    cols = set(dataset_df.columns)
    elements_col = "elements" if "elements" in cols else None
    for _, row in dataset_df.iterrows():
        if sample_id_col in cols:
            sample_id = str(row.get(sample_id_col))
        elif cas_col in cols:
            sample_id = str(row.get(cas_col))
        else:
            continue
        cas_val = str(row.get(cas_col)) if cas_col in cols else sample_id
        smiles_val = str(row.get(smiles_col)) if smiles_col and smiles_col in cols else None
        formula_val = str(row.get(formula_col)) if formula_col in cols else None
        name_val = str(row.get(name_col)) if name_col in cols else None
        meta_map[sample_id] = {
            "CAS": cas_val,
            "SMILES": smiles_val,
            "Name": name_val,
            "Formula": formula_val,
        }
        if elements_col:
            raw = row.get(elements_col)
            if isinstance(raw, str):
                elems = [e for e in raw.split(",") if e]
            else:
                elems = []
            elements_map[sample_id] = elems
    return meta_map, elements_map


def _get_mol_for_sample(
    sample_id: str,
    cas: Optional[str],
    smiles: Optional[str],
    sdf_dir: Optional[Path],
    cache: Dict[str, Optional["Chem.Mol"]],
) -> Optional["Chem.Mol"]:
    if Chem is None:
        return None
    key = cas or sample_id
    if key in cache:
        return cache[key]
    mol = None
    if smiles and isinstance(smiles, str) and smiles not in {"nan", "None"}:
        mol = Chem.MolFromSmiles(smiles)
    if mol is None and cas and sdf_dir is not None:
        mol = load_sdf_mol(sdf_path_from_cas(sdf_dir, cas))
    cache[key] = mol
    return mol


def _compute_coverage_curve(y_true, y_pred, y_std, ks: Iterable[float]) -> List[float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_std = np.asarray(y_std, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(y_std) & (y_std > 0)
    if not mask.any():
        return []
    err = np.abs(y_pred[mask] - y_true[mask])
    std = y_std[mask]
    coverages = []
    for k in ks:
        coverages.append(float(np.mean(err <= k * std)))
    return coverages


def _plot_corr_heatmap(matrix: np.ndarray, labels: List[str], out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    plt.figure()
    im = plt.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize predictions (parity/residual/hist).")
    ap.add_argument("--config", required=True, help="Path to a composed visualize config.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    validate_config(cfg)

    output_cfg = cfg.get("output", {})
    experiment_cfg = cfg.get("experiment", {})
    exp_name = str(output_cfg.get("exp_name", experiment_cfg.get("name", "visualize")))
    out_dir = ensure_dir(Path(output_cfg.get("out_dir", "runs/visualize")) / exp_name)
    plots_dir = ensure_dir(out_dir / "plots")
    logger = get_logger("visualize", log_file=out_dir / "visualize.log")

    dump_yaml(out_dir / "config.yaml", cfg)
    save_meta(
        out_dir,
        build_meta(process_name=str(cfg.get("process", {}).get("name", "visualize")), cfg=cfg),
    )

    pred_path = _resolve_predictions_path(cfg)
    if not pred_path.exists():
        raise FileNotFoundError(f"predictions.csv not found: {pred_path}")

    df = pd.read_csv(pred_path)

    eval_cfg = _resolve_eval_cfg(cfg)
    dataset_csv, sdf_dir, cas_col, sample_id_col, smiles_col, formula_col, name_col = _resolve_dataset_context(cfg, eval_cfg)
    dataset_df = _load_dataset_df(dataset_csv, logger)
    meta_map, elements_map = _build_meta_maps(
        dataset_df,
        sample_id_col=sample_id_col,
        cas_col=cas_col,
        smiles_col=smiles_col,
        formula_col=formula_col,
        name_col=name_col,
    )

    plots_cfg = cfg.get("plots", {})
    splits = plots_cfg.get("splits", ["val", "test"])
    include_train_hist = bool(plots_cfg.get("include_train_hist", True))
    target_mode = str(plots_cfg.get("target_mode", "primary")).lower()
    per_target_cfg = plots_cfg.get("per_target")
    primary_target = _resolve_primary_target(cfg, plots_cfg)

    plot_targets, available_targets = resolve_plot_targets(df.columns, target_mode, primary_target)
    per_target = bool(per_target_cfg) if per_target_cfg is not None else bool(available_targets)
    if available_targets and (per_target or target_mode == "all"):
        plot_targets, _ = resolve_plot_targets(df.columns, "all", primary_target)
    all_targets = plot_targets
    if available_targets:
        all_targets, _ = resolve_plot_targets(df.columns, "all", primary_target)
    required_cols = {t.y_true_col for t in plot_targets} | {t.y_pred_col for t in plot_targets}
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            "predictions.csv missing columns for visualization: "
            + ", ".join(sorted(missing_cols))
        )
    if target_mode == "all" and not available_targets:
        logger.warning("target_mode=all requested, but no per-target columns found; falling back to y_true/y_pred.")

    units = resolve_target_units(cfg, [t.name for t in plot_targets])
    logger.info(
        "Targets for plots: %s (mode=%s)",
        ", ".join([t.name for t in plot_targets]),
        target_mode,
    )

    has_split = "split" in df.columns
    for split_name in splits:
        if has_split:
            split_df = df[df["split"] == split_name]
        else:
            if split_name != "all":
                continue
            split_df = df
        if split_df.empty:
            continue
        for target in plot_targets:
            y_true = split_df[target.y_true_col].to_numpy(dtype=float)
            y_pred = split_df[target.y_pred_col].to_numpy(dtype=float)
            mask = np.isfinite(y_true) & np.isfinite(y_pred)
            y_true = y_true[mask]
            y_pred = y_pred[mask]
            if y_true.size == 0:
                continue
            label = _format_label(target.name, units)
            suffix = target.suffix
            title_suffix = f" - {label}" if suffix else ""
            save_parity_plot(
                y_true,
                y_pred,
                plots_dir / f"parity{suffix}_{split_name}.png",
                title=f"Parity ({split_name}){title_suffix}",
                xlabel="true",
                ylabel="pred",
            )
            save_residual_plot(
                y_true,
                y_pred,
                plots_dir / f"residual{suffix}_{split_name}.png",
                title=f"Residual ({split_name}){title_suffix}",
            )
            save_residual_hist(
                y_true,
                y_pred,
                plots_dir / f"residual_hist{suffix}_{split_name}.png",
                title=f"Residual distribution ({split_name}){title_suffix}",
            )
            save_abs_error_cdf(
                y_true,
                y_pred,
                plots_dir / f"abs_error_cdf{suffix}_{split_name}.png",
                title=f"Absolute error CDF ({split_name}){title_suffix}",
            )
            save_bland_altman_plot(
                y_true,
                y_pred,
                plots_dir / f"bland_altman{suffix}_{split_name}.png",
                title=f"Bland-Altman ({split_name}){title_suffix}",
            )

    if include_train_hist and has_split:
        train_df = df[df["split"] == "train"]
        if not train_df.empty:
            for target in plot_targets:
                y_train = train_df[target.y_true_col].to_numpy(dtype=float)
                y_train = y_train[np.isfinite(y_train)]
                if y_train.size == 0:
                    continue
                label = _format_label(target.name, units)
                suffix = target.suffix
                title_suffix = f" - {label}" if suffix else ""
                xlabel = label if label else "y_true"
                save_hist(
                    y_train,
                    plots_dir / f"y_train_hist{suffix}.png",
                    title=f"Target distribution (train){title_suffix}",
                    xlabel=xlabel,
                )

    embedding_cfg = plots_cfg.get("embedding", {}) or {}
    if bool(embedding_cfg.get("enabled", False)):
        from src.visualize.embedding import run_embedding_visualization

        run_embedding_visualization(
            cfg=cfg,
            pred_df=df,
            out_dir=out_dir,
            plots_dir=plots_dir,
            primary_target=primary_target,
            logger=logger,
        )

    metrics_path = _resolve_metrics_path(cfg)
    if metrics_path and metrics_path.exists():
        metrics = load_json(metrics_path)
        split_table, target_table = build_metrics_tables(metrics)
        if not split_table.empty:
            split_table.to_csv(plots_dir / "metrics_table.csv", index=False)
        if target_table is not None and not target_table.empty:
            target_table.to_csv(plots_dir / "metrics_by_target.csv", index=False)

    # Extra analyses
    try:
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover
        plt = None

    # Top-K worst errors table
    worst_cfg = plots_cfg.get("worst_errors", {}) or {}
    top_k = int(worst_cfg.get("top_k", 20))
    worst_rows: List[Dict[str, object]] = []
    if top_k > 0 and "sample_id" in df.columns:
        smiles_cache: Dict[str, Optional[str]] = {}
        mol_cache: Dict[str, Optional["Chem.Mol"]] = {}
        for split_name in splits:
            split_df = df[df["split"] == split_name] if has_split else df
            if split_df.empty:
                continue
            for target in all_targets:
                if target.y_true_col not in split_df.columns or target.y_pred_col not in split_df.columns:
                    continue
                tmp = split_df[["sample_id", target.y_true_col, target.y_pred_col]].copy()
                tmp["abs_error"] = (tmp[target.y_pred_col] - tmp[target.y_true_col]).abs()
                tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna(subset=["abs_error"])
                if tmp.empty:
                    continue
                tmp = tmp.sort_values("abs_error", ascending=False).head(top_k)
                for _, row in tmp.iterrows():
                    sample_id = str(row["sample_id"])
                    meta = meta_map.get(sample_id, {})
                    cas_val = meta.get("CAS") or sample_id
                    smiles_val = meta.get("SMILES")
                    if (not smiles_val or smiles_val in {"nan", "None"}) and Chem is not None and sdf_dir is not None:
                        if sample_id in smiles_cache:
                            smiles_val = smiles_cache[sample_id]
                        else:
                            mol = _get_mol_for_sample(sample_id, cas_val, None, sdf_dir, mol_cache)
                            if mol is not None:
                                smiles_val = Chem.MolToSmiles(mol)
                            else:
                                smiles_val = None
                            smiles_cache[sample_id] = smiles_val
                    worst_rows.append(
                        {
                            "split": split_name,
                            "target": target.name,
                            "sample_id": sample_id,
                            "CAS": cas_val,
                            "SMILES": smiles_val,
                            "Name": meta.get("Name"),
                            "Formula": meta.get("Formula"),
                            "y_true": float(row[target.y_true_col]),
                            "y_pred": float(row[target.y_pred_col]),
                            "abs_error": float(row["abs_error"]),
                        }
                    )
    if worst_rows:
        worst_df = pd.DataFrame(worst_rows)
        worst_df.to_csv(plots_dir / "worst_errors.csv", index=False)

    # Residual correlation between targets
    if len(all_targets) >= 2:
        for split_name in splits:
            split_df = df[df["split"] == split_name] if has_split else df
            if split_df.empty:
                continue
            resid_cols = {}
            for target in all_targets:
                if target.y_true_col in split_df.columns and target.y_pred_col in split_df.columns:
                    resid_cols[target.name] = split_df[target.y_pred_col] - split_df[target.y_true_col]
            if len(resid_cols) < 2:
                continue
            resid_df = pd.DataFrame(resid_cols).replace([np.inf, -np.inf], np.nan).dropna()
            if resid_df.empty:
                continue
            corr = resid_df.corr()
            corr.to_csv(plots_dir / f"residual_corr_{split_name}.csv", index=True)
            if plt is not None:
                _plot_corr_heatmap(
                    corr.to_numpy(),
                    list(corr.columns),
                    plots_dir / f"residual_corr_{split_name}.png",
                    title=f"Residual correlation ({split_name})",
                )
                if resid_df.shape[1] == 2:
                    names = list(resid_df.columns)
                    x = resid_df[names[0]].to_numpy()
                    y = resid_df[names[1]].to_numpy()
                    r = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else float("nan")
                    plt.figure()
                    plt.scatter(x, y, s=12, alpha=0.7)
                    plt.axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
                    plt.axvline(0.0, color="black", linewidth=1.0, alpha=0.5)
                    plt.title(f"Residual scatter ({split_name}) r={r:.3f}")
                    plt.xlabel(f"{names[0]} residual")
                    plt.ylabel(f"{names[1]} residual")
                    plt.tight_layout()
                    plt.savefig(plots_dir / f"residual_scatter_{split_name}.png", dpi=200)
                    plt.close()

    # Coverage curves (PICP) if uncertainty is available
    coverage_cfg = plots_cfg.get("coverage", {}) or {}
    ks = coverage_cfg.get("k_values", [0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    ks = [float(k) for k in ks]
    if "y_std" in df.columns or any(f"y_std_{t.name}" in df.columns for t in all_targets):
        for split_name in splits:
            split_df = df[df["split"] == split_name] if has_split else df
            if split_df.empty:
                continue
            for target in all_targets:
                std_col = f"y_std_{target.name}" if f"y_std_{target.name}" in split_df.columns else None
                if std_col is None and target.name == primary_target and "y_std" in split_df.columns:
                    std_col = "y_std"
                if std_col is None:
                    continue
                y_true = split_df[target.y_true_col].to_numpy(dtype=float)
                y_pred = split_df[target.y_pred_col].to_numpy(dtype=float)
                y_std = split_df[std_col].to_numpy(dtype=float)
                coverages = _compute_coverage_curve(y_true, y_pred, y_std, ks)
                if not coverages:
                    continue
                expected = [math.erf(k / math.sqrt(2.0)) for k in ks]
                suffix = target.suffix
                cov_df = pd.DataFrame({"k": ks, "coverage": coverages, "expected": expected})
                cov_df.to_csv(plots_dir / f"coverage{suffix}_{split_name}.csv", index=False)
                if plt is None:
                    continue
                plt.figure()
                plt.plot(ks, coverages, marker="o", label="empirical")
                plt.plot(ks, expected, linestyle="--", label="expected (normal)")
                plt.xlabel("k (std multiplier)")
                plt.ylabel("coverage")
                plt.title(f"Coverage curve ({split_name}) {target.name}")
                plt.ylim(0.0, 1.0)
                plt.legend()
                plt.tight_layout()
                plt.savefig(plots_dir / f"coverage{suffix}_{split_name}.png", dpi=200)
                plt.close()

    # Element-wise error boxplot (primary target)
    if elements_map and "sample_id" in df.columns and all_targets:
        elem_cfg = plots_cfg.get("element_errors", {}) or {}
        min_count = int(elem_cfg.get("min_count", 30))
        max_elements = int(elem_cfg.get("max_elements", 12))
        primary_t = next((t for t in all_targets if t.name == primary_target), all_targets[0])
        for split_name in splits:
            split_df = df[df["split"] == split_name] if has_split else df
            if split_df.empty:
                continue
            elem_errors: Dict[str, List[float]] = {}
            for _, row in split_df.iterrows():
                sample_id = str(row.get("sample_id"))
                elems = elements_map.get(sample_id, [])
                if not elems:
                    continue
                y_true = row.get(primary_t.y_true_col)
                y_pred = row.get(primary_t.y_pred_col)
                if not np.isfinite(y_true) or not np.isfinite(y_pred):
                    continue
                err = float(abs(y_pred - y_true))
                for el in elems:
                    elem_errors.setdefault(el, []).append(err)
            if not elem_errors:
                continue
            elem_items = [(el, errs) for el, errs in elem_errors.items() if len(errs) >= min_count]
            if not elem_items:
                continue
            elem_items.sort(key=lambda item: len(item[1]), reverse=True)
            elem_items = elem_items[:max_elements]
            elem_labels = [e for e, _ in elem_items]
            elem_vals = [errs for _, errs in elem_items]
            summary_rows = []
            for el, errs in elem_items:
                arr = np.asarray(errs, dtype=float)
                summary_rows.append(
                    {
                        "element": el,
                        "count": int(arr.size),
                        "mean": float(np.mean(arr)),
                        "median": float(np.median(arr)),
                        "q1": float(np.quantile(arr, 0.25)),
                        "q3": float(np.quantile(arr, 0.75)),
                    }
                )
            pd.DataFrame(summary_rows).to_csv(plots_dir / f"element_error_summary_{split_name}.csv", index=False)
            if plt is None:
                continue
            plt.figure()
            plt.boxplot(elem_vals, labels=elem_labels, showfliers=False)
            plt.ylabel("abs error")
            plt.title(f"Element-wise error (split={split_name})")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(plots_dir / f"element_error_boxplot_{split_name}.png", dpi=200)
            plt.close()

    # Functional group error distribution via SMARTS
    fg_cfg = plots_cfg.get("functional_groups", {}) or {}
    if Chem is not None and all_targets and "sample_id" in df.columns and (sdf_dir is not None or smiles_col):
        patterns = [
            ("alcohol", "[OX2H]"),
            ("phenol", "c[OX2H]"),
            ("amine", "[NX3;H2,H1;!$(NC=O)]"),
            ("amide", "C(=O)N"),
            ("carboxylic_acid", "C(=O)[OX2H]"),
            ("ester", "C(=O)O[#6]"),
            ("carbonyl", "[CX3]=[OX1]"),
            ("ether", "[OD2]([#6])[#6]"),
            ("halogen", "[F,Cl,Br,I]"),
            ("nitro", "[N+](=O)[O-]"),
            ("sulfonyl", "S(=O)(=O)"),
            ("thiol", "[SX2H]"),
            ("aromatic", "a"),
        ]
        fg_min_count = int(fg_cfg.get("min_count", 30))
        fg_max = int(fg_cfg.get("max_groups", 12))
        fg_max_samples = fg_cfg.get("max_samples")
        fg_max_samples = int(fg_max_samples) if fg_max_samples is not None else None
        fg_patterns = [(name, Chem.MolFromSmarts(smarts)) for name, smarts in patterns]
        mol_cache: Dict[str, Optional["Chem.Mol"]] = {}
        primary_t = next((t for t in all_targets if t.name == primary_target), all_targets[0])
        for split_name in splits:
            split_df = df[df["split"] == split_name] if has_split else df
            if split_df.empty:
                continue
            if fg_max_samples and len(split_df) > fg_max_samples:
                split_df = split_df.sample(fg_max_samples, random_state=42)
            fg_errors: Dict[str, List[float]] = {}
            for _, row in split_df.iterrows():
                sample_id = str(row.get("sample_id"))
                meta = meta_map.get(sample_id, {})
                cas_val = meta.get("CAS") or sample_id
                smiles_val = meta.get("SMILES")
                mol = _get_mol_for_sample(sample_id, cas_val, smiles_val, sdf_dir, mol_cache)
                if mol is None:
                    continue
                y_true = row.get(primary_t.y_true_col)
                y_pred = row.get(primary_t.y_pred_col)
                if not np.isfinite(y_true) or not np.isfinite(y_pred):
                    continue
                err = float(abs(y_pred - y_true))
                for name, patt in fg_patterns:
                    if patt is None:
                        continue
                    if mol.HasSubstructMatch(patt):
                        fg_errors.setdefault(name, []).append(err)
            if not fg_errors:
                continue
            fg_items = [(name, errs) for name, errs in fg_errors.items() if len(errs) >= fg_min_count]
            if not fg_items:
                continue
            fg_items.sort(key=lambda item: len(item[1]), reverse=True)
            fg_items = fg_items[:fg_max]
            fg_labels = [name for name, _ in fg_items]
            fg_vals = [errs for _, errs in fg_items]
            fg_summary = []
            for name, errs in fg_items:
                arr = np.asarray(errs, dtype=float)
                fg_summary.append(
                    {
                        "group": name,
                        "count": int(arr.size),
                        "mean": float(np.mean(arr)),
                        "median": float(np.median(arr)),
                        "q1": float(np.quantile(arr, 0.25)),
                        "q3": float(np.quantile(arr, 0.75)),
                    }
                )
            pd.DataFrame(fg_summary).to_csv(plots_dir / f"functional_group_error_summary_{split_name}.csv", index=False)
            if plt is None:
                continue
            plt.figure()
            plt.boxplot(fg_vals, labels=fg_labels, showfliers=False)
            plt.ylabel("abs error")
            plt.title(f"Functional-group error (split={split_name})")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(plots_dir / f"functional_group_error_boxplot_{split_name}.png", dpi=200)
            plt.close()

    # AD analysis (Tanimoto similarity + element out-of-domain)
    ad_cfg = plots_cfg.get("ad", {}) or {}
    if ad_cfg.get("enabled", True):
        model_dirs = None
        if "model_artifact_dir" in cfg:
            model_dirs = cfg.get("model_artifact_dir")
        elif eval_cfg and "model_artifact_dir" in eval_cfg:
            model_dirs = eval_cfg.get("model_artifact_dir")
        if isinstance(model_dirs, str):
            model_dirs = [model_dirs]
        model_dirs = model_dirs or []
        ad_path = Path(model_dirs[0]) / "artifacts" / "ad.pkl" if model_dirs else None
        if ad_path and ad_path.exists():
            try:
                from src.common.ad import compute_trust_score, tanimoto_top_k
                from src.fp.featurizer_fp import morgan_bitvect

                with open(ad_path, "rb") as f:
                    ad_artifact = pickle.load(f)
                training_elements = ad_artifact.get("training_elements", [])
                training_set = set(training_elements)
                train_fps = ad_artifact.get("train_fps", [])
                train_ids = ad_artifact.get("train_ids", [])
                tanimoto_warn = float(ad_artifact.get("tanimoto_warn_threshold", 0.5))
                top_k_sim = int(ad_artifact.get("top_k", 5))
                max_samples = ad_cfg.get("max_samples")
                max_samples = int(max_samples) if max_samples is not None else None

                mol_cache: Dict[str, Optional["Chem.Mol"]] = {}
                primary_t = next((t for t in all_targets if t.name == primary_target), all_targets[0]) if all_targets else None
                for split_name in splits:
                    split_df = df[df["split"] == split_name] if has_split else df
                    if split_df.empty:
                        continue
                    if max_samples and len(split_df) > max_samples:
                        split_df = split_df.sample(max_samples, random_state=42)
                    rows = []
                    unseen_counts: Dict[str, int] = {}
                    for _, row in split_df.iterrows():
                        sample_id = str(row.get("sample_id"))
                        meta = meta_map.get(sample_id, {})
                        cas_val = meta.get("CAS") or sample_id
                        smiles_val = meta.get("SMILES")
                        elems = elements_map.get(sample_id, [])
                        mol = _get_mol_for_sample(sample_id, cas_val, smiles_val, sdf_dir, mol_cache)
                        if mol is None:
                            continue
                        if not elems:
                            elems = sorted({atom.GetSymbol() for atom in mol.GetAtoms()})
                        query_fp = morgan_bitvect(mol, radius=int(ad_artifact.get("morgan_radius", 2)), n_bits=int(ad_artifact.get("n_bits", 2048)))
                        max_sim, top = tanimoto_top_k(query_fp, train_fps, train_ids, k=top_k_sim)
                        unseen = [e for e in elems if e not in training_set]
                        trust_score, warnings = compute_trust_score(max_sim, unseen, tanimoto_warn_threshold=tanimoto_warn)
                        for el in unseen:
                            unseen_counts[el] = unseen_counts.get(el, 0) + 1
                        abs_err = None
                        if primary_t is not None:
                            y_true = row.get(primary_t.y_true_col)
                            y_pred = row.get(primary_t.y_pred_col)
                            if np.isfinite(y_true) and np.isfinite(y_pred):
                                abs_err = float(abs(y_pred - y_true))
                        rows.append(
                            {
                                "sample_id": sample_id,
                                "CAS": cas_val,
                                "max_tanimoto": max_sim,
                                "tanimoto_distance": None if max_sim is None else float(1.0 - max_sim),
                                "trust_score": trust_score,
                                "unseen_elements": ",".join(unseen),
                                "n_unseen_elements": int(len(unseen)),
                                "abs_error": abs_err,
                            }
                        )
                    if rows:
                        ad_df = pd.DataFrame(rows)
                        ad_df.to_csv(plots_dir / f"ad_summary_{split_name}.csv", index=False)
                        if unseen_counts:
                            pd.DataFrame(
                                [{"element": k, "count": v} for k, v in sorted(unseen_counts.items(), key=lambda kv: kv[1], reverse=True)]
                            ).to_csv(plots_dir / f"ad_unseen_elements_{split_name}.csv", index=False)
                        if plt is not None and "max_tanimoto" in ad_df.columns:
                            vals = ad_df["max_tanimoto"].dropna().to_numpy(dtype=float)
                            if vals.size > 0:
                                plt.figure()
                                plt.hist(vals, bins=30)
                                plt.xlabel("max Tanimoto similarity")
                                plt.ylabel("count")
                                plt.title(f"AD similarity distribution ({split_name})")
                                plt.tight_layout()
                                plt.savefig(plots_dir / f"ad_similarity_hist_{split_name}.png", dpi=200)
                                plt.close()
                            if "abs_error" in ad_df.columns and ad_df["abs_error"].notna().any():
                                plt.figure()
                                plt.scatter(ad_df["max_tanimoto"], ad_df["abs_error"], s=12, alpha=0.7)
                                plt.xlabel("max Tanimoto similarity")
                                plt.ylabel("abs error")
                                plt.title(f"Error vs similarity ({split_name})")
                                plt.tight_layout()
                                plt.savefig(plots_dir / f"ad_error_vs_similarity_{split_name}.png", dpi=200)
                                plt.close()
            except Exception as e:
                logger.warning("AD analysis skipped: %s", e)

    run_error_slicing(
        cfg=cfg,
        pred_df=df,
        out_dir=out_dir,
        plots_dir=plots_dir,
        primary_target=primary_target,
        logger=logger,
    )

    logger.info("Done.")


if __name__ == "__main__":
    main()
