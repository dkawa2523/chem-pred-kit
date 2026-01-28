from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from rdkit import Chem
except Exception:  # pragma: no cover
    Chem = None

from src.common.chemistry import elements_string, n_elements, parse_formula
from src.common.config import dump_yaml
from src.common.dataset_selectors import SelectorContext, apply_selectors
from src.common.io import load_sdf_mol, read_csv, sdf_path_from_cas, write_csv
from src.common.lj import compute_lj
from src.common.meta import build_meta, save_meta
from src.common.splitters import (
    build_group_map,
    apply_leakage_check,
    group_split,
    random_split,
    resolve_group_column,
    save_split_indices,
    save_split_json,
    scaffold_split,
    time_split,
    validate_group_leakage,
    validate_scaffold_split,
    validate_split_indices,
    validate_time_split,
)
from src.common.utils import ensure_dir, set_seed
from src.data.adapters.base import DatasetAdapter, DatasetArtifacts, RawDatasetInputs
from src.data.adapters.registry import register_dataset_adapter
from src.data.adapters.utils import resolve_dataset_name
from src.targets.registry import load_target_registry, resolve_target_metadata, target_column_name
from src.utils.artifacts import compute_dataset_hash


def _needs_mols(selectors: List[Dict[str, Any]]) -> bool:
    need = {"diversity_farthest_point", "butina_cluster"}
    for s in selectors:
        name = str(s.get("name", "")).lower()
        if name in need:
            return True
    return False


def _to_float_or_none(x) -> Optional[float]:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        return float(x)
    except Exception:
        return None


def _build_dataset_index(df: pd.DataFrame, cas_col: str, indices: Dict[str, List[int]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    split_map: Dict[int, str] = {}
    for split_name, idxs in indices.items():
        for idx in idxs:
            split_map[int(idx)] = split_name

    if cas_col in df.columns:
        sample_ids = df[cas_col].astype(str).tolist()
    else:
        sample_ids = [str(i) for i in df.index.tolist()]

    payload = {
        "row_index": df.index.astype(int).tolist(),
        "sample_id": sample_ids,
        "split": [split_map.get(int(i), "") for i in df.index.tolist()],
    }
    if "raw_row_index" in df.columns:
        payload["raw_row_index"] = df["raw_row_index"].astype(int).tolist()
    pd.DataFrame(payload).to_csv(out_path, index=False)


@register_dataset_adapter("lj")
@register_dataset_adapter("lj_smoke")
@register_dataset_adapter("lj_quick")
@register_dataset_adapter("lj_fixture")
class LJDatasetAdapter(DatasetAdapter):
    name = "lj"

    def prepare_raw(self, cfg: Dict[str, Any], logger) -> RawDatasetInputs:
        paths = cfg.get("paths", {})
        raw_csv = Path(paths.get("raw_csv", "data/raw/tc_pc_tb_pubchem.csv"))
        sdf_dir = Path(paths.get("sdf_dir", "data/raw/sdf_files"))
        if not raw_csv.exists():
            raise FileNotFoundError(f"raw_csv not found: {raw_csv}")
        if not sdf_dir.exists():
            logger.warning(
                "sdf_dir does not exist: %s (SDF-based columns will be NaN; selectors requiring SDF will fail).",
                sdf_dir,
            )
        return RawDatasetInputs(raw_csv=raw_csv, sdf_dir=sdf_dir)

    def get_splits(self, cfg: Dict[str, Any], df: pd.DataFrame, logger) -> Dict[str, List[int]]:
        split_cfg = cfg.get("split", {})
        split_method = str(split_cfg.get("method", "random")).lower()
        ratios = split_cfg.get("fractions", split_cfg.get("ratios", [0.8, 0.1, 0.1]))
        ratios = [float(x) for x in ratios]
        split_seed = int(split_cfg.get("seed", cfg.get("seed", 42)))

        cols = cfg.get("columns", {})
        cas_col = cols.get("cas", "CAS")
        paths = cfg.get("paths", {})
        sdf_dir = Path(paths.get("sdf_dir", "data/raw/sdf_files"))

        if split_method == "random":
            indices = random_split(df, ratios=ratios, seed=split_seed)
        elif split_method == "scaffold":
            indices = scaffold_split(df, sdf_dir=sdf_dir, cas_col=cas_col, ratios=ratios, seed=split_seed)
            validate_scaffold_split(df, sdf_dir=sdf_dir, cas_col=cas_col, indices=indices)
        elif split_method == "group":
            group_key = split_cfg.get("group_key")
            if not group_key:
                raise ValueError("split.method=group requires split.group_key")
            group_col = cols.get(str(group_key), str(group_key))
            if group_col not in df.columns:
                raise ValueError(f"split.group_key '{group_key}' resolved to '{group_col}' not in dataset columns")
            indices = group_split(df, group_col=group_col, ratios=ratios, seed=split_seed)
            group_map = build_group_map(df, group_col=group_col)
            validate_group_leakage(indices, group_map, label=f"group:{group_col}")
        elif split_method == "time":
            time_key = split_cfg.get("time_key") or split_cfg.get("time_col")
            if not time_key:
                raise ValueError("split.method=time requires split.time_key or split.time_col")
            time_col = cols.get(str(time_key), str(time_key))
            if time_col not in df.columns:
                raise ValueError(f"split.time_key '{time_key}' resolved to '{time_col}' not in dataset columns")
            ascending = bool(split_cfg.get("ascending", True))
            indices = time_split(df, time_col=time_col, ratios=ratios, ascending=ascending)
            validate_time_split(df, time_col=time_col, indices=indices, ascending=ascending)
        else:
            raise ValueError(f"Unknown split.method: {split_method}")

        leak_cfg = split_cfg.get("leakage_check")
        if isinstance(leak_cfg, bool):
            leak_cfg = {"enabled": leak_cfg}
        if isinstance(leak_cfg, dict) and leak_cfg.get("enabled", False):
            group_key = leak_cfg.get("group_key") or split_cfg.get("group_key")
            group_col = resolve_group_column(df, cols, group_key, candidates=["canonical_smiles", "inchikey", cas_col])
            apply_leakage_check(
                df,
                indices,
                group_col=group_col,
                label=f"leakage:{group_col or 'unknown'}",
                mode=str(leak_cfg.get("mode", "error")),
                logger=logger,
            )

        validate_split_indices(indices)
        return indices

    def build_processed(self, cfg: Dict[str, Any], logger) -> DatasetArtifacts:
        seed = int(cfg.get("seed", 42))
        set_seed(seed)

        raw_inputs = self.prepare_raw(cfg, logger)
        paths = cfg.get("paths", {})
        out_csv = Path(paths.get("out_csv", "data/processed/dataset_with_lj.csv"))
        out_indices_dir = Path(paths.get("out_indices_dir", "data/processed/indices"))

        df = read_csv(raw_inputs.raw_csv)
        df["raw_row_index"] = df.index.astype(int)
        logger.info("Loaded %s rows from %s", df.shape[0], raw_inputs.raw_csv)

        limit_rows_cfg = cfg.get("limit_rows", None)
        limit_rows = int(limit_rows_cfg) if limit_rows_cfg else None
        if limit_rows is not None and int(limit_rows) > 0:
            before = df.shape[0]
            df = df.head(int(limit_rows)).copy()
            logger.warning("Limiting rows for debug: %s -> %s", before, df.shape[0])

        cols = cfg.get("columns", {})
        cas_col = cols.get("cas", "CAS")
        formula_col = cols.get("formula", "MolecularFormula")
        tc_col = cols.get("tc", "Tc [K]")
        pc_col = cols.get("pc", "Pc [Pa]")
        tb_col = cols.get("tb", "Tb [K]")

        element_counts_global: Dict[str, int] = {}
        elements_col_values = []
        n_elements_values = []

        logger.info("Parsing formulas and counting elements...")
        for formula in tqdm(df[formula_col].astype(str).tolist(), total=df.shape[0]):
            counts = parse_formula(formula)
            e_str = elements_string(counts)
            elements_col_values.append(e_str)
            n_elements_values.append(n_elements(counts))
            for el, n in counts.items():
                element_counts_global[el] = element_counts_global.get(el, 0) + 1

        df["elements"] = elements_col_values
        df["n_elements"] = n_elements_values

        n_atoms, n_heavy = [], []
        mols: List[Any] = []
        missing_sdf = 0
        logger.info("Reading SDF files to compute n_atoms / n_heavy_atoms ...")
        for cas in tqdm(df[cas_col].astype(str).tolist(), total=df.shape[0]):
            if raw_inputs.sdf_dir is None or not raw_inputs.sdf_dir.exists():
                mol = None
            else:
                p = sdf_path_from_cas(raw_inputs.sdf_dir, cas)
                mol = load_sdf_mol(p)
            if mol is None:
                missing_sdf += 1
                n_atoms.append(np.nan)
                n_heavy.append(np.nan)
            else:
                n_atoms.append(int(mol.GetNumAtoms()))
                n_heavy.append(int(mol.GetNumHeavyAtoms()))
            mols.append(mol)

        df["n_atoms"] = n_atoms
        df["n_heavy_atoms"] = n_heavy
        if missing_sdf > 0:
            logger.warning("Missing/invalid SDF for %s rows; they may be skipped later.", missing_sdf)

        canonical_smiles = []
        inchikeys = []
        if Chem is None:
            canonical_smiles = [None for _ in mols]
            inchikeys = [None for _ in mols]
        else:
            for mol in mols:
                if mol is None:
                    canonical_smiles.append(None)
                    inchikeys.append(None)
                    continue
                try:
                    canonical_smiles.append(Chem.MolToSmiles(mol, canonical=True))
                except Exception:
                    canonical_smiles.append(None)
                try:
                    inchikeys.append(Chem.inchi.MolToInchiKey(mol))
                except Exception:
                    inchikeys.append(None)
        df["canonical_smiles"] = canonical_smiles
        df["inchikey"] = inchikeys

        lj_cfg = cfg.get("lj", {})
        eps_method = str(lj_cfg.get("epsilon_method", "bird_critical"))
        sig_method = str(lj_cfg.get("sigma_method", "bird_critical"))
        eps_col = str(lj_cfg.get("epsilon_col", "lj_epsilon_over_k_K"))
        sig_col = str(lj_cfg.get("sigma_col", "lj_sigma_A"))

        eps_list, sig_list = [], []
        valid_eps_range = lj_cfg.get("valid_range", {}).get("epsilon_over_k_K", None)
        valid_sig_range = lj_cfg.get("valid_range", {}).get("sigma_A", None)

        logger.info("Computing LJ params (epsilon_method=%s, sigma_method=%s) ...", eps_method, sig_method)
        for Tc, Pc, Tb in tqdm(zip(df[tc_col].tolist(), df[pc_col].tolist(), df[tb_col].tolist()), total=df.shape[0]):
            out = compute_lj(
                Tc_K=_to_float_or_none(Tc),
                Pc_Pa=_to_float_or_none(Pc),
                Tb_K=_to_float_or_none(Tb),
                epsilon_method=eps_method,
                sigma_method=sig_method,
            )
            eps_list.append(out["lj_epsilon_over_k_K"])
            sig_list.append(out["lj_sigma_A"])

        df[eps_col] = eps_list
        df[sig_col] = sig_list

        registry = load_target_registry(cfg)
        if registry:
            target_names = registry.canonical_names()
            available_cols = df.columns.tolist()
            for name in target_names:
                source_col = registry.resolve_source_column(name, available_cols)
                if source_col is None:
                    spec = registry.get(name)
                    if spec is None or spec.required:
                        raise ValueError(f"Target '{name}' not found in dataset columns (aliases={spec.aliases if spec else []})")
                    continue
                df[target_column_name(name)] = df[source_col]

        df["lj_valid_flag"] = True
        df.loc[df[eps_col].isna() | df[sig_col].isna(), "lj_valid_flag"] = False
        if valid_eps_range:
            lo, hi = float(valid_eps_range[0]), float(valid_eps_range[1])
            df.loc[(df[eps_col] < lo) | (df[eps_col] > hi), "lj_valid_flag"] = False
        if valid_sig_range:
            lo, hi = float(valid_sig_range[0]), float(valid_sig_range[1])
            df.loc[(df[sig_col] < lo) | (df[sig_col] > hi), "lj_valid_flag"] = False

        selectors = cfg.get("selectors", []) or []
        ctx = SelectorContext(element_counts=element_counts_global, mols=mols if _needs_mols(selectors) else None, fps=None)

        if cfg.get("filter_invalid_lj", True):
            before = df.shape[0]
            df = df[df["lj_valid_flag"]].copy()
            logger.info("Filtered invalid LJ rows: %s -> %s", before, df.shape[0])

        if selectors:
            before = df.shape[0]
            df = apply_selectors(df, selectors=selectors, ctx=ctx)
            logger.info("Applied selectors: %s -> %s", before, df.shape[0])

        df = df.reset_index(drop=True)
        write_csv(df, out_csv)
        logger.info("Saved processed dataset: %s (rows=%s)", out_csv, df.shape[0])

        indices = self.get_splits(cfg, df, logger)
        save_split_indices(indices, out_indices_dir)
        split_meta = {
            "method": str(cfg.get("split", {}).get("method", "random")).lower(),
            "seed": int(cfg.get("split", {}).get("seed", seed)),
            "fractions": cfg.get("split", {}).get("fractions", cfg.get("split", {}).get("ratios", [0.8, 0.1, 0.1])),
        }
        if split_meta["method"] == "group":
            split_meta["group_key"] = str(cfg.get("split", {}).get("group_key"))
        if split_meta["method"] == "time":
            split_cfg = cfg.get("split", {})
            time_key = split_cfg.get("time_key") or split_cfg.get("time_col")
            if time_key:
                split_meta["time_key"] = str(time_key)
                split_meta["time_col"] = str(cols.get(str(time_key), str(time_key)))
            split_meta["ascending"] = bool(split_cfg.get("ascending", True))
        leak_cfg = cfg.get("split", {}).get("leakage_check")
        if isinstance(leak_cfg, bool):
            leak_cfg = {"enabled": leak_cfg}
        if isinstance(leak_cfg, dict):
            group_key = leak_cfg.get("group_key") or cfg.get("split", {}).get("group_key")
            group_col = resolve_group_column(df, cols, group_key, candidates=["canonical_smiles", "inchikey", cas_col])
            split_meta["leakage_check"] = {
                "enabled": bool(leak_cfg.get("enabled", True)),
                "group_key": group_key,
                "group_col": group_col,
                "mode": str(leak_cfg.get("mode", "error")),
            }
        split_json_path = out_indices_dir / "split.json"
        save_split_json(indices, split_json_path, metadata=split_meta)
        logger.info("Saved split indices to %s", out_indices_dir)

        dump_yaml(out_csv.parent / "dataset_config_snapshot.yaml", cfg)

        dataset_index_path = out_csv.parent / "dataset_index.csv"
        _build_dataset_index(df, cas_col=cas_col, indices=indices, out_path=dataset_index_path)

        dataset_hash = compute_dataset_hash(out_csv, out_indices_dir)

        run_dir = Path(cfg.get("output", {}).get("run_dir", out_csv.parent))
        ensure_dir(run_dir)
        dump_yaml(run_dir / "config.yaml", cfg)

        dataset_name = resolve_dataset_name(cfg, default=self.name)
        units_cfg = cfg.get("dataset", {}).get("units", {}) if isinstance(cfg.get("dataset", {}), dict) else {}
        default_units = {"lj_epsilon_over_k_K": "K", "lj_sigma_A": "A"}
        if registry:
            target_names = registry.canonical_names()
            units = registry.units_for(target_names)
            for name in target_names:
                if name not in units and name in default_units:
                    units[name] = default_units[name]
        else:
            target_names = [eps_col, sig_col]
            units = {name: default_units.get(name) for name in target_names}
        for name, unit in (units_cfg or {}).items():
            if unit is None:
                continue
            units[str(name)] = str(unit)

        meta_extra = {
            "dataset_name": dataset_name,
            "target_names": target_names,
            "units": units,
            "target_meta": resolve_target_metadata(cfg, target_names),
            "dataset_csv": str(out_csv),
            "indices_dir": str(out_indices_dir),
            "dataset_index": str(dataset_index_path),
            "split_json": str(split_json_path),
        }
        meta = build_meta(
            process_name=str(cfg.get("process", {}).get("name", "build_dataset")),
            cfg=cfg,
            dataset_hash=dataset_hash,
            extra=meta_extra,
        )
        meta_path = save_meta(run_dir, meta)

        return DatasetArtifacts(
            dataset_csv=out_csv,
            indices_dir=out_indices_dir,
            dataset_index=dataset_index_path,
            meta_path=meta_path,
            dataset_hash=dataset_hash,
        )
