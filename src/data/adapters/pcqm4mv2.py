from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.common.chemistry import elements_string, get_elements_from_mol
from src.common.config import dump_yaml
from src.common.io import sdf_path_from_cas, write_csv
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

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except Exception:  # pragma: no cover
    Chem = None
    AllChem = None

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    from ogb.lsc import PCQM4Mv2Dataset
except Exception:  # pragma: no cover
    PCQM4Mv2Dataset = None


def _to_numpy(values: Any) -> np.ndarray:
    if torch is not None and hasattr(values, "detach"):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def _resolve_smiles(dataset: Any, data: Any, idx: int) -> Optional[str]:
    if isinstance(data, tuple) and data:
        first = data[0]
        if isinstance(first, str):
            return first
        if isinstance(first, dict):
            for key in ("smiles", "smile"):
                if key in first and first[key]:
                    return str(first[key])
        for attr in ("smiles", "smile"):
            if hasattr(first, attr):
                value = getattr(first, attr)
                if isinstance(value, (list, tuple)) and value:
                    return str(value[0])
                if value:
                    return str(value)

    if isinstance(data, dict):
        for key in ("smiles", "smile"):
            if key in data and data[key]:
                return str(data[key])

    for attr in ("smiles", "smile"):
        if hasattr(data, attr):
            value = getattr(data, attr)
            if isinstance(value, (list, tuple)) and value:
                return str(value[0])
            if value:
                return str(value)

    for attr in ("smiles", "smiles_list"):
        if hasattr(dataset, attr):
            try:
                value = getattr(dataset, attr)[idx]
                if value:
                    return str(value)
            except Exception:
                pass
    return None


def _resolve_target(dataset: Any, data: Any, idx: int) -> Optional[float]:
    value = None
    if isinstance(data, tuple) and len(data) >= 2:
        value = data[1]
    elif isinstance(data, dict):
        for key in ("y", "target", "label", "labels"):
            if key in data:
                value = data[key]
                break
    else:
        for attr in ("y", "target", "label", "labels"):
            if hasattr(data, attr):
                value = getattr(data, attr)
                break

    if value is None:
        for attr in ("y", "target", "label", "labels"):
            if hasattr(dataset, attr):
                try:
                    value = getattr(dataset, attr)[idx]
                    break
                except Exception:
                    pass

    if value is None:
        return None

    arr = _to_numpy(value).reshape(-1)
    if arr.size == 0:
        return None
    try:
        return float(arr[0])
    except Exception:
        return None


def _canonical_smiles(smiles: str) -> Optional[str]:
    if Chem is None:
        raise ImportError("RDKit is required to canonicalize PCQM4Mv2 SMILES.")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def _mol_from_smiles(smiles: str, add_hs: bool, compute_2d: bool) -> Optional["Chem.Mol"]:
    if Chem is None:
        raise ImportError("RDKit is required to build PCQM4Mv2 molecules.")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if add_hs:
        mol = Chem.AddHs(mol)
    if compute_2d and AllChem is not None:
        AllChem.Compute2DCoords(mol)
    return mol


def _sanitize_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")


def _build_sample_id(dataset_name: str, canonical_smiles: str, raw_id: Optional[str], idx: int) -> str:
    digest = hashlib.sha1(canonical_smiles.encode("utf-8")).hexdigest()[:10]
    base = raw_id if raw_id else str(idx)
    safe = _sanitize_id(base) or str(idx)
    dataset_tag = _sanitize_id(dataset_name)
    return f"{dataset_tag}_{safe}_{digest}"


def _build_dataset_index(df: pd.DataFrame, id_col: str, indices: Dict[str, List[int]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    split_map: Dict[int, str] = {}
    for split_name, idxs in indices.items():
        for idx in idxs:
            split_map[int(idx)] = split_name

    if id_col in df.columns:
        sample_ids = df[id_col].astype(str).tolist()
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


def _normalize_split_name(name: str) -> str:
    key = str(name).strip().lower()
    if "test" in key:
        return "test"
    if key in {"valid", "validation", "val"}:
        return "val"
    if key in {"train"}:
        return "train"
    return key


def _to_index_list(values: Any) -> List[int]:
    if values is None:
        return []
    if torch is not None and hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    arr = np.asarray(values).reshape(-1)
    return [int(v) for v in arr.tolist()]


@register_dataset_adapter("pcqm4mv2")
class PCQM4Mv2DatasetAdapter(DatasetAdapter):
    name = "pcqm4mv2"

    def __init__(self) -> None:
        self._split_indices: Optional[Dict[str, List[int]]] = None

    def prepare_raw(self, cfg: Dict[str, Any], logger) -> RawDatasetInputs:
        paths = cfg.get("paths", {})
        raw_csv = Path(paths.get("raw_csv", "data/raw/pcqm4mv2/pcqm4mv2.csv"))
        sdf_dir = Path(paths.get("sdf_dir", "data/processed/pcqm4mv2/sdf"))
        pcqm_cfg = cfg.get("pcqm4mv2", {}) or {}
        root = Path(pcqm_cfg.get("root", paths.get("pcqm4mv2_root", raw_csv.parent)))
        root.mkdir(parents=True, exist_ok=True)
        return RawDatasetInputs(raw_csv=raw_csv, sdf_dir=sdf_dir, metadata={"root": str(root)})

    def get_splits(self, cfg: Dict[str, Any], df: pd.DataFrame, logger) -> Dict[str, List[int]]:
        if self._split_indices is not None:
            return self._split_indices

        split_cfg = cfg.get("split", {})
        split_method = str(split_cfg.get("method", "ogb")).lower()
        ratios = split_cfg.get("fractions", split_cfg.get("ratios", [0.8, 0.1, 0.1]))
        ratios = [float(x) for x in ratios]
        split_seed = int(split_cfg.get("seed", cfg.get("seed", 42)))

        cols = cfg.get("columns", {})
        cas_col = cols.get("cas", "sample_id")
        paths = cfg.get("paths", {})
        sdf_dir = Path(paths.get("sdf_dir", "data/processed/pcqm4mv2/sdf"))

        if split_method in {"ogb", "official", "provided"}:
            raise ValueError("OGB split indices not available; build_processed must set split indices.")
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
            group_col = resolve_group_column(df, cols, group_key, candidates=["canonical_smiles", "smiles", cas_col])
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
        if PCQM4Mv2Dataset is None:
            raise ImportError("ogb is required for PCQM4Mv2 adapter. Install ogb to use this adapter.")
        if Chem is None:
            raise ImportError("RDKit is required for PCQM4Mv2 adapter. Install rdkit to build SDF files.")

        seed = int(cfg.get("seed", 42))
        set_seed(seed)

        raw_inputs = self.prepare_raw(cfg, logger)
        pcqm_cfg = cfg.get("pcqm4mv2", {}) or {}

        root = Path(pcqm_cfg.get("root", raw_inputs.metadata.get("root", "data/raw/pcqm4mv2")))
        allow_download = bool(pcqm_cfg.get("allow_download", True))
        if not allow_download:
            dataset_dir = root / "pcqm4mv2"
            if not dataset_dir.exists():
                raise FileNotFoundError(
                    f"PCQM4Mv2 data not found at {dataset_dir}. Set pcqm4mv2.allow_download=true or pre-download the dataset."
                )

        dataset_kwargs: Dict[str, Any] = {}
        if "only_smiles" in pcqm_cfg:
            dataset_kwargs["only_smiles"] = bool(pcqm_cfg.get("only_smiles"))

        try:
            dataset = PCQM4Mv2Dataset(root=str(root), **dataset_kwargs)
        except TypeError:
            dataset = PCQM4Mv2Dataset(root=str(root))

        paths = cfg.get("paths", {})
        out_csv = Path(paths.get("out_csv", "data/processed/pcqm4mv2/dataset.csv"))
        out_indices_dir = Path(paths.get("out_indices_dir", "data/processed/pcqm4mv2/indices"))
        sdf_dir = Path(paths.get("sdf_dir", "data/processed/pcqm4mv2/sdf"))
        sdf_dir.mkdir(parents=True, exist_ok=True)

        limit_rows_cfg = cfg.get("limit_rows", None)
        limit_rows = int(limit_rows_cfg) if limit_rows_cfg else None
        n_rows = len(dataset)
        if limit_rows is not None and limit_rows > 0:
            n_rows = min(n_rows, limit_rows)
            logger.warning("Limiting rows for debug: %s -> %s", len(dataset), n_rows)

        if n_rows == 0:
            raise RuntimeError("PCQM4Mv2 dataset is empty; check dataset root and configuration.")

        target_name = str(pcqm_cfg.get("target_name", "homo_lumo_gap"))
        if target_name.startswith("target."):
            target_name = target_name[len("target.") :]
        add_hs = bool(pcqm_cfg.get("add_hs", True))
        compute_2d = bool(pcqm_cfg.get("compute_2d", True))
        overwrite_sdf = bool(pcqm_cfg.get("overwrite_sdf", False))
        write_sdf = bool(pcqm_cfg.get("write_sdf", True))
        canonicalize = bool(pcqm_cfg.get("canonicalize_smiles", True))
        dataset_name = resolve_dataset_name(cfg, default=self.name) or self.name

        records: List[Dict[str, Any]] = []
        row_map: Dict[int, int] = {}
        missing_smiles = 0
        invalid_mol = 0
        missing_targets = 0

        for idx in tqdm(range(n_rows), total=n_rows, desc="PCQM4Mv2 build"):
            data = dataset[idx]
            smiles = _resolve_smiles(dataset, data, idx)
            if not smiles:
                missing_smiles += 1
                continue

            canonical = smiles
            if canonicalize:
                canonical = _canonical_smiles(smiles)
                if not canonical:
                    invalid_mol += 1
                    continue

            target_value = _resolve_target(dataset, data, idx)
            if target_value is None or not np.isfinite(target_value):
                missing_targets += 1
                continue

            raw_id = None
            for key in ("idx", "id", "name"):
                if hasattr(data, key):
                    raw_id = getattr(data, key)
                    break
                if isinstance(data, dict) and key in data:
                    raw_id = data[key]
                    break
            if raw_id is not None and torch is not None and hasattr(raw_id, "item"):
                raw_id = raw_id.item()
            raw_id_str = str(raw_id) if raw_id is not None else None

            sample_id = _build_sample_id(dataset_name, canonical, raw_id_str, idx)

            mol = _mol_from_smiles(canonical, add_hs=add_hs, compute_2d=compute_2d)
            if mol is None:
                invalid_mol += 1
                continue
            mol.SetProp("_Name", sample_id)
            mol.SetProp("smiles", canonical)

            elements = elements_string(get_elements_from_mol(mol))
            n_heavy_atoms = int(mol.GetNumHeavyAtoms())

            if write_sdf:
                sdf_path = sdf_path_from_cas(sdf_dir, sample_id)
                if overwrite_sdf or not sdf_path.exists():
                    writer = Chem.SDWriter(str(sdf_path))
                    writer.write(mol)
                    writer.close()

            record = {
                "sample_id": sample_id,
                "smiles": canonical,
                "canonical_smiles": canonical,
                "raw_row_index": int(idx),
                "elements": elements,
                "n_heavy_atoms": n_heavy_atoms,
                str(target_name): float(target_value),
            }
            records.append(record)
            row_map[int(idx)] = len(records) - 1

        if missing_smiles:
            logger.warning("PCQM4Mv2 rows missing SMILES: %s", missing_smiles)
        if invalid_mol:
            logger.warning("PCQM4Mv2 rows with invalid molecules: %s", invalid_mol)
        if missing_targets:
            logger.warning("PCQM4Mv2 rows with missing targets: %s", missing_targets)

        if not records:
            raise RuntimeError("PCQM4Mv2 adapter produced no records; check dataset integrity and SMILES parsing.")

        df = pd.DataFrame(records).reset_index(drop=True)

        registry = load_target_registry(cfg)
        if registry:
            canonical_targets: List[str] = []
            available_cols = df.columns.tolist()
            for name in registry.canonical_names():
                source_col = registry.resolve_source_column(name, available_cols)
                if source_col is None:
                    spec = registry.get(name)
                    if spec is None or spec.required:
                        raise ValueError(f"Target '{name}' not found in dataset columns (aliases={spec.aliases if spec else []})")
                    continue
                df[target_column_name(name)] = df[source_col]
                canonical_targets.append(name)
        else:
            canonical_targets = [target_name]
            df[target_column_name(target_name)] = df[target_name]

        write_csv(df, out_csv)
        logger.info("Saved processed dataset: %s (rows=%s)", out_csv, df.shape[0])

        split_cfg = cfg.get("split", {})
        split_method = str(split_cfg.get("method", "ogb")).lower()
        if split_method in {"ogb", "official", "provided"} and hasattr(dataset, "get_idx_split"):
            raw_split = dataset.get_idx_split()
            indices = {"train": [], "val": [], "test": []}
            missing_split = 0
            for split_name, raw_idxs in raw_split.items():
                normalized = _normalize_split_name(split_name)
                if normalized not in indices:
                    continue
                for raw_idx in _to_index_list(raw_idxs):
                    mapped = row_map.get(int(raw_idx))
                    if mapped is None:
                        missing_split += 1
                        continue
                    indices[normalized].append(mapped)
            if missing_split:
                logger.warning("PCQM4Mv2 split indices skipped due to filtered rows: %s", missing_split)
            self._split_indices = indices
        else:
            indices = self.get_splits(cfg, df, logger)

        validate_split_indices(indices)
        save_split_indices(indices, out_indices_dir)
        split_meta = {
            "method": split_method,
            "seed": int(split_cfg.get("seed", seed)),
            "fractions": split_cfg.get("fractions", split_cfg.get("ratios", [0.8, 0.1, 0.1])),
        }
        if split_meta["method"] == "group":
            split_meta["group_key"] = str(split_cfg.get("group_key"))
        if split_meta["method"] == "time":
            time_key = split_cfg.get("time_key") or split_cfg.get("time_col")
            if time_key:
                split_meta["time_key"] = str(time_key)
                cols = cfg.get("columns", {})
                split_meta["time_col"] = str(cols.get(str(time_key), str(time_key)))
            split_meta["ascending"] = bool(split_cfg.get("ascending", True))
        split_json_path = out_indices_dir / "split.json"
        save_split_json(indices, split_json_path, metadata=split_meta)
        logger.info("Saved split indices to %s", out_indices_dir)

        dump_yaml(out_csv.parent / "dataset_config_snapshot.yaml", cfg)

        cols = cfg.get("columns", {})
        dataset_index_path = out_csv.parent / "dataset_index.csv"
        _build_dataset_index(df, id_col=str(cols.get("sample_id", cols.get("cas", "sample_id"))), indices=indices, out_path=dataset_index_path)

        dataset_hash = compute_dataset_hash(out_csv, out_indices_dir)

        run_dir = Path(cfg.get("output", {}).get("run_dir", out_csv.parent))
        ensure_dir(run_dir)
        dump_yaml(run_dir / "config.yaml", cfg)

        dataset_name = resolve_dataset_name(cfg, default=self.name)
        units_cfg = cfg.get("dataset", {}).get("units", {}) if isinstance(cfg.get("dataset", {}), dict) else {}
        if registry:
            units = registry.units_for(canonical_targets)
        else:
            units = {name: None for name in canonical_targets}
        for name, unit in (units_cfg or {}).items():
            if unit is None:
                continue
            units[str(name)] = str(unit)

        meta_extra = {
            "dataset_name": dataset_name,
            "target_names": canonical_targets,
            "units": units,
            "target_meta": resolve_target_metadata(cfg, canonical_targets),
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
