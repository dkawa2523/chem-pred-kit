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
    from torch_geometric.datasets import QM9
except Exception:  # pragma: no cover
    torch = None
    QM9 = None

DEFAULT_QM9_TARGETS_12 = [
    "mu",
    "alpha",
    "homo",
    "lumo",
    "gap",
    "r2",
    "zpve",
    "U0",
    "U",
    "H",
    "G",
    "Cv",
]

DEFAULT_QM9_TARGETS_19 = [
    "A",
    "B",
    "C",
    "mu",
    "alpha",
    "homo",
    "lumo",
    "gap",
    "r2",
    "zpve",
    "U0",
    "U",
    "H",
    "G",
    "Cv",
    "U0_atom",
    "U_atom",
    "H_atom",
    "G_atom",
]


def _to_numpy(values: Any) -> np.ndarray:
    if torch is not None and hasattr(values, "detach"):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def _resolve_target_names(cfg: Dict[str, Any], dataset: Any, y_dim: int, logger) -> List[str]:
    qm9_cfg = cfg.get("qm9", {}) or {}
    override = qm9_cfg.get("target_names")
    if override:
        if not isinstance(override, (list, tuple)):
            raise ValueError("qm9.target_names must be a list")
        names = [str(v) for v in override]
        if len(names) != y_dim:
            raise ValueError(f"qm9.target_names length ({len(names)}) != QM9 y_dim ({y_dim})")
        return names

    for attr in ("target_names", "targets", "target"):
        if hasattr(dataset, attr):
            names = getattr(dataset, attr)
            if isinstance(names, (list, tuple)) and len(names) == y_dim:
                return [str(v) for v in names]

    if y_dim == len(DEFAULT_QM9_TARGETS_12):
        return list(DEFAULT_QM9_TARGETS_12)
    if y_dim == len(DEFAULT_QM9_TARGETS_19):
        return list(DEFAULT_QM9_TARGETS_19)

    logger.warning("QM9 target count=%s has no known mapping; using generic names.", y_dim)
    return [f"y{i}" for i in range(y_dim)]


def _resolve_smiles(dataset: Any, data: Any, idx: int) -> Optional[str]:
    if hasattr(data, "smiles"):
        smiles = getattr(data, "smiles")
        if isinstance(smiles, (list, tuple)) and smiles:
            return str(smiles[0])
        if smiles:
            return str(smiles)
    if hasattr(dataset, "smiles"):
        try:
            smiles = dataset.smiles[idx]
            if smiles:
                return str(smiles)
        except Exception:
            pass
    return None


def _canonical_smiles(smiles: str) -> Optional[str]:
    if Chem is None:
        raise ImportError("RDKit is required to canonicalize QM9 SMILES.")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def _mol_from_smiles(smiles: str, add_hs: bool, compute_2d: bool) -> Optional["Chem.Mol"]:
    if Chem is None:
        raise ImportError("RDKit is required to build QM9 molecules.")
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


def _build_sample_id(canonical_smiles: str, raw_id: Optional[str], idx: int) -> str:
    digest = hashlib.sha1(canonical_smiles.encode("utf-8")).hexdigest()[:10]
    base = raw_id if raw_id else str(idx)
    safe = _sanitize_id(base) or str(idx)
    return f"qm9_{safe}_{digest}"


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


@register_dataset_adapter("qm9")
class QM9DatasetAdapter(DatasetAdapter):
    name = "qm9"

    def prepare_raw(self, cfg: Dict[str, Any], logger) -> RawDatasetInputs:
        paths = cfg.get("paths", {})
        raw_csv = Path(paths.get("raw_csv", "data/raw/qm9/qm9.csv"))
        sdf_dir = Path(paths.get("sdf_dir", "data/processed/qm9/sdf"))
        qm9_cfg = cfg.get("qm9", {}) or {}
        root = Path(qm9_cfg.get("root", paths.get("qm9_root", raw_csv.parent)))
        root.mkdir(parents=True, exist_ok=True)
        return RawDatasetInputs(raw_csv=raw_csv, sdf_dir=sdf_dir, metadata={"root": str(root)})

    def get_splits(self, cfg: Dict[str, Any], df: pd.DataFrame, logger) -> Dict[str, List[int]]:
        split_cfg = cfg.get("split", {})
        split_method = str(split_cfg.get("method", "random")).lower()
        ratios = split_cfg.get("fractions", split_cfg.get("ratios", [0.8, 0.1, 0.1]))
        ratios = [float(x) for x in ratios]
        split_seed = int(split_cfg.get("seed", cfg.get("seed", 42)))

        cols = cfg.get("columns", {})
        cas_col = cols.get("cas", "sample_id")
        paths = cfg.get("paths", {})
        sdf_dir = Path(paths.get("sdf_dir", "data/processed/qm9/sdf"))

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
        if QM9 is None:
            raise ImportError(
                "torch_geometric is required for QM9 adapter. Install torch_geometric (and torch) to use this adapter."
            )
        if Chem is None:
            raise ImportError("RDKit is required for QM9 adapter. Install rdkit to build SDF files.")

        seed = int(cfg.get("seed", 42))
        set_seed(seed)

        raw_inputs = self.prepare_raw(cfg, logger)
        qm9_cfg = cfg.get("qm9", {}) or {}

        root = Path(qm9_cfg.get("root", raw_inputs.metadata.get("root", "data/raw/qm9")))
        allow_download = bool(qm9_cfg.get("allow_download", True))
        if not allow_download:
            raw_sdf = root / "raw" / "gdb9.sdf"
            if not raw_sdf.exists():
                raise FileNotFoundError(
                    f"QM9 raw data not found at {raw_sdf}. Set qm9.allow_download=true or pre-download the dataset."
                )

        dataset = QM9(str(root))

        paths = cfg.get("paths", {})
        out_csv = Path(paths.get("out_csv", "data/processed/qm9/dataset.csv"))
        out_indices_dir = Path(paths.get("out_indices_dir", "data/processed/qm9/indices"))
        sdf_dir = Path(paths.get("sdf_dir", "data/processed/qm9/sdf"))
        sdf_dir.mkdir(parents=True, exist_ok=True)

        limit_rows_cfg = cfg.get("limit_rows", None)
        limit_rows = int(limit_rows_cfg) if limit_rows_cfg else None
        n_rows = len(dataset)
        if limit_rows is not None and limit_rows > 0:
            n_rows = min(n_rows, limit_rows)
            logger.warning("Limiting rows for debug: %s -> %s", len(dataset), n_rows)

        if n_rows == 0:
            raise RuntimeError("QM9 dataset is empty; check dataset root and configuration.")

        first_y = _to_numpy(dataset[0].y).reshape(-1)
        y_dim = int(first_y.shape[0])
        target_names = _resolve_target_names(cfg, dataset, y_dim, logger)

        add_hs = bool(qm9_cfg.get("add_hs", True))
        compute_2d = bool(qm9_cfg.get("compute_2d", True))
        overwrite_sdf = bool(qm9_cfg.get("overwrite_sdf", False))

        records: List[Dict[str, Any]] = []
        missing_smiles = 0
        invalid_mol = 0

        for idx in tqdm(range(n_rows), total=n_rows, desc="QM9 build"):
            data = dataset[idx]
            smiles = _resolve_smiles(dataset, data, idx)
            if not smiles:
                missing_smiles += 1
                continue
            canonical = _canonical_smiles(smiles)
            if not canonical:
                invalid_mol += 1
                continue

            raw_id = None
            for key in ("idx", "id", "name"):
                if hasattr(data, key):
                    raw_id = getattr(data, key)
                    break
            if raw_id is not None and torch is not None and hasattr(raw_id, "item"):
                raw_id = raw_id.item()
            raw_id_str = str(raw_id) if raw_id is not None else None
            try:
                raw_row_index = int(raw_id) if raw_id is not None else int(idx)
            except Exception:
                raw_row_index = int(idx)

            sample_id = _build_sample_id(canonical, raw_id_str, idx)
            mol = _mol_from_smiles(canonical, add_hs=add_hs, compute_2d=compute_2d)
            if mol is None:
                invalid_mol += 1
                continue
            mol.SetProp("_Name", sample_id)
            mol.SetProp("smiles", canonical)

            element_counts = get_elements_from_mol(mol)
            elements = elements_string(element_counts)
            n_heavy_atoms = int(mol.GetNumHeavyAtoms())

            sdf_path = sdf_path_from_cas(sdf_dir, sample_id)
            if overwrite_sdf or not sdf_path.exists():
                writer = Chem.SDWriter(str(sdf_path))
                writer.write(mol)
                writer.close()

            y = _to_numpy(data.y).reshape(-1)
            if y.shape[0] != y_dim:
                logger.warning("Skipping row %s due to y dim mismatch (%s != %s).", idx, y.shape[0], y_dim)
                continue

            record = {
                "sample_id": sample_id,
                "smiles": canonical,
                "canonical_smiles": canonical,
                "raw_row_index": raw_row_index,
                "elements": elements,
                "n_heavy_atoms": n_heavy_atoms,
            }
            for name, value in zip(target_names, y.tolist()):
                record[str(name)] = float(value)
            records.append(record)

        if missing_smiles:
            logger.warning("QM9 rows missing SMILES: %s", missing_smiles)
        if invalid_mol:
            logger.warning("QM9 rows with invalid molecules: %s", invalid_mol)

        if not records:
            raise RuntimeError("QM9 adapter produced no records; check dataset integrity and SMILES parsing.")

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
            canonical_targets = list(target_names)
            for name in canonical_targets:
                df[target_column_name(name)] = df[name]

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
