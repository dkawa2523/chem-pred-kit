from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.common.chemistry import elements_string, get_elements_from_mol
from src.common.config import dump_yaml
from src.common.io import load_sdf_mol, sdf_path_from_cas, write_csv
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
    from rdkit.Chem import rdDetermineBonds
    from rdkit.Geometry import Point3D
except Exception:  # pragma: no cover
    Chem = None
    rdDetermineBonds = None
    Point3D = None


def _sanitize_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none"}


def _to_float_or_none(value: Any) -> Optional[float]:
    if _is_missing(value):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def _parse_json(value: Any, field: str) -> Optional[Any]:
    if _is_missing(value):
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{field} must be valid JSON array") from exc
    return None


def _coerce_atomic_numbers(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=int)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr


def _coerce_matrix(value: Any, field: str) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 1:
        if arr.size % 3 != 0:
            raise ValueError(f"{field} length must be multiple of 3")
        arr = arr.reshape(-1, 3)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{field} must have shape (N, 3); got {arr.shape}")
    return arr


def _build_sample_id(dataset_name: str, trajectory_id: Optional[str], time_value: Any, idx: int, atomic_numbers: Optional[np.ndarray]) -> str:
    safe_dataset = _sanitize_id(dataset_name)
    base = _sanitize_id(trajectory_id) if trajectory_id else safe_dataset
    time_str = _sanitize_id(str(time_value)) if not _is_missing(time_value) else str(idx)
    digest = ""
    if atomic_numbers is not None and atomic_numbers.size:
        payload = ",".join(str(int(v)) for v in atomic_numbers.tolist()).encode("utf-8")
        digest = hashlib.sha1(payload).hexdigest()[:8]
    parts = [safe_dataset, base, time_str, digest]
    safe = "_".join([p for p in parts if p])
    return safe or f"{safe_dataset}_{idx}"


def _mol_has_3d(mol) -> bool:
    if mol is None:
        return False
    if mol.GetNumConformers() == 0:
        return False
    conf = mol.GetConformer()
    return conf is not None and conf.Is3D()


def _build_mol_with_pos(
    atomic_numbers: np.ndarray,
    positions: np.ndarray,
    infer_bonds: bool,
) -> "Chem.Mol":
    if Chem is None or Point3D is None:
        raise ImportError("RDKit is required to build MD17 SDF files.")
    mol = Chem.RWMol()
    for z in atomic_numbers.tolist():
        atom = Chem.Atom(int(z))
        atom.SetNoImplicit(True)
        mol.AddAtom(atom)
    conf = Chem.Conformer(int(len(atomic_numbers)))
    for i, (x, y, z) in enumerate(positions.tolist()):
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    mol.AddConformer(conf, assignId=True)
    mol = mol.GetMol()
    if infer_bonds and rdDetermineBonds is not None:
        try:
            rdDetermineBonds.DetermineBonds(mol)
        except Exception:
            pass
    return mol


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


@register_dataset_adapter("md17")
class MD17DatasetAdapter(DatasetAdapter):
    name = "md17"

    def prepare_raw(self, cfg: Dict[str, Any], logger) -> RawDatasetInputs:
        paths = cfg.get("paths", {})
        raw_csv = Path(paths.get("raw_csv", "data/raw/md17/md17.csv"))
        sdf_dir = Path(paths.get("sdf_dir", "data/raw/md17/sdf"))
        if not raw_csv.exists():
            raise FileNotFoundError(f"raw_csv not found: {raw_csv}")
        sdf_dir.mkdir(parents=True, exist_ok=True)
        return RawDatasetInputs(raw_csv=raw_csv, sdf_dir=sdf_dir)

    def get_splits(self, cfg: Dict[str, Any], df: pd.DataFrame, logger) -> Dict[str, List[int]]:
        split_cfg = cfg.get("split", {})
        split_method = str(split_cfg.get("method", "time")).lower()
        ratios = split_cfg.get("fractions", split_cfg.get("ratios", [0.8, 0.1, 0.1]))
        ratios = [float(x) for x in ratios]
        split_seed = int(split_cfg.get("seed", cfg.get("seed", 42)))

        cols = cfg.get("columns", {})
        cas_col = cols.get("cas", "sample_id")
        paths = cfg.get("paths", {})
        sdf_dir = Path(paths.get("sdf_dir", "data/raw/md17/sdf"))

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
            group_col = resolve_group_column(df, cols, group_key, candidates=[cas_col])
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
        if Chem is None:
            raise ImportError("RDKit is required for MD17 adapter. Please install rdkit.")

        seed = int(cfg.get("seed", 42))
        set_seed(seed)

        raw_inputs = self.prepare_raw(cfg, logger)
        paths = cfg.get("paths", {})
        out_csv = Path(paths.get("out_csv", "data/processed/md17/dataset.csv"))
        out_indices_dir = Path(paths.get("out_indices_dir", "data/processed/md17/indices"))

        df = pd.read_csv(raw_inputs.raw_csv)
        df["raw_row_index"] = df.index.astype(int)
        logger.info("Loaded %s rows from %s", df.shape[0], raw_inputs.raw_csv)

        limit_rows_cfg = cfg.get("limit_rows", None)
        limit_rows = int(limit_rows_cfg) if limit_rows_cfg else None
        if limit_rows is not None and int(limit_rows) > 0:
            before = df.shape[0]
            df = df.head(int(limit_rows)).copy()
            logger.warning("Limiting rows for debug: %s -> %s", before, df.shape[0])

        cols = cfg.get("columns", {})
        sample_id_col = cols.get("sample_id", "sample_id")
        time_col = cols.get("time", "time")
        energy_col = cols.get("energy", "energy")
        forces_col = cols.get("forces", "forces")
        positions_col = cols.get("positions", cols.get("pos", "positions"))
        atomic_numbers_col = cols.get("atomic_numbers", cols.get("atomic_nums", cols.get("z", "atomic_numbers")))
        sdf_path_col = cols.get("sdf_path", cols.get("sdf", cols.get("sdf_file", None)))
        traj_col = cols.get("trajectory", cols.get("molecule", cols.get("traj", None)))
        smiles_col = cols.get("smiles", None)

        md17_cfg = cfg.get("md17", {}) or {}
        overwrite_sdf = bool(md17_cfg.get("overwrite_sdf", False))
        write_sdf = bool(md17_cfg.get("write_sdf", True))
        require_3d = bool(md17_cfg.get("require_3d", True))
        infer_bonds = bool(md17_cfg.get("infer_bonds", False))

        dataset_name = resolve_dataset_name(cfg, default=self.name)

        records: List[Dict[str, Any]] = []
        missing_3d = 0
        invalid_rows = 0

        for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="MD17 build"):
            raw_row_index = int(row.get("raw_row_index", idx))

            time_value = row.get(time_col) if time_col in df.columns else None
            trajectory_id = None
            if traj_col and traj_col in df.columns and not _is_missing(row.get(traj_col)):
                trajectory_id = str(row.get(traj_col))

            sample_id = None
            if sample_id_col in df.columns and not _is_missing(row.get(sample_id_col)):
                sample_id = str(row.get(sample_id_col))
            if not sample_id:
                atom_json = _parse_json(row.get(atomic_numbers_col) if atomic_numbers_col in df.columns else None, "atomic_numbers")
                atomic_numbers = _coerce_atomic_numbers(atom_json)
                sample_id = _build_sample_id(dataset_name, trajectory_id, time_value, idx, atomic_numbers)

            energy_val = _to_float_or_none(row.get(energy_col) if energy_col in df.columns else None)

            forces_json = _parse_json(row.get(forces_col) if forces_col in df.columns else None, "forces")
            forces_arr = _coerce_matrix(forces_json, "forces") if forces_json is not None else None
            forces_payload = _json_dumps(forces_arr.tolist()) if forces_arr is not None else None

            positions_json = _parse_json(row.get(positions_col) if positions_col in df.columns else None, "positions")
            positions_arr = _coerce_matrix(positions_json, "positions") if positions_json is not None else None

            atomic_numbers_json = _parse_json(row.get(atomic_numbers_col) if atomic_numbers_col in df.columns else None, "atomic_numbers")
            atomic_numbers_arr = _coerce_atomic_numbers(atomic_numbers_json) if atomic_numbers_json is not None else None

            mol = None
            sdf_path = None
            if sdf_path_col and sdf_path_col in df.columns and not _is_missing(row.get(sdf_path_col)):
                sdf_path = Path(str(row.get(sdf_path_col)))
            else:
                sdf_path = sdf_path_from_cas(raw_inputs.sdf_dir, sample_id)

            if sdf_path.exists() and not overwrite_sdf:
                mol = load_sdf_mol(sdf_path)

            needs_sdf = (write_sdf and mol is None) or (require_3d and not _mol_has_3d(mol))
            if needs_sdf:
                if positions_arr is None or atomic_numbers_arr is None:
                    missing_3d += 1
                    if require_3d:
                        raise ValueError(f"Missing positions/atomic_numbers for sample_id={sample_id}")
                else:
                    if positions_arr.shape[0] != atomic_numbers_arr.shape[0]:
                        invalid_rows += 1
                        logger.warning("Skipping row %s due to atom/pos mismatch (%s != %s).", idx, positions_arr.shape[0], atomic_numbers_arr.shape[0])
                        continue
                    mol = _build_mol_with_pos(atomic_numbers_arr, positions_arr, infer_bonds=infer_bonds)
                    mol.SetProp("_Name", str(sample_id))
                    if trajectory_id:
                        mol.SetProp("trajectory_id", str(trajectory_id))
                    if time_value is not None:
                        mol.SetProp("time", str(time_value))
                    sdf_path.parent.mkdir(parents=True, exist_ok=True)
                    writer = Chem.SDWriter(str(sdf_path))
                    writer.write(mol)
                    writer.close()

            if require_3d and not _mol_has_3d(mol):
                missing_3d += 1
                raise ValueError(f"3D positions required but missing for sample_id={sample_id}")

            if mol is None:
                invalid_rows += 1
                logger.warning("Skipping row %s due to missing SDF/mol.", idx)
                continue

            if atomic_numbers_arr is None:
                atomic_numbers_arr = np.array([int(atom.GetAtomicNum()) for atom in mol.GetAtoms()], dtype=int)

            elements = elements_string(get_elements_from_mol(mol))
            n_atoms = int(atomic_numbers_arr.shape[0]) if atomic_numbers_arr is not None else int(mol.GetNumAtoms())
            n_heavy = int(np.sum(atomic_numbers_arr > 1)) if atomic_numbers_arr is not None else int(mol.GetNumHeavyAtoms())

            record: Dict[str, Any] = {
                "sample_id": str(sample_id),
                "raw_row_index": raw_row_index,
                "time": time_value,
                "elements": elements,
                "n_atoms": n_atoms,
                "n_heavy_atoms": n_heavy,
                str(energy_col): energy_val,
                str(forces_col): forces_payload,
            }
            if smiles_col and smiles_col in df.columns and not _is_missing(row.get(smiles_col)):
                record["smiles"] = str(row.get(smiles_col))
            if trajectory_id is not None and traj_col:
                record[str(traj_col)] = trajectory_id
            records.append(record)

        if missing_3d:
            logger.warning("Rows missing 3D positions: %s", missing_3d)
        if invalid_rows:
            logger.warning("Rows skipped due to invalid inputs: %s", invalid_rows)

        if not records:
            raise RuntimeError("MD17 adapter produced no records; check raw CSV and SDF/positions.")

        out_df = pd.DataFrame(records).reset_index(drop=True)

        registry = load_target_registry(cfg)
        if registry:
            canonical_targets: List[str] = []
            available_cols = out_df.columns.tolist()
            for name in registry.canonical_names():
                source_col = registry.resolve_source_column(name, available_cols)
                if source_col is None:
                    spec = registry.get(name)
                    if spec is None or spec.required:
                        raise ValueError(f"Target '{name}' not found in dataset columns (aliases={spec.aliases if spec else []})")
                    continue
                out_df[target_column_name(name)] = out_df[source_col]
                canonical_targets.append(name)
        else:
            canonical_targets = []
            for col_name in [energy_col, forces_col]:
                if col_name in out_df.columns:
                    out_df[target_column_name(col_name)] = out_df[col_name]
                    canonical_targets.append(str(col_name))

        write_csv(out_df, out_csv)
        logger.info("Saved processed dataset: %s (rows=%s)", out_csv, out_df.shape[0])

        indices = self.get_splits(cfg, out_df, logger)
        save_split_indices(indices, out_indices_dir)
        split_meta = {
            "method": str(cfg.get("split", {}).get("method", "time")).lower(),
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
        split_json_path = out_indices_dir / "split.json"
        save_split_json(indices, split_json_path, metadata=split_meta)
        logger.info("Saved split indices to %s", out_indices_dir)

        dump_yaml(out_csv.parent / "dataset_config_snapshot.yaml", cfg)

        dataset_index_path = out_csv.parent / "dataset_index.csv"
        _build_dataset_index(out_df, id_col=str(cols.get("sample_id", "sample_id")), indices=indices, out_path=dataset_index_path)

        dataset_hash = compute_dataset_hash(out_csv, out_indices_dir)

        run_dir = Path(cfg.get("output", {}).get("run_dir", out_csv.parent))
        ensure_dir(run_dir)
        dump_yaml(run_dir / "config.yaml", cfg)

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
