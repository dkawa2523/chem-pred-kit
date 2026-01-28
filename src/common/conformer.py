from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except Exception:  # pragma: no cover
    Chem = None
    AllChem = None


class ConformerError(ValueError):
    pass


@dataclass
class ConformerConfig:
    enabled: bool = True
    seed: int = 42
    method: str = "etkdg"
    forcefield: str = "uff"
    max_iters: int = 200
    max_attempts: int = 1
    num_conformers: int = 1
    aggregation: str = "mean"
    on_failure: str = "skip"  # skip | fallback_2d | retry
    cache_enabled: bool = True
    use_smiles_fallback: bool = True

    def normalized(self) -> "ConformerConfig":
        cfg = ConformerConfig(**asdict(self))
        cfg.method = str(cfg.method).lower()
        cfg.forcefield = str(cfg.forcefield).lower()
        cfg.aggregation = str(cfg.aggregation or "mean").lower()
        cfg.num_conformers = max(1, int(cfg.num_conformers))
        cfg.on_failure = str(cfg.on_failure).lower()
        if cfg.on_failure == "retry" and cfg.max_attempts < 2:
            cfg.max_attempts = 3
        return cfg


def conformer_config_from_cfg(cfg: Dict[str, Any]) -> ConformerConfig:
    cfg = cfg or {}
    feat_cfg = cfg.get("featurizer", {}) if isinstance(cfg, dict) else {}
    raw: Dict[str, Any] = {}
    top_cfg = cfg.get("conformer", {})
    if isinstance(top_cfg, dict):
        raw.update(top_cfg)
    if isinstance(feat_cfg, dict) and isinstance(feat_cfg.get("conformer", {}), dict):
        raw.update(feat_cfg.get("conformer", {}))

    seed_default = int(cfg.get("train", {}).get("seed", 42)) if isinstance(cfg, dict) else 42
    seed = int(raw.get("seed", seed_default))

    on_failure = str(raw.get("on_failure", raw.get("failure_strategy", "skip"))).lower()
    max_attempts = int(raw.get("max_attempts", raw.get("retry", 1)))
    if on_failure == "retry" and max_attempts < 2:
        max_attempts = 3
    num_conformers_raw = raw.get("num_conformers", raw.get("n_conformers", raw.get("k", 1)))
    try:
        num_conformers = int(num_conformers_raw)
    except Exception:
        num_conformers = 1
    aggregation_raw = raw.get("aggregation", raw.get("aggregate", "mean"))
    aggregation = "mean" if aggregation_raw is None else str(aggregation_raw)

    if "cache_enabled" in raw:
        cache_enabled = bool(raw.get("cache_enabled"))
    else:
        cache_cfg = raw.get("cache", None)
        if isinstance(cache_cfg, dict):
            cache_enabled = bool(cache_cfg.get("enabled", True))
        elif cache_cfg is None:
            cache_enabled = True
        else:
            cache_enabled = bool(cache_cfg)

    return ConformerConfig(
        enabled=bool(raw.get("enabled", True)),
        seed=seed,
        method=str(raw.get("method", "etkdg")),
        forcefield=str(raw.get("forcefield", "uff")),
        max_iters=int(raw.get("max_iters", 200)),
        max_attempts=max_attempts,
        num_conformers=num_conformers,
        aggregation=aggregation,
        on_failure=on_failure,
        cache_enabled=cache_enabled,
        use_smiles_fallback=bool(raw.get("use_smiles_fallback", True)),
    ).normalized()


def conformer_config_payload(cfg: ConformerConfig) -> Dict[str, Any]:
    return asdict(cfg.normalized())


def mol_from_smiles(smiles: str):
    if Chem is None:
        raise ImportError("RDKit is required to create molecules from SMILES.")
    if smiles is None:
        return None
    s = str(smiles).strip()
    if not s:
        return None
    try:
        return Chem.MolFromSmiles(s)
    except Exception:
        return None


def _require_rdkit() -> None:
    if Chem is None or AllChem is None:
        raise ImportError("RDKit is required for conformer generation.")


def _seed_for_sample(sample_id: str, base_seed: int, attempt: int) -> int:
    payload = f"{base_seed}:{sample_id}:{attempt}".encode("utf-8")
    digest = hashlib.sha1(payload).hexdigest()[:8]
    value = int(digest, 16)
    return value % (2**31)


def _seed_for_conformer(sample_id: str, base_seed: int, attempt: int, conformer_idx: int) -> int:
    payload = f"{base_seed}:{sample_id}:{attempt}:{conformer_idx}".encode("utf-8")
    digest = hashlib.sha1(payload).hexdigest()[:8]
    value = int(digest, 16)
    return value % (2**31)


def _pos_from_conformer(conf, n_atoms: int) -> np.ndarray:
    coords = conf.GetPositions()
    arr = np.array(coords, dtype=np.float32)
    if arr.shape != (n_atoms, 3):
        raise ConformerError(f"conformer coords shape mismatch: {arr.shape} vs ({n_atoms}, 3)")
    return arr


def _aggregate_positions(positions: Sequence[np.ndarray], method: str) -> np.ndarray:
    if not positions:
        raise ConformerError("No conformers available for aggregation.")
    method = str(method).lower()
    if method != "mean":
        raise ConformerError(f"Unsupported conformer aggregation: {method}")
    stack = np.stack([np.asarray(pos, dtype=np.float32) for pos in positions], axis=0)
    return np.mean(stack, axis=0).astype(np.float32)


class ConformerCache:
    version = 1

    def __init__(self, path: str | Path, config: Optional[Dict[str, Any]] = None) -> None:
        self.path = Path(path)
        self.config: Dict[str, Any] = dict(config or {})
        self._entries: Dict[str, Dict[str, Dict[str, Any]]] = {}
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ConformerError(f"Invalid conformer cache format: {self.path}")
        if data.get("version") not in (None, self.version):
            raise ConformerError(f"Unsupported conformer cache version: {data.get('version')}")
        if isinstance(data.get("config"), dict):
            self.config = data["config"]
        entries = data.get("entries", []) or []
        for entry in entries:
            sample_id = str(entry.get("sample_id", ""))
            if not sample_id:
                continue
            conformers = entry.get("conformers", []) or []
            for conf in conformers:
                conformer_id = str(conf.get("conformer_id", "0"))
                pos = conf.get("pos")
                meta = conf.get("meta", {}) or {}
                self._entries.setdefault(sample_id, {})[conformer_id] = {"pos": pos, "meta": meta}

    def get(self, sample_id: str, conformer_id: str = "0") -> Optional[Dict[str, Any]]:
        entry = self._entries.get(str(sample_id), {}).get(str(conformer_id))
        if entry is None:
            return None
        pos = entry.get("pos")
        pos_arr = np.array(pos, dtype=np.float32) if pos is not None else None
        return {"pos": pos_arr, "meta": entry.get("meta", {}) or {}, "conformer_id": str(conformer_id)}

    def set(self, sample_id: str, pos: Optional[np.ndarray], meta: Dict[str, Any], conformer_id: str = "0") -> None:
        pos_list = None
        if pos is not None:
            arr = np.asarray(pos, dtype=np.float32)
            pos_list = [[float(v) for v in row] for row in arr.tolist()]
        self._entries.setdefault(str(sample_id), {})[str(conformer_id)] = {"pos": pos_list, "meta": dict(meta)}

    def save(self) -> None:
        entries = []
        for sample_id in sorted(self._entries.keys()):
            confs = []
            for conformer_id in sorted(self._entries[sample_id].keys(), key=str):
                record = self._entries[sample_id][conformer_id]
                confs.append(
                    {
                        "conformer_id": conformer_id,
                        "pos": record.get("pos"),
                        "meta": record.get("meta", {}) or {},
                    }
                )
            entries.append({"sample_id": sample_id, "conformers": confs})
        payload = {"version": self.version, "config": self.config, "entries": entries}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2), encoding="utf-8")

    def __len__(self) -> int:
        return sum(len(v) for v in self._entries.values())


class ConformerService:
    AGGREGATE_CONFORMER_ID = "agg"

    def __init__(self, cfg: ConformerConfig, cache: Optional[ConformerCache] = None) -> None:
        self.cfg = cfg.normalized()
        self.cache = cache

    def _base_meta(self) -> Dict[str, Any]:
        return {
            "seed": int(self.cfg.seed),
            "forcefield": self.cfg.forcefield,
            "max_iters": int(self.cfg.max_iters),
            "num_conformers": int(self.cfg.num_conformers),
            "aggregation": self.cfg.aggregation,
        }

    def get_pos(
        self,
        sample_id: str,
        mol,
        smiles: Optional[str] = None,
        allow_generate: bool = True,
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        sample_id = str(sample_id)
        if not self.cfg.enabled:
            return None, {"status": "disabled"}
        if self.cache is not None:
            if self.cfg.num_conformers > 1:
                cached = self.cache.get(sample_id, conformer_id=self.AGGREGATE_CONFORMER_ID)
                if cached is not None:
                    return cached["pos"], cached.get("meta", {})
                pos, meta = self._aggregate_cached_conformers(sample_id)
                if pos is not None:
                    return pos, meta
            else:
                cached = self.cache.get(sample_id)
                if cached is not None:
                    return cached["pos"], cached.get("meta", {})
        if not allow_generate:
            return None, {"status": "cache_miss"}

        if self.cfg.num_conformers > 1:
            pos, meta, conformer_entries = self._generate_multi_pos(sample_id, mol, smiles)
            if self.cache is not None and self.cfg.cache_enabled:
                for entry in conformer_entries:
                    self.cache.set(sample_id, entry["pos"], entry["meta"], conformer_id=entry["conformer_id"])
                self.cache.set(sample_id, pos, meta, conformer_id=self.AGGREGATE_CONFORMER_ID)
            return pos, meta

        pos, meta = self._generate_pos(sample_id, mol, smiles)
        if self.cache is not None and self.cfg.cache_enabled:
            self.cache.set(sample_id, pos, meta)
        return pos, meta

    def _aggregate_cached_conformers(self, sample_id: str) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        if self.cache is None:
            return None, {}
        positions = []
        conformer_ids = []
        cached_meta = None
        for idx in range(int(self.cfg.num_conformers)):
            cached = self.cache.get(sample_id, conformer_id=str(idx))
            if cached is None:
                continue
            if cached_meta is None:
                cached_meta = cached.get("meta", {})
            if cached["pos"] is None:
                continue
            positions.append(cached["pos"])
            conformer_ids.append(str(idx))
        if not positions:
            return None, {}
        pos = _aggregate_positions(positions, self.cfg.aggregation)
        status = "ok" if len(positions) >= self.cfg.num_conformers else "partial"
        meta = self._base_meta()
        meta.update(
            {
                "status": status,
                "source": cached_meta.get("source", "cache") if cached_meta else "cache",
                "method": cached_meta.get("method", "cache") if cached_meta else "cache",
                "attempts": cached_meta.get("attempts", 0) if cached_meta else 0,
                "num_conformers_used": len(positions),
                "conformer_ids": conformer_ids,
            }
        )
        return pos, meta

    def _generate_pos(
        self,
        sample_id: str,
        mol,
        smiles: Optional[str],
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        _require_rdkit()
        base_meta = self._base_meta()

        if mol is not None and mol.GetNumConformers() > 0:
            conf = mol.GetConformer()
            if conf is not None and conf.Is3D():
                pos = _pos_from_conformer(conf, mol.GetNumAtoms())
                meta = dict(base_meta)
                meta.update({"status": "ok", "source": "sdf", "method": "sdf_3d", "attempts": 0})
                return pos, meta

        work_mol = mol
        if work_mol is None and self.cfg.use_smiles_fallback:
            work_mol = mol_from_smiles(smiles)

        if work_mol is None:
            return None, {**base_meta, "status": "failed", "reason": "missing_mol"}

        attempts = max(1, int(self.cfg.max_attempts))
        for attempt in range(attempts):
            seed = _seed_for_sample(sample_id, int(self.cfg.seed), attempt)
            pos, meta = self._embed_and_optimize(work_mol, seed, attempt)
            if pos is not None:
                meta = {**base_meta, **meta}
                return pos, meta
            if self.cfg.on_failure != "retry":
                break

        if self.cfg.on_failure == "fallback_2d":
            pos = self._fallback_2d(work_mol)
            meta = dict(base_meta)
            status = "fallback_2d" if pos is not None else "failed"
            meta.update({"status": status, "source": "fallback", "method": "2d", "attempts": attempts})
            return pos, meta

        meta = dict(base_meta)
        meta.update({"status": "failed", "source": "embed", "method": self.cfg.method, "attempts": attempts})
        return None, meta

    def _generate_multi_pos(
        self,
        sample_id: str,
        mol,
        smiles: Optional[str],
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any], list[Dict[str, Any]]]:
        _require_rdkit()
        base_meta = self._base_meta()
        conformer_entries: list[Dict[str, Any]] = []

        if mol is not None and mol.GetNumConformers() > 0:
            for conf in mol.GetConformers():
                if conf is None or not conf.Is3D():
                    continue
                pos = _pos_from_conformer(conf, mol.GetNumAtoms())
                conformer_id = str(len(conformer_entries))
                meta = dict(base_meta)
                meta.update(
                    {
                        "status": "ok",
                        "source": "sdf",
                        "method": "sdf_3d",
                        "attempts": 0,
                        "conformer_id": conformer_id,
                    }
                )
                conformer_entries.append({"conformer_id": conformer_id, "pos": pos, "meta": meta})
                if len(conformer_entries) >= self.cfg.num_conformers:
                    break
            if conformer_entries:
                pos = _aggregate_positions([entry["pos"] for entry in conformer_entries], self.cfg.aggregation)
                status = "ok" if len(conformer_entries) >= self.cfg.num_conformers else "partial"
                agg_meta = dict(base_meta)
                agg_meta.update(
                    {
                        "status": status,
                        "source": "sdf",
                        "method": "sdf_3d",
                        "attempts": 0,
                        "num_conformers_used": len(conformer_entries),
                        "conformer_ids": [entry["conformer_id"] for entry in conformer_entries],
                    }
                )
                return pos, agg_meta, conformer_entries

        work_mol = mol
        if work_mol is None and self.cfg.use_smiles_fallback:
            work_mol = mol_from_smiles(smiles)

        if work_mol is None:
            meta = dict(base_meta)
            meta.update({"status": "failed", "reason": "missing_mol", "num_conformers_used": 0})
            return None, meta, conformer_entries

        attempts = max(1, int(self.cfg.max_attempts))
        for attempt in range(attempts):
            conformer_entries = []
            positions = []
            for idx in range(int(self.cfg.num_conformers)):
                seed = _seed_for_conformer(sample_id, int(self.cfg.seed), attempt, idx)
                pos, meta = self._embed_and_optimize(work_mol, seed, attempt)
                conformer_id = str(idx)
                meta = {**base_meta, **meta}
                meta.update({"conformer_id": conformer_id, "conformer_index": int(idx)})
                conformer_entries.append({"conformer_id": conformer_id, "pos": pos, "meta": meta})
                if pos is not None:
                    positions.append(pos)
            if positions:
                pos = _aggregate_positions(positions, self.cfg.aggregation)
                status = "ok" if len(positions) >= self.cfg.num_conformers else "partial"
                agg_meta = dict(base_meta)
                agg_meta.update(
                    {
                        "status": status,
                        "source": "embed",
                        "method": self.cfg.method,
                        "attempts": attempt + 1,
                        "num_conformers_used": len(positions),
                        "conformer_ids": [entry["conformer_id"] for entry in conformer_entries if entry["pos"] is not None],
                    }
                )
                return pos, agg_meta, conformer_entries
            if self.cfg.on_failure != "retry":
                break

        if self.cfg.on_failure == "fallback_2d":
            pos = self._fallback_2d(work_mol)
            status = "fallback_2d" if pos is not None else "failed"
            conformer_entries = []
            if pos is not None:
                conformer_id = "0"
                meta = dict(base_meta)
                meta.update(
                    {
                        "status": status,
                        "source": "fallback",
                        "method": "2d",
                        "attempts": attempts,
                        "conformer_id": conformer_id,
                    }
                )
                conformer_entries.append({"conformer_id": conformer_id, "pos": pos, "meta": meta})
            agg_meta = dict(base_meta)
            agg_meta.update(
                {
                    "status": status,
                    "source": "fallback",
                    "method": "2d",
                    "attempts": attempts,
                    "num_conformers_used": len(conformer_entries),
                    "conformer_ids": [entry["conformer_id"] for entry in conformer_entries],
                }
            )
            return pos, agg_meta, conformer_entries

        meta = dict(base_meta)
        meta.update(
            {
                "status": "failed",
                "source": "embed",
                "method": self.cfg.method,
                "attempts": attempts,
                "num_conformers_used": 0,
            }
        )
        return None, meta, conformer_entries

    def _embed_and_optimize(self, mol, seed: int, attempt: int) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        work = Chem.Mol(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = int(seed)
        try:
            embed_status = AllChem.EmbedMolecule(work, params)
        except Exception as exc:
            return (
                None,
                {
                    "status": "failed",
                    "source": "embed",
                    "method": self.cfg.method,
                    "seed": int(seed),
                    "attempts": attempt + 1,
                    "embed_error": str(exc),
                },
            )
        meta = {
            "status": "failed",
            "source": "embed",
            "method": self.cfg.method,
            "seed": int(seed),
            "attempts": attempt + 1,
        }
        if embed_status != 0:
            meta["embed_status"] = int(embed_status)
            return None, meta

        opt_status = None
        opt_error = None
        if self.cfg.forcefield == "mmff":
            props = AllChem.MMFFGetMoleculeProperties(work, mmffVariant="MMFF94")
            if props is None:
                opt_error = "mmff_props_none"
            else:
                try:
                    opt_status = AllChem.MMFFOptimizeMolecule(work, mmffVariant="MMFF94", maxIters=int(self.cfg.max_iters))
                except Exception as exc:
                    opt_error = str(exc)
        else:
            try:
                opt_status = AllChem.UFFOptimizeMolecule(work, maxIters=int(self.cfg.max_iters))
            except Exception as exc:
                opt_error = str(exc)

        meta.update(
            {
                "status": "ok",
                "embed_status": int(embed_status),
                "opt_status": None if opt_status is None else int(opt_status),
            }
        )
        if opt_error:
            meta["opt_error"] = opt_error
        conf = work.GetConformer()
        if conf is None or not conf.Is3D():
            meta["status"] = "failed"
            return None, meta
        pos = _pos_from_conformer(conf, work.GetNumAtoms())
        return pos, meta

    def _fallback_2d(self, mol) -> Optional[np.ndarray]:
        work = Chem.Mol(mol)
        status = AllChem.Compute2DCoords(work)
        if status != 0:
            return None
        conf = work.GetConformer()
        if conf is None:
            return None
        pos = _pos_from_conformer(conf, work.GetNumAtoms())
        pos[:, 2] = 0.0
        return pos
