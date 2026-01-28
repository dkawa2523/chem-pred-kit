from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

from src.common.feature_pipeline import (
    FEATURE_PIPELINE_FILENAME,
    EmbeddingFeaturePipeline,
    FingerprintFeaturePipeline,
    GraphFeaturePipeline,
    load_feature_pipeline,
    resolve_tabular_pipeline,
    save_feature_pipeline,
)
from src.common.conformer import conformer_config_from_cfg, conformer_config_payload

FEATURESET_VERSION = 1
FEATURESET_MANIFEST_FILENAME = "featureset.json"
FEATURESET_HASH_FILENAME = "featureset_hash.txt"


def _iter_file_bytes(path: Path, chunk_size: int = 1024 * 1024) -> Iterable[bytes]:
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk


def _manifest_payload(manifest: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(manifest)
    payload.pop("featureset_hash", None)
    return payload


def compute_featureset_hash(manifest: Dict[str, Any], artifact_paths: Sequence[Path]) -> str:
    payload = _manifest_payload(manifest)
    digest = hashlib.sha256()
    digest.update(json.dumps(payload, sort_keys=True, default=str, ensure_ascii=True).encode("utf-8"))
    for path in sorted(artifact_paths, key=lambda p: p.name):
        if not path.exists():
            continue
        digest.update(path.name.encode("utf-8"))
        for chunk in _iter_file_bytes(path):
            digest.update(chunk)
    return digest.hexdigest()


def save_featureset_manifest(artifacts_dir: Path, manifest: Dict[str, Any]) -> Path:
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    path = artifacts_dir / FEATURESET_MANIFEST_FILENAME
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=True, sort_keys=True, indent=2, default=str)
    return path


def load_featureset_manifest(artifacts_dir: Path) -> Optional[Dict[str, Any]]:
    path = Path(artifacts_dir) / FEATURESET_MANIFEST_FILENAME
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_featureset_hash(artifacts_dir: Path, featureset_hash: str) -> Path:
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    path = artifacts_dir / FEATURESET_HASH_FILENAME
    path.write_text(featureset_hash, encoding="utf-8")
    return path


def load_featureset_hash(artifacts_dir: Path) -> Optional[str]:
    path = Path(artifacts_dir) / FEATURESET_HASH_FILENAME
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8").strip() or None


class FeatureSetBase:
    def __init__(self, name: str, kind: str, config: Dict[str, Any], version: int = FEATURESET_VERSION) -> None:
        self.name = name
        self.kind = kind
        self.version = int(version)
        self.config = config
        self.feature_meta: Optional[Dict[str, Any]] = None

    def manifest(self) -> Dict[str, Any]:
        manifest: Dict[str, Any] = {
            "name": self.name,
            "kind": self.kind,
            "version": self.version,
            "config": self.config,
        }
        if self.feature_meta:
            manifest["feature_meta"] = self.feature_meta
        return manifest

    def artifact_paths(self, artifacts_dir: Path) -> Sequence[Path]:
        return []

    def _save_manifest_and_hash(self, artifacts_dir: Path, artifact_paths: Sequence[Path]) -> str:
        manifest = self.manifest()
        featureset_hash = compute_featureset_hash(manifest, artifact_paths)
        manifest["featureset_hash"] = featureset_hash
        save_featureset_manifest(artifacts_dir, manifest)
        save_featureset_hash(artifacts_dir, featureset_hash)
        return featureset_hash


def _tabular_config_from_pipeline(
    pipeline: FingerprintFeaturePipeline | EmbeddingFeaturePipeline,
) -> Dict[str, Any]:
    if isinstance(pipeline, FingerprintFeaturePipeline):
        featurizer = asdict(pipeline.fp_cfg)
    else:
        featurizer = asdict(pipeline.emb_cfg)
    preprocess = {
        "impute_nan": pipeline.impute_strategy,
        "standardize": bool(pipeline.standardize),
    }
    return {"featurizer": featurizer, "preprocess": preprocess}


class TabularFeatureSet(FeatureSetBase):
    def __init__(
        self,
        name: str,
        pipeline: FingerprintFeaturePipeline | EmbeddingFeaturePipeline,
        config: Dict[str, Any],
    ) -> None:
        kind = getattr(pipeline, "pipeline_type", "fp")
        super().__init__(name=name, kind=kind, config=config)
        self.pipeline = pipeline

    @classmethod
    def from_config(cls, name: str, cfg: Dict[str, Any]) -> "TabularFeatureSet":
        pipeline = resolve_tabular_pipeline(cfg)
        config = _tabular_config_from_pipeline(pipeline)
        return cls(name=name, pipeline=pipeline, config=config)

    @classmethod
    def load(cls, name: str, artifacts_dir: Path, cfg: Dict[str, Any]) -> "TabularFeatureSet":
        pipeline = load_feature_pipeline(artifacts_dir)
        if isinstance(pipeline, (FingerprintFeaturePipeline, EmbeddingFeaturePipeline)) and pipeline.imputer is not None:
            config = _tabular_config_from_pipeline(pipeline)
            return cls(name=name, pipeline=pipeline, config=config)
        pipeline = resolve_tabular_pipeline(cfg)
        pipeline.load_preprocess_artifacts(artifacts_dir)
        config = _tabular_config_from_pipeline(pipeline)
        return cls(name=name, pipeline=pipeline, config=config)

    def build_features(
        self,
        df,
        sdf_dir: Path,
        cas_col: str,
        cache_dir: Optional[Path],
        cache_key: str,
        logger,
    ) -> Tuple[Any, list[Any], list[str], Dict[str, Any]]:
        X, ids, elements, meta = self.pipeline.build_features(
            df=df,
            sdf_dir=sdf_dir,
            cas_col=cas_col,
            cache_dir=cache_dir,
            cache_key=cache_key,
            logger=logger,
        )
        self.feature_meta = meta
        return X, ids, elements, meta

    def fit(self, X) -> None:
        self.pipeline.fit(X)

    def transform(self, X):
        return self.pipeline.transform_features(X)

    def transform_mol(self, mol):
        return self.pipeline.transform_mol(mol)

    def save(self, artifacts_dir: Path) -> str:
        artifacts_dir = Path(artifacts_dir)
        save_feature_pipeline(self.pipeline, artifacts_dir)
        self.pipeline.save_preprocess_artifacts(artifacts_dir)
        return self._save_manifest_and_hash(artifacts_dir, self.artifact_paths(artifacts_dir))

    def artifact_paths(self, artifacts_dir: Path) -> Sequence[Path]:
        return [
            Path(artifacts_dir) / FEATURE_PIPELINE_FILENAME,
            Path(artifacts_dir) / "imputer.pkl",
            Path(artifacts_dir) / "scaler.pkl",
        ]


class SmilesFeatureSet(FeatureSetBase):
    def __init__(self, name: str, pipeline, config: Dict[str, Any]) -> None:
        kind = getattr(pipeline, "pipeline_type", "smiles")
        super().__init__(name=name, kind=kind, config=config)
        self.pipeline = pipeline

    @classmethod
    def from_config(cls, name: str, cfg: Dict[str, Any]) -> "SmilesFeatureSet":
        from src.smiles.tokenizer import SmilesTokenPipeline

        pipeline = SmilesTokenPipeline.from_config(cfg)
        config = {"tokenizer": pipeline.cfg.as_payload()}
        return cls(name=name, pipeline=pipeline, config=config)

    @classmethod
    def load(cls, name: str, artifacts_dir: Path, cfg: Dict[str, Any]) -> "SmilesFeatureSet":
        from src.smiles.tokenizer import SmilesTokenPipeline

        pipeline = SmilesTokenPipeline.from_artifacts(artifacts_dir, cfg)
        config = {"tokenizer": pipeline.cfg.as_payload()}
        return cls(name=name, pipeline=pipeline, config=config)

    def fit(self, X=None) -> None:
        return None

    def tokenize_smiles(self, smiles: str) -> Dict[str, Any]:
        return self.pipeline.encode_smiles(smiles)

    def tokenize_batch(self, smiles_list: Sequence[str]) -> Dict[str, Any]:
        return self.pipeline.encode(smiles_list)

    def save(self, artifacts_dir: Path) -> str:
        artifacts_dir = Path(artifacts_dir)
        saved_paths = list(self.pipeline.save_tokenizer(artifacts_dir))
        tokenizer_files = [Path(path).name for path in saved_paths if path]
        feature_meta = dict(self.feature_meta or {})
        if tokenizer_files:
            feature_meta["tokenizer_files"] = tokenizer_files
        self.feature_meta = feature_meta or None
        artifact_paths = saved_paths if saved_paths else self.artifact_paths(artifacts_dir)
        return self._save_manifest_and_hash(artifacts_dir, artifact_paths)

    def artifact_paths(self, artifacts_dir: Path) -> Sequence[Path]:
        tokenizer_dir = Path(artifacts_dir) / "tokenizer"
        if not tokenizer_dir.exists():
            return []
        if isinstance(self.feature_meta, dict):
            names = self.feature_meta.get("tokenizer_files")
            if isinstance(names, list) and names:
                paths = [tokenizer_dir / str(name) for name in names]
                return [path for path in paths if path.exists()]
        return sorted([p for p in tokenizer_dir.iterdir() if p.is_file()], key=lambda p: p.name)


class GraphFeatureSet(FeatureSetBase):
    def __init__(self, name: str, pipeline: GraphFeaturePipeline, config: Dict[str, Any]) -> None:
        kind = getattr(pipeline, "pipeline_type", "gnn")
        super().__init__(name=name, kind=kind, config=config)
        self.pipeline = pipeline

    @classmethod
    def from_config(cls, name: str, cfg: Dict[str, Any]) -> "GraphFeatureSet":
        pipeline = GraphFeaturePipeline.from_config(cfg)
        conformer_cfg = conformer_config_from_cfg(cfg)
        config = {"featurizer": asdict(pipeline.graph_cfg), "conformer": conformer_config_payload(conformer_cfg)}
        return cls(name=name, pipeline=pipeline, config=config)

    @classmethod
    def load(cls, name: str, artifacts_dir: Path, cfg: Dict[str, Any]) -> "GraphFeatureSet":
        pipeline = load_feature_pipeline(artifacts_dir)
        if not isinstance(pipeline, GraphFeaturePipeline):
            pipeline = GraphFeaturePipeline.from_artifacts(artifacts_dir)
        conformer_cfg = conformer_config_from_cfg(cfg)
        config = {"featurizer": asdict(pipeline.graph_cfg), "conformer": conformer_config_payload(conformer_cfg)}
        return cls(name=name, pipeline=pipeline, config=config)

    def fit(self, X=None) -> None:
        return None

    def transform(self, mol, y: Optional[Sequence[float]] = None, pos=None, mask_y: Optional[Sequence[float]] = None):
        return self.pipeline.featurize_mol(mol, y=y, pos=pos, mask_y=mask_y)

    def save(self, artifacts_dir: Path) -> str:
        artifacts_dir = Path(artifacts_dir)
        save_feature_pipeline(self.pipeline, artifacts_dir)
        graph_cfg_path = artifacts_dir / "graph_featurizer.pkl"
        if not graph_cfg_path.exists():
            with graph_cfg_path.open("wb") as f:
                import pickle

                pickle.dump(self.pipeline.graph_cfg, f)
        return self._save_manifest_and_hash(artifacts_dir, self.artifact_paths(artifacts_dir))

    def artifact_paths(self, artifacts_dir: Path) -> Sequence[Path]:
        paths = [
            Path(artifacts_dir) / FEATURE_PIPELINE_FILENAME,
            Path(artifacts_dir) / "graph_featurizer.pkl",
        ]
        conformer_cache = Path(artifacts_dir) / "conformer_cache.json"
        if conformer_cache.exists():
            paths.append(conformer_cache)
        return paths
