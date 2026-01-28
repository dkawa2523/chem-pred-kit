from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

try:
    from rdkit import Chem
except Exception:  # pragma: no cover
    Chem = None


class EmbeddingFeaturizerError(ValueError):
    pass


@dataclass
class PretrainedEmbeddingConfig:
    name: str = "pretrained_embedding"
    backend: str = "stub"
    embedding_dim: int = 256
    seed: int = 0
    normalize: bool = True


def _hash_seed(smiles: str, seed: int) -> int:
    payload = f"{seed}:{smiles}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _stub_embed_smiles(smiles: str, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(_hash_seed(smiles, seed))
    vec = rng.normal(0.0, 1.0, size=dim)
    return vec.astype(float)


def featurize_mol(mol, cfg: PretrainedEmbeddingConfig) -> Tuple[np.ndarray, Dict[str, Any]]:
    if Chem is None:
        raise ImportError("RDKit is required for embedding featurization.")
    if mol is None:
        raise EmbeddingFeaturizerError("mol is None")

    backend = str(cfg.backend).lower()
    if backend != "stub":
        raise EmbeddingFeaturizerError(f"Unknown embedding backend: {cfg.backend}")

    smiles = Chem.MolToSmiles(mol, canonical=True)
    vec = _stub_embed_smiles(smiles, int(cfg.embedding_dim), int(cfg.seed))
    if cfg.normalize:
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm

    meta: Dict[str, Any] = {
        "embedding_backend": backend,
        "embedding_dim": int(cfg.embedding_dim),
        "embedding_stub": True,
    }
    return vec.astype(float), meta
