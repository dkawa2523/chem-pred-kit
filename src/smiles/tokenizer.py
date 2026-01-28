from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np

try:
    from transformers import AutoTokenizer
except Exception:  # pragma: no cover
    AutoTokenizer = None


class SmilesTokenizerError(ValueError):
    pass


def _require_transformers() -> None:
    if AutoTokenizer is None:
        raise ImportError(
            "transformers/tokenizers are required for SMILES tokenization. "
            "Install optional deps: pip install -e .[transformer]"
        )


@dataclass(frozen=True)
class SmilesTokenizerConfig:
    name: str = "smiles_tokenizer"
    pretrained_name: str = "seyonec/ChemBERTa-zinc-base-v1"
    max_length: int = 256
    padding: str = "max_length"
    truncation: bool = True
    add_special_tokens: bool = True
    use_fast: bool = True
    local_files_only: bool = False
    cache_dir: Optional[str] = None

    def as_payload(self) -> Dict[str, Any]:
        return asdict(self)


def _normalize_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def build_smiles_tokenizer_config(feat_cfg: Dict[str, Any]) -> SmilesTokenizerConfig:
    feat_cfg = feat_cfg or {}
    pretrained_name = feat_cfg.get("pretrained_name") or feat_cfg.get("tokenizer_name") or feat_cfg.get("model_name")
    if not pretrained_name:
        pretrained_name = SmilesTokenizerConfig.pretrained_name
    return SmilesTokenizerConfig(
        name=str(feat_cfg.get("name", SmilesTokenizerConfig.name)),
        pretrained_name=str(pretrained_name),
        max_length=int(feat_cfg.get("max_length", SmilesTokenizerConfig.max_length)),
        padding=str(feat_cfg.get("padding", SmilesTokenizerConfig.padding)),
        truncation=_normalize_bool(feat_cfg.get("truncation", SmilesTokenizerConfig.truncation), True),
        add_special_tokens=_normalize_bool(feat_cfg.get("add_special_tokens", SmilesTokenizerConfig.add_special_tokens), True),
        use_fast=_normalize_bool(feat_cfg.get("use_fast", SmilesTokenizerConfig.use_fast), True),
        local_files_only=_normalize_bool(
            feat_cfg.get("local_files_only", SmilesTokenizerConfig.local_files_only), False
        ),
        cache_dir=str(feat_cfg.get("cache_dir")) if feat_cfg.get("cache_dir") else None,
    )


def _extract_tokenizer_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(cfg, dict):
        return {}
    tokenizer_cfg = cfg.get("tokenizer")
    if isinstance(tokenizer_cfg, dict):
        return tokenizer_cfg
    return cfg.get("featurizer", {}) or {}


def load_tokenizer(cfg: SmilesTokenizerConfig, tokenizer_dir: Optional[Path] = None):
    _require_transformers()
    if tokenizer_dir is not None and Path(tokenizer_dir).exists():
        return AutoTokenizer.from_pretrained(
            str(tokenizer_dir),
            use_fast=cfg.use_fast,
            local_files_only=True,
        )
    return AutoTokenizer.from_pretrained(
        cfg.pretrained_name,
        use_fast=cfg.use_fast,
        cache_dir=cfg.cache_dir,
        local_files_only=cfg.local_files_only,
    )


def encode_smiles_batch(
    smiles_list: Sequence[str],
    tokenizer,
    cfg: SmilesTokenizerConfig,
) -> Dict[str, np.ndarray]:
    if not smiles_list:
        raise SmilesTokenizerError("smiles_list is empty.")
    encode_kwargs: Dict[str, Any] = {
        "padding": cfg.padding,
        "truncation": cfg.truncation,
        "add_special_tokens": cfg.add_special_tokens,
        "return_attention_mask": True,
        "return_tensors": "np",
    }
    if cfg.max_length and cfg.max_length > 0:
        encode_kwargs["max_length"] = int(cfg.max_length)
    encoded = tokenizer(list(smiles_list), **encode_kwargs)
    out: Dict[str, np.ndarray] = {}
    for key, value in encoded.items():
        out[key] = np.asarray(value)
    return out


class SmilesTokenPipeline:
    pipeline_type: str = "smiles"
    version: int = 1

    def __init__(self, tokenizer, cfg: SmilesTokenizerConfig) -> None:
        self.tokenizer = tokenizer
        self.cfg = cfg

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "SmilesTokenPipeline":
        feat_cfg = _extract_tokenizer_cfg(cfg)
        tok_cfg = build_smiles_tokenizer_config(feat_cfg)
        tokenizer = load_tokenizer(tok_cfg)
        return cls(tokenizer=tokenizer, cfg=tok_cfg)

    @classmethod
    def from_artifacts(cls, artifacts_dir: Path, cfg: Dict[str, Any]) -> "SmilesTokenPipeline":
        feat_cfg = _extract_tokenizer_cfg(cfg)
        tok_cfg = build_smiles_tokenizer_config(feat_cfg)
        tokenizer_dir = Path(artifacts_dir) / "tokenizer"
        tokenizer = load_tokenizer(tok_cfg, tokenizer_dir=tokenizer_dir if tokenizer_dir.exists() else None)
        return cls(tokenizer=tokenizer, cfg=tok_cfg)

    def encode(self, smiles_list: Sequence[str]) -> Dict[str, np.ndarray]:
        return encode_smiles_batch(smiles_list, self.tokenizer, self.cfg)

    def encode_smiles(self, smiles: str) -> Dict[str, np.ndarray]:
        batch = self.encode([smiles])
        return {k: v[0] for k, v in batch.items()}

    def save_tokenizer(self, artifacts_dir: Path) -> Iterable[Path]:
        tokenizer_dir = Path(artifacts_dir) / "tokenizer"
        tokenizer_dir.mkdir(parents=True, exist_ok=True)
        saved = self.tokenizer.save_pretrained(str(tokenizer_dir))
        return [Path(p) for p in saved]

    def tokenizer_state(self) -> Dict[str, Any]:
        vocab_size = getattr(self.tokenizer, "vocab_size", None)
        return {"config": self.cfg.as_payload(), "vocab_size": vocab_size}
