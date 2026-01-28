from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


_MODE_ALIASES = {
    "fine_tune": "finetune",
    "fine-tune": "finetune",
    "linear-probe": "linear_probe",
    "linearprobe": "linear_probe",
    "partial-freeze": "partial_freeze",
}


@dataclass
class PretrainMetadata:
    ckpt_path: Path
    ckpt_hash: Optional[str]
    upstream_run_id: Optional[str]
    upstream_artifact_dir: Optional[Path]
    mode: str
    strict: bool

    def meta_fields(self) -> Dict[str, Any]:
        return {
            "pretrain_ckpt_path": str(self.ckpt_path),
            "pretrain_ckpt_hash": self.ckpt_hash,
            "pretrain_run_id": self.upstream_run_id,
            "pretrain_mode": self.mode,
            "pretrain_strict": self.strict,
            "pretrain_upstream_artifact": str(self.upstream_artifact_dir) if self.upstream_artifact_dir else None,
        }


@dataclass
class PretrainLoadReport:
    loaded_keys: List[str]
    skipped_keys: List[str]
    missing_keys: List[str]
    unexpected_keys: List[str]

    def to_meta(self, max_keys: int = 20) -> Dict[str, Any]:
        return {
            "loaded_keys": len(self.loaded_keys),
            "skipped_keys": len(self.skipped_keys),
            "missing_keys": len(self.missing_keys),
            "unexpected_keys": len(self.unexpected_keys),
            "skipped_keys_sample": self.skipped_keys[:max_keys],
        }


@dataclass
class FreezeReport:
    mode: str
    trainable_params: int
    frozen_params: int
    trainable_param_names: List[str]

    def to_meta(self, max_keys: int = 20) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "trainable_params": self.trainable_params,
            "frozen_params": self.frozen_params,
            "trainable_param_names_sample": self.trainable_param_names[:max_keys],
        }


def normalize_mode(mode: Any) -> str:
    value = str(mode or "finetune").strip().lower()
    return _MODE_ALIASES.get(value, value)


def _normalize_patterns(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if str(v).strip()]
    if isinstance(value, str):
        return [value]
    return [str(value)]


def _compile_patterns(patterns: Iterable[str]) -> List[re.Pattern[str]]:
    compiled = []
    for pat in patterns:
        try:
            compiled.append(re.compile(pat))
        except re.error:
            continue
    return compiled


def resolve_checkpoint_path(ckpt_path: Any, base_dir: Optional[Path] = None) -> Path:
    if ckpt_path is None:
        raise ValueError("pretrain.ckpt_path is required when pretrain is enabled.")
    path = Path(str(ckpt_path)).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    if path.is_dir():
        candidate = path / "model" / "model.ckpt"
        if candidate.exists():
            return candidate
        candidate = path / "model.ckpt"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"model.ckpt not found under pretrain dir: {path}")
    if not path.exists():
        raise FileNotFoundError(f"pretrain.ckpt_path not found: {path}")
    return path


def compute_file_hash(path: Path) -> Optional[str]:
    if path is None or not path.exists():
        return None
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def find_upstream_run(ckpt_path: Path) -> Tuple[Optional[str], Optional[Path]]:
    current = ckpt_path.resolve().parent
    while True:
        meta_path = current / "meta.json"
        if meta_path.exists():
            try:
                with meta_path.open("r", encoding="utf-8") as f:
                    meta = json.load(f)
                run_id = meta.get("run_id")
            except Exception:
                run_id = None
            return (run_id, current)
        if current.parent == current:
            break
        current = current.parent
    return (None, None)


def build_pretrain_metadata(pretrain_cfg: Dict[str, Any], base_dir: Optional[Path] = None) -> Optional[PretrainMetadata]:
    if not isinstance(pretrain_cfg, dict):
        return None
    ckpt_raw = pretrain_cfg.get("ckpt_path")
    if ckpt_raw is None or str(ckpt_raw).strip() == "":
        return None
    mode = normalize_mode(pretrain_cfg.get("mode", "finetune"))
    strict = bool(pretrain_cfg.get("strict", False))
    ckpt_path = resolve_checkpoint_path(ckpt_raw, base_dir=base_dir)
    ckpt_hash = compute_file_hash(ckpt_path)
    run_id, run_dir = find_upstream_run(ckpt_path)
    return PretrainMetadata(
        ckpt_path=ckpt_path,
        ckpt_hash=ckpt_hash,
        upstream_run_id=run_id,
        upstream_artifact_dir=run_dir,
        mode=mode,
        strict=strict,
    )


def _extract_state_dict(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, dict):
        if all(isinstance(k, str) for k in payload.keys()):
            if torch and payload and all(torch.is_tensor(v) for v in payload.values()):
                return payload
        for key in ("state_dict", "model_state_dict", "model"):
            if key in payload and isinstance(payload[key], dict):
                return payload[key]
    raise ValueError("Unsupported checkpoint format: expected a state_dict or dict with state_dict/model.")


def _maybe_strip_prefix(state_dict: Dict[str, Any], model_state: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    keys = list(state_dict.keys())
    if not keys:
        return state_dict
    if not all(k.startswith(prefix) for k in keys):
        return state_dict
    stripped = {k[len(prefix):]: v for k, v in state_dict.items()}
    before = sum(1 for k in state_dict.keys() if k in model_state)
    after = sum(1 for k in stripped.keys() if k in model_state)
    if after > before:
        return stripped
    return state_dict


def _filter_state_dict(model, state_dict: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], List[str]]:
    model_state = model.state_dict()
    filtered: Dict[str, Any] = {}
    skipped: List[str] = []
    unexpected: List[str] = []
    for key, value in state_dict.items():
        if key not in model_state:
            unexpected.append(key)
            continue
        if not torch.is_tensor(value):
            skipped.append(key)
            continue
        ref = model_state[key]
        if getattr(value, "shape", None) != getattr(ref, "shape", None):
            skipped.append(key)
            continue
        filtered[key] = value
    return filtered, skipped, unexpected


def load_pretrained_weights(
    model,
    metadata: PretrainMetadata,
    pretrain_cfg: Optional[Dict[str, Any]] = None,
    logger=None,
) -> PretrainLoadReport:
    if torch is None:
        raise ImportError("PyTorch is required for loading pretrained weights.")
    pretrain_cfg = pretrain_cfg or {}
    strict = bool(pretrain_cfg.get("strict", metadata.strict))
    payload = torch.load(metadata.ckpt_path, map_location="cpu")
    state_dict = _extract_state_dict(payload)
    model_state = model.state_dict()
    for prefix in ("module.", "model."):
        state_dict = _maybe_strip_prefix(state_dict, model_state, prefix)
    if strict:
        model.load_state_dict(state_dict, strict=True)
        return PretrainLoadReport(loaded_keys=list(state_dict.keys()), skipped_keys=[], missing_keys=[], unexpected_keys=[])
    filtered, skipped, unexpected = _filter_state_dict(model, state_dict)
    incompatible = model.load_state_dict(filtered, strict=False)
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", [])) + unexpected
    if logger is not None:
        logger.info(
            "Loaded pretrained weights: loaded=%d skipped=%d missing=%d unexpected=%d",
            len(filtered),
            len(skipped),
            len(missing),
            len(unexpected),
        )
        if skipped:
            logger.warning("Skipped incompatible pretrained keys: %s", ", ".join(skipped[:20]))
    return PretrainLoadReport(
        loaded_keys=list(filtered.keys()),
        skipped_keys=skipped,
        missing_keys=missing,
        unexpected_keys=unexpected,
    )


def apply_pretrain_freeze(model, pretrain_cfg: Dict[str, Any], logger=None) -> FreezeReport:
    mode = normalize_mode(pretrain_cfg.get("mode", "finetune"))
    head_patterns = _normalize_patterns(
        pretrain_cfg.get("head_patterns", ["head", "classifier", "predictor", "regressor"])
    )
    freeze_patterns = _normalize_patterns(pretrain_cfg.get("freeze_patterns"))
    trainable_patterns = _normalize_patterns(pretrain_cfg.get("trainable_patterns"))

    if mode == "finetune":
        for _, param in model.named_parameters():
            param.requires_grad = True
    elif mode == "linear_probe":
        for _, param in model.named_parameters():
            param.requires_grad = False
        compiled = _compile_patterns(head_patterns)
        for name, param in model.named_parameters():
            if any(p.search(name) for p in compiled):
                param.requires_grad = True
        if not any(p.requires_grad for p in model.parameters()):
            raise ValueError("linear_probe mode found no trainable head parameters. Check pretrain.head_patterns.")
    elif mode == "partial_freeze":
        if trainable_patterns:
            for _, param in model.named_parameters():
                param.requires_grad = False
            compiled = _compile_patterns(trainable_patterns)
            for name, param in model.named_parameters():
                if any(p.search(name) for p in compiled):
                    param.requires_grad = True
        elif freeze_patterns:
            for _, param in model.named_parameters():
                param.requires_grad = True
            compiled = _compile_patterns(freeze_patterns)
            for name, param in model.named_parameters():
                if any(p.search(name) for p in compiled):
                    param.requires_grad = False
        else:
            raise ValueError("partial_freeze mode requires pretrain.freeze_patterns or pretrain.trainable_patterns.")
    else:
        raise ValueError(f"Unknown pretrain.mode: {mode}")

    trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    if logger is not None:
        logger.info(
            "Pretrain freeze mode=%s trainable_params=%d frozen_params=%d",
            mode,
            trainable_params,
            frozen_params,
        )
    return FreezeReport(
        mode=mode,
        trainable_params=trainable_params,
        frozen_params=frozen_params,
        trainable_param_names=trainable_names,
    )


def collect_trainable_params(model) -> List[Any]:
    return [p for p in model.parameters() if p.requires_grad]
