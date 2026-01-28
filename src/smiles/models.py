from __future__ import annotations

from typing import Any, Dict, Optional

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None
    nn = None

try:
    from transformers import AutoConfig, AutoModel
except Exception:  # pragma: no cover
    AutoConfig = None
    AutoModel = None


class SmilesModelError(ValueError):
    pass


def _require_smiles_deps() -> None:
    if torch is None or nn is None:
        raise ImportError("PyTorch is required for SMILES transformer models.")
    if AutoModel is None or AutoConfig is None:
        raise ImportError(
            "transformers is required for SMILES transformer models. "
            "Install optional deps: pip install -e .[transformer]"
        )


def _resolve_pretrained_name(model_cfg: Dict[str, Any]) -> str:
    pretrained = model_cfg.get("pretrained_name") or model_cfg.get("model_name") or model_cfg.get("tokenizer_name")
    if not pretrained:
        pretrained = "seyonec/ChemBERTa-zinc-base-v1"
    return str(pretrained)


def _maybe_enable_gradient_checkpointing(encoder, flag: bool) -> None:
    if not flag:
        return
    if hasattr(encoder, "gradient_checkpointing_enable"):
        encoder.gradient_checkpointing_enable()


_BASE_SMILES_MODULE = nn.Module if nn is not None else object


class SmilesTransformerRegressor(_BASE_SMILES_MODULE):
    def __init__(
        self,
        pretrained_name: str,
        out_dim: int,
        pooling: str = "cls",
        dropout: float = 0.1,
        head_hidden_dim: int = 0,
        use_pretrained: bool = True,
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        _require_smiles_deps()
        if use_pretrained:
            self.encoder = AutoModel.from_pretrained(
                pretrained_name,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
        else:
            config = AutoConfig.from_pretrained(
                pretrained_name,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
            self.encoder = AutoModel.from_config(config)
        _maybe_enable_gradient_checkpointing(self.encoder, gradient_checkpointing)

        hidden_size = int(getattr(self.encoder.config, "hidden_size", 0))
        if hidden_size <= 0:
            raise SmilesModelError("encoder hidden_size is missing/invalid.")

        self.pooling = str(pooling).lower()
        self.dropout = nn.Dropout(float(dropout))
        if head_hidden_dim and int(head_hidden_dim) > 0:
            head_dim = int(head_hidden_dim)
            self.head = nn.Sequential(
                nn.Linear(hidden_size, head_dim),
                nn.ReLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(head_dim, int(out_dim)),
            )
        else:
            self.head = nn.Linear(hidden_size, int(out_dim))

    def _pool(self, outputs, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if self.pooling == "pooler" and getattr(outputs, "pooler_output", None) is not None:
            return outputs.pooler_output
        last_hidden = outputs.last_hidden_state
        if self.pooling == "mean":
            if attention_mask is None:
                return last_hidden.mean(dim=1)
            mask = attention_mask.unsqueeze(-1).float()
            denom = mask.sum(dim=1).clamp(min=1.0)
            return (last_hidden * mask).sum(dim=1) / denom
        return last_hidden[:, 0, :]

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled = self._pool(outputs, attention_mask=attention_mask)
        return self.dropout(pooled)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pooled = self.encode(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return self.head(pooled)


def build_smiles_model(model_cfg: Dict[str, Any], context: Dict[str, Any]) -> SmilesTransformerRegressor:
    model_cfg = model_cfg or {}
    out_dim = int(context.get("out_dim", 1))
    if out_dim <= 0:
        raise SmilesModelError("context.out_dim must be positive for SMILES transformer models.")
    pretrained_name = _resolve_pretrained_name(model_cfg)
    pooling = model_cfg.get("pooling", "cls")
    dropout = float(model_cfg.get("dropout", 0.1))
    head_hidden_dim = int(model_cfg.get("head_hidden_dim", 0))
    use_pretrained = bool(model_cfg.get("use_pretrained", True))
    cache_dir = model_cfg.get("cache_dir", None)
    local_files_only = bool(model_cfg.get("local_files_only", False))
    gradient_checkpointing = bool(model_cfg.get("gradient_checkpointing", False))
    return SmilesTransformerRegressor(
        pretrained_name=pretrained_name,
        out_dim=out_dim,
        pooling=pooling,
        dropout=dropout,
        head_hidden_dim=head_hidden_dim,
        use_pretrained=use_pretrained,
        cache_dir=str(cache_dir) if cache_dir else None,
        local_files_only=local_files_only,
        gradient_checkpointing=gradient_checkpointing,
    )
