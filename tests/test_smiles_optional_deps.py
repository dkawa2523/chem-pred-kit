from __future__ import annotations

import pytest

from src.smiles import models as smiles_models
from src.smiles import tokenizer as smiles_tokenizer


def test_smiles_tokenizer_requires_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(smiles_tokenizer, "AutoTokenizer", None)
    with pytest.raises(ImportError, match=r"transformers.+pip install -e \.\[transformer\]"):
        smiles_tokenizer._require_transformers()


def test_smiles_models_require_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(smiles_models, "torch", None)
    monkeypatch.setattr(smiles_models, "nn", None)
    with pytest.raises(ImportError, match="PyTorch"):
        smiles_models._require_smiles_deps()


def test_smiles_models_require_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(smiles_models, "torch", object())
    monkeypatch.setattr(smiles_models, "nn", object())
    monkeypatch.setattr(smiles_models, "AutoConfig", None)
    monkeypatch.setattr(smiles_models, "AutoModel", None)
    with pytest.raises(ImportError, match=r"transformers.+pip install -e \.\[transformer\]"):
        smiles_models._require_smiles_deps()
