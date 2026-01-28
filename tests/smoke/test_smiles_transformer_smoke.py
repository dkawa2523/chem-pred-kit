from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.optional


def _build_local_bert(pretrained_dir: Path) -> None:
    from transformers import BertConfig, BertModel, BertTokenizerFast

    pretrained_dir.mkdir(parents=True, exist_ok=True)
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "C", "O", "N", "=", "#", "(", ")"]
    vocab_path = pretrained_dir / "vocab.txt"
    vocab_path.write_text("\n".join(vocab), encoding="utf-8")

    tokenizer = BertTokenizerFast(vocab_file=str(vocab_path), do_lower_case=False)
    tokenizer.save_pretrained(str(pretrained_dir))

    config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=64,
        type_vocab_size=2,
    )
    model = BertModel(config)
    model.save_pretrained(str(pretrained_dir))


@pytest.fixture(scope="session")
def smiles_transformer_smoke_artifacts(tmp_path_factory: pytest.TempPathFactory) -> Dict[str, Any]:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    from src.common.splitters import save_split_indices
    from src.smiles import evaluate as smiles_evaluate
    from src.smiles import train as smiles_train

    root = tmp_path_factory.mktemp("smiles_transformer_smoke")
    pretrained_dir = root / "tiny_bert"
    _build_local_bert(pretrained_dir)

    smiles_list = [
        "C",
        "CC",
        "CCC",
        "CO",
        "CCO",
        "CCN",
        "C=O",
        "C#N",
        "CN",
        "COC",
        "CNC",
        "CCOCC",
        "O",
        "N",
        "CON",
        "CC=O",
        "CNO",
        "COO",
        "CCCN",
        "CCCO",
    ]
    dataset_csv = root / "dataset.csv"
    df = pd.DataFrame(
        {
            "sample_id": [f"id{i}" for i in range(len(smiles_list))],
            "smiles": smiles_list,
            "y": np.linspace(0.1, 2.0, len(smiles_list)),
        }
    )
    df.to_csv(dataset_csv, index=False)

    indices_dir = root / "indices"
    indices = {
        "train": list(range(0, 12)),
        "val": list(range(12, 16)),
        "test": list(range(16, len(smiles_list))),
    }
    save_split_indices(indices, indices_dir)

    train_run_root = root / "runs" / "train" / "smiles"
    train_cfg = {
        "process": {"name": "train", "backend": "smiles"},
        "data": {
            "dataset_csv": str(dataset_csv),
            "indices_dir": str(indices_dir),
            "sample_id_col": "sample_id",
            "smiles_col": "smiles",
        },
        "task": {
            "name": "smiles_smoke",
            "type": "regression",
            "target_columns": ["y"],
            "metrics": "regression",
        },
        "featurizer": {
            "name": "smiles_tokenizer",
            "pretrained_name": str(pretrained_dir),
            "max_length": 64,
            "padding": "max_length",
            "truncation": True,
            "add_special_tokens": True,
            "use_fast": True,
            "local_files_only": True,
        },
        "model": {
            "name": "chemberta",
            "pretrained_name": str(pretrained_dir),
            "use_pretrained": True,
            "local_files_only": True,
            "dropout": 0.1,
        },
        "train": {
            "seed": 42,
            "device": "cpu",
            "epochs": 1,
            "batch_size": 4,
            "lr": 1.0e-4,
            "weight_decay": 0.0,
            "progress_bar": False,
            "log_interval_sec": 0.0,
            "max_batches_per_epoch": 2,
        },
        "output": {"run_dir": str(train_run_root), "exp_name": "smiles_smoke", "plots": False},
    }

    train_run_dir = smiles_train.run(train_cfg)

    eval_run_root = root / "runs" / "evaluate" / "smiles"
    eval_cfg = {
        "process": {"name": "evaluate", "backend": "smiles"},
        "model_artifact_dir": str(train_run_dir),
        "output": {"run_dir": str(eval_run_root), "exp_name": "smiles_smoke"},
    }
    eval_run_dir = smiles_evaluate.run(eval_cfg)

    return {
        "train_run_dir": train_run_dir,
        "eval_run_dir": eval_run_dir,
    }


def test_smiles_transformer_smoke_artifacts(smiles_transformer_smoke_artifacts) -> None:
    from src.utils.artifacts import validate_evaluate_artifacts, validate_train_artifacts

    validate_train_artifacts(smiles_transformer_smoke_artifacts["train_run_dir"])
    validate_evaluate_artifacts(smiles_transformer_smoke_artifacts["eval_run_dir"])
