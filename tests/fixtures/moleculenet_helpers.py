from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

MOLECULENET_TARGET_REGISTRY: Dict[str, List[Dict[str, Any]]] = {
    "esol": [{"name": "esol", "aliases": ["esol", "ESOL"], "units": "log mol/L"}],
    "freesolv": [{"name": "freesolv", "aliases": ["freesolv", "FreeSolv"], "units": "kcal/mol"}],
    "lipo": [{"name": "lipo", "aliases": ["lipo", "Lipophilicity"], "units": "logD"}],
}

MOLECULENET_PYG_NAMES = {
    "esol": "ESOL",
    "freesolv": "FreeSolv",
    "lipo": "Lipophilicity",
}


def build_moleculenet_dataset_cfg(run_dir: Path, root: Path, dataset_key: str, n_rows: int = 256) -> Dict[str, Any]:
    dataset_dir = run_dir / "dataset" / dataset_key
    return {
        "process": {"name": "build_dataset"},
        "paths": {
            "raw_csv": str(run_dir / f"{dataset_key}_dummy.csv"),
            "sdf_dir": str(dataset_dir / "sdf"),
            "out_csv": str(dataset_dir / "dataset.csv"),
            "out_indices_dir": str(dataset_dir / "indices"),
        },
        "columns": {
            "sample_id": "sample_id",
            "cas": "sample_id",
            "smiles": "smiles",
        },
        "split": {"method": "random", "seed": 42, "fractions": [0.8, 0.1, 0.1]},
        "moleculenet": {
            "root": str(root),
            "allow_download": False,
            "add_hs": True,
            "compute_2d": True,
        },
        "seed": 42,
        "dataset": {"name": "moleculenet", "dataset_name": dataset_key},
        "target_registry": MOLECULENET_TARGET_REGISTRY[dataset_key],
        "output": {"run_dir": str(run_dir)},
        "limit_rows": int(n_rows),
    }


def build_moleculenet_train_cfg(
    run_dir: Path,
    dataset_csv: Path,
    indices_dir: Path,
    sdf_dir: Path,
    dataset_key: str,
) -> Dict[str, Any]:
    return {
        "process": {"name": "train", "backend": "fp"},
        "data": {
            "dataset_csv": str(dataset_csv),
            "indices_dir": str(indices_dir),
            "sdf_dir": str(sdf_dir),
            "cas_col": "sample_id",
        },
        "task": {
            "name": dataset_key,
            "type": "regression",
            "target_names": [dataset_key],
            "metrics": "regression",
        },
        "target_registry": MOLECULENET_TARGET_REGISTRY[dataset_key],
        "preprocess": {"impute_nan": "mean", "standardize": False},
        "featurizer": {"fingerprint": "morgan", "morgan_radius": 2, "n_bits": 256},
        "model": {"name": "rf", "params": {"n_estimators": 10, "random_state": 42}},
        "train": {"seed": 42},
        "output": {"run_dir": str(run_dir), "exp_name": f"{dataset_key}_fp_smoke", "plots": False},
    }


def build_moleculenet_eval_cfg(run_dir: Path, model_artifact_dir: Path, dataset_key: str) -> Dict[str, Any]:
    return {
        "process": {"name": "evaluate", "backend": "fp"},
        "model_artifact_dir": str(model_artifact_dir),
        "output": {"run_dir": str(run_dir), "exp_name": f"{dataset_key}_fp_smoke"},
    }


def build_moleculenet_gnn_train_cfg(
    run_dir: Path,
    dataset_csv: Path,
    indices_dir: Path,
    sdf_dir: Path,
    dataset_key: str,
) -> Dict[str, Any]:
    return {
        "process": {"name": "train", "backend": "gnn"},
        "data": {
            "dataset_csv": str(dataset_csv),
            "indices_dir": str(indices_dir),
            "sdf_dir": str(sdf_dir),
            "cas_col": "sample_id",
        },
        "task": {
            "name": dataset_key,
            "type": "regression",
            "target_names": [dataset_key],
            "metrics": "regression",
        },
        "target_registry": MOLECULENET_TARGET_REGISTRY[dataset_key],
        "featureset": {"name": "gnn_graph_quick"},
        "featurizer": {
            "node_features": ["atomic_num", "degree", "formal_charge", "aromatic", "num_h", "in_ring"],
            "edge_features": ["bond_type", "conjugated", "aromatic"],
            "use_3d_pos": False,
            "edge_mode": "bond",
            "conformer": {"enabled": False},
        },
        "model": {
            "name": "gcn",
            "family": "gnn",
            "hidden_dim": 32,
            "num_layers": 2,
            "dropout": 0.1,
        },
        "train": {
            "seed": 42,
            "device": "cpu",
            "epochs": 2,
            "batch_size": 8,
            "lr": 1e-3,
            "weight_decay": 1e-6,
            "progress_bar": False,
            "max_batches_per_epoch": 2,
        },
        "output": {"run_dir": str(run_dir), "exp_name": f"{dataset_key}_gcn_smoke", "plots": False},
    }


def build_moleculenet_gnn_eval_cfg(run_dir: Path, model_artifact_dir: Path, dataset_key: str) -> Dict[str, Any]:
    return {
        "process": {"name": "evaluate", "backend": "gnn"},
        "model_artifact_dir": str(model_artifact_dir),
        "eval": {"batch_size": 16, "device": "cpu"},
        "output": {"run_dir": str(run_dir), "exp_name": f"{dataset_key}_gcn_smoke"},
    }
