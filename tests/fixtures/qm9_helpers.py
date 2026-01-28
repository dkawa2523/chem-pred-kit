from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

QM9_TARGET_REGISTRY: List[Dict[str, Any]] = [
    {"name": "gap", "units": "eV"},
]


def build_qm9_dataset_cfg(run_dir: Path, qm9_root: Path, n_rows: int = 256) -> Dict[str, Any]:
    dataset_dir = run_dir / "dataset"
    return {
        "process": {"name": "build_dataset"},
        "paths": {
            "raw_csv": str(run_dir / "qm9_dummy.csv"),
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
        "qm9": {
            "root": str(qm9_root),
            "allow_download": False,
            "add_hs": True,
            "compute_2d": True,
        },
        "seed": 42,
        "dataset": {"name": "qm9"},
        "target_registry": QM9_TARGET_REGISTRY,
        "output": {"run_dir": str(run_dir)},
        "limit_rows": int(n_rows),
    }


def build_qm9_train_cfg(
    run_dir: Path,
    dataset_csv: Path,
    indices_dir: Path,
    sdf_dir: Path,
    cache_dir: Path,
) -> Dict[str, Any]:
    return {
        "process": {"name": "train", "backend": "fp"},
        "data": {
            "dataset_csv": str(dataset_csv),
            "indices_dir": str(indices_dir),
            "sdf_dir": str(sdf_dir),
            "cas_col": "sample_id",
            "cache_dir": str(cache_dir),
        },
        "task": {
            "name": "gap",
            "type": "regression",
            "target_names": ["gap"],
            "metrics": "regression",
        },
        "target_registry": QM9_TARGET_REGISTRY,
        "preprocess": {"impute_nan": "mean", "standardize": False},
        "featurizer": {"fingerprint": "morgan", "morgan_radius": 2, "n_bits": 256},
        "model": {"name": "rf", "params": {"n_estimators": 10, "random_state": 42}},
        "train": {"seed": 42},
        "output": {"run_dir": str(run_dir), "exp_name": "qm9_smoke", "plots": False},
    }


def build_qm9_eval_cfg(run_dir: Path, model_artifact_dir: Path) -> Dict[str, Any]:
    return {
        "process": {"name": "evaluate", "backend": "fp"},
        "model_artifact_dir": str(model_artifact_dir),
        "output": {"run_dir": str(run_dir), "exp_name": "qm9_smoke"},
    }


def build_qm9_gnn_train_cfg(
    run_dir: Path,
    dataset_csv: Path,
    indices_dir: Path,
    sdf_dir: Path,
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
            "name": "gap",
            "type": "regression",
            "target_names": ["gap"],
            "metrics": "regression",
        },
        "target_registry": QM9_TARGET_REGISTRY,
        "featureset": {"name": "gnn_radius_graph"},
        "featurizer": {
            "node_features": ["atomic_num"],
            "edge_features": ["distance", "gaussian"],
            "edge_mode": "radius",
            "radius_cutoff": 4.0,
            "max_neighbors": 16,
            "gaussian_expansion": {"enabled": True, "num_centers": 8, "min": 0.0, "max": 4.0},
            "use_3d_pos": True,
            "conformer": {
                "enabled": True,
                "seed": 42,
                "method": "etkdg",
                "forcefield": "uff",
                "max_iters": 50,
                "max_attempts": 1,
                "on_failure": "retry",
                "cache": True,
                "use_smiles_fallback": True,
            },
        },
        "model": {
            "name": "schnet",
            "family": "gnn",
            "hidden_dim": 32,
            "num_filters": 32,
            "num_interactions": 2,
            "num_gaussians": 8,
            "cutoff": 4.0,
            "readout": "add",
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
        "output": {"run_dir": str(run_dir), "exp_name": "qm9_schnet_smoke", "plots": False},
    }


def build_qm9_gnn_eval_cfg(run_dir: Path, model_artifact_dir: Path) -> Dict[str, Any]:
    return {
        "process": {"name": "evaluate", "backend": "gnn"},
        "model_artifact_dir": str(model_artifact_dir),
        "eval": {"batch_size": 16, "device": "cpu"},
        "output": {"run_dir": str(run_dir), "exp_name": "qm9_schnet_smoke"},
    }


def build_qm9_gnn_dimenetpp_train_cfg(
    run_dir: Path,
    dataset_csv: Path,
    indices_dir: Path,
    sdf_dir: Path,
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
            "name": "gap",
            "type": "regression",
            "target_names": ["gap"],
            "metrics": "regression",
        },
        "target_registry": QM9_TARGET_REGISTRY,
        "featureset": {"name": "gnn_radius_graph"},
        "featurizer": {
            "node_features": ["atomic_num"],
            "edge_features": [],
            "edge_mode": "radius",
            "radius_cutoff": 4.0,
            "max_neighbors": 16,
            "angle_features": ["angle"],
            "use_3d_pos": True,
            "conformer": {
                "enabled": True,
                "seed": 42,
                "method": "etkdg",
                "forcefield": "uff",
                "max_iters": 50,
                "max_attempts": 1,
                "on_failure": "retry",
                "cache": True,
                "use_smiles_fallback": True,
            },
        },
        "model": {
            "name": "dimenetpp",
            "family": "gnn",
            "hidden_dim": 32,
            "num_blocks": 2,
            "num_bilinear": 4,
            "num_spherical": 3,
            "num_radial": 3,
            "cutoff": 4.0,
            "max_num_neighbors": 16,
            "envelope_exponent": 5,
            "num_before_skip": 1,
            "num_after_skip": 2,
            "num_output_layers": 2,
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
        "output": {"run_dir": str(run_dir), "exp_name": "qm9_dimenetpp_smoke", "plots": False},
    }


def build_qm9_gnn_dimenetpp_eval_cfg(run_dir: Path, model_artifact_dir: Path) -> Dict[str, Any]:
    return {
        "process": {"name": "evaluate", "backend": "gnn"},
        "model_artifact_dir": str(model_artifact_dir),
        "eval": {"batch_size": 16, "device": "cpu"},
        "output": {"run_dir": str(run_dir), "exp_name": "qm9_dimenetpp_smoke"},
    }


def build_qm9_gnn_painn_train_cfg(
    run_dir: Path,
    dataset_csv: Path,
    indices_dir: Path,
    sdf_dir: Path,
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
            "name": "gap",
            "type": "regression",
            "target_names": ["gap"],
            "metrics": "regression",
        },
        "target_registry": QM9_TARGET_REGISTRY,
        "featureset": {"name": "gnn_radius_graph"},
        "featurizer": {
            "node_features": ["atomic_num"],
            "edge_features": ["distance", "gaussian"],
            "edge_mode": "radius",
            "radius_cutoff": 4.0,
            "max_neighbors": 16,
            "gaussian_expansion": {"enabled": True, "num_centers": 8, "min": 0.0, "max": 4.0},
            "use_3d_pos": True,
            "conformer": {
                "enabled": True,
                "seed": 42,
                "method": "etkdg",
                "forcefield": "uff",
                "max_iters": 50,
                "max_attempts": 1,
                "on_failure": "retry",
                "cache": True,
                "use_smiles_fallback": True,
            },
        },
        "model": {
            "name": "painn",
            "family": "gnn",
            "hidden_dim": 32,
            "num_filters": 32,
            "num_interactions": 2,
            "num_rbf": 8,
            "cutoff": 4.0,
            "max_num_neighbors": 16,
            "num_output_layers": 2,
            "readout": "add",
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
        "output": {"run_dir": str(run_dir), "exp_name": "qm9_painn_smoke", "plots": False},
    }


def build_qm9_gnn_painn_eval_cfg(run_dir: Path, model_artifact_dir: Path) -> Dict[str, Any]:
    return {
        "process": {"name": "evaluate", "backend": "gnn"},
        "model_artifact_dir": str(model_artifact_dir),
        "eval": {"batch_size": 16, "device": "cpu"},
        "output": {"run_dir": str(run_dir), "exp_name": "qm9_painn_smoke"},
    }


def build_qm9_gnn_egnn_train_cfg(
    run_dir: Path,
    dataset_csv: Path,
    indices_dir: Path,
    sdf_dir: Path,
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
            "name": "gap",
            "type": "regression",
            "target_names": ["gap"],
            "metrics": "regression",
        },
        "target_registry": QM9_TARGET_REGISTRY,
        "featureset": {"name": "gnn_radius_graph"},
        "featurizer": {
            "node_features": ["atomic_num"],
            "edge_features": ["distance", "gaussian"],
            "edge_mode": "radius",
            "radius_cutoff": 4.0,
            "max_neighbors": 16,
            "gaussian_expansion": {"enabled": True, "num_centers": 8, "min": 0.0, "max": 4.0},
            "use_3d_pos": True,
            "conformer": {
                "enabled": True,
                "seed": 42,
                "method": "etkdg",
                "forcefield": "uff",
                "max_iters": 50,
                "max_attempts": 1,
                "on_failure": "retry",
                "cache": True,
                "use_smiles_fallback": True,
            },
        },
        "model": {
            "name": "egnn",
            "family": "gnn",
            "hidden_dim": 32,
            "num_layers": 2,
            "dropout": 0.1,
            "use_edge_attr": True,
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
        "output": {"run_dir": str(run_dir), "exp_name": "qm9_egnn_smoke", "plots": False},
    }


def build_qm9_gnn_egnn_eval_cfg(run_dir: Path, model_artifact_dir: Path) -> Dict[str, Any]:
    return {
        "process": {"name": "evaluate", "backend": "gnn"},
        "model_artifact_dir": str(model_artifact_dir),
        "eval": {"batch_size": 16, "device": "cpu"},
        "output": {"run_dir": str(run_dir), "exp_name": "qm9_egnn_smoke"},
    }


def run_qm9_featurize(
    train_cfg: Dict[str, Any],
    dataset_csv: Path,
    sdf_dir: Path,
    cache_dir: Path,
    run_dir: Path,
    dataset_hash: str | None,
    upstream_artifact: Path,
) -> str:
    from src.common.config import dump_yaml
    from src.common.feature_pipeline import resolve_tabular_pipeline
    from src.common.io import read_csv
    from src.common.meta import build_meta, save_meta
    from src.common.utils import get_logger, save_json
    from src.fp.feature_utils import hash_cfg
    from src.utils.artifacts import resolve_featureset_name

    run_dir.mkdir(parents=True, exist_ok=True)
    features_dir = run_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = train_cfg.get("data", {}) or {}
    cas_col = str(data_cfg.get("cas_col", "sample_id"))

    featurize_cfg = {
        "process": {"name": "featurize", "backend": "fp"},
        "data": {
            "dataset_csv": str(dataset_csv),
            "indices_dir": str(data_cfg.get("indices_dir", "")),
            "sdf_dir": str(sdf_dir),
            "cas_col": cas_col,
            "cache_dir": str(cache_dir),
        },
        "featurizer": train_cfg.get("featurizer", {}),
        "preprocess": train_cfg.get("preprocess", {}),
        "output": {"run_dir": str(run_dir)},
    }

    dump_yaml(run_dir / "config.yaml", featurize_cfg)
    meta = build_meta(
        process_name="featurize",
        cfg=featurize_cfg,
        dataset_hash=dataset_hash,
        upstream_artifacts=[str(upstream_artifact)],
    )
    save_meta(run_dir, meta)

    logger = get_logger("qm9_featurize", log_file=run_dir / "featurize.log")
    df = read_csv(dataset_csv)
    pipeline = resolve_tabular_pipeline(train_cfg)
    cache_key = hash_cfg({"featurizer": train_cfg.get("featurizer", {}), "dataset": str(dataset_csv)})
    _, _, _, meta_info = pipeline.build_features(
        df=df,
        sdf_dir=sdf_dir,
        cas_col=cas_col,
        cache_dir=cache_dir,
        cache_key=cache_key,
        logger=logger,
    )

    manifest = {
        "featureset_name": resolve_featureset_name(train_cfg),
        "pipeline_type": getattr(pipeline, "pipeline_type", "fp"),
        "feature_dim": int(meta_info.get("feature_dim")) if meta_info.get("feature_dim") is not None else None,
        "feature_meta": meta_info,
        "cache_key": cache_key,
        "cache_files": [
            f"fp_features_{cache_key}.npz",
            f"fp_features_{cache_key}_meta.pkl",
        ],
    }
    save_json(features_dir / "features_manifest.json", manifest)
    return cache_key
