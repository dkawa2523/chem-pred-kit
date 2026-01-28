from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.common.config import dump_yaml, load_config
from src.common.leaderboard import run as run_leaderboard
from src.common.meta import build_meta, save_meta
from src.common.utils import ensure_dir, get_logger, save_json
from src.fp import train as fp_train
from src.gnn import train as gnn_train
from src.fp import evaluate as fp_evaluate
from src.gnn import evaluate as gnn_evaluate
from src.utils.validate_config import validate_config

DEFAULT_PREPROCESS = {"fp": "fp_default", "gnn": "none"}
DEFAULT_TRAIN = {"fp": "fp_default", "gnn": "gnn_default"}
DEFAULT_PRETRAIN = "default"
DEFAULT_EVAL = "default"
DEFAULT_NAME_TEMPLATE = "{dataset}_{target}_{entry}"
DEFAULT_VISUALIZE = {
    "splits": ["val", "test"],
    "include_train_hist": True,
    "target_mode": "primary",
}


@dataclass
class BenchmarkCase:
    case_name: str
    dataset: str
    target: str
    entry_name: str
    model_group: str
    model_name: str
    features_group: str
    featureset_name: Optional[str]
    backend: str
    preprocess_group: str
    train_group: str
    pretrain_group: str
    eval_group: str
    tags: List[str]


def run(cfg: Dict[str, Any]) -> Path:
    validate_config(cfg)
    bench_cfg = cfg.get("benchmark_suite", {}) or {}
    output_cfg = cfg.get("output", {}) or {}
    experiment_cfg = cfg.get("experiment", {}) or {}

    suite_name = str(bench_cfg.get("name") or output_cfg.get("exp_name") or experiment_cfg.get("name") or "benchmark")
    exp_name = str(output_cfg.get("exp_name", suite_name))
    run_dir_root = Path(output_cfg.get("run_dir", "runs/benchmark_suite"))
    suite_run_dir = ensure_dir(run_dir_root / exp_name)
    configs_dir = ensure_dir(suite_run_dir / "configs")
    logger = get_logger("benchmark_suite", log_file=suite_run_dir / "benchmark_suite.log")

    dump_yaml(suite_run_dir / "config.yaml", cfg)

    cases = expand_matrix(bench_cfg, suite_name=suite_name)
    if not cases:
        raise ValueError("benchmark_suite.matrix did not produce any cases.")

    plan = _build_plan(cases, suite_run_dir)
    save_json(suite_run_dir / "benchmark_plan.json", plan)

    run_cfg = bench_cfg.get("run", {}) or {}
    dry_run = bool(run_cfg.get("dry_run", False))
    run_evaluate = bool(run_cfg.get("evaluate", True))
    run_visualize = bool(run_cfg.get("visualize", True))
    run_leaderboard_flag = bool(run_cfg.get("leaderboard", True))
    stop_on_error = bool(run_cfg.get("stop_on_error", True))

    visualize_defaults = bench_cfg.get("defaults", {}).get("visualize", {}) or DEFAULT_VISUALIZE

    run_records: List[Dict[str, Any]] = []
    upstream: List[str] = []

    for case in cases:
        record = {
            "case_name": case.case_name,
            "dataset": case.dataset,
            "target": case.target,
            "entry_name": case.entry_name,
            "model_group": case.model_group,
            "features_group": case.features_group,
            "backend": case.backend,
            "train_run_dir": str(_train_run_dir(suite_run_dir, case.case_name)),
            "evaluate_run_dir": str(_evaluate_run_dir(suite_run_dir, case.case_name)),
            "visualize_run_dir": str(_visualize_run_dir(suite_run_dir, case.case_name)),
            "train_status": "planned",
            "evaluate_status": "planned" if run_evaluate else "skipped",
            "visualize_status": "planned" if run_visualize else "skipped",
            "error": None,
        }

        try:
            train_cfg_path = configs_dir / f"{case.case_name}_train.yaml"
            train_cfg = _build_train_config(case, suite_run_dir)
            dump_yaml(train_cfg_path, train_cfg)

            if dry_run:
                record["train_status"] = "dry_run"
                if run_evaluate:
                    record["evaluate_status"] = "dry_run"
                if run_visualize:
                    record["visualize_status"] = "dry_run"
                run_records.append(record)
                continue

            train_run_dir = _run_train(train_cfg_path, case.backend)
            record["train_status"] = "done"
            record["train_run_dir"] = str(train_run_dir)
            upstream.append(str(train_run_dir))

            eval_run_dir = None
            if run_evaluate:
                eval_cfg_path = configs_dir / f"{case.case_name}_evaluate.yaml"
                eval_cfg = _build_evaluate_config(case, suite_run_dir, train_run_dir)
                dump_yaml(eval_cfg_path, eval_cfg)
                eval_run_dir = _run_evaluate(eval_cfg_path)
                record["evaluate_status"] = "done"
                record["evaluate_run_dir"] = str(eval_run_dir)
                upstream.append(str(eval_run_dir))

            if run_visualize:
                if eval_run_dir is None:
                    raise ValueError("visualize requested but evaluate is disabled.")
                viz_cfg_path = configs_dir / f"{case.case_name}_visualize.yaml"
                viz_cfg = _build_visualize_config(case, suite_run_dir, eval_run_dir, visualize_defaults)
                dump_yaml(viz_cfg_path, viz_cfg)
                _run_visualize(viz_cfg_path)
                record["visualize_status"] = "done"
                record["visualize_run_dir"] = str(_visualize_run_dir(suite_run_dir, case.case_name))
                upstream.append(record["visualize_run_dir"])

            run_records.append(record)
        except Exception as exc:
            record["error"] = str(exc)
            if record["train_status"] == "planned":
                record["train_status"] = "failed"
            elif run_evaluate and record["evaluate_status"] == "planned":
                record["evaluate_status"] = "failed"
            elif run_visualize and record["visualize_status"] == "planned":
                record["visualize_status"] = "failed"
            run_records.append(record)
            logger.error("Case failed: %s (%s)", case.case_name, exc)
            if stop_on_error:
                raise

    save_json(suite_run_dir / "benchmark_runs.json", {"cases": run_records})

    leaderboard_run_dir = None
    if run_leaderboard_flag and not dry_run:
        leaderboard_cfg = _build_leaderboard_config(bench_cfg, suite_run_dir, suite_name)
        leaderboard_run_dir = run_leaderboard(leaderboard_cfg)
        upstream.append(str(leaderboard_run_dir))

    report_path = suite_run_dir / "benchmark_report.md"
    report_payload = _build_report(
        suite_name=suite_name,
        cases=cases,
        run_records=run_records,
        leaderboard_run_dir=leaderboard_run_dir,
        bench_cfg=bench_cfg,
    )
    report_path.write_text(report_payload, encoding="utf-8")

    meta = build_meta(
        process_name=str(cfg.get("process", {}).get("name", "benchmark_suite")),
        cfg=cfg,
        upstream_artifacts=upstream,
        extra={
            "benchmark_suite_name": suite_name,
            "benchmark_case_count": len(cases),
            "benchmark_report": str(report_path),
            "leaderboard_run_dir": str(leaderboard_run_dir) if leaderboard_run_dir else None,
        },
    )
    save_meta(suite_run_dir, meta)

    logger.info("Benchmark suite done. Report: %s", report_path)
    return suite_run_dir


def expand_matrix(bench_cfg: Dict[str, Any], suite_name: str) -> List[BenchmarkCase]:
    matrix_cfg = bench_cfg.get("matrix", {}) or {}
    datasets = _normalize_list(matrix_cfg.get("datasets") or bench_cfg.get("datasets"))
    targets = _normalize_list(matrix_cfg.get("targets") or bench_cfg.get("targets"))
    entries = matrix_cfg.get("entries") or []

    if not datasets:
        raise ValueError("benchmark_suite.matrix.datasets is required.")
    if not targets:
        raise ValueError("benchmark_suite.matrix.targets is required.")
    if not entries:
        raise ValueError("benchmark_suite.matrix.entries is required.")

    defaults_cfg = bench_cfg.get("defaults", {}) or {}
    preprocess_defaults = defaults_cfg.get("preprocess", {}) or {}
    train_defaults = defaults_cfg.get("train", {}) or {}
    pretrain_default = defaults_cfg.get("pretrain", DEFAULT_PRETRAIN)
    eval_default = defaults_cfg.get("eval", DEFAULT_EVAL)
    name_template = str(bench_cfg.get("name_template") or defaults_cfg.get("name_template") or DEFAULT_NAME_TEMPLATE)

    base_tags = _normalize_list(bench_cfg.get("tags"))
    suite_tag = f"suite:{suite_name}"
    base_tags = _dedupe_tags(["benchmark", suite_tag] + base_tags)

    repo_root = Path(__file__).resolve().parents[2]
    model_cache: Dict[str, Dict[str, Any]] = {}
    features_cache: Dict[str, Dict[str, Any]] = {}

    cases: List[BenchmarkCase] = []

    for dataset in datasets:
        for target in targets:
            for entry in entries:
                if not isinstance(entry, dict):
                    raise ValueError("benchmark_suite.matrix.entries must be list of mappings.")
                model_group = str(entry.get("model", "")).strip()
                features_group = str(entry.get("features", "")).strip()
                if not model_group or not features_group:
                    raise ValueError("Each matrix entry must include model and features.")

                entry_name = str(entry.get("name") or f"{model_group}_{features_group}")

                model_cfg = _load_group_config(repo_root, "model", model_group, model_cache)
                model_info = model_cfg.get("model", {}) or {}
                model_name = str(model_info.get("name") or model_group)
                backend = str(model_info.get("family") or "").lower().strip()
                if not backend:
                    backend = "gnn" if model_name not in {"lightgbm", "lgbm", "rf", "random_forest", "catboost", "gpr"} else "fp"

                features_cfg = _load_group_config(repo_root, "features", features_group, features_cache)
                featureset_name = None
                if isinstance(features_cfg.get("featureset"), dict):
                    featureset_name = features_cfg.get("featureset", {}).get("name")

                preprocess_group = str(entry.get("preprocess") or preprocess_defaults.get(backend) or DEFAULT_PREPROCESS.get(backend, "none"))
                train_group = str(entry.get("train") or train_defaults.get(backend) or DEFAULT_TRAIN.get(backend, "gnn_default"))
                pretrain_group = str(entry.get("pretrain") or pretrain_default)
                eval_group = str(entry.get("eval") or eval_default)

                tags = _dedupe_tags(
                    base_tags
                    + _normalize_list(entry.get("tags"))
                    + [f"dataset:{dataset}", f"target:{target}", f"entry:{entry_name}"]
                )

                case_name = _slugify(name_template.format(dataset=dataset, target=target, entry=entry_name))

                cases.append(
                    BenchmarkCase(
                        case_name=case_name,
                        dataset=str(dataset),
                        target=str(target),
                        entry_name=entry_name,
                        model_group=model_group,
                        model_name=model_name,
                        features_group=features_group,
                        featureset_name=str(featureset_name) if featureset_name else None,
                        backend=backend,
                        preprocess_group=preprocess_group,
                        train_group=train_group,
                        pretrain_group=pretrain_group,
                        eval_group=eval_group,
                        tags=tags,
                    )
                )

    _ensure_unique_case_names(cases)
    return cases


def _run_train(cfg_path: Path, backend: str) -> Path:
    cfg = load_config(cfg_path)
    if backend == "fp":
        return fp_train.run(cfg)
    if backend == "gnn":
        return gnn_train.run(cfg)
    raise ValueError(f"Unknown backend: {backend}")


def _run_evaluate(cfg_path: Path) -> Path:
    cfg = load_config(cfg_path)
    model_artifact_dir = Path(cfg["model_artifact_dir"])
    backend = _resolve_backend_from_model_dir(model_artifact_dir)
    if backend == "fp":
        return fp_evaluate.run(cfg)
    if backend == "gnn":
        return gnn_evaluate.run(cfg)
    raise ValueError(f"Unknown backend: {backend}")


def _run_visualize(cfg_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    cmd = [sys.executable, str(repo_root / "scripts" / "visualize.py"), "--config", str(cfg_path)]
    subprocess.run(cmd, check=True)


def _resolve_backend_from_model_dir(model_dir: Path) -> str:
    train_cfg_path = model_dir / "config_snapshot.yaml"
    if not train_cfg_path.exists():
        raise FileNotFoundError(f"config_snapshot.yaml not found in model dir: {train_cfg_path}")
    train_cfg = load_config(train_cfg_path)
    model_cfg = train_cfg.get("model", {}) if isinstance(train_cfg, dict) else {}
    family = None
    if isinstance(model_cfg, dict):
        family = model_cfg.get("family")
    if family:
        return str(family).lower()
    name = model_cfg.get("name") if isinstance(model_cfg, dict) else None
    name = str(name).lower() if name else ""
    return "fp" if name in {"lightgbm", "lgbm", "rf", "random_forest", "catboost", "gpr"} else "gnn"


def _build_train_config(case: BenchmarkCase, suite_run_dir: Path) -> Dict[str, Any]:
    return {
        "defaults": [
            {"process": "train"},
            {"dataset": case.dataset},
            {"task": case.target},
            {"preprocess": case.preprocess_group},
            {"features": case.features_group},
            {"model": case.model_group},
            {"pretrain": case.pretrain_group},
            {"train": case.train_group},
            {"eval": case.eval_group},
            {"hydra": "default"},
            "_self_",
        ],
        "experiment": {"name": case.case_name},
        "process": {"backend": case.backend},
        "output": {
            "run_dir": str(suite_run_dir / "train"),
            "exp_name": case.case_name,
        },
        "tags": case.tags,
    }


def _build_evaluate_config(case: BenchmarkCase, suite_run_dir: Path, train_run_dir: Path) -> Dict[str, Any]:
    return {
        "defaults": [
            {"process": "evaluate"},
            {"dataset": case.dataset},
            {"task": case.target},
            {"eval": case.eval_group},
            {"hydra": "default"},
            "_self_",
        ],
        "model_artifact_dir": str(train_run_dir),
        "experiment": {"name": case.case_name},
        "output": {
            "run_dir": str(suite_run_dir / "evaluate"),
            "exp_name": case.case_name,
        },
        "tags": case.tags,
    }


def _build_visualize_config(
    case: BenchmarkCase,
    suite_run_dir: Path,
    evaluate_run_dir: Path,
    visualize_defaults: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "defaults": [
            {"process": "visualize"},
            {"task": case.target},
            {"hydra": "default"},
            "_self_",
        ],
        "input": {"evaluate_run_dir": str(evaluate_run_dir)},
        "plots": visualize_defaults,
        "experiment": {"name": case.case_name},
        "output": {
            "out_dir": str(suite_run_dir / "visualize"),
            "exp_name": case.case_name,
        },
        "tags": case.tags,
    }


def _build_leaderboard_config(bench_cfg: Dict[str, Any], suite_run_dir: Path, suite_name: str) -> Dict[str, Any]:
    leaderboard_cfg = bench_cfg.get("leaderboard", {}) or {}
    filters = leaderboard_cfg.get("filters", {}) or {}
    suite_tag = f"suite:{suite_name}"
    tags_filter = _normalize_list(filters.get("tags"))
    if suite_tag not in tags_filter:
        tags_filter.append(suite_tag)
    filters.setdefault("process_name", "evaluate")
    filters["tags"] = tags_filter

    leaderboard_payload = {
        "process": {"name": "leaderboard"},
        "leaderboard": {
            "root_dir": str(suite_run_dir / "evaluate"),
            "metric_key": str(leaderboard_cfg.get("metric_key", "mae")),
            "metric_split": leaderboard_cfg.get("metric_split", "val"),
            "sort_order": str(leaderboard_cfg.get("sort_order", "asc")),
            "top_n": int(leaderboard_cfg.get("top_n", 20)),
            "filters": filters,
        },
        "output": {
            "run_dir": str(suite_run_dir),
            "exp_name": "leaderboard",
        },
    }

    if "target_mode" in leaderboard_cfg:
        leaderboard_payload["leaderboard"]["target_mode"] = leaderboard_cfg.get("target_mode")
    if "target_names" in leaderboard_cfg:
        leaderboard_payload["leaderboard"]["target_names"] = leaderboard_cfg.get("target_names")
    if "target_top_n" in leaderboard_cfg:
        leaderboard_payload["leaderboard"]["target_top_n"] = leaderboard_cfg.get("target_top_n")

    return leaderboard_payload


def _build_plan(cases: List[BenchmarkCase], suite_run_dir: Path) -> Dict[str, Any]:
    payload = []
    for case in cases:
        payload.append(
            {
                **asdict(case),
                "train_run_dir": str(_train_run_dir(suite_run_dir, case.case_name)),
                "evaluate_run_dir": str(_evaluate_run_dir(suite_run_dir, case.case_name)),
                "visualize_run_dir": str(_visualize_run_dir(suite_run_dir, case.case_name)),
            }
        )
    return {"cases": payload}


def _train_run_dir(suite_run_dir: Path, case_name: str) -> Path:
    return suite_run_dir / "train" / case_name


def _evaluate_run_dir(suite_run_dir: Path, case_name: str) -> Path:
    return suite_run_dir / "evaluate" / case_name


def _visualize_run_dir(suite_run_dir: Path, case_name: str) -> Path:
    return suite_run_dir / "visualize" / case_name


def _build_report(
    suite_name: str,
    cases: List[BenchmarkCase],
    run_records: List[Dict[str, Any]],
    leaderboard_run_dir: Optional[Path],
    bench_cfg: Dict[str, Any],
) -> str:
    lines = [f"# Benchmark Report: {suite_name}", ""]

    description = bench_cfg.get("description")
    if description:
        lines.append(str(description))
        lines.append("")

    lines.append("## Matrix")
    lines.append(_matrix_table(cases))
    lines.append("")

    lines.append("## Runs")
    lines.append(_runs_table(run_records))
    lines.append("")

    lines.append("## Convergence")
    lines.append(_convergence_table(run_records))
    lines.append("")

    if leaderboard_run_dir:
        lines.append("## Leaderboard")
        md_path = Path(leaderboard_run_dir) / "leaderboard.md"
        if md_path.exists():
            lines.append(md_path.read_text(encoding="utf-8"))
        else:
            lines.append("leaderboard.md not found.")
        lines.append("")
    else:
        lines.append("## Leaderboard")
        lines.append("Leaderboard generation skipped or dry-run.")
        lines.append("")

    return "\n".join(lines)


def _matrix_table(cases: Iterable[BenchmarkCase]) -> str:
    header = "|case|dataset|target|entry|model|features|backend|"
    sep = "|---|---|---|---|---|---|---|"
    rows = [header, sep]
    for case in cases:
        rows.append(
            "|{case}|{dataset}|{target}|{entry}|{model}|{features}|{backend}|".format(
                case=case.case_name,
                dataset=case.dataset,
                target=case.target,
                entry=case.entry_name,
                model=case.model_group,
                features=case.features_group,
                backend=case.backend,
            )
        )
    return "\n".join(rows)


def _runs_table(run_records: Iterable[Dict[str, Any]]) -> str:
    header = "|case|train|evaluate|visualize|status|"
    sep = "|---|---|---|---|---|"
    rows = [header, sep]
    repo_root = Path(__file__).resolve().parents[2]
    for record in run_records:
        train_dir = _format_path(record.get("train_run_dir"), repo_root)
        eval_dir = _format_path(record.get("evaluate_run_dir"), repo_root)
        viz_dir = _format_path(record.get("visualize_run_dir"), repo_root)
        status = "/".join(
            [
                str(record.get("train_status")),
                str(record.get("evaluate_status")),
                str(record.get("visualize_status")),
            ]
        )
        rows.append(f"|{record.get('case_name')}|{train_dir}|{eval_dir}|{viz_dir}|{status}|")
    return "\n".join(rows)


def _convergence_table(run_records: Iterable[Dict[str, Any]]) -> str:
    header = "|case|learning_curve|"
    sep = "|---|---|"
    rows = [header, sep]
    repo_root = Path(__file__).resolve().parents[2]
    for record in run_records:
        train_status = str(record.get("train_status") or "").strip().lower()
        curve_cell = ""
        if train_status and train_status != "done":
            curve_cell = train_status
        else:
            train_dir = record.get("train_run_dir")
            if train_dir:
                candidates = [
                    Path(train_dir) / "plots" / "learning_curve.png",
                    Path(train_dir) / "plots" / "learning_curve_val.png",
                ]
                curve_path = next((p for p in candidates if p.exists()), None)
                if curve_path:
                    curve_cell = _format_path(str(curve_path), repo_root)
                else:
                    curve_cell = "missing"
            else:
                curve_cell = "missing"
        rows.append(f"|{record.get('case_name')}|{curve_cell}|")
    return "\n".join(rows)


def _normalize_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value]
    return [str(value)]


def _dedupe_tags(tags: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for tag in tags:
        tag = str(tag)
        if tag in seen or tag == "":
            continue
        seen.add(tag)
        out.append(tag)
    return out


def _slugify(value: str) -> str:
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value or "run"


def _ensure_unique_case_names(cases: List[BenchmarkCase]) -> None:
    counts: Dict[str, int] = {}
    for case in cases:
        name = case.case_name
        if name not in counts:
            counts[name] = 0
            continue
        counts[name] += 1
        case.case_name = f"{name}-{counts[name]}"


def _load_group_config(
    repo_root: Path,
    group: str,
    name: str,
    cache: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    key = f"{group}:{name}"
    if key in cache:
        return cache[key]
    path = repo_root / "configs" / group / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    cfg = load_config(path)
    cache[key] = cfg
    return cfg


def _format_path(path: Optional[str], repo_root: Path) -> str:
    if not path:
        return ""
    p = Path(path)
    try:
        return str(p.relative_to(repo_root))
    except ValueError:
        return str(p)
