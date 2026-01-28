from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.common.config import dump_yaml
from src.common.meta import build_meta, save_meta
from src.common.utils import ensure_dir, get_logger, load_json
from src.utils.validate_config import validate_config


DEFAULT_FALLBACK_SPLITS = ("val", "test", "train")
CSV_COLUMNS = (
    "rank",
    "metric_value",
    "metric_key",
    "metric_split",
    "metric_source",
    "process_name",
    "task_name",
    "model_name",
    "featureset_name",
    "dataset_hash",
    "n_train",
    "n_val",
    "n_test",
    "seed",
    "run_id",
    "created_at",
    "git_sha",
    "config_hash",
    "run_dir",
    "tags",
)
TARGET_CSV_COLUMNS = (
    "rank",
    "metric_value",
    "metric_key",
    "metric_split",
    "metric_source",
    "target_name",
    "process_name",
    "task_name",
    "model_name",
    "featureset_name",
    "dataset_hash",
    "n_train",
    "n_val",
    "n_test",
    "seed",
    "run_id",
    "created_at",
    "git_sha",
    "config_hash",
    "run_dir",
    "tags",
)
MD_COLUMNS = (
    "rank",
    "metric_value",
    "process_name",
    "model_name",
    "featureset_name",
    "task_name",
    "run_dir",
)


def run(cfg: Dict[str, Any]) -> Path:
    validate_config(cfg)

    leaderboard_cfg = cfg.get("leaderboard", {}) or {}
    root_dirs = _normalize_roots(leaderboard_cfg.get("root_dir", "runs"))
    metric_key = str(leaderboard_cfg.get("metric_key", "r2"))
    metric_split = _normalize_optional_str(leaderboard_cfg.get("metric_split"))
    sort_order = str(leaderboard_cfg.get("sort_order", "desc")).lower()
    top_n = int(leaderboard_cfg.get("top_n", 20))
    target_mode = str(leaderboard_cfg.get("target_mode", "all")).lower()
    target_names = _normalize_list(leaderboard_cfg.get("target_names"))
    target_top_n = leaderboard_cfg.get("target_top_n", top_n)
    if target_top_n is None:
        target_top_n = top_n
    target_top_n = int(target_top_n)
    filters = leaderboard_cfg.get("filters", {}) or {}

    out_cfg = cfg.get("output", {}) or {}
    experiment_cfg = cfg.get("experiment", {}) or {}
    exp_name = str(out_cfg.get("exp_name", experiment_cfg.get("name", "leaderboard")))
    run_dir_root = Path(out_cfg.get("run_dir", "runs/leaderboard"))
    run_dir = ensure_dir(run_dir_root / exp_name)
    logger = get_logger("leaderboard", log_file=run_dir / "leaderboard.log")

    dump_yaml(run_dir / "config.yaml", cfg)

    if target_mode not in {"all", "primary", "none"}:
        raise ValueError(f"Unknown leaderboard.target_mode: {target_mode}")

    rows, target_rows, upstream = _collect_rows(
        root_dirs=root_dirs,
        metric_key=metric_key,
        metric_split=metric_split,
        filters=filters,
        target_mode=target_mode,
        target_names=target_names,
        logger=logger,
    )

    sorted_rows = _sort_rows(rows, sort_order=sort_order)
    _assign_ranks(sorted_rows)

    csv_path = run_dir / "leaderboard.csv"
    _write_csv(csv_path, sorted_rows, CSV_COLUMNS)

    md_payload = _build_markdown(
        rows=sorted_rows,
        metric_key=metric_key,
        metric_split=metric_split,
        sort_order=sort_order,
        top_n=top_n,
    )
    target_md = ""
    if target_rows and target_mode != "none":
        target_order = _resolve_target_order(target_rows, target_names)
        target_groups = _group_target_rows(target_rows, target_order, sort_order)
        target_csv_rows = _flatten_target_groups(target_groups)
        if target_csv_rows:
            target_csv_path = run_dir / "leaderboard_by_target.csv"
            _write_csv(target_csv_path, target_csv_rows, TARGET_CSV_COLUMNS)
        target_md = _build_target_markdown(
            target_groups=target_groups,
            metric_key=metric_key,
            metric_split=metric_split,
            sort_order=sort_order,
            top_n=target_top_n,
        )
        if target_md:
            target_md_standalone = _build_target_markdown(
                target_groups=target_groups,
                metric_key=metric_key,
                metric_split=metric_split,
                sort_order=sort_order,
                top_n=target_top_n,
                header_level=1,
            )
            (run_dir / "leaderboard_by_target.md").write_text(target_md_standalone, encoding="utf-8")
            md_payload = md_payload.rstrip() + "\n" + target_md
    md_path = run_dir / "leaderboard.md"
    md_path.write_text(md_payload, encoding="utf-8")

    meta = build_meta(
        process_name=str(cfg.get("process", {}).get("name", "leaderboard")),
        cfg=cfg,
        upstream_artifacts=upstream,
        extra={
            "leaderboard_metric_key": metric_key,
            "leaderboard_metric_split": metric_split,
            "leaderboard_sort_order": sort_order,
            "leaderboard_target_mode": target_mode,
            "leaderboard_target_names": target_names or None,
        },
    )
    save_meta(run_dir, meta)

    with_metric = sum(1 for row in sorted_rows if row.get("metric_value") is not None)
    logger.info("Collected %d runs (with metric: %d).", len(sorted_rows), with_metric)
    if target_rows and target_mode != "none":
        with_target_metric = sum(1 for row in target_rows if row.get("metric_value") is not None)
        logger.info(
            "Collected %d target rows (with metric: %d).",
            len(target_rows),
            with_target_metric,
        )
    logger.info("Saved leaderboard to %s", csv_path)
    return run_dir


def _normalize_roots(root_value: Any) -> List[Path]:
    if root_value is None or root_value == "":
        return [Path("runs")]
    if isinstance(root_value, (list, tuple, set)):
        return [Path(str(v)) for v in root_value]
    return [Path(str(root_value))]


def _normalize_optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    return value


def _normalize_list(value: Any) -> List[str]:
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value if str(v).strip()]
    value = str(value).strip()
    return [value] if value else []


def _collect_rows(
    root_dirs: Iterable[Path],
    metric_key: str,
    metric_split: Optional[str],
    filters: Dict[str, Any],
    target_mode: str,
    target_names: List[str],
    logger,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    rows: List[Dict[str, Any]] = []
    target_rows: List[Dict[str, Any]] = []
    upstream: List[str] = []
    seen_dirs: set[Path] = set()
    repo_root = Path(__file__).resolve().parents[2]

    for run_dir, meta_path, metrics_path in _iter_runs(root_dirs):
        if run_dir in seen_dirs:
            continue
        seen_dirs.add(run_dir)

        meta = _load_json_safe(meta_path, logger=logger)
        metrics = _load_json_safe(metrics_path, logger=logger)
        if not isinstance(meta, dict) or not isinstance(metrics, dict):
            continue

        if not _matches_filters(meta, filters):
            continue

        metric_value, metric_source, resolved_split = _extract_metric(
            metrics,
            metric_key=metric_key,
            metric_split=metric_split,
            fallback_splits=list(DEFAULT_FALLBACK_SPLITS),
        )

        run_dir_str = _format_path(run_dir, repo_root)
        upstream.append(run_dir_str)

        row_base = {
            "rank": None,
            "process_name": meta.get("process_name"),
            "task_name": meta.get("task_name"),
            "model_name": meta.get("model_name"),
            "featureset_name": meta.get("featureset_name"),
            "dataset_hash": meta.get("dataset_hash"),
            "n_train": metrics.get("n_train"),
            "n_val": metrics.get("n_val"),
            "n_test": metrics.get("n_test"),
            "seed": metrics.get("seed"),
            "run_id": meta.get("run_id"),
            "created_at": meta.get("created_at"),
            "git_sha": meta.get("git_sha"),
            "config_hash": meta.get("config_hash"),
            "run_dir": run_dir_str,
            "tags": _normalize_tags(meta.get("tags")),
        }
        rows.append(
            {
                **row_base,
                "metric_value": metric_value,
                "metric_key": metric_key,
                "metric_split": resolved_split or metric_split,
                "metric_source": metric_source,
            }
        )

        if target_mode != "none":
            available_targets = _extract_target_names(metrics)
            selected_targets = _select_targets(available_targets, target_mode, target_names)
            for target_name in selected_targets:
                t_value, t_source, t_split = _extract_target_metric(
                    metrics,
                    metric_key=metric_key,
                    metric_split=metric_split,
                    target_name=target_name,
                    fallback_splits=list(DEFAULT_FALLBACK_SPLITS),
                )
                target_rows.append(
                    {
                        **row_base,
                        "metric_value": t_value,
                        "metric_key": metric_key,
                        "metric_split": t_split or metric_split,
                        "metric_source": t_source,
                        "target_name": target_name,
                    }
                )

    return rows, target_rows, upstream


def _iter_runs(root_dirs: Iterable[Path]) -> Iterable[Tuple[Path, Path, Path]]:
    for root in root_dirs:
        if not root.exists():
            continue
        for meta_path in root.rglob("meta.json"):
            run_dir = meta_path.parent
            metrics_path = run_dir / "metrics.json"
            if metrics_path.exists():
                yield run_dir, meta_path, metrics_path


def _load_json_safe(path: Path, logger) -> Optional[Dict[str, Any]]:
    try:
        return load_json(path)
    except Exception as exc:
        logger.warning("Failed to load %s (%s)", path, exc)
        return None


def _normalize_tags(tags: Any) -> str:
    if tags is None:
        return ""
    if isinstance(tags, str):
        return tags
    if isinstance(tags, (list, tuple, set)):
        return ",".join(str(t) for t in tags)
    return str(tags)


def _extract_target_names(metrics: Dict[str, Any]) -> List[str]:
    names: List[str] = []
    seen = set()

    def _add(name: Any) -> None:
        if name is None:
            return
        name_str = str(name)
        if not name_str or name_str in seen:
            return
        names.append(name_str)
        seen.add(name_str)

    target_names = metrics.get("target_names")
    if isinstance(target_names, (list, tuple)):
        for name in target_names:
            _add(name)
    elif isinstance(target_names, str):
        _add(target_names)

    by_target = metrics.get("by_target")
    if isinstance(by_target, dict):
        for name in by_target.keys():
            _add(name)

    by_split = metrics.get("by_split")
    if isinstance(by_split, dict):
        for split_name in DEFAULT_FALLBACK_SPLITS:
            split_metrics = by_split.get(split_name)
            if not isinstance(split_metrics, dict):
                continue
            split_targets = split_metrics.get("by_target")
            if isinstance(split_targets, dict):
                for name in split_targets.keys():
                    _add(name)
        for split_metrics in by_split.values():
            if not isinstance(split_metrics, dict):
                continue
            split_targets = split_metrics.get("by_target")
            if isinstance(split_targets, dict):
                for name in split_targets.keys():
                    _add(name)

    for split_name in DEFAULT_FALLBACK_SPLITS:
        split_metrics = metrics.get(split_name)
        if not isinstance(split_metrics, dict):
            continue
        split_targets = split_metrics.get("by_target")
        if isinstance(split_targets, dict):
            for name in split_targets.keys():
                _add(name)

    return names


def _select_targets(
    available: List[str],
    target_mode: str,
    target_names: List[str],
) -> List[str]:
    if not available:
        return []
    if target_names:
        filtered = [name for name in available if name in target_names]
    else:
        filtered = list(available)
    if not filtered:
        return []
    if target_mode == "primary":
        return [filtered[0]]
    if target_mode == "all":
        return filtered
    return []


def _matches_filters(meta: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    if not filters:
        return True

    def _match_value(value: Any, criterion: Any) -> bool:
        if criterion is None or criterion == "" or criterion == []:
            return True
        if isinstance(criterion, (list, tuple, set)):
            acceptable = {str(v) for v in criterion}
            return str(value) in acceptable
        return str(value) == str(criterion)

    for key in (
        "task_name",
        "model_name",
        "featureset_name",
        "process_name",
        "dataset_hash",
        "run_id",
        "git_sha",
    ):
        if not _match_value(meta.get(key), filters.get(key)):
            return False

    tags_filter = filters.get("tags")
    if tags_filter:
        tags = meta.get("tags") or []
        if isinstance(tags, str):
            tags = [tags]
        if isinstance(tags_filter, str):
            tags_filter = [tags_filter]
        tag_set = {str(t) for t in tags}
        if not tag_set.intersection({str(t) for t in tags_filter}):
            return False

    created_after = _parse_iso(filters.get("created_after"))
    created_before = _parse_iso(filters.get("created_before"))
    if created_after or created_before:
        created_at = _parse_iso(meta.get("created_at"))
        if created_after and (created_at is None or created_at < created_after):
            return False
        if created_before and (created_at is None or created_at > created_before):
            return False

    return True


def _parse_iso(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        text = str(value)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text)
    except Exception:
        return None


def _extract_metric(
    metrics: Dict[str, Any],
    metric_key: str,
    metric_split: Optional[str],
    fallback_splits: List[str],
) -> Tuple[Optional[float], str, Optional[str]]:
    if metric_split:
        value, source = _find_metric(metrics, metric_key, metric_split)
        if value is not None:
            return value, source, metric_split

    by_split = metrics.get("by_split")
    if isinstance(by_split, dict):
        for split_name in fallback_splits:
            value, source = _find_metric(metrics, metric_key, split_name)
            if value is not None:
                return value, source, split_name
        for split_name, split_metrics in by_split.items():
            if isinstance(split_metrics, dict) and metric_key in split_metrics:
                return _coerce_float(split_metrics.get(metric_key)), f"by_split.{split_name}", split_name

    for split_name in fallback_splits:
        value, source = _find_metric(metrics, metric_key, split_name)
        if value is not None:
            return value, source, split_name

    if metric_key in metrics:
        value = _coerce_float(metrics.get(metric_key))
        if value is not None:
            return value, "metrics", None

    return None, "", None


def _extract_target_metric(
    metrics: Dict[str, Any],
    metric_key: str,
    metric_split: Optional[str],
    target_name: str,
    fallback_splits: List[str],
) -> Tuple[Optional[float], str, Optional[str]]:
    if metric_split:
        value, source = _find_target_metric(metrics, metric_key, target_name, metric_split)
        if value is not None:
            return value, source, metric_split

    for split_name in fallback_splits:
        value, source = _find_target_metric(metrics, metric_key, target_name, split_name)
        if value is not None:
            return value, source, split_name

    by_target = metrics.get("by_target")
    if isinstance(by_target, dict):
        target_metrics = by_target.get(target_name)
        if isinstance(target_metrics, dict) and metric_key in target_metrics:
            value = _coerce_float(target_metrics.get(metric_key))
            if value is not None:
                return value, "by_target", None

    return None, "", None


def _find_metric(metrics: Dict[str, Any], metric_key: str, split_name: str) -> Tuple[Optional[float], str]:
    by_split = metrics.get("by_split")
    if isinstance(by_split, dict):
        split_metrics = by_split.get(split_name)
        if isinstance(split_metrics, dict) and metric_key in split_metrics:
            return _coerce_float(split_metrics.get(metric_key)), f"by_split.{split_name}"

    direct_split = metrics.get(split_name)
    if isinstance(direct_split, dict) and metric_key in direct_split:
        return _coerce_float(direct_split.get(metric_key)), split_name

    return None, ""


def _find_target_metric(
    metrics: Dict[str, Any],
    metric_key: str,
    target_name: str,
    split_name: str,
) -> Tuple[Optional[float], str]:
    by_target = metrics.get("by_target")
    if isinstance(by_target, dict):
        target_block = by_target.get(target_name)
        if isinstance(target_block, dict):
            split_metrics = target_block.get(split_name)
            if isinstance(split_metrics, dict) and metric_key in split_metrics:
                return _coerce_float(split_metrics.get(metric_key)), f"by_target.{split_name}"

    by_split = metrics.get("by_split")
    if isinstance(by_split, dict):
        split_metrics = by_split.get(split_name)
        if isinstance(split_metrics, dict):
            split_targets = split_metrics.get("by_target")
            if isinstance(split_targets, dict):
                target_metrics = split_targets.get(target_name)
                if isinstance(target_metrics, dict) and metric_key in target_metrics:
                    return _coerce_float(target_metrics.get(metric_key)), f"by_split.{split_name}.by_target"

    direct_split = metrics.get(split_name)
    if isinstance(direct_split, dict):
        split_targets = direct_split.get("by_target")
        if isinstance(split_targets, dict):
            target_metrics = split_targets.get(target_name)
            if isinstance(target_metrics, dict) and metric_key in target_metrics:
                return _coerce_float(target_metrics.get(metric_key)), f"{split_name}.by_target"

    return None, ""


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sort_rows(rows: List[Dict[str, Any]], sort_order: str) -> List[Dict[str, Any]]:
    reverse = sort_order != "asc"

    def sort_key(row: Dict[str, Any]) -> float:
        value = row.get("metric_value")
        if value is None:
            return float("-inf") if reverse else float("inf")
        return float(value)

    return sorted(rows, key=sort_key, reverse=reverse)


def _assign_ranks(rows: List[Dict[str, Any]]) -> None:
    rank = 0
    for row in rows:
        if row.get("metric_value") is None:
            row["rank"] = None
            continue
        rank += 1
        row["rank"] = rank


def _write_csv(path: Path, rows: List[Dict[str, Any]], columns: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({col: _csv_value(row.get(col)) for col in columns})


def _resolve_target_order(target_rows: List[Dict[str, Any]], preferred: List[str]) -> List[str]:
    order: List[str] = []
    seen = set()

    def _add(name: Any) -> None:
        if name is None:
            return
        name_str = str(name)
        if not name_str or name_str in seen:
            return
        order.append(name_str)
        seen.add(name_str)

    for name in preferred:
        _add(name)
    for row in target_rows:
        _add(row.get("target_name"))
    return order


def _group_target_rows(
    target_rows: List[Dict[str, Any]],
    target_order: List[str],
    sort_order: str,
) -> List[Tuple[str, List[Dict[str, Any]]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in target_rows:
        name = row.get("target_name")
        if name is None:
            continue
        name_str = str(name)
        grouped.setdefault(name_str, []).append(row)

    groups: List[Tuple[str, List[Dict[str, Any]]]] = []
    remaining = set(grouped.keys())
    for name in target_order:
        if name not in grouped:
            continue
        sorted_rows = _sort_rows(grouped[name], sort_order=sort_order)
        _assign_ranks(sorted_rows)
        groups.append((name, sorted_rows))
        remaining.discard(name)
    for name in sorted(remaining):
        sorted_rows = _sort_rows(grouped[name], sort_order=sort_order)
        _assign_ranks(sorted_rows)
        groups.append((name, sorted_rows))
    return groups


def _flatten_target_groups(target_groups: List[Tuple[str, List[Dict[str, Any]]]]) -> List[Dict[str, Any]]:
    return [row for _, group_rows in target_groups for row in group_rows]


def _build_markdown(
    rows: List[Dict[str, Any]],
    metric_key: str,
    metric_split: Optional[str],
    sort_order: str,
    top_n: int,
) -> str:
    rows_with_metric = [row for row in rows if row.get("metric_value") is not None]
    md_rows = rows_with_metric[: max(top_n, 0)]
    header_lines = [
        "# Leaderboard",
        f"- metric_key: {metric_key}",
        f"- metric_split: {metric_split or 'auto'}",
        f"- sort_order: {sort_order}",
        f"- total_runs: {len(rows)}",
        f"- with_metric: {len(rows_with_metric)}",
        "",
    ]

    if not md_rows:
        return "\n".join(header_lines + ["No runs matched filters or metric_key."]) + "\n"

    table = _format_markdown_table(md_rows, MD_COLUMNS)
    return "\n".join(header_lines + [table, ""])


def _build_target_markdown(
    target_groups: List[Tuple[str, List[Dict[str, Any]]]],
    metric_key: str,
    metric_split: Optional[str],
    sort_order: str,
    top_n: int,
    header_level: int = 2,
) -> str:
    if not target_groups:
        return ""
    header_prefix = "#" * max(1, header_level)
    total_rows = sum(len(rows) for _, rows in target_groups)
    header_lines = [
        f"{header_prefix} Leaderboard by Target",
        f"- metric_key: {metric_key}",
        f"- metric_split: {metric_split or 'auto'}",
        f"- sort_order: {sort_order}",
        f"- targets: {len(target_groups)}",
        f"- total_rows: {total_rows}",
        "",
    ]
    lines = list(header_lines)
    subheader_prefix = "#" * max(2, header_level + 1)
    for target_name, rows in target_groups:
        lines.append(f"{subheader_prefix} {target_name}")
        rows_with_metric = [row for row in rows if row.get("metric_value") is not None]
        md_rows = rows_with_metric[: max(top_n, 0)]
        if not md_rows:
            lines.append("No runs matched filters or metric_key.")
            lines.append("")
            continue
        table = _format_markdown_table(md_rows, MD_COLUMNS)
        lines.append(table)
        lines.append("")
    return "\n".join(lines)


def _format_markdown_table(rows: List[Dict[str, Any]], columns: Iterable[str]) -> str:
    col_list = list(columns)
    lines = [
        "| " + " | ".join(col_list) + " |",
        "| " + " | ".join(["---"] * len(col_list)) + " |",
    ]
    for row in rows:
        values = []
        for col in col_list:
            value = row.get(col)
            if value is None:
                value_str = ""
            else:
                value_str = str(value)
            value_str = value_str.replace("\n", " ").replace("|", "\\|")
            values.append(value_str)
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _format_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (list, dict, tuple, set)):
        return str(value)
    return value
