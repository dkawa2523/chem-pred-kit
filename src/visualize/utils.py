from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd


@dataclass(frozen=True)
class PlotTarget:
    name: str
    y_true_col: str
    y_pred_col: str
    suffix: str


def sanitize_filename(name: str) -> str:
    out = []
    for ch in str(name):
        if ch.isalnum() or ch in {".", "-", "_"}:
            out.append(ch)
        else:
            out.append("_")
    sanitized = "".join(out).strip("_")
    return sanitized or "target"


def _infer_target_names(columns: Sequence[str]) -> List[str]:
    column_set = {str(c) for c in columns}
    names: List[str] = []
    seen = set()
    for col in columns:
        col = str(col)
        if not col.startswith("y_true_"):
            continue
        name = col[len("y_true_") :]
        if name in seen:
            continue
        if f"y_pred_{name}" in column_set:
            names.append(name)
            seen.add(name)
    return names


def resolve_plot_targets(
    columns: Sequence[str],
    target_mode: str = "primary",
    primary_target: Optional[str] = None,
) -> Tuple[List[PlotTarget], List[str]]:
    mode = str(target_mode or "primary").lower()
    if mode not in {"primary", "all"}:
        raise ValueError(f"Unknown target_mode: {target_mode}")

    available = _infer_target_names(columns)
    if available:
        if primary_target and primary_target not in available:
            primary_target = None
        if mode == "all":
            names = available
            include_suffix = True
        else:
            names = [primary_target or available[0]]
            include_suffix = False

        targets: List[PlotTarget] = []
        for name in names:
            suffix = f"_{sanitize_filename(name)}" if include_suffix else ""
            targets.append(
                PlotTarget(
                    name=name,
                    y_true_col=f"y_true_{name}",
                    y_pred_col=f"y_pred_{name}",
                    suffix=suffix,
                )
            )
        return targets, available

    fallback_name = primary_target or "target"
    return [
        PlotTarget(
            name=fallback_name,
            y_true_col="y_true",
            y_pred_col="y_pred",
            suffix="",
        )
    ], []


def build_metrics_tables(metrics: Dict[str, Any]) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    by_split = metrics.get("by_split", {}) if isinstance(metrics, dict) else {}
    split_rows: List[Dict[str, Any]] = []
    target_rows: List[Dict[str, Any]] = []

    if isinstance(by_split, dict):
        for split, split_metrics in by_split.items():
            if not isinstance(split_metrics, dict):
                continue
            row: Dict[str, Any] = {"split": split}
            for key, value in split_metrics.items():
                if key == "by_target":
                    continue
                if isinstance(value, (int, float)):
                    row[key] = float(value)
            if len(row) > 1:
                split_rows.append(row)

            by_target = split_metrics.get("by_target")
            if isinstance(by_target, dict):
                for target, target_metrics in by_target.items():
                    if not isinstance(target_metrics, dict):
                        continue
                    trow: Dict[str, Any] = {"split": split, "target": str(target)}
                    for key, value in target_metrics.items():
                        if isinstance(value, (int, float)):
                            trow[key] = float(value)
                    if len(trow) > 2:
                        target_rows.append(trow)

    split_df = pd.DataFrame(split_rows)
    target_df = pd.DataFrame(target_rows) if target_rows else None
    return split_df, target_df
