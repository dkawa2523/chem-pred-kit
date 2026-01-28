from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

# Allow running as `python scripts/audit_dataset.py ...` without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.config import dump_yaml, load_config
from src.common.meta import build_meta, save_meta
from src.common.plots import save_hist
from src.common.utils import ensure_dir, get_logger, save_json
from src.data.audit import audit_dataset
from src.utils.validate_config import validate_config


def _save_element_plot(element_counts: Dict[str, int], out_path: Path, max_elements: int) -> None:
    if not element_counts:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    items = sorted(element_counts.items(), key=lambda x: x[1], reverse=True)[:max_elements]
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    plt.figure(figsize=(max(6, len(labels) * 0.4), 4))
    plt.bar(labels, values)
    plt.ylabel("count")
    plt.title("Element occurrence (per molecule)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit dataset quality (duplicates/leakage/invalid/mol stats).")
    ap.add_argument("--config", required=True, help="Path to configs/audit_dataset.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    validate_config(cfg)

    out_cfg = cfg.get("output", {})
    run_dir_root = Path(out_cfg.get("run_dir", "runs/audit"))
    experiment_cfg = cfg.get("experiment", {})
    exp_name = str(out_cfg.get("exp_name", experiment_cfg.get("name", "audit_dataset")))
    run_dir = ensure_dir(run_dir_root / exp_name)
    audit_dir = ensure_dir(run_dir / "audit")
    plots_dir = ensure_dir(run_dir / "plots")

    logger = get_logger("audit_dataset", log_file=run_dir / "audit.log")

    dump_yaml(run_dir / "config.yaml", cfg)

    report, report_md, plot_data = audit_dataset(cfg)
    meta = build_meta(
        process_name=str(cfg.get("process", {}).get("name", "audit_dataset")),
        cfg=cfg,
        dataset_hash=report.get("dataset_hash"),
    )
    save_meta(run_dir, meta)

    save_json(audit_dir / "audit_report.json", report)
    (audit_dir / "audit_report.md").write_text(report_md, encoding="utf-8")

    plot_cfg = cfg.get("audit", {}).get("plots", {}) or {}
    if plot_cfg.get("target_hist", True):
        target_stats = report.get("target_stats", {})
        if target_stats.get("available"):
            cols = list((target_stats.get("columns") or {}).keys())
            if cols:
                col = cols[0]
                values = plot_data.get("target_values_by_col", {}).get(col, plot_data.get("target_values"))
                if values:
                    save_hist(values, plots_dir / "target_hist.png", title="Target distribution", xlabel=str(col))

    if plot_cfg.get("mol_wt_hist", True):
        if plot_data.get("mol_weights"):
            save_hist(plot_data["mol_weights"], plots_dir / "mol_wt_hist.png", title="Molecular weight distribution", xlabel="MolWt")

    if plot_cfg.get("element_bar", True):
        max_elements = int(plot_cfg.get("max_elements", 20))
        _save_element_plot(report.get("element_counts", {}), plots_dir / "element_counts.png", max_elements)

    logger.info("Saved audit report to %s", audit_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
