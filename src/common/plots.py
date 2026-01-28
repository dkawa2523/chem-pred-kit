from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence


def _configure_matplotlib_cache() -> None:
    """
    Ensure Matplotlib uses a writable config/cache directory.

    Some environments (e.g. sandboxed runs, containers) cannot write to $HOME,
    which can make Matplotlib imports very slow or noisy due to repeated cache builds.
    """
    if os.environ.get("MPLCONFIGDIR"):
        return
    cache_root = Path.cwd() / ".cache"
    cache_dir = cache_root / "matplotlib"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return
    os.environ["MPLCONFIGDIR"] = str(cache_dir)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))


_configure_matplotlib_cache()

import matplotlib

matplotlib.use("Agg")  # headless-friendly
import matplotlib.pyplot as plt
import numpy as np


def _apply_log_scale(values: np.ndarray) -> None:
    if np.all(values > 0):
        plt.yscale("log")
    else:
        plt.yscale("symlog", linthresh=1e-6)


def save_learning_curve(
    loss_train: Sequence[float],
    loss_val: Sequence[float],
    out_path: str | Path,
    ylabel: str = "loss",
    yscale: str = "linear",
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    loss_train = np.asarray(loss_train, dtype=float)
    loss_val_arr = np.asarray(loss_val, dtype=float) if loss_val is not None else None
    x = np.arange(1, len(loss_train) + 1)
    plt.figure()
    plt.plot(x, loss_train, label="train")
    if loss_val_arr is not None and len(loss_val_arr) == len(loss_train):
        plt.plot(x, loss_val_arr, label="val")
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    if yscale == "log":
        if loss_val_arr is not None and len(loss_val_arr) == len(loss_train):
            _apply_log_scale(np.concatenate([loss_train, loss_val_arr]))
        else:
            _apply_log_scale(loss_train)
    elif yscale and yscale != "linear":
        plt.yscale(yscale)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_parity_plot(y_true, y_pred, out_path: str | Path, title: Optional[str] = None, xlabel: str = "true", ylabel: str = "pred") -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    plt.figure()
    plt.scatter(y_true, y_pred, s=12, alpha=0.7)
    if y_true.size == 0:
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if title:
            plt.title(title)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        return
    mn = float(np.nanmin([y_true.min(), y_pred.min()]))
    mx = float(np.nanmax([y_true.max(), y_pred.max()]))
    plt.plot([mn, mx], [mn, mx], color="black", linewidth=1.0, alpha=0.6, label="y=x")
    if y_true.size >= 2 and np.nanstd(y_true) > 0:
        slope, intercept = np.polyfit(y_true, y_pred, 1)
        x_line = np.array([mn, mx])
        plt.plot(x_line, slope * x_line + intercept, color="tab:orange", linestyle="--", linewidth=1.5, label="fit")
        denom = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = float("nan") if denom == 0 else 1.0 - float(np.sum((y_true - y_pred) ** 2) / denom)
        plt.gca().text(
            0.05,
            0.95,
            f"R$^2$ = {r2:.3f}",
            transform=plt.gca().transAxes,
            ha="left",
            va="top",
            fontsize="small",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6, linewidth=0.0),
        )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.legend(loc="best", fontsize="small", frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_residual_plot(y_true, y_pred, out_path: str | Path, title: Optional[str] = None) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    resid = y_pred - y_true
    plt.figure()
    plt.scatter(y_true, resid, s=12, alpha=0.7)
    plt.axhline(0.0)
    plt.xlabel("true")
    plt.ylabel("residual (pred-true)")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_residual_hist(y_true, y_pred, out_path: str | Path, title: Optional[str] = None, bins: int = 50) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    resid = y_pred - y_true
    resid = resid[np.isfinite(resid)]
    plt.figure()
    plt.hist(resid, bins=bins)
    plt.axvline(0.0, color="black", linewidth=1.0, alpha=0.6)
    plt.xlabel("residual (pred-true)")
    plt.ylabel("count")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_abs_error_cdf(y_true, y_pred, out_path: str | Path, title: Optional[str] = None) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    abs_err = np.abs(y_pred - y_true)
    abs_err = abs_err[np.isfinite(abs_err)]
    if abs_err.size == 0:
        return
    x = np.sort(abs_err)
    y = np.arange(1, len(x) + 1) / len(x)
    plt.figure()
    plt.plot(x, y, color="tab:blue")
    plt.xlabel("|pred-true|")
    plt.ylabel("CDF")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_bland_altman_plot(y_true, y_pred, out_path: str | Path, title: Optional[str] = None) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    mean_vals = 0.5 * (y_true + y_pred)
    diff = y_pred - y_true
    mean_diff = float(np.mean(diff))
    sd_diff = float(np.std(diff))
    plt.figure()
    plt.scatter(mean_vals, diff, s=12, alpha=0.7)
    plt.axhline(mean_diff, color="tab:orange", linestyle="--", linewidth=1.2, label="mean diff")
    plt.axhline(mean_diff + 1.96 * sd_diff, color="tab:green", linestyle=":", linewidth=1.0, label="±1.96 SD")
    plt.axhline(mean_diff - 1.96 * sd_diff, color="tab:green", linestyle=":", linewidth=1.0)
    plt.xlabel("mean of true & pred")
    plt.ylabel("pred-true")
    if title:
        plt.title(title)
    plt.legend(loc="best", fontsize="small", frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_hist(values, out_path: str | Path, title: str, xlabel: str) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    plt.figure()
    plt.hist(v, bins=50)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_embedding_plot(
    coords,
    out_path: str | Path,
    color=None,
    title: Optional[str] = None,
    color_label: Optional[str] = None,
    max_categories: int = 12,
    point_size: float = 12.0,
    alpha: float = 0.75,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape (N, 2) for embedding plot.")
    x = coords[:, 0]
    y = coords[:, 1]

    plt.figure()
    if color is None:
        plt.scatter(x, y, s=point_size, alpha=alpha)
    else:
        values = np.asarray(color)
        numeric_values = None
        is_numeric = False
        try:
            numeric_values = values.astype(float)
            is_numeric = np.isfinite(numeric_values).any()
        except Exception:
            is_numeric = False

        if is_numeric and numeric_values is not None:
            scatter = plt.scatter(x, y, c=numeric_values, s=point_size, alpha=alpha, cmap="viridis")
            plt.colorbar(scatter, label=color_label or "value")
        else:
            labels = []
            for v in values:
                if v is None:
                    labels.append("unknown")
                elif isinstance(v, float) and np.isnan(v):
                    labels.append("unknown")
                else:
                    labels.append(str(v))
            counts = {}
            for label in labels:
                counts[label] = counts.get(label, 0) + 1
            sorted_labels = sorted(counts.items(), key=lambda item: item[1], reverse=True)
            if len(sorted_labels) > max_categories:
                keep = {name for name, _ in sorted_labels[:max_categories - 1]}
                labels = [lab if lab in keep else "other" for lab in labels]
            unique = sorted({lab for lab in labels})
            cmap = plt.cm.get_cmap("tab20", max(len(unique), 1))
            for idx, label in enumerate(unique):
                mask = np.array([lab == label for lab in labels], dtype=bool)
                if not mask.any():
                    continue
                plt.scatter(
                    x[mask],
                    y[mask],
                    s=point_size,
                    alpha=alpha,
                    color=cmap(idx),
                    label=label,
                )
            plt.legend(title=color_label or "category", loc="best", fontsize="small", frameon=False)

    plt.xlabel("dim-1")
    plt.ylabel("dim-2")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
