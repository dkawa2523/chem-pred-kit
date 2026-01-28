from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdMolDescriptors
    from rdkit.ML.Cluster import Butina
except Exception:  # pragma: no cover
    Chem = None
    DataStructs = None
    rdMolDescriptors = None
    Butina = None


class SelectorError(ValueError):
    pass


@dataclass
class SelectorContext:
    """Optional shared context for selectors."""
    # element frequency in dataset
    element_counts: Optional[Dict[str, int]] = None
    # RDKit mols aligned with df rows (same order)
    mols: Optional[List[Any]] = None
    # fingerprints aligned with df rows
    fps: Optional[List[Any]] = None


def _elements_set_from_string(s: str) -> set[str]:
    if s is None:
        return set()
    s = str(s).strip()
    if not s:
        return set()
    return set([x for x in s.split(",") if x])


def _align_ctx_list_to_df(ctx_values: List[Any], df: pd.DataFrame, *, label: str) -> List[Any]:
    """
    Align a context list (mols/fps) to the current df row order.

    Assumptions:
      - If len(ctx_values) == len(df), it is already aligned to df order.
      - Otherwise, ctx_values is assumed to be aligned to the ORIGINAL df order,
        and df.index contains positional integer indices into ctx_values.
    """
    if len(ctx_values) == len(df):
        return ctx_values
    out: List[Any] = []
    for idx in df.index.tolist():
        try:
            j = int(idx)
        except Exception as e:
            raise SelectorError(
                f"Cannot align selector context '{label}' to df because df.index is not integer-like. "
                "Reset index (df.reset_index(drop=True)) or provide aligned context."
            ) from e
        if j < 0 or j >= len(ctx_values):
            raise SelectorError(
                f"Cannot align selector context '{label}': df.index contains {j} but context length is {len(ctx_values)}. "
                "Ensure the context was created from the same pre-filtered dataframe."
            )
        out.append(ctx_values[j])
    return out


def selector_element_whitelist(df: pd.DataFrame, allowed: Sequence[str], elements_col: str = "elements") -> pd.Series:
    allowed_set = set(allowed)
    mask = []
    for e_str in df[elements_col].astype(str).tolist():
        es = _elements_set_from_string(e_str)
        mask.append(es.issubset(allowed_set))
    return pd.Series(mask, index=df.index)


def selector_element_blacklist(df: pd.DataFrame, banned: Sequence[str], elements_col: str = "elements") -> pd.Series:
    banned_set = set(banned)
    mask = []
    for e_str in df[elements_col].astype(str).tolist():
        es = _elements_set_from_string(e_str)
        mask.append(len(es.intersection(banned_set)) == 0)
    return pd.Series(mask, index=df.index)


def selector_min_element_frequency(
    df: pd.DataFrame,
    ctx: SelectorContext,
    min_count: int,
    elements_col: str = "elements",
) -> pd.Series:
    if ctx.element_counts is None:
        raise SelectorError("element_counts missing in SelectorContext for min_element_frequency selector.")
    mask = []
    for e_str in df[elements_col].astype(str).tolist():
        es = _elements_set_from_string(e_str)
        ok = True
        for el in es:
            if ctx.element_counts.get(el, 0) < int(min_count):
                ok = False
                break
        mask.append(ok)
    return pd.Series(mask, index=df.index)


def selector_max_size(df: pd.DataFrame, max_heavy_atoms: int, heavy_atoms_col: str = "n_heavy_atoms") -> pd.Series:
    return df[heavy_atoms_col].fillna(1e9).astype(float) <= float(max_heavy_atoms)


def selector_target_range(
    df: pd.DataFrame,
    target_col: str,
    valid_range: Tuple[float, float],
) -> pd.Series:
    lo, hi = valid_range
    y = df[target_col].astype(float)
    return (y >= lo) & (y <= hi)


def sampler_target_stratified(
    df: pd.DataFrame,
    target_col: str,
    n_bins: int,
    samples_per_bin: Optional[int] = None,
    total_samples: Optional[int] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Downsample df to balance target distribution."""
    rng = np.random.default_rng(seed)
    y = df[target_col].astype(float).values
    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.quantile(y, quantiles)
    # Ensure unique edges
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 3:
        # Not enough variation
        return df

    bins = np.digitize(y, bin_edges[1:-1], right=True)
    out_idx = []
    # Determine samples per bin
    if samples_per_bin is None and total_samples is not None:
        samples_per_bin = max(1, int(total_samples) // (bins.max() + 1))
    if samples_per_bin is None:
        samples_per_bin = int(min(df.shape[0], 1000))  # safe default
    for b in range(bins.max() + 1):
        idx_b = df.index[bins == b].to_numpy()
        if len(idx_b) == 0:
            continue
        take = min(len(idx_b), int(samples_per_bin))
        chosen = rng.choice(idx_b, size=take, replace=False)
        out_idx.extend(chosen.tolist())
    return df.loc[out_idx].copy()


def _compute_morgan_fps(mols: List[Any], radius: int = 2, n_bits: int = 2048):
    if rdMolDescriptors is None:
        raise ImportError("RDKit is required for fingerprint-based selection.")
    fps = []
    for m in mols:
        if m is None:
            fps.append(None)
        else:
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)
            fps.append(fp)
    return fps


def sampler_diversity_farthest_point(
    df: pd.DataFrame,
    ctx: SelectorContext,
    n_samples: int,
    radius: int = 2,
    n_bits: int = 2048,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Greedy farthest-point sampling in Tanimoto space.

    This is O(N*K) with similarity computations; use for moderate sizes or after pre-filtering.
    """
    if DataStructs is None:
        raise ImportError("RDKit is required for diversity sampling.")
    if ctx.mols is None:
        raise SelectorError("mols missing in SelectorContext for diversity sampling.")
    if ctx.fps is None:
        ctx.fps = _compute_morgan_fps(ctx.mols, radius=radius, n_bits=n_bits)

    # Filter only valid fps
    fps_aligned = _align_ctx_list_to_df(ctx.fps, df, label="fps")
    valid = [(i, fp) for i, fp in zip(df.index.tolist(), fps_aligned) if fp is not None]
    if len(valid) == 0:
        return df.iloc[0:0].copy()

    rng = np.random.default_rng(seed)
    n_samples = min(int(n_samples), len(valid))
    # Start from random seed
    start = rng.integers(0, len(valid))
    selected = [valid[start][0]]
    selected_fps = [valid[start][1]]

    remaining = [v for j, v in enumerate(valid) if j != start]

    # Precompute to speed: keep best similarity to selected for each remaining
    best_sim = np.zeros(len(remaining), dtype=float)
    for idx, (row_idx, fp) in enumerate(remaining):
        best_sim[idx] = max(DataStructs.BulkTanimotoSimilarity(fp, selected_fps))

    for _ in range(1, n_samples):
        # Choose point with minimal best similarity (farthest)
        k = int(np.argmin(best_sim))
        row_idx, fp = remaining.pop(k)
        selected.append(row_idx)
        selected_fps.append(fp)
        best_sim = np.delete(best_sim, k)
        if len(remaining) == 0:
            break
        # Update best similarity after adding new selected fp
        sims = np.array(DataStructs.BulkTanimotoSimilarity(fp, [r[1] for r in remaining]))
        # best_sim[j] = max(old, sim(new_selected, remaining[j]))
        best_sim = np.maximum(best_sim, sims)

    return df.loc[selected].copy()


def sampler_butina_cluster(
    df: pd.DataFrame,
    ctx: SelectorContext,
    dist_thresh: float = 0.4,
    radius: int = 2,
    n_bits: int = 2048,
    max_per_cluster: Optional[int] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Cluster molecules with Butina clustering in Tanimoto distance; sample up to max_per_cluster."""
    if Butina is None or DataStructs is None:
        raise ImportError("RDKit is required for Butina clustering.")
    if ctx.mols is None:
        raise SelectorError("mols missing in SelectorContext for butina clustering.")
    if ctx.fps is None:
        ctx.fps = _compute_morgan_fps(ctx.mols, radius=radius, n_bits=n_bits)

    fps_aligned = _align_ctx_list_to_df(ctx.fps, df, label="fps")
    pairs = [(idx, fp) for idx, fp in zip(df.index.tolist(), fps_aligned) if fp is not None]
    fps = [p[1] for p in pairs]
    row_indices = [p[0] for p in pairs]
    if len(fps) == 0:
        return df.iloc[0:0].copy()

    # Distance matrix upper triangle as list
    dists = []
    for i in range(1, len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1.0 - x for x in sims])

    clusters = Butina.ClusterData(dists, len(fps), dist_thresh, isDistData=True)
    rng = np.random.default_rng(seed)
    selected = []
    for c in clusters:
        members = [row_indices[i] for i in c]
        if max_per_cluster is None:
            selected.extend(members)
        else:
            take = min(len(members), int(max_per_cluster))
            chosen = rng.choice(members, size=take, replace=False)
            selected.extend(chosen.tolist())
    return df.loc[selected].copy()


def apply_selectors(df: pd.DataFrame, selectors: List[Dict[str, Any]], ctx: SelectorContext) -> pd.DataFrame:
    """
    Apply a sequence of selectors/samplers described in YAML.

    Selector item examples:
      - {name: element_whitelist, params: {allowed: [C,H,O,N]}}
      - {name: min_element_frequency, params: {min_count: 50}}
      - {name: max_size, params: {max_heavy_atoms: 60}}
      - {name: target_range, params: {target_col: lj_epsilon_over_k_K, valid_range: [1.0, 5000.0]}}
      - {name: target_stratified, params: {target_col: ..., n_bins: 20, samples_per_bin: 200}}
      - {name: diversity_farthest_point, params: {n_samples: 2000}}
      - {name: butina_cluster, params: {dist_thresh: 0.4, max_per_cluster: 50}}
    """
    out = df.copy()
    for item in selectors:
        name = str(item.get("name", "")).strip()
        params = item.get("params", {}) or {}
        name_l = name.lower()

        if name_l == "element_whitelist":
            allowed = params.get("allowed", [])
            out = out[selector_element_whitelist(out, allowed=allowed, elements_col=params.get("elements_col", "elements"))]
        elif name_l == "element_blacklist":
            banned = params.get("banned", [])
            out = out[selector_element_blacklist(out, banned=banned, elements_col=params.get("elements_col", "elements"))]
        elif name_l == "min_element_frequency":
            out = out[selector_min_element_frequency(out, ctx=ctx, min_count=int(params.get("min_count", 1)), elements_col=params.get("elements_col", "elements"))]
        elif name_l == "max_size":
            out = out[selector_max_size(out, max_heavy_atoms=int(params.get("max_heavy_atoms", 9999)), heavy_atoms_col=params.get("heavy_atoms_col", "n_heavy_atoms"))]
        elif name_l == "target_range":
            target_col = params["target_col"]
            lo, hi = params["valid_range"]
            out = out[selector_target_range(out, target_col=target_col, valid_range=(float(lo), float(hi)))]
        elif name_l == "target_stratified":
            out = sampler_target_stratified(
                out,
                target_col=params["target_col"],
                n_bins=int(params.get("n_bins", 20)),
                samples_per_bin=params.get("samples_per_bin", None),
                total_samples=params.get("total_samples", None),
                seed=int(params.get("seed", 42)),
            )
        elif name_l == "diversity_farthest_point":
            out = sampler_diversity_farthest_point(
                out,
                ctx=ctx,
                n_samples=int(params.get("n_samples", 1000)),
                radius=int(params.get("radius", 2)),
                n_bits=int(params.get("n_bits", 2048)),
                seed=int(params.get("seed", 42)),
            )
        elif name_l == "butina_cluster":
            out = sampler_butina_cluster(
                out,
                ctx=ctx,
                dist_thresh=float(params.get("dist_thresh", 0.4)),
                radius=int(params.get("radius", 2)),
                n_bits=int(params.get("n_bits", 2048)),
                max_per_cluster=params.get("max_per_cluster", None),
                seed=int(params.get("seed", 42)),
            )
        else:
            raise SelectorError(f"Unknown selector name: {name}")

    return out
