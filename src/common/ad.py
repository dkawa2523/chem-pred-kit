from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from rdkit import DataStructs
except Exception:  # pragma: no cover
    DataStructs = None


@dataclass
class ADResult:
    trust_score: int
    max_tanimoto: Optional[float]
    top_k: List[Tuple[float, Any]]
    unseen_elements: List[str]
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trust_score": int(self.trust_score),
            "max_tanimoto": None if self.max_tanimoto is None else float(self.max_tanimoto),
            "top_k": [(float(s), str(i)) for s, i in self.top_k],
            "unseen_elements": list(self.unseen_elements),
            "warnings": list(self.warnings),
        }


def tanimoto_top_k(query_fp, train_fps: Sequence[Any], train_ids: Sequence[Any], k: int = 5) -> Tuple[Optional[float], List[Tuple[float, Any]]]:
    if DataStructs is None:
        raise ImportError("RDKit is required for Tanimoto similarity.")
    if query_fp is None or train_fps is None or len(train_fps) == 0:
        return None, []
    sims = DataStructs.BulkTanimotoSimilarity(query_fp, list(train_fps))
    sims = np.asarray(sims, dtype=float)
    if len(sims) == 0:
        return None, []
    max_sim = float(np.max(sims))
    # Top-k indices
    k = min(int(k), len(sims))
    top_idx = np.argpartition(-sims, k - 1)[:k]
    top_sorted = top_idx[np.argsort(-sims[top_idx])]
    top = [(float(sims[i]), train_ids[i]) for i in top_sorted.tolist()]
    return max_sim, top


def compute_trust_score(
    max_tanimoto: Optional[float],
    unseen_elements: Sequence[str],
    tanimoto_warn_threshold: float = 0.5,
) -> Tuple[int, List[str]]:
    warnings: List[str] = []
    if unseen_elements:
        warnings.append(f"Unseen element(s) in training data: {', '.join(unseen_elements)}")
        return 5, warnings  # almost not reliable

    if max_tanimoto is None:
        warnings.append("No similarity available; cannot assess applicability domain.")
        return 30, warnings

    # heuristic mapping similarity -> score
    if max_tanimoto >= 0.85:
        score = 90
    elif max_tanimoto >= 0.70:
        score = 75
    elif max_tanimoto >= 0.55:
        score = 60
    elif max_tanimoto >= 0.40:
        score = 45
    else:
        score = 25

    if max_tanimoto < tanimoto_warn_threshold:
        warnings.append(
            f"Low similarity to training set: max Tanimoto={max_tanimoto:.3f} < {tanimoto_warn_threshold:.2f} (extrapolation risk)"
        )
    return score, warnings


def applicability_domain(
    query_elements: Sequence[str],
    training_elements: Sequence[str],
    query_fp,
    train_fps: Sequence[Any],
    train_ids: Sequence[Any],
    top_k: int = 5,
    tanimoto_warn_threshold: float = 0.5,
) -> ADResult:
    training_set = set(training_elements)
    unseen = sorted([e for e in set(query_elements) if e not in training_set])
    max_sim, top = tanimoto_top_k(query_fp, train_fps, train_ids, k=top_k)
    score, warnings = compute_trust_score(max_tanimoto=max_sim, unseen_elements=unseen, tanimoto_warn_threshold=tanimoto_warn_threshold)
    return ADResult(trust_score=score, max_tanimoto=max_sim, top_k=top, unseen_elements=unseen, warnings=warnings)
