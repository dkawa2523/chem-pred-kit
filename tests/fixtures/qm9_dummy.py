from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

DEFAULT_QM9_TARGETS_12: List[str] = [
    "mu",
    "alpha",
    "homo",
    "lumo",
    "gap",
    "r2",
    "zpve",
    "U0",
    "U",
    "H",
    "G",
    "Cv",
]

DEFAULT_SMILES: List[str] = [
    "C",
    "CC",
    "CCC",
    "CO",
    "CCO",
    "CCN",
    "O",
    "N",
    "C=O",
    "CC=O",
    "C#N",
    "CC#N",
    "COC",
    "CCOCC",
    "c1ccccc1",
    "C1CC1",
]


@dataclass(frozen=True)
class DummyQM9Data:
    idx: int
    smiles: str
    y: np.ndarray


class DummyQM9Dataset:
    target_names = list(DEFAULT_QM9_TARGETS_12)
    n_rows = 256

    def __init__(self, root: str) -> None:
        self.root = root
        self._n = int(self.n_rows)
        self.smiles = [DEFAULT_SMILES[i % len(DEFAULT_SMILES)] for i in range(self._n)]
        self._ys = [self._build_y(i) for i in range(self._n)]

    def _build_y(self, idx: int) -> np.ndarray:
        base = (idx % 37) * 0.01
        values = [base + j * 0.1 for j in range(len(DEFAULT_QM9_TARGETS_12))]
        return np.asarray(values, dtype=float)

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> DummyQM9Data:
        return DummyQM9Data(idx=idx, smiles=self.smiles[idx], y=self._ys[idx])
