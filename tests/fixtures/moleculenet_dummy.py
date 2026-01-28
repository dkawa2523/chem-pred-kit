from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

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
class DummyMoleculeNetData:
    idx: int
    smiles: str
    y: np.ndarray


class DummyMoleculeNetDataset:
    n_rows = 256

    def __init__(self, root: str, name: str) -> None:
        self.root = root
        self.name = name
        self._n = int(self.n_rows)
        self.smiles = [DEFAULT_SMILES[i % len(DEFAULT_SMILES)] for i in range(self._n)]
        self._ys = [self._build_y(i) for i in range(self._n)]

    def _build_y(self, idx: int) -> np.ndarray:
        value = (idx % 29) * 0.05
        return np.asarray([value], dtype=float)

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> DummyMoleculeNetData:
        return DummyMoleculeNetData(idx=idx, smiles=self.smiles[idx], y=self._ys[idx])
