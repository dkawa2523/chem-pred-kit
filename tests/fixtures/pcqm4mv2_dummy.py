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
class DummyPCQM4Mv2Data:
    idx: int
    smiles: str
    y: np.ndarray


class DummyPCQM4Mv2Dataset:
    n_rows = 256

    def __init__(self, root: str, **_: object) -> None:
        self.root = root
        self._n = int(self.n_rows)
        self.smiles = [DEFAULT_SMILES[i % len(DEFAULT_SMILES)] for i in range(self._n)]
        self._ys = [self._build_y(i) for i in range(self._n)]

    def _build_y(self, idx: int) -> np.ndarray:
        value = (idx % 53) * 0.02
        return np.asarray([value], dtype=float)

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> DummyPCQM4Mv2Data:
        return DummyPCQM4Mv2Data(idx=idx, smiles=self.smiles[idx], y=self._ys[idx])

    def get_idx_split(self) -> dict[str, np.ndarray]:
        idx = np.arange(self._n)
        n_train = int(self._n * 0.8)
        n_val = int(self._n * 0.1)
        return {
            "train": idx[:n_train],
            "valid": idx[n_train : n_train + n_val],
            "test": idx[n_train + n_val :],
        }
