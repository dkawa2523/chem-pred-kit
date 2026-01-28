from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from logging import Logger
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class RawDatasetInputs:
    raw_csv: Path
    sdf_dir: Optional[Path] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetArtifacts:
    dataset_csv: Path
    indices_dir: Path
    dataset_index: Path
    meta_path: Path
    dataset_hash: Optional[str]


class DatasetAdapter(ABC):
    name: str

    @abstractmethod
    def prepare_raw(self, cfg: Dict[str, Any], logger: Logger) -> RawDatasetInputs:
        raise NotImplementedError

    @abstractmethod
    def build_processed(self, cfg: Dict[str, Any], logger: Logger) -> DatasetArtifacts:
        raise NotImplementedError

    @abstractmethod
    def get_splits(self, cfg: Dict[str, Any], df, logger: Logger) -> Dict[str, List[int]]:
        raise NotImplementedError
