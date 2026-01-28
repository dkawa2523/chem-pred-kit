from __future__ import annotations

from abc import ABC, abstractmethod
from logging import Logger
from typing import Any, Dict

from src.data_collection.types import DataCollectionResult, RawPayload


class DataFormatter(ABC):
    @abstractmethod
    def format(self, payload: RawPayload, cfg: Dict[str, Any], logger: Logger) -> DataCollectionResult:
        raise NotImplementedError
