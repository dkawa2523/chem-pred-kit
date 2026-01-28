from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from src.data_collection.types import DataCollectionQuery, RawPayload


class DataClient(ABC):
    def __init__(self, cfg: Dict[str, Any], api_key: Optional[str] = None) -> None:
        self.cfg = cfg
        self.api_key = api_key

    @abstractmethod
    def fetch(self, query: DataCollectionQuery) -> RawPayload:
        raise NotImplementedError
