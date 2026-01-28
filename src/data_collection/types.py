from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass(frozen=True)
class DataCollectionQuery:
    identifiers: List[str] = field(default_factory=list)
    filters: Dict[str, Any] = field(default_factory=dict)
    limit: Optional[int] = None

    def cache_key_payload(self) -> Dict[str, Any]:
        return {
            "identifiers": list(self.identifiers),
            "filters": dict(self.filters),
            "limit": self.limit,
        }


@dataclass
class RawPayload:
    records: List[Dict[str, Any]]
    sdf_records: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataCollectionResult:
    table: pd.DataFrame
    sdf_records: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
