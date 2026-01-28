from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

from src.data_collection.types import DataCollectionQuery, RawPayload


class FileCache:
    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def build_key(self, source_name: str, query: DataCollectionQuery, extra: Optional[dict[str, Any]] = None) -> str:
        payload = {
            "source": source_name,
            "query": query.cache_key_payload(),
            "extra": extra or {},
        }
        blob = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def load(self, key: str) -> Optional[RawPayload]:
        path = self._cache_path(key)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return RawPayload(
            records=list(data.get("records", [])),
            sdf_records=dict(data.get("sdf_records", {})),
            metadata=dict(data.get("metadata", {})),
        )

    def save(self, key: str, payload: RawPayload) -> Path:
        path = self._cache_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "records": payload.records,
            "sdf_records": payload.sdf_records,
            "metadata": payload.metadata,
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=True, indent=2)
        return path
