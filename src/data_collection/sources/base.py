from __future__ import annotations

from logging import Logger
from typing import Any, Dict, Optional

from src.data_collection.cache import FileCache
from src.data_collection.clients.base import DataClient
from src.data_collection.formatters.base import DataFormatter
from src.data_collection.types import DataCollectionQuery, DataCollectionResult


class DataSource:
    def __init__(
        self,
        name: str,
        client: DataClient,
        formatter: DataFormatter,
        cache: Optional[FileCache] = None,
        cache_extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.name = name
        self.client = client
        self.formatter = formatter
        self.cache = cache
        self.cache_extra = cache_extra or {}

    def collect(self, query: DataCollectionQuery, formatter_cfg: Dict[str, Any], logger: Logger) -> DataCollectionResult:
        payload = None
        cache_key = None
        if self.cache is not None:
            cache_key = self.cache.build_key(self.name, query, extra=self.cache_extra)
            payload = self.cache.load(cache_key)
            if payload is not None:
                logger.info("Cache hit for data source '%s'", self.name)

        if payload is None:
            payload = self.client.fetch(query)
            if self.cache is not None and cache_key is not None:
                self.cache.save(cache_key, payload)

        return self.formatter.format(payload, formatter_cfg, logger)
