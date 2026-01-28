from __future__ import annotations

from typing import Any, Dict, Optional

from src.data_collection.cache import FileCache
from src.data_collection.clients.dummy import DummyClient
from src.data_collection.formatters.mapping import MappingFormatter
from src.data_collection.registry import register_data_source
from src.data_collection.sources.base import DataSource
from src.data_collection.utils import resolve_api_key


@register_data_source("dummy")
def build_dummy_source(
    source_cfg: Dict[str, Any],
    collection_cfg: Dict[str, Any],
    cache: Optional[FileCache],
) -> DataSource:
    client_cfg = source_cfg.get("client", {}) or {}
    api_key = resolve_api_key(client_cfg, required=bool(client_cfg.get("api_key_required", False)))
    client = DummyClient(client_cfg, api_key=api_key)

    formatter_cfg = source_cfg.get("formatter", {}) or {}
    formatter_cfg.setdefault("sample_id_column", collection_cfg.get("sample_id_column", "sample_id"))
    formatter = MappingFormatter(formatter_cfg)

    return DataSource(name="dummy", client=client, formatter=formatter, cache=cache)
