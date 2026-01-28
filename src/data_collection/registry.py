from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from src.data_collection.cache import FileCache
from src.data_collection.sources.base import DataSource

Factory = Callable[[Dict[str, Any], Dict[str, Any], Optional[FileCache]], DataSource]


_DATA_SOURCE_FACTORIES: Dict[str, Factory] = {}


def register_data_source(name: str) -> Callable[[Factory], Factory]:
    def _decorator(factory: Factory) -> Factory:
        _DATA_SOURCE_FACTORIES[name] = factory
        return factory

    return _decorator


def create_data_source(
    name: str,
    source_cfg: Dict[str, Any],
    collection_cfg: Dict[str, Any],
    cache: Optional[FileCache],
) -> DataSource:
    if not _DATA_SOURCE_FACTORIES:
        import importlib

        importlib.import_module("src.data_collection.sources.dummy")

    if name not in _DATA_SOURCE_FACTORIES:
        available = ", ".join(sorted(_DATA_SOURCE_FACTORIES.keys()))
        raise ValueError(f"Unknown data_source '{name}'. Available: {available}")
    return _DATA_SOURCE_FACTORIES[name](source_cfg, collection_cfg, cache)
