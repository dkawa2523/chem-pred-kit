from __future__ import annotations

from typing import Callable, Dict

from src.data.adapters.base import DatasetAdapter

Factory = Callable[[], DatasetAdapter]

_DATASET_ADAPTER_FACTORIES: Dict[str, Factory] = {}
_ADAPTER_IMPORT_ALIASES = {
    "lj": "src.data.adapters.lj",
    "lj_smoke": "src.data.adapters.lj",
    "lj_quick": "src.data.adapters.lj",
    "lj_fixture": "src.data.adapters.lj",
    "qm9": "src.data.adapters.qm9",
    "qm9_quick": "src.data.adapters.qm9",
}


def register_dataset_adapter(name: str) -> Callable[[Factory], Factory]:
    def _decorator(factory: Factory) -> Factory:
        _DATASET_ADAPTER_FACTORIES[name] = factory
        return factory

    return _decorator


def create_dataset_adapter(name: str) -> DatasetAdapter:
    if not _DATASET_ADAPTER_FACTORIES:
        import importlib

        importlib.import_module("src.data.adapters.lj")

    if name not in _DATASET_ADAPTER_FACTORIES:
        import importlib

        try:
            module = _ADAPTER_IMPORT_ALIASES.get(name, f"src.data.adapters.{name}")
            importlib.import_module(module)
        except ModuleNotFoundError:
            pass

    if name not in _DATASET_ADAPTER_FACTORIES:
        available = ", ".join(sorted(_DATASET_ADAPTER_FACTORIES.keys()))
        raise ValueError(f"Unknown dataset adapter '{name}'. Available: {available}")
    return _DATASET_ADAPTER_FACTORIES[name]()
