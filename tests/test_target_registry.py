from __future__ import annotations

from src.targets.registry import load_target_registry, resolve_target_names, target_column_name


def test_target_registry_resolves_aliases() -> None:
    cfg = {
        "target_registry": [
            {"name": "foo", "aliases": ["FOO"], "units": "K"},
        ]
    }
    registry = load_target_registry(cfg)
    assert registry is not None
    assert registry.resolve_target_columns(["foo"]) == [target_column_name("foo")]
    assert registry.resolve_target_columns(["FOO"]) == [target_column_name("foo")]
    assert registry.resolve_source_column("foo", ["FOO", "bar"]) == "FOO"


def test_resolve_target_names_from_columns() -> None:
    cfg = {
        "target_registry": [
            {"name": "foo", "aliases": ["FOO"], "units": "K"},
        ]
    }
    names = resolve_target_names(cfg, ["target.foo", "FOO"])
    assert names == ["foo", "foo"]
