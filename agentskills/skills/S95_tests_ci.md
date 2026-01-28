# S95 Tests & CI

## Purpose
pytest + smoke + contract test を整備し、改修で壊れないようにする。

## Allowed Changes
- tests/**
- pyproject.toml（任意）
- .github/workflows/**（任意）

## Pitfalls
- smoke が重くなりすぎる
- 乱数により flaky になる
