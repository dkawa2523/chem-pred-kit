# ADR 0002: Plugin Registry for Models/Features/Tasks

- Status: Proposed
- Date: 2025-12-25

## Context
モデルや特徴量が増えると if/elif が肥大化し、保守が困難になる。

## Decision
- registry を導入し、`name -> class` で動的に選択する
- config の `model.name`, `features.name`, `task.name` と一致させる

## Consequences
- 新規追加が局所化される
- 無効な組合せは validation で防ぐ必要がある
