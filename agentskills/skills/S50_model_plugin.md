# S50 Model Plugin

## Purpose
新しいモデルを追加し、registry + config で選択可能にする。

## Inputs
- docs/01_ARCHITECTURE.md
- docs/03_CONFIG_CONVENTIONS.md
- work/tasks/060_add_gnn_model_gin.md（または NEW_MODEL タスク）

## Allowed Changes
- src/**/models.py
- configs/model/**
- tests/**

## Steps
1) 既存モデルの I/F を確認
2) 新モデルを実装し、name で選択可能にする
3) smoke test（1epoch）を追加
4) artifact が契約どおり出ることを確認

## Common Pitfalls
- if/elif の増殖（registry化）
- modelごとに別の出力形式にしてしまう
