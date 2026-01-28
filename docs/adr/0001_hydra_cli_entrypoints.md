# ADR 0001: Hydra CLI Entry Points

- Status: Proposed
- Date: 2025-12-25

## Context
現状のスクリプト群を維持しつつ、Hydraで設定を統一したい。

## Decision
- 入口は `scripts/train.py`, `scripts/predict.py`, `scripts/evaluate.py` を基本とする
- 将来 `scripts/main.py` 統合は別ADRで検討

## Consequences
- 互換性を保ちつつ移行しやすい
- スクリプトは増えるが責務が明確
