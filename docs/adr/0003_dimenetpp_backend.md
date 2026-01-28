# ADR 0003: DimeNet++ Backend Selection

- Status: Accepted
- Date: 2025-12-27

## Context
DimeNet++ は角度/三体相互作用を使うため、実装・依存が重くなりやすい。
PyG/DIG/外部実装が候補となるため、採用方針を明確にする必要がある。

## Decision
- PyG の `DimeNetPlusPlus` 実装を採用する
- 依存は optional とし、torch_geometric + torch_scatter + torch_cluster を要求する
- Feature pipeline 側で角度特徴の生成を提供し、capability に `angles` を宣言する

## Consequences
- 良い影響: 既存の PyG 連携に合わせて実装が最小化される
- 悪い影響: 依存が重く、base 環境では動作しない
- 移行計画: なし（optional dependency として共存）

## Alternatives Considered
- DIG の DimeNet++ 実装
- 外部実装（torchmd-net など）
