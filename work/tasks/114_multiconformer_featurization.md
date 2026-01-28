# Task 114 (P1): Multi-conformer Featurization

## 目的
3D座標が1つだと不安定なケースに備え、複数コンフォーマを扱える設計に拡張する。

## In scope
- K個コンフォーマ生成（Kはconfig）
- 集約（mean / attention / Boltzmann-weighted など）をまずは mean で実装
- 生成に失敗した場合のfallback

## Acceptance Criteria
- [x] K=3 などで特徴量生成が動く
- [x] artifactに conformer数と集約法が残る
