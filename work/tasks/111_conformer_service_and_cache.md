# Task 111 (P0): Conformer Service + Cache

## 目的
3Dモデルに必要な `pos` を安定に生成・保存し、再利用できるようにする。

## In scope
- conformer generator（SDF優先、無ければSMILES→Embed→Optimize）
- 失敗時の戦略（skip / 2D fallback / retry）をconfig化
- cache（sample_id→conformer_id→pos）をartifactとして保存
- メタ情報（生成法、seed、力場、最適化回数）を meta に残す

## Acceptance Criteria
- [ ] 同じseed/configなら同じposが再現できる（近似でOKだがハッシュは一致）
- [ ] predict 時に再生成せず cache を読む（skew防止）
