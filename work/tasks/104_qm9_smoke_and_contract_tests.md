# Task 104 (P0): QM9 E2E Smoke + Contract Tests

## 目的
QM9対応が「壊れずに回る」ことを、CI/ローカルで最小データで保証する。

## In scope
- QM9 small fixture（N=256程度）を用意（download不要なら尚良い）
- Process: build_dataset → featurize → train → evaluate を smoke で通す
- contract test（meta/metrics/predsの必須項目）
- split保存と再利用が確認できる

## Acceptance Criteria
- [ ] `pytest -q` で QM9 fixture のテストが通る
- [ ] artifacts が spec に合致
- [ ] compare が成立（同seedなら同split/同hash）

## Verification
- `pytest tests/smoke -q`
- `pytest tests/contract -q`
