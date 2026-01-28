# Task 113 (P1): Partial Charges / Electronic Descriptors

## 目的
LJ/物性に効きやすい電子情報（Gasteiger charge等）を特徴量として追加し、性能改善を狙う。

## In scope
- RDKit Gasteiger charge を node feature として追加（optional feature flag）
- 欠損/NaN の扱い（0埋め + mask など）を決める
- 2D/3D両方で一貫して使えるようにする

## Acceptance Criteria
- [ ] `features.add_partial_charge=true` で学習が回る
- [ ] train/predict skew が無い（同一実装）
