# Task 100 (P0): ChemKit Dataset Adapter Registry 導入

## 目的
外部データセット（QM9/MD17/MoleculeNet/PCQM4Mv2/独自SDF等）を **Adapter追加**で取り込めるようにするため、
`DatasetAdapter` の共通インターフェースと registry を導入する。

## 背景
- 今後データセットが増えると、if/elif が増えて保守不能になりやすい
- データ形式（SDF/XYZ/CSV/JSON、Trajectory）が多様で、I/O契約が崩れやすい
- 比較可能性（split/units/targets）を壊さないために “Adapter境界” が必要

## In scope
- dataset adapter interface（最低：`prepare_raw()` / `build_processed()` / `get_splits()`）
- dataset registry（`dataset.name -> adapter`）
- adapterが出力する **dataset artifact 契約**（metaに dataset_hash/units/target_names 等）
- 最小の smoke adapter（既存LJ datasetを adapterとしてラップでもOK）

## Out of scope
- QM9/MD17等の実装（Task 103以降）

## Acceptance Criteria
- [ ] Adapterの抽象クラス（or protocol）があり、最低限のI/Oが定義されている
- [ ] registryで `dataset.name` を切替できる
- [ ] dataset artifact 生成後、contract test が通る（config/meta/…）
- [ ] 既存の LJ パイプラインが壊れない

## Verification
- `python -m tools.dataset_doctor --dataset lj_smoke`（例）
- `pytest tests/contract -q`
