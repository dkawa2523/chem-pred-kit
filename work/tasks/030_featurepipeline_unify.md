# Task 030 (P0): FeaturePipeline の統一（train/infer skew 排除）

## 目的
- 学習と推論で特徴量がズレる問題を防ぐ（精度と運用の地雷）

## Plan
1) `src/common/feature_pipeline.py`（案）を作り、前処理→特徴量化の一本化
2) 学習側は pipeline を fit/transform、推論側は保存済み pipeline を load/transform
3) FP と GNN で共通化できる範囲を決め、差分は adapter で吸収
4) smoke test：同一サンプルで train→save→load→predict が一致すること

## Acceptance Criteria
- [x] 推論時に学習と同じ pipeline が使われる
- [x] pipeline 状態が artifact に保存される

## Implementation Notes
- `src/common/feature_pipeline.py` に FP/GNN の共通パイプラインを追加
- train/evaluate/predict で pipeline の save/load を統一し、旧 artifact も互換対応
- `tests/test_feature_pipeline.py` を追加して roundtrip を検証

## Verification
- `python - <<'PY'\nimport ast\nfrom pathlib import Path\npath = Path('src/common/feature_pipeline.py')\nast.parse(path.read_text())\nprint('feature_pipeline parsed')\nPY`
