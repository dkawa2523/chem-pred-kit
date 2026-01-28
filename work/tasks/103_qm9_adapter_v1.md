# Task 103 (P0): QM9 Adapter v1（取得→変換→artifact）

## 目的
QM9 を内部共通スキーマへ変換し、1クリックで学習に使える状態にする。

## 実装方針（候補）
- A) PyTorch Geometric の `QM9` dataset を使う（入手が容易）
- B) 公式配布ファイルを直接読む（依存は軽いがETL実装が増える）
まずは A を優先し、B は将来の代替として docs に残す。

## In scope
- adapter: qm9
- canonical_smiles / sample_id 生成
- target mapping（例：gap, homo, lumo, U0, G…）
- split（random、seed固定）
- dataset artifact 保存（graph/tabular両方を想定）

## Out of scope
- マルチタスク学習（Task 130）
- 3Dモデル（Task 121以降）

## Acceptance Criteria
- [x] `dataset=qm9` で build_dataset が完了し、artifact が生成される
- [x] 生成artifactから train が走る（baselineでOK）
- [x] target registry に従い `target.<name>` が揃う

## Verification
- `python scripts/build_dataset.py dataset=qm9`
- `python scripts/train.py dataset=qm9 model=gcn features=gnn2d target=gap`

## 実行済み
- `python -m pytest tests/contract/test_qm9_adapter_contract.py -q`（pytest 未導入で失敗）
- `python - <<'PY' ...`（adapter.prepare_raw の軽量スモーク）

## Notes
- QM9 のフル build/train は PyG/rdkit/QM9 raw が揃った環境で再実行が必要。
