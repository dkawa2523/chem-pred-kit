# Task 106 (P1): MoleculeNet Adapter Bundle（ESOL/FreeSolv/Lipo）

## 目的
小規模ベンチマーク（ESOL/FreeSolv/Lipo）を簡単に回せるようにする。

## In scope
- adapter: moleculenet（dataset_nameで切替）
- SMILES入力→canonicalize→graph/tabular生成
- scaffold split を標準
- 回帰/分類どちらにも対応できる設計（本タスクは回帰中心）

## Acceptance Criteria
- [x] `dataset=esol` 等で artifact を生成できる
- [x] baseline（FP/GCN）で train/eval が通る

## Verification
- `python scripts/build_dataset.py --config configs/dataset_esol.yaml`
- `python scripts/build_dataset.py --config configs/dataset_freesolv.yaml`
- `python scripts/build_dataset.py --config configs/dataset_lipo.yaml`
- `python -m pytest tests/contract/test_moleculenet_adapter_import.py`
- `python -m pytest tests/contract/test_moleculenet_adapter_contract.py`
- `python -m pytest tests/smoke/test_moleculenet_fp_smoke.py`
- `python -m pytest tests/smoke/test_moleculenet_gcn_smoke.py`

## 実行済み
- `python -m pytest tests/contract/test_moleculenet_adapter_import.py tests/contract/test_moleculenet_adapter_contract.py tests/smoke/test_moleculenet_fp_smoke.py tests/smoke/test_moleculenet_gcn_smoke.py`（pytest 未導入で失敗）
- `python - <<'PY' ...`（`create_dataset_adapter("moleculenet")` の import 確認）

## Notes
- pytest 未導入のためテストは未実行。rdkit/torch/torch_geometric 依存の smoke は環境次第で skip される想定。
