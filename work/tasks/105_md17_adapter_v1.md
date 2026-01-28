# Task 105 (P1): MD17 Adapter v1（trajectory対応）

## 目的
MD17（energy/forces）の trajectory データを内部スキーマへ取り込み、3Dモデル学習の足場を作る。

## 背景
MD17 は時間相関が強く、random split はリークしやすい。
time-based split を標準にする。

## In scope
- adapter: md17
- `target.energy`, `target.forces` の出力（forcesは可変長：N_atoms×3）
- time-based split
- 3D pos を必須として保存

## Out of scope
- force学習を全モデルへ適用（まずはSchNet等の一部で）

## Acceptance Criteria
- [x] md17 artifact を生成できる
- [x] 1モデルで energy 回帰が学習できる（forceはevaluateまででも可）

## Verification
- `python scripts/build_dataset.py dataset=md17`
- `python scripts/train.py dataset=md17 model=schnet target=energy`

## 実行済み
- `python -m pytest tests/contract/test_md17_adapter_contract.py`（pytest 未導入で失敗）
- `python - <<'PY' ...`（`work/tmp_md17_check` で adapter.build_processed を実行）

## Notes
- GNN 学習は torch_geometric 未導入のため未実行。
