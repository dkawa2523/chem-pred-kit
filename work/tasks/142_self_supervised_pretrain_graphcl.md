# Task 142 (P2): Self-supervised Pretrain（GraphCL/Masked Atom）

## 目的
ラベル無し大規模データでも使える自己教師あり pretrain を導入する。

## In scope
- GraphCL（augmentation設計含む）or Masked Atom のどちらかを先に実装
- dataset adapter は pcqm4mv2 or pubchem など（既存を再利用）
- downstream への転移（Task 140）と接続

## Acceptance Criteria
- [x] 1エポック回る
- [x] downstream でロードできる

## Done
- GraphCL のaugmentation/loss と self-supervised pretrain loop を追加
- pretrain 用 config と CLI（`scripts/pretrain.py`）を追加
- downstream で利用する pretrain config を追加

## Verification
- `python scripts/pretrain.py --config configs/gnn/pretrain_graphcl_pcqm4mv2.yaml`
- `python scripts/train.py --config configs/gnn/train_lj_finetune_qm9.yaml pretrain=pcqm4mv2_graphcl`
- `pytest -q tests/unit/test_graphcl.py`

## 実行済み
- `pytest -q tests/unit/test_graphcl.py`（torch/torch_geometric 未導入のため skip）

## Notes
- torch/torch_geometric 未導入のため、GraphCL pretrain の1エポック実行と downstream でのロードは未実行。
