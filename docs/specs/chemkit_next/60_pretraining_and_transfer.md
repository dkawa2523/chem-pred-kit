# Pretraining & Transfer（ChemKit Next）

## 目的
- 大規模データで事前学習→LJ/物性へ転移して精度・収束を改善する

## 方式（候補）
### 教師あり pretrain
- PCQM4Mv2（HOMO-LUMO gap）
- QM9（複数物性をマルチタスクで）

### 自己教師あり pretrain
- GraphCL / Masked Atom/Edge / Context prediction
- SMILES MLM（ChemBERTa 等）

## 転移モード
- fine-tune（全層更新）
- linear probe（encoder freeze + headのみ）
- partial freeze（凍結範囲を設定可能に）

## Artifact/Meta
- 事前学習 run_id を downstream の meta に必ず記録（upstream_artifacts）
