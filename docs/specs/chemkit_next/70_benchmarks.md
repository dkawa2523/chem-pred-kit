# Benchmarks（ChemKit Next）

## 対象
- QM9（量子化学物性、~134k）
- MD17（分子動力学、energy/forces）
- MoleculeNet（ESOL/FreeSolv/Lipo 等）
- （任意）PCQM4Mv2（大規模、pretrain用途）

## 推奨 split
- QM9: random split（seed固定）
- MD17: time-based split（リーク防止）
- MoleculeNet: scaffold split（構造リーク防止）

## 推奨 metrics
- 回帰：MAE/RMSE/R2（データセット慣習に合わせる）
- forces：MAE（vector）、energy：MAE
