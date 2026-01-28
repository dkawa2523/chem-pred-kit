# Target Registry & Units（ChemKit Next）

## 目的
- 物性名の揺れ、単位の揺れ、変換（log/standardize）を **設定とartifact**で一貫管理する

## 推奨：canonical target 名（例）
### QM9（回帰）
- `target.mu`（dipole moment）
- `target.alpha`
- `target.homo`
- `target.lumo`
- `target.gap`
- `target.r2`
- `target.zpve`
- `target.U0`, `target.U`, `target.H`, `target.G`
- `target.Cv`

### MD17（回帰）
- `target.energy`
- `target.forces`（多次元：N_atoms x 3）

### MoleculeNet（回帰/分類）
- `target.esol_logS`
- `target.freesolv_G`
- `target.lipo`

## 単位と変換
- 内部は「入力単位のまま」でも良いが、**必ず meta に units を残す**
- 変換（例：log、標準化）は Feature/Target pipeline の一部として扱い、
  train/infer skew を起こさないよう fit state を保存する

## Multi-target（マルチタスク）
- `y` は shape `(batch, T)`（T=ターゲット数）
- 欠損がある場合は `mask_y` で loss/metric を制御する
