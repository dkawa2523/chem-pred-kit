# Feature Sets（ChemKit Next）

## 目的
- 特徴量を「FeatureSet」として宣言し、train/eval/predictで同一にする（skew排除）

## FeatureSet の種類
### 2D（RDKit）
- Morgan FP / MACCS / AtomPair
- グローバル記述子（MolWt, TPSA, LogP, HBD, HBA, RingCount…）
- 原子特徴（Z、degree、aromatic、formal charge…）

### 3D（幾何）
- `pos`（3D座標）
- 距離ベース edge_attr（distance / gaussian expansion）
- 角度/トーション（DimeNet++系で必要）

### 電子特徴（準3D/補助）
- Gasteiger charge（RDKit）
- （任意）QM由来の電荷・双極子（データがある場合）

## Conformer（3D座標）方針
- SDFに3Dがあるならそれを優先
- 無い場合は SMILES から生成（Embed → optimize）
- 失敗時の扱い（skip/2D fallback）を明確にし、ログとmetaへ残す
