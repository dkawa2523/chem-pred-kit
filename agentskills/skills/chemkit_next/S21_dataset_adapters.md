# S21: ChemKit Dataset Adapters

## 目的
- 新データセット（QM9/MD17/MoleculeNet 等）を Adapter 追加で取り込めるようにする

## チェックリスト
- [ ] dataset_name が一意
- [ ] sample_id が安定（canonical_smiles + index 等）
- [ ] target registry にマッピング（target.<name>）
- [ ] split が保存され、seed/strategy が meta に残る
- [ ] dataset_hash が再現可能
- [ ] contract test を追加（最低：読み込み→1バッチ→shape確認）

## 事故りやすい点
- 同一分子の重複（データリーク）
- units変換ミス
- split漏洩（MD17は時間相関）
