# Dataset Contracts（ChemKit Next）

## 1. Dataset Adapter の責務
- 外部データ（QM9/MD17/MoleculeNet/PCQM4Mv2/独自SDF/CSV 等）を
  **内部共通スキーマ**へ変換する
- 変換後のデータを「artifact」として保存し、train/eval/predict が同じ契約で扱えるようにする

## 2. 内部共通スキーマ（最小）
### 必須カラム（tabular view）
- `sample_id`（安定ID。原則：canonical_smiles + dataset_name + source_index から生成）
- `dataset_name`
- `split`（train/valid/test）
- `smiles`（canonical 推奨）
- `target.*`（複数可、例：target.gap, target.U0）
- `units.*`（任意：targetの単位を明示したい場合）
- `meta.*`（任意：source、conformer_id、trajectory_step 等）

### Graph view（GNN）
- `x`（node features）
- `edge_index`（2D graph）
- `edge_attr`（任意：距離/ボンド特徴/角度など）
- `pos`（3Dが必要なモデル用。任意だが3Dモデルでは必須）
- `u`（global descriptors。任意）
- `y`（target tensor。単一または多次元）
- `mask_y`（任意：欠損ターゲットのmask）

## 3. target registry（重要）
データセットごとに「同名でも単位や意味が違う」問題を避けるため、
内部では `target.<canonical_name>` に揃える。
例：QM9 の "gap" は `target.gap`、MD17 の energy は `target.energy` 等。

## 4. split の原則（比較可能性）
- splitは **必ずartifactとして保存**し、再現可能にする
- データセットごとに推奨splitが違う場合でも、
  `split_strategy` と `split_seed` を meta/config に保存する

## 5. ハッシュ（再現性）
- dataset_hash は「入力データ + adapter config + split」のハッシュで決まる
- featureset_hash は「feature config +（必要なら）fit state」まで含める
