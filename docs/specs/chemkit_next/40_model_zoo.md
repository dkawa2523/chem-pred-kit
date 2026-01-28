# Model Zoo（ChemKit Next）

## 目的
- モデル追加を「実装＋登録＋config」で完結し、比較評価を容易にする

## 2D GNN
- GCN / MPNN / GIN / Graphormer（2D Transformer）

## 3D GNN（距離・等変性）
- SchNet（距離）
- DimeNet++（角度）
- PaiNN（E(3) equivariant）
- EGNN（equivariant）

## SMILES Transformer
- ChemBERTa / MolFormer など（入力：tokenized SMILES）

## optional dependency
- 3Dモデルは依存が重くなりやすい（torch-scatter, torch-sparse, e3nn, torchmd-net 等）
- 「ベース環境では import しない」設計にする（lazy import / extras）
