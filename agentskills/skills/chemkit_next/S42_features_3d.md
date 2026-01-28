# S42: 3D Features & Conformer

## 目的
- 3Dモデルに必要な pos/edge_attr を一貫して生成・保存する

## 推奨フロー
1) 入力に3Dがある（SDF等）→ それを使う
2) 無い場合：SMILES→Embed→Optimize→pos生成（失敗は扱いを明確化）
3) radius graph / distance を edge_attr に（モデル要件で切替）
4) fitが必要な変換（scaler等）は artifact として保存

## 事故りやすい点
- trainだけ3D生成し、predictで再生成→skew
- conformer失敗でsilentに欠損pos
