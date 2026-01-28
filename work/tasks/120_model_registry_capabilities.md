# Task 120 (P0): Model Registry + Capability Contract

## 目的
モデルを registry 管理し、必要入力（pos/edge_attr/angles/SMILES tokens）を宣言できるようにする。

## In scope
- model registry（model.name -> constructor）
- capability flags
  - supports_3d, supports_multitask, supports_force, input_requires (pos/edge_attr/angles/tokens)
- config で選択し、比較実験を容易にする

## Acceptance Criteria
- [x] `model.name` の切替で動作する
- [x] 必要入力が無い場合に分かりやすくfailする（silentに壊れない）
