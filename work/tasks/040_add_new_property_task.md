# Task 040 (P1): 新しい目的変数（物性）追加の枠組み

## 目的
- CSV/SDF 形式は同じまま、目的変数だけを差し替えて学習できるようにする

## Plan
1) `configs/task/<property>.yaml` を追加できる形へ
2) `src/tasks/` に Task I/F を導入（loss/metrics/target_columns）
3) 既存 LJ タスクを Task 化し、後方互換を保つ
4) サンプルとして 1つ新規物性（ダミーでも可）を追加する

## Acceptance Criteria
- [x] `task=<property>` で学習が走る
- [x] metrics が出る
