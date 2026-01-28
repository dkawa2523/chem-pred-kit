# S00 Repo Orientation

## Purpose
現状のリポジトリ構造・既存パイプライン・既存config/スクリプトの関係を可視化し、
後続タスクの前提を固める。

## When to Use
- devkit導入直後
- 大規模リファクタ前
- 「どこを触れば良いか」不明な時

## Inputs
- docs/00_INVARIANTS.md
- work/tasks/000_bootstrap_devkit.md（または該当タスク）

## Allowed Changes
- 原則：コードは変更しない（調査のみ）
- 例外：調査結果を `work/` や `docs/` に追記するのはOK

## Steps
1) ルートのディレクトリ一覧と役割をまとめる
2) `scripts/` の実行動線（dataset→train→predict）を図にする
3) `configs/` の現状を列挙し、どこが重複/混乱か指摘
4) “次に整理すべき P0” を 3 つ提示する

## Outputs
- 現状マップ（Markdown）
- 主要スクリプトの入出力一覧
