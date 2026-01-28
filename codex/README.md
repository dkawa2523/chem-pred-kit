# Codex 運用ガイド

このディレクトリは「Codexへ渡す指示」を標準化するためのものです。

## 基本
Codexへの指示は原則、次の 3 点セットを貼り付けます：
1) `codex/SESSION_CONTEXT.md`
2) 使うスキルカード（`agentskills/skills/Sxx_*.md`）
3) タスク定義（`work/tasks/NNN_*.md`）

## 推奨の1ターン構造（Codexに要求する出力）
- (A) 変更計画（ファイル単位）
- (B) 実装（差分がわかる形）
- (C) 追加/更新したテスト
- (D) 検証コマンド
- (E) 互換性への影響

## 重要
- 不変条件（docs/00）を破る提案が必要なら、まず RFC を書くこと


## AUTO（タスク/スキル指定なし）
- `codex/AUTO.md` を使うと、Codexが `work/queue.json` を読み、優先度順にタスクを選びます。
- ローカルで生成したい場合は `python tools/codex_prompt.py next` を使ってください。

- Process追加: codex/prompt_templates/06_add_process.md を参照
