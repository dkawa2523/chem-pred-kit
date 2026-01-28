# AgentSkills 運用ガイド

AgentSkills は「繰り返し発生する開発タスク」を **スキルカード** として定義し、
Codex 指示を安定化する仕組みです。

## 使い方（基本）
1) `work/tasks/NNN_*.md` を作る（タスク）
2) `agentskills/ROUTER.md` で使うスキルを選ぶ
3) Codexへ「SESSION_CONTEXT + スキルカード + タスク」をまとめて渡す

## 目標
- 指示のぶれ（スコープ逸脱、互換破壊、skew）を防ぐ
- タスクが増えても同じ型で実装できる
