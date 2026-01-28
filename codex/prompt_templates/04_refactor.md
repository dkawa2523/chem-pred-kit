# Prompt Template: Refactor

前提：
- codex/SESSION_CONTEXT.md
- agentskills/skills/S60_trainer_loop.md（必要なら）
- work/tasks/<TASK_ID>_*.md

要求：
- 外部挙動（config/CLI/artifact）を壊さない
- 変更前後で smoke test が通ること
- 影響範囲を明確にする


注意: 変更が大きくなりそうなら Process を分割し、scripts/visualize.py のように独立させてください（docs/10参照）。
