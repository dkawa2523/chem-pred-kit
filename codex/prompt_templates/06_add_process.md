# Prompt Template: Add Process

前提：
- codex/SESSION_CONTEXT.md
- docs/10_PROCESS_CATALOG.md
- agentskills/skills/S10_hydra_config.md（必要ならS80/S70など）
- work/tasks/<TASK_ID>_*.md

要求：
1) 新ProcessのI/O（入力artifact・出力artifact）を最初に整理
2) scripts/<process>.py を追加（1 script = 1 process）
3) configs/process/<process>.yaml を追加（入口config）
4) 最小smoke test を追加
5) docs/10_PROCESS_CATALOG.md を更新
