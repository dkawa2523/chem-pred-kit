#!/usr/bin/env python3
"""
tools/codex_autopilot.py

Codex + tools/codex_prompt.py を自動でループ実行する “軽量オートパイロット”。

設計意図:
- VSCode UI の自動入力は壊れやすいので、Codex CLI の `codex exec` を使う。
- `codex exec` は PROMPT に "-" を渡すと stdin からプロンプトを読める（安全に長文対応）。
- 反復実行時の事故（dirty tree, doctor失敗, 同一タスク無限ループ）を減らす。

必要:
- Python 3.10+
- Codex CLI installed (`codex`)

例:
  python tools/codex_autopilot.py --max-iters 50 --full-auto
  python tools/codex_autopilot.py --max-iters 10 --sandbox workspace-write --ask-for-approval on-request
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


def find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(12):
        if (cur / "work" / "queue.json").exists() and (cur / "tools" / "codex_prompt.py").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    raise FileNotFoundError("repo root not found (expected work/queue.json, tools/codex_prompt.py)")


def run(cmd: list[str], *, cwd: Path, input_text: Optional[str] = None, capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        input=input_text,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )


def git_clean(cwd: Path) -> bool:
    cp = run(["git", "status", "--porcelain"], cwd=cwd)
    if cp.returncode != 0:
        return True  # not a git repo
    return (cp.stdout.strip() == "")


def extract_task_id(prompt_text: str) -> Optional[str]:
    m = re.search(r"(?m)^\s*-\s*id:\s*([0-9]{3,})\s*$", prompt_text)
    return m.group(1) if m else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-iters", type=int, default=20)
    ap.add_argument("--codex-bin", type=str, default="codex")
    ap.add_argument("--full-auto", action="store_true")
    ap.add_argument("--sandbox", type=str, default="")
    ap.add_argument("--ask-for-approval", type=str, default="")
    ap.add_argument("--json", action="store_true", help="use codex exec --json and save events.jsonl")
    ap.add_argument("--max-same-task-repeats", type=int, default=2)
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()

    repo_root = find_repo_root(Path(__file__).parent)

    if not args.allow_dirty and not git_clean(repo_root):
        print("ERROR: git working tree is dirty. Commit/stash changes before autopilot.")
        print("  (Override with --allow-dirty)")
        return 3

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = repo_root / "work" / ".autopilot" / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    # Build codex exec flags
    codex_flags: list[str] = []
    if args.full_auto:
        codex_flags.append("--full-auto")
    if args.sandbox:
        codex_flags += ["--sandbox", args.sandbox]
    if args.ask_for_approval:
        codex_flags += ["--ask-for-approval", args.ask_for_approval]
    if args.json:
        codex_flags.append("--json")

    same_task_count = 0
    last_task_id: Optional[str] = None

    print(f"Autopilot: repo={repo_root}")
    print(f"Autopilot: logs={run_dir}")
    print(f"Autopilot: max_iters={args.max_iters}")
    print(f"Autopilot: codex_flags={' '.join(codex_flags) or '(none)'}")

    for i in range(1, args.max_iters + 1):
        print(f"\n===== Iteration {i}/{args.max_iters} =====")

        # doctor pre-check
        pre = run([sys.executable, "tools/codex_prompt.py", "doctor"], cwd=repo_root)
        (run_dir / f"doctor_before_{i}.txt").write_text((pre.stdout or "") + "\n" + (pre.stderr or ""), encoding="utf-8")
        if pre.returncode != 0:
            print("doctor failed (before). stop.")
            return 10

        # Generate prompt (also updates queue status)
        nxt = run([sys.executable, "tools/codex_prompt.py", "next"], cwd=repo_root)
        prompt = nxt.stdout or ""
        (run_dir / f"prompt_{i}.md").write_text(prompt, encoding="utf-8")

        if "No tasks available" in prompt:
            print("No tasks available. stop.")
            return 0

        task_id = extract_task_id(prompt) or "unknown"
        print(f"Selected task: {task_id}")

        if task_id == last_task_id:
            same_task_count += 1
        else:
            same_task_count = 0
        last_task_id = task_id

        if same_task_count >= args.max_same_task_repeats:
            print(f"Same task repeated {same_task_count+1} times. stop to avoid infinite loop.")
            return 20

        # Run codex exec with stdin prompt by passing "-" as PROMPT.
        last_msg_path = run_dir / f"codex_last_{i}.md"
        stderr_path = run_dir / f"codex_stderr_{i}.log"
        stdout_path = run_dir / (f"codex_events_{i}.jsonl" if args.json else f"codex_stdout_{i}.txt")

        cmd = [args.codex_bin, "exec"] + codex_flags + ["-", "--output-last-message", str(last_msg_path)]
        cp = subprocess.run(
            cmd,
            cwd=str(repo_root),
            input=prompt,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout_path.write_text(cp.stdout or "", encoding="utf-8")
        stderr_path.write_text(cp.stderr or "", encoding="utf-8")

        print(f"codex exit code: {cp.returncode}")
        if cp.returncode != 0:
            print("Codex failed. check logs.")
            return cp.returncode

        # doctor post-check
        post = run([sys.executable, "tools/codex_prompt.py", "doctor"], cwd=repo_root)
        (run_dir / f"doctor_after_{i}.txt").write_text((post.stdout or "") + "\n" + (post.stderr or ""), encoding="utf-8")
        if post.returncode != 0:
            print("doctor failed (after). stop.")
            return 11

        # git diff stat
        if (repo_root / ".git").exists():
            diff = run(["git", "diff", "--stat"], cwd=repo_root)
            (run_dir / f"git_diff_stat_{i}.txt").write_text(diff.stdout or "", encoding="utf-8")

        print("Iteration done.")

    print("Reached max iters. stop.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
