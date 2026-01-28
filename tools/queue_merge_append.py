#!/usr/bin/env python3

"""Merge tasks from an append file into an existing work/queue.json.

Why: avoid overwriting an existing queue.json that already contains 000-090 tasks.
Usage:
  python tools/queue_merge_append.py --queue work/queue.json --append work/queue.append.chemkit_next.json

Behavior:
- Keeps existing tasks as-is
- Appends new tasks by id (if id doesn't exist)
- If id exists: by default, do NOT overwrite. Use --overwrite to replace that task entry.
"""

import argparse, json
from pathlib import Path

def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--append", required=True)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    qpath = Path(args.queue)
    apath = Path(args.append)

    q = load_json(qpath)
    a = load_json(apath)

    # Support both {"tasks":[...]} and {"queue":[...]} shapes
    q_tasks = q.get("tasks") or q.get("queue") or []
    a_tasks = a.get("tasks") or a.get("queue") or []

    # Build index
    idx = {t.get("id"): i for i, t in enumerate(q_tasks) if "id" in t}
    added, overwritten, skipped = 0, 0, 0

    for t in a_tasks:
        tid = t.get("id")
        if not tid:
            continue
        if tid in idx:
            if args.overwrite:
                q_tasks[idx[tid]] = t
                overwritten += 1
            else:
                skipped += 1
        else:
            q_tasks.append(t)
            idx[tid] = len(q_tasks) - 1
            added += 1

    # Put back preserving original key
    if "tasks" in q:
        q["tasks"] = q_tasks
    elif "queue" in q:
        q["queue"] = q_tasks
    else:
        q["tasks"] = q_tasks

    q.setdefault("updated_at", None)
    q["updated_at"] = a.get("generated_at")

    if args.dry_run:
        print(f"DRY RUN: added={added} overwritten={overwritten} skipped(existing)={skipped}")
        return

    qpath.write_text(json.dumps(q, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Updated {qpath}: added={added} overwritten={overwritten} skipped(existing)={skipped}")

if __name__ == "__main__":
    main()
