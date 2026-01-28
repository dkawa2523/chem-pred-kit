#!/usr/bin/env python3
"""
tools/codex_prompt.py

work/queue.json から「次にやるべきタスク」を自動選択し、Codex に貼る統合プロンプトを生成します。

v3.7 追加/強化:
- `--mode autopilot` を追加（非対話・自動ループ前提）
  - ユーザー確認を求めない（確認質問は禁止）
  - queue.json の status を single source of truth として扱う
  - task md の "Blocked" 記述は参考情報。queue が todo/in_progress なら実装を進める。
  - どうしても進められない場合は queue を blocked に更新し、解除用子タスクを起票（unblocks付与）

- 依存関係:
  - `depends_on` があるタスクは依存が done になるまで選ばない（順序事故防止）

- blocked:
  - blocked 親タスクがあれば解除タスク（queue.jsonの unblocks / mdのBlock記述）を優先提示
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional


PRIORITY_RANK = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
VALID_STATUS = {"todo", "in_progress", "blocked", "done"}
VALID_MODE = {"interactive", "autopilot"}


# -------------------------
# IO helpers
# -------------------------
def find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(12):
        if (cur / "work" / "queue.json").exists() and (cur / "codex").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    raise FileNotFoundError("repo root not found (expected work/queue.json and codex/)")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_text_if_exists(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def normalize_task_id(task_id: Any) -> str:
    s = str(task_id).strip()
    if re.fullmatch(r"\d+", s):
        return s.zfill(3)
    return s


def now_utc_iso() -> str:
    import datetime
    return datetime.datetime.utcnow().isoformat() + "Z"


# -------------------------
# Task indexing / sorting
# -------------------------
def index_tasks(queue: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    id2task: Dict[str, Dict[str, Any]] = {}
    for t in queue.get("tasks", []):
        tid = normalize_task_id(t.get("id"))
        id2task[tid] = t
    return id2task


def task_sort_key(task: Dict[str, Any]) -> Tuple[int, int]:
    """priority -> id の順（同priority内で昇順）。"""
    pri = PRIORITY_RANK.get(task.get("priority", "P9"), 99)
    tid = normalize_task_id(task.get("id", "999"))
    try:
        tid_int = int(tid) if re.fullmatch(r"\d+", tid) else 999
    except ValueError:
        tid_int = 999
    return pri, tid_int


# -------------------------
# Dependency / unblock resolution
# -------------------------
def _as_id_list(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, str):
        parts = re.split(r"[,\s]+", v.strip())
        return [normalize_task_id(p) for p in parts if p]
    if isinstance(v, list):
        out = []
        for x in v:
            if x is None:
                continue
            out.append(normalize_task_id(x))
        return out
    return []


def deps_satisfied(task: Dict[str, Any], id2task: Dict[str, Dict[str, Any]]) -> bool:
    deps = _as_id_list(task.get("depends_on"))
    if not deps:
        return True
    for dep_id in deps:
        dep = id2task.get(dep_id)
        if not dep:
            return False
        if str(dep.get("status")) != "done":
            return False
    return True


def find_unblockers_by_field(parent_id: str, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """queue.json の `unblocks: ["010"]` を最優先で見る。"""
    hits: List[Dict[str, Any]] = []
    for t in tasks:
        st = str(t.get("status", "todo"))
        if st not in ("todo", "in_progress"):
            continue
        unblocks = _as_id_list(t.get("unblocks"))
        if parent_id in unblocks:
            hits.append(t)
    hits.sort(key=task_sort_key)
    hits.sort(key=lambda x: 0 if str(x.get("status")) == "in_progress" else 1)
    return hits


def extract_blocked_section(md: str) -> str:
    if not md:
        return ""
    m = re.search(r"(?im)^\s*##\s+Blocked\s*$", md) or re.search(r"(?im)^\s*##\s+ブロック\s*$", md)
    if not m:
        return ""
    start = m.end()
    tail = md[start:]
    m2 = re.search(r"(?im)^\s*##\s+", tail)
    if m2:
        return tail[: m2.start()].strip()
    return tail.strip()


def parse_unblock_task_ids_from_text(text: str, parent_id: str) -> List[str]:
    """md の Blocked セクションから解除タスクIDをヒューリスティックに拾う。"""
    if not text:
        return []
    scores: Dict[str, int] = {}

    # unblock_tasks: [011, 012]
    for group in re.findall(r"(?im)^\s*-\s*unblock_tasks\s*:\s*\[([^\]]+)\]\s*$", text):
        for tid in re.findall(r"\b(\d{1,4})\b", group):
            tidn = normalize_task_id(tid)
            if tidn != parent_id:
                scores[tidn] = scores.get(tidn, 0) + 10

    # 011_xxx.md
    for fn in re.findall(r"\b(\d{3}_[A-Za-z0-9_\-]+\.md)\b", text):
        tidn = normalize_task_id(fn.split("_", 1)[0])
        if tidn != parent_id:
            scores[tidn] = scores.get(tidn, 0) + 6

    # "unblock 011" or "解除 011"
    for tid in re.findall(r"(?i)\bunblock\b[^\n]*?\b(\d{1,4})\b", text):
        tidn = normalize_task_id(tid)
        if tidn != parent_id:
            scores[tidn] = scores.get(tidn, 0) + 8
    for tid in re.findall(r"解除[^\n]*?\b(\d{1,4})\b", text):
        tidn = normalize_task_id(tid)
        if tidn != parent_id:
            scores[tidn] = scores.get(tidn, 0) + 8

    # fallback: any 3-digit id
    for tid in re.findall(r"\b(\d{3})\b", text):
        tidn = normalize_task_id(tid)
        if tidn != parent_id:
            scores[tidn] = scores.get(tidn, 0) + 1

    candidates = [k for k, v in scores.items() if re.fullmatch(r"\d{3}", k) and v > 0]
    candidates.sort(key=lambda x: (-scores[x], x))
    return candidates


def resolve_unblocker_task(repo_root: Path, queue: Dict[str, Any], parent_task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = queue.get("tasks", [])
    id2task = index_tasks(queue)
    parent_id = normalize_task_id(parent_task.get("id"))

    hits = find_unblockers_by_field(parent_id, tasks)
    if hits:
        return hits[0]

    md_path = repo_root / str(parent_task.get("path", ""))
    md = _read_text_if_exists(md_path)
    blocked_text = extract_blocked_section(md)
    candidates = parse_unblock_task_ids_from_text(blocked_text, parent_id)
    live: List[Dict[str, Any]] = []
    for cid in candidates:
        t = id2task.get(cid)
        if t and str(t.get("status")) in ("todo", "in_progress"):
            live.append(t)
    if live:
        live.sort(key=task_sort_key)
        live.sort(key=lambda x: 0 if str(x.get("status")) == "in_progress" else 1)
        return live[0]

    # search other task md for "Unblocks: parent_id"
    hint_patterns = [
        re.compile(rf"(?im)^\s*Unblocks\s*:\s*{re.escape(parent_id)}\s*$"),
        re.compile(rf"(?im)^\s*-\s*Unblocks\s*:\s*{re.escape(parent_id)}\s*$"),
        re.compile(rf"(?i)\bunblock(?:s|ing)?\b[^\n]*?\b{re.escape(parent_id)}\b"),
        re.compile(rf"解除[^\n]*?\b{re.escape(parent_id)}\b"),
    ]
    hits2 = []
    for t in tasks:
        st = str(t.get("status"))
        if st not in ("todo", "in_progress"):
            continue
        p = repo_root / str(t.get("path", ""))
        txt = _read_text_if_exists(p)
        if txt and any(pat.search(txt) for pat in hint_patterns):
            hits2.append(t)
    if hits2:
        hits2.sort(key=task_sort_key)
        hits2.sort(key=lambda x: 0 if str(x.get("status")) == "in_progress" else 1)
        return hits2[0]

    return None


# -------------------------
# Selection
# -------------------------
def pick_next_task(repo_root: Path, queue: Dict[str, Any], *, skip_blocked: bool = False) -> Optional[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = queue.get("tasks", [])
    id2task = index_tasks(queue)

    # 1) WIP first
    inprog = [t for t in tasks if str(t.get("status")) == "in_progress"]
    if inprog:
        inprog.sort(key=lambda t: (t.get("started_at") or "9999",) + task_sort_key(t))
        return inprog[0]

    # 2) blocked (but prefer unblockers)
    blocked = [t for t in tasks if str(t.get("status")) == "blocked"]
    if blocked and not skip_blocked:
        blocked.sort(key=task_sort_key)
        for parent in blocked:
            unblocker = resolve_unblocker_task(repo_root, queue, parent)
            if unblocker is not None:
                return unblocker
        return blocked[0]

    # 3) todo (dependency-aware)
    todo = [t for t in tasks if str(t.get("status")) == "todo"]
    if not todo:
        return None

    eligible = [t for t in todo if deps_satisfied(t, id2task)]
    if eligible:
        eligible.sort(key=task_sort_key)
        return eligible[0]

    # If none eligible, pull an unmet dependency task
    unmet: List[Dict[str, Any]] = []
    for t in todo:
        for dep_id in _as_id_list(t.get("depends_on")):
            dep = id2task.get(dep_id)
            if dep and str(dep.get("status")) != "done":
                unmet.append(dep)

    if unmet:
        uniq = {normalize_task_id(u.get("id")): u for u in unmet}
        unmet = list(uniq.values())

        # in_progress > blocked > todo
        inprog2 = [t for t in unmet if str(t.get("status")) == "in_progress"]
        if inprog2:
            inprog2.sort(key=lambda t: (t.get("started_at") or "9999",) + task_sort_key(t))
            return inprog2[0]

        if not skip_blocked:
            blocked2 = [t for t in unmet if str(t.get("status")) == "blocked"]
            if blocked2:
                blocked2.sort(key=task_sort_key)
                for parent in blocked2:
                    unblocker = resolve_unblocker_task(repo_root, queue, parent)
                    if unblocker is not None:
                        return unblocker
                return blocked2[0]

        todo2 = [t for t in unmet if str(t.get("status")) == "todo"]
        if todo2:
            todo2.sort(key=task_sort_key)
            return todo2[0]

    todo.sort(key=task_sort_key)
    return todo[0]


# -------------------------
# Prompt building
# -------------------------
def get_skill_paths(repo_root: Path, skill_ids: List[str]) -> List[Path]:
    reg_path = repo_root / "agentskills" / "skill_registry.json"
    if not reg_path.exists():
        return []
    reg = load_json(reg_path)
    id2path = {s.get("id"): s.get("path") for s in reg.get("skills", [])}
    out: List[Path] = []
    for sid in skill_ids:
        p = id2path.get(sid)
        if p:
            out.append(repo_root / p)
    return out


def autopilot_preamble() -> str:
    return (
        "# ===== AUTOPILOT MODE (non-interactive) =====\n"
        "あなたは自動実行ループの中で動いています。**停止しないための規約**として次を厳守してください：\n"
        "\n"
        "【禁止】\n"
        "- ユーザーへの確認・質問・承認依頼（\"確認してください\" / \"進めてよいですか\" / \"Approve\" 等）は一切しない。\n"
        "- \"今回は未実装\" / \"差分なし\" のまま終了しない（必ず前進する）。\n"
        "\n"
        "【single source of truth】\n"
        "- タスクの status の唯一の真実は work/queue.json。task.md の 'Blocked' 記述は参考情報。\n"
        "  - queue が todo/in_progress なら実装を進める（task.md 側の Blocked を理由に止めない）。\n"
        "  - queue が blocked なら blocked-handling を行う（reason/unblock_condition/next_action を整備し、解除子タスクを起票）。\n"
        "\n"
        "【stale Blocked の扱い】\n"
        "- task.md に '## Blocked' が残っていても queue が todo/in_progress の場合、それは **stale**。\n"
        "  - task.md を更新し、Blocked を Notes/History に移す or 解除した上で実装を進める。\n"
        "\n"
        "【書き込み権限】\n"
        "- **書き込みは可能**な前提。work/queue.json 更新を含む必要な変更は実施する。\n"
        "  - もし実際にツール/環境エラーで書き込めない場合：\n"
        "    1) エラーメッセージを短く引用\n"
        "    2) 最小の対処（sandbox/approval/flags 見直し、権限確認）を提示\n"
        "    3) タスクを blocked にして解除子タスクを起票（子に unblocks を付与）\n"
        "    ※それでも『確認してください』は書かない。\n"
        "\n"
        "【必ずどれかを実施（Decision Protocol）】\n"
        "A) 実装を進める → テスト/検証 → work/queue.json を done に更新\n"
        "B) 進められない → work/queue.json を blocked に更新し、解除子タスク（unblocks付き）を作成\n"
        "C) 依存/状態がズレている → work/queue.json と task.md を修正して整合させた上で A へ\n"
    )


def build_prompt(repo_root: Path, task: Dict[str, Any], *, note: str = "", mode: str = "interactive") -> str:
    if mode not in VALID_MODE:
        mode = "interactive"

    session = _read_text_if_exists(repo_root / "codex" / "SESSION_CONTEXT.md").strip()

    task_id = normalize_task_id(task.get("id"))
    task_title = str(task.get("title", "")).strip()
    task_priority = str(task.get("priority", "")).strip()
    task_path = str(task.get("path", "")).strip()

    contract_paths = task.get("contracts") or ["docs/00_INVARIANTS.md"]
    contracts_list = "\n".join([f"- {p}" for p in contract_paths])

    skill_ids = task.get("skills") or []
    skill_paths = [str(p.relative_to(repo_root)) for p in get_skill_paths(repo_root, list(skill_ids))]
    skills_list = "\n".join([f"- {p}" for p in skill_paths]) if skill_paths else "- (auto) agentskills/ROUTER.md を参照"

    md_path = repo_root / task_path if task_path else None
    task_text = (_read_text_if_exists(md_path) if md_path else "").strip()

    template_path = repo_root / "work" / "templates" / "TASK.md"
    template_text = _read_text_if_exists(template_path).strip()
    if not task_text:
        missing_note = (
            "## ⚠️ Task file is missing or empty\n"
            f"- expected: `{task_path}`\n"
            "- まずこのファイルを作成/記入してから実装に着手してください。\n"
            "- 下のテンプレをベースに **このタスクに合わせて具体化**してください。\n\n"
        )
        task_text = missing_note + (template_text or "(TASK template missing)")

    status = str(task.get("status", "todo"))
    extra_mode = ""
    if status == "blocked":
        extra_mode = (
            "\n# ===== BLOCKED MODE (mandatory) =====\n"
            "このタスクは queue.json 上で blocked です。Codex は次を必ず実施：\n"
            "1) task.md の `## Blocked` に reason / unblock_condition / next_action を明記\n"
            "2) unblock 用の子タスクを作る場合：\n"
            "   - work/tasks/NNN_*.md を追加\n"
            "   - work/queue.json に追加（**unblocks: [\"<parent_id>\"]** を必ず付ける）\n"
            "3) 依存が解消して進められるなら queue.json の status を todo/in_progress へ戻し、そのまま実装を進める\n"
        )

    autopilot = ""
    if mode == "autopilot":
        autopilot = autopilot_preamble() + "\n"

    prompt = f"""# ===== SESSION CONTEXT =====
{session}

{autopilot}# ===== SELECTED TASK =====
- id: {task_id}
- title: {task_title}
- priority: {task_priority}
- status: {status}
- path: {task_path}

# ===== CONTRACTS TO FOLLOW (open & read) =====
{contracts_list}

# ===== SKILLS TO FOLLOW (open & follow) =====
{skills_list}

{note}{extra_mode}
# ===== TASK FILE (single source of truth for *intent*) =====
{task_text}

# ===== OUTPUT REQUIREMENTS (mandatory) =====
1) 変更計画（ファイル単位）
2) 実装（差分が分かるように）
3) 追加/更新したテスト
4) 検証コマンド
5) 互換性影響（config/CLI/artifact）
6) タスク完了時：work/queue.json の status を done に更新（満たせない場合は blocked と理由）
"""
    return prompt


# -------------------------
# Commands
# -------------------------
def cmd_list(repo_root: Path) -> int:
    queue = load_json(repo_root / "work" / "queue.json")
    tasks = sorted(queue.get("tasks", []), key=task_sort_key)
    print("id   pri  status         title")
    print("---- ---- -------------- ------------------------------")
    for t in tasks:
        tid = normalize_task_id(t.get("id"))
        print(f"{tid:>4} {t.get('priority',''):>4} {t.get('status',''):>14} {t.get('title','')}")
    print("\n(next order) in_progress -> blocked(unblockers first) -> todo(dep-aware)")
    return 0


def cmd_doctor(repo_root: Path) -> int:
    queue_path = repo_root / "work" / "queue.json"
    queue = load_json(queue_path)
    tasks: List[Dict[str, Any]] = queue.get("tasks", [])
    id2task = index_tasks(queue)

    issues = 0

    # duplicate ids
    seen = {}
    for t in tasks:
        tid = normalize_task_id(t.get("id"))
        seen.setdefault(tid, 0)
        seen[tid] += 1
    dups = {k: v for k, v in seen.items() if v > 1}
    if dups:
        issues += len(dups)
        print("❌ Duplicate task IDs (normalized):")
        for k, v in sorted(dups.items()):
            print(f"  - {k}: {v} entries")

    # invalid status
    bad_status = []
    for t in tasks:
        st = str(t.get("status", "todo"))
        if st not in VALID_STATUS:
            bad_status.append((normalize_task_id(t.get("id")), st))
    if bad_status:
        issues += len(bad_status)
        print("❌ Invalid status values:")
        for tid, st in bad_status:
            print(f"  - {tid}: {st} (expected one of {sorted(VALID_STATUS)})")

    # missing task files (warning)
    missing = []
    for t in tasks:
        p = str(t.get("path", "")).strip()
        if not p:
            missing.append((normalize_task_id(t.get("id")), "(no path)"))
            continue
        if not (repo_root / p).exists():
            missing.append((normalize_task_id(t.get("id")), p))
    if missing:
        print("⚠️ Missing task markdown files (will be auto-templated by next):")
        for tid, p in missing:
            print(f"  - {tid}: {p}")

    # blocked tasks with no unblockers (error) — keep strict by default
    blocked = [t for t in tasks if str(t.get("status")) == "blocked"]
    if blocked:
        print("\nBlocked analysis:")
        for parent in sorted(blocked, key=task_sort_key):
            pid = normalize_task_id(parent.get("id"))
            unblocker = resolve_unblocker_task(repo_root, queue, parent)
            if unblocker is None:
                issues += 1
                print(f"  ❌ {pid} has no unblocker task detected. Add one with queue.json field `unblocks: [\"{pid}\"]`.")
            else:
                uid = normalize_task_id(unblocker.get("id"))
                print(f"  ✅ {pid} -> unblocker candidate: {uid} ({unblocker.get('status')})")

    # dependency sanity
    dep_issues = 0
    for t in tasks:
        deps = _as_id_list(t.get("depends_on"))
        for dep in deps:
            if dep not in id2task:
                dep_issues += 1
                print(f"❌ {normalize_task_id(t.get('id'))} depends_on missing task id: {dep}")
    issues += dep_issues

    if issues == 0:
        print("✅ doctor: no issues found.")
        return 0
    print(f"\ndoctor: found {issues} issue(s).")
    return 2


def cmd_next(repo_root: Path, *, dry_run: bool, skip_blocked: bool, mode: str) -> int:
    queue_path = repo_root / "work" / "queue.json"
    queue = load_json(queue_path)
    task = pick_next_task(repo_root, queue, skip_blocked=skip_blocked)

    if task is None:
        print("# No tasks available.")
        return 0

    if not dry_run:
        st = str(task.get("status", "todo"))
        if st == "todo" and queue.get("policy", {}).get("auto_set_in_progress_on_next", True):
            task["status"] = "in_progress"
            task["started_at"] = task.get("started_at") or now_utc_iso()
        task["last_presented_at"] = now_utc_iso()
        queue["updated_at"] = now_utc_iso()
        save_json(queue_path, queue)

    note = ""
    id2task = index_tasks(queue)
    if str(task.get("status")) in ("todo", "in_progress") and not deps_satisfied(task, id2task):
        note = (
            "# ===== NOTE =====\n"
            "⚠️ このタスクは `depends_on` が未完了の可能性があります。\n"
            "まず依存タスクを done にするか、依存関係を見直してください。\n\n"
        )

    prompt = build_prompt(repo_root, task, note=note, mode=mode)
    print(prompt)
    return 0


def cmd_done(repo_root: Path, task_id: str) -> int:
    queue_path = repo_root / "work" / "queue.json"
    queue = load_json(queue_path)
    tid = normalize_task_id(task_id)

    found = False
    for t in queue.get("tasks", []):
        if normalize_task_id(t.get("id")) == tid:
            t["status"] = "done"
            t["done_at"] = now_utc_iso()
            found = True
    if not found:
        print(f"Task id {tid} not found")
        return 1

    queue["updated_at"] = now_utc_iso()
    save_json(queue_path, queue)
    print(f"Marked {tid} as done")
    return 0


def cmd_set(repo_root: Path, task_id: str, new_status: str) -> int:
    new_status = new_status.strip()
    if new_status not in VALID_STATUS:
        print(f"Invalid status: {new_status} (expected one of {sorted(VALID_STATUS)})")
        return 2

    queue_path = repo_root / "work" / "queue.json"
    queue = load_json(queue_path)
    tid = normalize_task_id(task_id)

    found = False
    for t in queue.get("tasks", []):
        if normalize_task_id(t.get("id")) == tid:
            t["status"] = new_status
            if new_status == "in_progress":
                t["started_at"] = t.get("started_at") or now_utc_iso()
            found = True
    if not found:
        print(f"Task id {tid} not found")
        return 1

    queue["updated_at"] = now_utc_iso()
    save_json(queue_path, queue)
    print(f"Set {tid} -> {new_status}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "next", "done", "set", "doctor"])
    ap.add_argument("task_id", nargs="?")
    ap.add_argument("status", nargs="?")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-blocked", action="store_true", help="skip blocked tasks (exception)")
    ap.add_argument("--mode", choices=sorted(VALID_MODE), default=os.environ.get("CODEX_PROMPT_MODE", "interactive"))

    args = ap.parse_args()
    repo_root = find_repo_root(Path(__file__).parent)

    if args.cmd == "list":
        return cmd_list(repo_root)
    if args.cmd == "doctor":
        return cmd_doctor(repo_root)
    if args.cmd == "next":
        return cmd_next(repo_root, dry_run=args.dry_run, skip_blocked=args.skip_blocked, mode=args.mode)
    if args.cmd == "done":
        if not args.task_id:
            raise SystemExit("done requires task_id")
        return cmd_done(repo_root, args.task_id)
    if args.cmd == "set":
        if not args.task_id or not args.status:
            raise SystemExit("set requires task_id and status")
        return cmd_set(repo_root, args.task_id, args.status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
