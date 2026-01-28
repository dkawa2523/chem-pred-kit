#!/usr/bin/env python3

"""Merge skills from an append file into an existing skill_registry.json.

Usage:
  python tools/skills_merge_append.py --registry work/agentskills/skill_registry.json --append work/agentskills/skill_registry.append.chemkit_next.json

- Adds new skills by id
- Does not overwrite existing by default; use --overwrite
"""

import argparse, json
from pathlib import Path

def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", required=True)
    ap.add_argument("--append", required=True)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rpath = Path(args.registry)
    apath = Path(args.append)
    r = load_json(rpath)
    a = load_json(apath)

    r_skills = r.get("skills", [])
    a_skills = a.get("skills", [])

    idx = {s.get("id"): i for i, s in enumerate(r_skills) if "id" in s}
    added, overwritten, skipped = 0, 0, 0

    for s in a_skills:
        sid = s.get("id")
        if not sid:
            continue
        if sid in idx:
            if args.overwrite:
                r_skills[idx[sid]] = s
                overwritten += 1
            else:
                skipped += 1
        else:
            r_skills.append(s)
            idx[sid] = len(r_skills) - 1
            added += 1

    r["skills"] = r_skills

    if args.dry_run:
        print(f"DRY RUN: added={added} overwritten={overwritten} skipped(existing)={skipped}")
        return

    rpath.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Updated {rpath}: added={added} overwritten={overwritten} skipped(existing)={skipped}")

if __name__ == "__main__":
    main()
