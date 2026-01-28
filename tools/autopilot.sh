#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Autopilot: work/queue.json 駆動で Codex を複数回回すループ
#
# v3.7: “止まりやすい要因” を深めに潰す
# - doctor 失敗は自動修復を試す（AUTO_FIX_DOCTOR）
# - “確認お願いします/未実装/差分なし” 等の停止文言を検出し、強制前進プロンプトを追加で流す
# - 同一タスク/無変更が続く場合に停止（無限ループ防止）
# - Codex CLI のフラグ差分（global/exec）を安全側に正規化しつつログへ残す
#
# Usage:
#   ./tools/autopilot.sh [MAX_ITERS]
#
# Env:
#   MAX_ITERS                : 最大反復回数（引数でも指定可）
#   AUTO_FIX_DOCTOR          : doctor 失敗を Codex で自動修復する (default 1)
#   STOP_WHEN_ONLY_BLOCKED   : blocked だけ残ったら停止 (default 1)
#   FORCE_PROGRESS_ON_STALL  : “確認/未実装/差分なし” を検出したら追加プロンプトで前進 (default 1)
#   RETRY_CODEX_ON_ERROR     : Codex 実行失敗を何回リトライするか (default 1)
#   MAX_SAME_TASK            : 同一タスク連続回数の上限 (default 3)
#   MAX_NO_CHANGE            : 無変更（queue hash + git diff）連続回数の上限 (default 2)
#   CODEX_GLOBAL_FLAGS       : codex の global flags（例: "--ask-for-approval never"）
#   CODEX_EXEC_FLAGS         : codex exec の flags（例: "--sandbox workspace-write"）
#   CODEX_FLAGS              : 旧形式（global/exec混在）※可能な範囲で自動分離
#   CODEX_JSON               : codex exec --json を保存 (default 0; 対応版のみ)
#   REQUIRE_CLEAN_GIT        : dirty working tree なら停止 (default 0)
#   AUTO_GIT_COMMIT          : 反復ごとに自動コミット (default 0)
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

MAX_ITERS="${1:-${MAX_ITERS:-30}}"
AUTO_FIX_DOCTOR="${AUTO_FIX_DOCTOR:-1}"
STOP_WHEN_ONLY_BLOCKED="${STOP_WHEN_ONLY_BLOCKED:-1}"
FORCE_PROGRESS_ON_STALL="${FORCE_PROGRESS_ON_STALL:-1}"
RETRY_CODEX_ON_ERROR="${RETRY_CODEX_ON_ERROR:-1}"
MAX_SAME_TASK="${MAX_SAME_TASK:-3}"
MAX_NO_CHANGE="${MAX_NO_CHANGE:-2}"
CODEX_JSON="${CODEX_JSON:-0}"
REQUIRE_CLEAN_GIT="${REQUIRE_CLEAN_GIT:-0}"
AUTO_GIT_COMMIT="${AUTO_GIT_COMMIT:-0}"

if [[ ! -f "work/queue.json" ]]; then
  echo "ERROR: work/queue.json not found."
  exit 2
fi

if ! command -v codex >/dev/null 2>&1; then
  echo "ERROR: codex command not found in PATH."
  exit 2
fi

# -------------------------
# Logging dir
# -------------------------
RUN_DIR="work/.autopilot/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN_DIR"

echo "LOGDIR=$REPO_ROOT/$RUN_DIR"
echo "MAX_ITERS=$MAX_ITERS"
echo "AUTO_FIX_DOCTOR=$AUTO_FIX_DOCTOR"
echo "STOP_WHEN_ONLY_BLOCKED=$STOP_WHEN_ONLY_BLOCKED"
echo "FORCE_PROGRESS_ON_STALL=$FORCE_PROGRESS_ON_STALL"
echo "RETRY_CODEX_ON_ERROR=$RETRY_CODEX_ON_ERROR"
echo "MAX_SAME_TASK=$MAX_SAME_TASK"
echo "MAX_NO_CHANGE=$MAX_NO_CHANGE"
echo "CODEX_JSON=$CODEX_JSON"
echo "REQUIRE_CLEAN_GIT=$REQUIRE_CLEAN_GIT"
echo "AUTO_GIT_COMMIT=$AUTO_GIT_COMMIT"

# Save basic environment info + preflight (fail fast with actionable logs)
set +e
codex --version > "$RUN_DIR/codex_version.txt" 2>&1
RC_VER=$?
codex --help > "$RUN_DIR/codex_help.txt" 2>&1
RC_HELP=$?
codex exec --help > "$RUN_DIR/codex_exec_help.txt" 2>&1
RC_EXEC_HELP=$?
set -e

if [[ "$RC_VER" -ne 0 || "$RC_HELP" -ne 0 || "$RC_EXEC_HELP" -ne 0 ]]; then
  echo "ERROR: codex CLI preflight failed."
  echo "See logs:"
  echo "  - $REPO_ROOT/$RUN_DIR/codex_version.txt"
  echo "  - $REPO_ROOT/$RUN_DIR/codex_help.txt"
  echo "  - $REPO_ROOT/$RUN_DIR/codex_exec_help.txt"

  if grep -qiE "model_reasoning_effort|unknown variant|Failed to deserialize" "$RUN_DIR/codex_version.txt" "$RUN_DIR/codex_help.txt" 2>/dev/null; then
    echo ""
    echo "Hint: ~/.codex/config.toml の model_reasoning_effort が CLI と不整合の可能性があります。"
    echo "  - 例: xhigh が認識されず high しか受け付けない等"
    echo "  - 対処: model_reasoning_effort=\"high\" へ変更、または codex CLI を更新"
  fi
  if grep -qiE "OPENAI_API_KEY|api key|not authenticated|login" "$RUN_DIR/codex_version.txt" "$RUN_DIR/codex_help.txt" 2>/dev/null; then
    echo ""
    echo "Hint: 認証/環境変数が不足している可能性があります。"
    echo "  - OPENAI_API_KEY を設定するか、codex login が必要です。"
  fi

  exit 2
fi

# -------------------------
# Optional: git guard
# -------------------------
IN_GIT=0
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  IN_GIT=1
  if [[ "$REQUIRE_CLEAN_GIT" -eq 1 ]]; then
    if [[ -n "$(git status --porcelain)" ]]; then
      echo "ERROR: working tree is dirty (REQUIRE_CLEAN_GIT=1)."
      git status --porcelain > "$RUN_DIR/git_status_initial.txt" || true
      exit 2
    fi
  fi
fi

# -------------------------
# Local write probe (OS permission sanity)
# -------------------------
WRITE_PROBE_PATH="$RUN_DIR/_write_probe.txt"
if ! printf "ok\n" > "$WRITE_PROBE_PATH" 2>/dev/null; then
  echo "ERROR: local write probe failed (OS permission)."
  exit 2
fi

# -------------------------
# Flag resolution / normalization (robust)
# -------------------------
CODEX_FLAGS="${CODEX_FLAGS:-}"
CODEX_GLOBAL_FLAGS="${CODEX_GLOBAL_FLAGS:-}"
CODEX_EXEC_FLAGS="${CODEX_EXEC_FLAGS:-}"

# Backward compatibility: split CODEX_FLAGS into global/exec if explicit ones are not provided.
if [[ -z "$CODEX_GLOBAL_FLAGS" && -z "$CODEX_EXEC_FLAGS" && -n "$CODEX_FLAGS" ]]; then
  TOKENS=($CODEX_FLAGS)
  GLOBAL=()
  EXEC=()
  i=0
  while [[ $i -lt ${#TOKENS[@]} ]]; do
    t="${TOKENS[$i]}"
    # Heuristic: ask-for-approval is global
    if [[ "$t" == "--ask-for-approval" ]]; then
      GLOBAL+=("$t")
      ((i+=1))
      if [[ $i -lt ${#TOKENS[@]} ]]; then
        GLOBAL+=("${TOKENS[$i]}")
      fi
    else
      EXEC+=("$t")
    fi
    ((i+=1))
  done
  CODEX_GLOBAL_FLAGS="${GLOBAL[*]}"
  CODEX_EXEC_FLAGS="${EXEC[*]}"
fi

# Feature detect supported flags to avoid version mismatch
HAS_ASK_FOR_APPROVAL=0
HAS_FULL_AUTO=0
HAS_OUTPUT_LAST_MESSAGE=0
HAS_JSON_FLAG=0
HAS_SANDBOX_IN_EXEC=0
HAS_SANDBOX_IN_GLOBAL=0

if codex --help 2>/dev/null | grep -q "ask-for-approval"; then
  HAS_ASK_FOR_APPROVAL=1
fi
if codex --help 2>/dev/null | grep -q "full-auto"; then
  HAS_FULL_AUTO=1
fi
if codex exec --help 2>/dev/null | grep -q "output-last-message"; then
  HAS_OUTPUT_LAST_MESSAGE=1
fi
if codex exec --help 2>/dev/null | grep -qE "\s--json(\s|$)"; then
  HAS_JSON_FLAG=1
fi
if codex exec --help 2>/dev/null | grep -qE "\s--sandbox(\s|$)"; then
  HAS_SANDBOX_IN_EXEC=1
fi
if codex --help 2>/dev/null | grep -qE "\s--sandbox(\s|$)"; then
  HAS_SANDBOX_IN_GLOBAL=1
fi

# Provide safe defaults if not set
if [[ -z "$CODEX_GLOBAL_FLAGS" ]]; then
  if [[ "$HAS_ASK_FOR_APPROVAL" -eq 1 ]]; then
    CODEX_GLOBAL_FLAGS="--ask-for-approval never"
  elif [[ "$HAS_FULL_AUTO" -eq 1 ]]; then
    # fallback (may still ask on failure depending on implementation)
    CODEX_GLOBAL_FLAGS="--full-auto"
  else
    CODEX_GLOBAL_FLAGS=""
  fi
fi

if [[ -z "$CODEX_EXEC_FLAGS" ]]; then
  if [[ "$HAS_SANDBOX_IN_EXEC" -eq 1 ]]; then
    CODEX_EXEC_FLAGS="--sandbox workspace-write"
  elif [[ "$HAS_SANDBOX_IN_GLOBAL" -eq 1 ]]; then
    # will be moved to global normalization below
    CODEX_EXEC_FLAGS="--sandbox workspace-write"
  else
    CODEX_EXEC_FLAGS=""
  fi
fi

# Normalize sandbox placement if needed
# - If exec doesn't support --sandbox but global does, move sandbox tokens from exec->global.
if [[ "$HAS_SANDBOX_IN_EXEC" -eq 0 && "$HAS_SANDBOX_IN_GLOBAL" -eq 1 ]]; then
  # move "--sandbox X" from EXEC_FLAGS into GLOBAL_FLAGS
  EXEC_TOK=($CODEX_EXEC_FLAGS)
  NEW_EXEC=()
  SANDBOX_PAIR=()
  j=0
  while [[ $j -lt ${#EXEC_TOK[@]} ]]; do
    if [[ "${EXEC_TOK[$j]}" == "--sandbox" ]]; then
      SANDBOX_PAIR+=("${EXEC_TOK[$j]}")
      ((j+=1))
      if [[ $j -lt ${#EXEC_TOK[@]} ]]; then
        SANDBOX_PAIR+=("${EXEC_TOK[$j]}")
      fi
    else
      NEW_EXEC+=("${EXEC_TOK[$j]}")
    fi
    ((j+=1))
  done
  if [[ ${#SANDBOX_PAIR[@]} -gt 0 ]]; then
    CODEX_GLOBAL_FLAGS="$CODEX_GLOBAL_FLAGS ${SANDBOX_PAIR[*]}"
    CODEX_EXEC_FLAGS="${NEW_EXEC[*]}"
  fi
fi

echo "CODEX_GLOBAL_FLAGS=$CODEX_GLOBAL_FLAGS" | tee "$RUN_DIR/codex_flags.txt"
echo "CODEX_EXEC_FLAGS=$CODEX_EXEC_FLAGS" | tee -a "$RUN_DIR/codex_flags.txt"
echo "HAS_SANDBOX_IN_EXEC=$HAS_SANDBOX_IN_EXEC HAS_SANDBOX_IN_GLOBAL=$HAS_SANDBOX_IN_GLOBAL" >> "$RUN_DIR/codex_flags.txt"
echo "HAS_ASK_FOR_APPROVAL=$HAS_ASK_FOR_APPROVAL HAS_FULL_AUTO=$HAS_FULL_AUTO HAS_OUTPUT_LAST_MESSAGE=$HAS_OUTPUT_LAST_MESSAGE HAS_JSON_FLAG=$HAS_JSON_FLAG" >> "$RUN_DIR/codex_flags.txt"

# -------------------------
# Helpers
# -------------------------
queue_counts () {
python - <<'PY'
import json
q=json.load(open("work/queue.json"))
c={"todo":0,"in_progress":0,"blocked":0,"done":0}
for t in q.get("tasks",[]):
    s=t.get("status","todo")
    c[s]=c.get(s,0)+1
print(f"todo={c['todo']} in_progress={c['in_progress']} blocked={c['blocked']} done={c['done']}")
PY
}

queue_hash () {
python - <<'PY'
import hashlib, pathlib
p=pathlib.Path("work/queue.json")
h=hashlib.sha256(p.read_bytes()).hexdigest()
print(h)
PY
}

queue_status_of () {
  local TASK_ID="$1"
python - <<PY
import json
tid = "$TASK_ID"
q=json.load(open("work/queue.json"))
for t in q.get("tasks",[]):
    if str(t.get("id")).zfill(3)==tid:
        print(t.get("status","todo"))
        raise SystemExit(0)
print("missing")
PY
}

extract_task_id_from_prompt () {
  local PROMPT_FILE="$1"
python - <<PY
import re, pathlib
txt=pathlib.Path(r"$PROMPT_FILE").read_text(encoding="utf-8", errors="ignore")
m=re.search(r"(?m)^- id:\s*([0-9]{1,4})\s*$", txt)
if m:
    print(m.group(1).zfill(3))
else:
    print("000")
PY
}

git_fingerprint () {
  if [[ "$IN_GIT" -eq 1 ]]; then
    python - <<'PY'
import subprocess, hashlib
try:
    out=subprocess.check_output(["git","diff","--name-only"]).decode()
except Exception:
    out=""
h=hashlib.sha256(out.encode()).hexdigest()
print(h)
PY
  else
    echo "nogit"
  fi
}

make_doctor_fix_prompt () {
  local DOCTOR_TXT="$1"
  local OUT_PROMPT="$2"
  cat > "$OUT_PROMPT" <<'PROMPT'
# ===== AUTOPILOT: FIX DOCTOR ISSUES ONLY =====
今回は **work/queue.json と work/tasks/*.md の整合性修復だけ**を行います（実装を進めない）。

目的:
- `python tools/codex_prompt.py doctor` が成功（exit 0）する状態にする。

やってよいこと:
- work/queue.json の修正（重複ID、invalid status、missing path、unblocks/depends_on の追加）
- 存在しない work/tasks/*.md を作成（テンプレでOK）
- blocked 親タスクがあるなら解除用の子タスクを作る/修正する
  - 子タスク側の queue.json エントリに必ず `unblocks: ["<親ID>"]` を付ける

禁止:
- 学習/モデル/特徴量の実装を進める
- 大規模リファクタ

出力:
- 変更計画（ファイル単位）
- 変更点の説明
- doctor が通る理由

以下が doctor の出力です：
PROMPT
  echo "" >> "$OUT_PROMPT"
  echo "```" >> "$OUT_PROMPT"
  cat "$DOCTOR_TXT" >> "$OUT_PROMPT"
  echo "```" >> "$OUT_PROMPT"
}

make_force_progress_prompt () {
  local TASK_ID="$1"
  local LAST_MSG_FILE="$2"
  local OUT_PROMPT="$3"
  cat > "$OUT_PROMPT" <<PROMPT
# ===== AUTOPILOT: FORCE PROGRESS (NO QUESTIONS) =====
あなたは自動ループ中です。**確認質問は禁止**です。次を必ず実行してください。

対象タスク: $TASK_ID

状況:
- 直前の出力が「未実装/差分なし/確認お願いします/blockedだから保留」等で停止している可能性があります。
- ここでは “前に進める” ことが最優先です。

指示（必ずどれか）:
A) 実装を進める（最小でもよい）→ テスト/検証 → work/queue.json を done に更新
B) 本当に進められないなら、work/queue.json を blocked に更新し、解除子タスクを起票（子に unblocks を付与）
C) task.md が stale (Blocked が残っている等) なら task.md/queue.json を修正して整合させてから A へ

禁止:
- 「確認をお願いします」「進めてよいですか」等の質問
- “差分なし” で終了

参考: 直前の最終メッセージ（抜粋）
\`\`\`
$(tail -n 120 "$LAST_MSG_FILE" 2>/dev/null || true)
\`\`\`
PROMPT
}

run_codex () {
  # Args:
  #   $1: prompt_file
  #   $2: stdout_file
  #   $3: stderr_file
  #   $4: last_file (optional)
  #   $5: events_file (optional)
  local PROMPT_FILE="$1"
  local STDOUT_FILE="$2"
  local STDERR_FILE="$3"
  local LAST_FILE="${4:-}"
  local EVENTS_FILE="${5:-}"

  read -r -a GLOBAL_ARR <<< "$CODEX_GLOBAL_FLAGS"
  read -r -a EXEC_ARR <<< "$CODEX_EXEC_FLAGS"

  CMD=(codex)
  CMD+=("${GLOBAL_ARR[@]}")
  CMD+=(exec)
  CMD+=("${EXEC_ARR[@]}")

  # Optional flags (only if supported)
  if [[ -n "$LAST_FILE" && "$HAS_OUTPUT_LAST_MESSAGE" -eq 1 ]]; then
    CMD+=(--output-last-message "$LAST_FILE")
  fi
  if [[ "$CODEX_JSON" -eq 1 && -n "$EVENTS_FILE" && "$HAS_JSON_FLAG" -eq 1 ]]; then
    CMD+=(--json)
  fi

  # PROMPT from stdin by passing "-"
  CMD+=(-)

  # log command
  printf '%q ' "${CMD[@]}" > "${STDERR_FILE}.cmd"
  echo "" >> "${STDERR_FILE}.cmd"

  set +e
  "${CMD[@]}" < "$PROMPT_FILE" > "$STDOUT_FILE" 2> "$STDERR_FILE"
  local RC=$?
  set -e

  # Save events if requested and supported (codex writes to stdout in json mode; we already captured stdout)
  # We keep STDOUT_FILE as the source of truth.

  return $RC
}

doctor_or_fix () {
  local STAGE="$1"
  local OUT_TXT="$RUN_DIR/doctor_${STAGE}.txt"
  set +e
  python tools/codex_prompt.py doctor > "$OUT_TXT" 2>&1
  local RC=$?
  set -e

  if [[ "$RC" -eq 0 ]]; then
    return 0
  fi

  echo "doctor failed at stage=$STAGE (rc=$RC). report=$REPO_ROOT/$OUT_TXT"

  if [[ "$AUTO_FIX_DOCTOR" -ne 1 ]]; then
    exit 2
  fi

  local FIX_PROMPT="$RUN_DIR/prompt_fix_doctor_${STAGE}.md"
  local FIX_STDOUT="$RUN_DIR/codex_fix_doctor_${STAGE}.out"
  local FIX_STDERR="$RUN_DIR/codex_fix_doctor_${STAGE}.err"
  local FIX_LAST="$RUN_DIR/last_fix_doctor_${STAGE}.md"
  local FIX_EVENTS="$RUN_DIR/events_fix_doctor_${STAGE}.jsonl"

  make_doctor_fix_prompt "$OUT_TXT" "$FIX_PROMPT"
  run_codex "$FIX_PROMPT" "$FIX_STDOUT" "$FIX_STDERR" "$FIX_LAST" "$FIX_EVENTS" || true

  python tools/codex_prompt.py doctor > "$RUN_DIR/doctor_${STAGE}_after_fix.txt" 2>&1 || {
    echo "doctor still failing after attempted fix. See $REPO_ROOT/$RUN_DIR/doctor_${STAGE}_after_fix.txt"
    exit 2
  }
}

detect_stall_patterns () {
  local STDOUT_FILE="$1"
  local STDERR_FILE="$2"
  local LAST_FILE="$3"

  # patterns that typically mean “it stopped / no progress”
  local PAT="確認|進めてよい|Approve|approval|未実装|差分なし|No new changes|Change Plan[[:space:]]*None|Implementation[[:space:]]*None|blocked.*保留|read-?only|書き込み不可|write access"
  if grep -qiE "$PAT" "$STDOUT_FILE" "$STDERR_FILE" "$LAST_FILE" 2>/dev/null; then
    return 0
  fi
  return 1
}

detect_stall_patterns () {
  local STDOUT_FILE="$1"
  local STDERR_FILE="$2"
  local LAST_FILE="$3"

  # patterns that typically mean “it stopped / no progress”
  local PAT="確認|進めてよい|Approve|approval|未実装|差分なし|No new changes|Change Plan[[:space:]]*None|Implementation[[:space:]]*None|blocked.*保留|read-?only|書き込み不可|write access"
  if grep -qiE "$PAT" "$STDOUT_FILE" "$STDERR_FILE" "$LAST_FILE" 2>/dev/null; then
    return 0
  fi
  return 1
}

print_codex_failure_hints () {
  local STDOUT_FILE="$1"
  local STDERR_FILE="$2"

  if grep -qiE "model_reasoning_effort|unknown variant|Failed to deserialize" "$STDOUT_FILE" "$STDERR_FILE" 2>/dev/null; then
    echo ""
    echo "Hint: Codex config の reasoning 設定が CLI と不整合の可能性があります。"
    echo "  - ~/.codex/config.toml の model_reasoning_effort を high/medium 等へ変更"
    echo "  - または codex CLI を更新"
  fi

  if grep -qiE "OPENAI_API_KEY|api key|not authenticated|login" "$STDOUT_FILE" "$STDERR_FILE" 2>/dev/null; then
    echo ""
    echo "Hint: 認証が不足している可能性があります。"
    echo "  - OPENAI_API_KEY を設定するか、codex login を実行"
  fi

  if grep -qiE "unexpected argument.*ask-for-approval|unexpected argument.*sandbox" "$STDOUT_FILE" "$STDERR_FILE" 2>/dev/null; then
    echo ""
    echo "Hint: CLI フラグの並び/所属（global vs exec）が不整合の可能性があります。"
    echo "  - $REPO_ROOT/$RUN_DIR/codex_help.txt と codex_exec_help.txt を確認"
    echo "  - CODEX_GLOBAL_FLAGS / CODEX_EXEC_FLAGS を見直し"
  fi
}

# -------------------------
# Initial doctor
# -------------------------
doctor_or_fix "initial"

LAST_TASK_ID=""
SAME_TASK_STREAK=0
NO_CHANGE_STREAK=0
LAST_QUEUE_HASH="$(queue_hash)"
LAST_GIT_FP="$(git_fingerprint)"

for ((i=1; i<=MAX_ITERS; i++)); do
  echo "=== iter $i ==="
  echo "$(queue_counts)" | tee "$RUN_DIR/counts_$i.txt"

  # Stop if all done
  NON_DONE=$(python - <<'PY'
import json
q=json.load(open("work/queue.json"))
print(sum(1 for t in q.get("tasks",[]) if t.get("status")!="done"))
PY
)
  if [[ "$NON_DONE" -eq 0 ]]; then
    echo "All tasks are done. stop."
    break
  fi

  # Stop if only blocked remains (manual decision point)
  if [[ "$STOP_WHEN_ONLY_BLOCKED" -eq 1 ]]; then
    ONLY_BLOCKED=$(python - <<'PY'
import json
q=json.load(open("work/queue.json"))
c={"todo":0,"in_progress":0,"blocked":0,"done":0}
for t in q.get("tasks",[]):
    c[t.get("status","todo")] = c.get(t.get("status","todo"),0)+1
print(1 if (c["todo"]==0 and c["in_progress"]==0 and c["blocked"]>0) else 0)
PY
)
    if [[ "$ONLY_BLOCKED" -eq 1 ]]; then
      echo "Only blocked tasks remain. stop for manual decision."
      break
    fi
  fi

  PROMPT_FILE="$RUN_DIR/prompt_$i.md"
  python tools/codex_prompt.py next --mode autopilot > "$PROMPT_FILE" || true
  if grep -q "No tasks available" "$PROMPT_FILE"; then
    echo "No tasks available. stop."
    break
  fi

  TASK_ID="$(extract_task_id_from_prompt "$PROMPT_FILE")"
  echo "selected_task=$TASK_ID" | tee "$RUN_DIR/selected_task_$i.txt"

  # streak bookkeeping
  if [[ "$TASK_ID" == "$LAST_TASK_ID" ]]; then
    SAME_TASK_STREAK=$((SAME_TASK_STREAK + 1))
  else
    SAME_TASK_STREAK=1
  fi
  LAST_TASK_ID="$TASK_ID"
  echo "same_task_streak=$SAME_TASK_STREAK" | tee -a "$RUN_DIR/selected_task_$i.txt"

  if [[ "$SAME_TASK_STREAK" -gt "$MAX_SAME_TASK" ]]; then
    echo "STOP: same task repeated too many times (task=$TASK_ID streak=$SAME_TASK_STREAK)."
    echo "See logs in $REPO_ROOT/$RUN_DIR"
    exit 40
  fi

  QUEUE_HASH_BEFORE="$(queue_hash)"
  GIT_FP_BEFORE="$(git_fingerprint)"
  echo "queue_hash_before=$QUEUE_HASH_BEFORE" > "$RUN_DIR/fingerprint_$i.txt"
  echo "git_fp_before=$GIT_FP_BEFORE" >> "$RUN_DIR/fingerprint_$i.txt"

  STDOUT_FILE="$RUN_DIR/codex_out_${i}.txt"
  STDERR_FILE="$RUN_DIR/codex_err_${i}.txt"
  LAST_FILE="$RUN_DIR/last_message_${i}.md"
  EVENTS_FILE="$RUN_DIR/events_${i}.jsonl"

  # Run codex with retries
  ATTEMPT=0
  RC=0
  while true; do
    ATTEMPT=$((ATTEMPT + 1))
    echo "codex_attempt=$ATTEMPT" > "$RUN_DIR/codex_attempt_${i}.txt"

    set +e
    run_codex "$PROMPT_FILE" "$STDOUT_FILE" "$STDERR_FILE" "$LAST_FILE" "$EVENTS_FILE"
    RC=$?
    set -e

    if [[ "$RC" -eq 0 ]]; then
      break
    fi
    if [[ "$ATTEMPT" -le "$RETRY_CODEX_ON_ERROR" ]]; then
      echo "Codex failed (rc=$RC). retrying... (attempt=$ATTEMPT)" | tee -a "$RUN_DIR/codex_attempt_${i}.txt"
      continue
    fi
    echo "Codex failed (rc=$RC). stop. See: $REPO_ROOT/$STDERR_FILE"
    print_codex_failure_hints "$STDOUT_FILE" "$STDERR_FILE"
    exit 50
  done

  # If output smells like “stall”, optionally force progress
  if [[ "$FORCE_PROGRESS_ON_STALL" -eq 1 ]]; then
    if detect_stall_patterns "$STDOUT_FILE" "$STDERR_FILE" "$LAST_FILE"; then
      FORCE_PROMPT="$RUN_DIR/prompt_force_progress_${i}.md"
      FORCE_OUT="$RUN_DIR/codex_force_progress_${i}.out"
      FORCE_ERR="$RUN_DIR/codex_force_progress_${i}.err"
      FORCE_LAST="$RUN_DIR/last_force_progress_${i}.md"
      FORCE_EVENTS="$RUN_DIR/events_force_progress_${i}.jsonl"
      make_force_progress_prompt "$TASK_ID" "$LAST_FILE" "$FORCE_PROMPT"
      run_codex "$FORCE_PROMPT" "$FORCE_OUT" "$FORCE_ERR" "$FORCE_LAST" "$FORCE_EVENTS" || true
      # Replace last message file pointer for downstream checks
      LAST_FILE="$FORCE_LAST"
      STDOUT_FILE="$FORCE_OUT"
      STDERR_FILE="$FORCE_ERR"
    fi
  fi

  # doctor check + fix
  doctor_or_fix "iter_${i}"

  # diff stats
  if [[ "$IN_GIT" -eq 1 ]]; then
    git diff --stat > "$RUN_DIR/git_diff_stat_${i}.txt" || true
    git status --porcelain > "$RUN_DIR/git_status_${i}.txt" || true
  fi

  QUEUE_HASH_AFTER="$(queue_hash)"
  GIT_FP_AFTER="$(git_fingerprint)"
  echo "queue_hash_after=$QUEUE_HASH_AFTER" >> "$RUN_DIR/fingerprint_$i.txt"
  echo "git_fp_after=$GIT_FP_AFTER" >> "$RUN_DIR/fingerprint_$i.txt"

  # no-change detection
  if [[ "$QUEUE_HASH_AFTER" == "$QUEUE_HASH_BEFORE" && "$GIT_FP_AFTER" == "$GIT_FP_BEFORE" ]]; then
    NO_CHANGE_STREAK=$((NO_CHANGE_STREAK + 1))
  else
    NO_CHANGE_STREAK=0
  fi
  echo "no_change_streak=$NO_CHANGE_STREAK" >> "$RUN_DIR/fingerprint_$i.txt"

  if [[ "$NO_CHANGE_STREAK" -gt "$MAX_NO_CHANGE" ]]; then
    echo "STOP: no changes detected repeatedly (streak=$NO_CHANGE_STREAK)."
    echo "Likely stuck on approvals/blocked mismatch. Review:"
    echo "  - $REPO_ROOT/$RUN_DIR/prompt_$i.md"
    echo "  - $REPO_ROOT/$RUN_DIR/codex_out_${i}.txt"
    echo "  - $REPO_ROOT/$RUN_DIR/codex_err_${i}.txt"
    exit 60
  fi

  # Optional auto-commit
  if [[ "$AUTO_GIT_COMMIT" -eq 1 && "$IN_GIT" -eq 1 ]]; then
    if [[ -n "$(git status --porcelain)" ]]; then
      git add -A || true
      git commit -m "autopilot: iter $i (task $TASK_ID)" || true
      git rev-parse HEAD > "$RUN_DIR/git_commit_${i}.txt" || true
    fi
  fi

  echo "Iteration $i done."
done

echo "Autopilot finished. Logs: $REPO_ROOT/$RUN_DIR"
