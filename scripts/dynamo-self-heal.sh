#!/usr/bin/env bash
# dynamo-self-heal.sh — poll Dynamo CI, extract failures, apply known fixes, push.
#
# Usage:
#   ./scripts/dynamo-self-heal.sh              # one cycle: preflight → push → poll → report
#   ./scripts/dynamo-self-heal.sh --loop 6     # up to 6 fix/push/poll cycles
#   ./scripts/dynamo-self-heal.sh --no-push    # preflight + parse latest run only
#   ./scripts/dynamo-self-heal.sh --fix-local  # run known local fixers without push
#
# Requires: gh, git, python3, harbor (optional but recommended for oracle preflight)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_DIR="$ROOT/task"
REPORT_DIR="$ROOT/.dynamo-heal"
BRANCH="${BRANCH:-submission}"
PR_NUMBER="${PR_NUMBER:-3}"
REPO="${REPO:-handshake-project-dynamo/dynamo-e28fc47-data-science-and-reporting}"
PUSH_REMOTE="${PUSH_REMOTE:-origin}"
POLL_INTERVAL="${POLL_INTERVAL:-90}"
POLL_TIMEOUT="${POLL_TIMEOUT:-7200}"
MAX_CYCLES="${MAX_CYCLES:-1}"
DO_PUSH=1
DO_FIX_LOCAL=0
HARBOR="${HARBOR:-$(command -v harbor 2>/dev/null || true)}"

usage() {
  sed -n '2,12p' "$0"
}

log() { printf '[heal] %s\n' "$*"; }
warn() { printf '[heal] WARN: %s\n' "$*" >&2; }
die() { printf '[heal] ERROR: %s\n' "$*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --loop) MAX_CYCLES="${2:-6}"; shift 2 ;;
    --no-push) DO_PUSH=0; shift ;;
    --fix-local) DO_FIX_LOCAL=1; shift ;;
    --branch) BRANCH="$2"; shift 2 ;;
    --repo) REPO="$2"; shift 2 ;;
    --pr) PR_NUMBER="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown arg: $1" ;;
  esac
done

mkdir -p "$REPORT_DIR"

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

require_cmd gh
require_cmd git
require_cmd python3

instruction_tokens() {
  python3 - "$TASK_DIR/instruction.md" << 'PY'
import sys
from pathlib import Path
text = Path(sys.argv[1]).read_text()
try:
    import tiktoken
    n = len(tiktoken.get_encoding("cl100k_base").encode(text))
except Exception:
    n = len(text) // 4
print(n)
PY
}

preflight_local() {
  log "Local preflight…"
  local tokens
  tokens="$(instruction_tokens)"
  log "instruction.md tokens: $tokens (limit 1500)"
  if [[ "$tokens" -gt 1500 ]]; then
    warn "instruction exceeds 1500 tokens — static review will fail"
    return 1
  fi

  if [[ -n "$HARBOR" && -x "$HARBOR" ]]; then
    log "Running harbor oracle (cpus=2)…"
    if ! "$HARBOR" run -p "$TASK_DIR" --agent oracle \
      -o "$ROOT/jobs" --job-name "heal-oracle-$(date +%Y%m%d-%H%M%S)" \
      --ek "cpus=2" -q 2>&1 | tee "$REPORT_DIR/last-oracle.log" | tail -5; then
      warn "oracle run failed — fix task before pushing"
      return 1
    fi
    if ! rg -q "Mean: 1\.000|reward.*1\.0" "$REPORT_DIR/last-oracle.log" 2>/dev/null; then
      warn "oracle may not have passed — check $REPORT_DIR/last-oracle.log"
      return 1
    fi
    log "oracle preflight OK"
  else
    warn "harbor not found — skipping oracle preflight"
  fi
  return 0
}

git_push_if_needed() {
  [[ "$DO_PUSH" -eq 1 ]] || return 0
  cd "$ROOT"
  if git diff --quiet && git diff --cached --quiet && [[ -z "$(git ls-files --others --exclude-standard task/ scripts/)" ]]; then
    log "No task/script changes to push"
    return 0
  fi
  git add task/ scripts/dynamo-self-heal.sh 2>/dev/null || git add task/
  if git diff --cached --quiet; then
    log "Nothing staged"
    return 0
  fi
  local msg="${1:-chore: dynamo self-heal automated fix}"
  git commit -m "$msg" || true
  log "Pushing $BRANCH to $PUSH_REMOTE…"
  git push -u "$PUSH_REMOTE" "$BRANCH"
}

latest_run_for_branch() {
  gh run list --repo "$REPO" --branch "$BRANCH" --limit 5 \
    --json databaseId,headSha,status,conclusion,createdAt \
    | python3 -c "
import json,sys
runs=json.load(sys.stdin)
head=__import__('subprocess').check_output(['git','rev-parse','HEAD'], cwd='$ROOT', text=True).strip()
for r in runs:
    if r.get('headSha','').startswith(head[:12]) or r.get('headSha')==head:
        print(r['databaseId']); break
else:
    print(runs[0]['databaseId'] if runs else '')
"
}

fetch_run_report() {
  local run_id="$1"
  local out="$REPORT_DIR/run-${run_id}.log"
  gh run view "$run_id" --repo "$REPO" --log-failed > "$out" 2>&1 || gh run view "$run_id" --repo "$REPO" --log > "$out" 2>&1 || true
  echo "$out"
}

parse_failures() {
  local logfile="$1"
  local summary="$REPORT_DIR/last-failures.txt"
  : > "$summary"

  if rg -q "static checks failed" "$logfile" 2>/dev/null; then
    echo "STATIC_FAIL: instruction/token/path/static check" >> "$summary"
    rg "FAIL submission|instruction|token|relative path" "$logfile" 2>/dev/null | head -20 >> "$summary" || true
  fi

  if rg -q "eval verdict: FAIL|Verdict: FAIL" "$logfile" 2>/dev/null; then
    echo "RUBRIC_FAIL: Harbor rubric review" >> "$summary"
    rg "\| [0-9]+ \| [a-z_]+ \| FAIL" "$logfile" 2>/dev/null >> "$summary" || true
    rg "\*\*[a-z_]+ \(FAIL\)" "$logfile" 2>/dev/null >> "$summary" || true
  fi

  if rg -q "pass@2: 2/2 passed|too easy|no valid fail" "$logfile" 2>/dev/null; then
    echo "PASS2_TOO_EASY: need >=1 valid agent failure" >> "$summary"
  fi

  if rg -q "validation.*fail|oracle" "$logfile" 2>/dev/null; then
    rg -i "validation|oracle|force.*speedup" "$logfile" 2>/dev/null | head -15 >> "$summary" || true
  fi

  if rg -q "Pipeline not green" "$logfile" 2>/dev/null; then
    rg "review=|pass2=|validation=" "$logfile" 2>/dev/null | tail -3 >> "$summary" || true
  fi

  if [[ ! -s "$summary" ]]; then
    echo "UNKNOWN: see $logfile" >> "$summary"
  fi
  cat "$summary"
}

poll_until_complete() {
  local run_id="$1"
  local deadline=$(( $(date +%s) + POLL_TIMEOUT ))
  while [[ $(date +%s) -lt deadline ]]; do
    local status conclusion
    read -r status conclusion < <(gh run view "$run_id" --repo "$REPO" \
      --json status,conclusion -q '[.status,.conclusion]|@tsv' 2>/dev/null || echo "unknown unknown")
    log "Run $run_id: status=$status conclusion=${conclusion:-pending}"
    if [[ "$status" == "completed" ]]; then
      echo "$conclusion"
      return 0
    fi
    sleep "$POLL_INTERVAL"
  done
  die "timed out waiting for run $run_id"
}

apply_known_fixes() {
  local failures="$1"
  local fixed=0

  if echo "$failures" | rg -q "solution_quality|giant heredoc|analyze\.py"; then
    if [[ -f "$TASK_DIR/solution/analyze.py" ]] && ! rg -q 'python3 << .PY.' "$TASK_DIR/solution/solve.sh" 2>/dev/null; then
      log "analyze.py already extracted"
    else
      warn "solution_quality: extract analysis to task/solution/analyze.py manually or re-run after merge"
    fi
  fi

  if echo "$failures" | rg -q "difficulty_explanation"; then
    if rg -q "computational chemists|HPC engineers" "$TASK_DIR/task.toml" 2>/dev/null; then
      log "difficulty_explanation already mentions real-world audience"
    else
      warn "difficulty_explanation needs who/why prose in task.toml"
    fi
  fi

  if echo "$failures" | rg -q "STATIC_FAIL|1500|token|instruction"; then
    local tokens
    tokens="$(instruction_tokens)"
    if [[ "$tokens" -gt 1500 ]]; then
      warn "instruction still $tokens tokens — condense task/instruction.md below 1500"
    else
      log "instruction token count OK ($tokens)"
    fi
  fi

  if echo "$failures" | rg -q "relative path|\./simulation"; then
    if rg -q '\./simulation' "$TASK_DIR/instruction.md" 2>/dev/null; then
      warn "replace ./simulation with /app/simulation in instruction.md"
    fi
  fi

  if echo "$failures" | rg -q "PASS2_TOO_EASY"; then
    warn "pass@2 too easy — task needs more verifier surface or subtler traps (50-test version may help once CI runs pass@2)"
  fi

  return "$fixed"
}

fetch_pr_pass2_comment() {
  gh pr view "$PR_NUMBER" --repo "$REPO" --comments 2>/dev/null \
    | rg -A30 "pass@2|Pre-check" | tail -35 || true
}

print_report() {
  local run_id="$1" conclusion="$2" failures="$3"
  log "========== HEAL REPORT =========="
  log "Run: https://github.com/$REPO/actions/runs/$run_id"
  log "Conclusion: $conclusion"
  log "Failures parsed:"
  sed 's/^/  /' "$REPORT_DIR/last-failures.txt"
  log "PR pass@2 snippet:"
  fetch_pr_pass2_comment | sed 's/^/  /' | head -25
  log "Full log: $REPORT_DIR/run-${run_id}.log"
  log "================================="
}

maybe_cursor_agent() {
  local failures="$1"
  if ! command -v cursor >/dev/null 2>&1; then
    return 0
  fi
  if [[ "${USE_CURSOR_AGENT:-0}" != "1" ]]; then
    log "Set USE_CURSOR_AGENT=1 to invoke cursor agent for unhandled failures"
    return 0
  fi
  local prompt
  prompt="Fix Dynamo task CI failures in $ROOT. Failures:
$failures
Read task/instruction.md, task/task.toml, task/solution/, task/tests/. Apply minimal honest fixes. Run harbor oracle. Do not commit submission/ or jobs/."
  log "Invoking cursor agent…"
  cursor agent --prompt "$prompt" || warn "cursor agent invocation failed"
}

cycle_once() {
  local cycle="$1"
  log "——— cycle $cycle / $MAX_CYCLES ———"

  if [[ "$DO_FIX_LOCAL" -eq 1 ]]; then
    apply_known_fixes "$(cat "$REPORT_DIR/last-failures.txt" 2>/dev/null || echo "")" || true
    preflight_local || true
    return 0
  fi

  preflight_local || warn "preflight failed — continuing anyway (use --no-push to abort push)"

  if [[ "$DO_PUSH" -eq 1 ]]; then
    git_push_if_needed "chore(heal): cycle $cycle — rubric/static fixes"
  fi

  local run_id
  run_id="$(latest_run_for_branch)"
  [[ -n "$run_id" ]] || die "no CI run found for branch $BRANCH"
  log "Watching run $run_id…"

  local conclusion
  conclusion="$(poll_until_complete "$run_id")"
  local logfile
  logfile="$(fetch_run_report "$run_id")"
  parse_failures "$logfile" | tee "$REPORT_DIR/last-failures.txt" >/dev/null
  print_report "$run_id" "$conclusion" "$(cat "$REPORT_DIR/last-failures.txt")"

  if [[ "$conclusion" == "success" ]]; then
    log "CI green — done"
    fetch_pr_pass2_comment
    return 0
  fi

  apply_known_fixes "$(cat "$REPORT_DIR/last-failures.txt")" || true
  maybe_cursor_agent "$(cat "$REPORT_DIR/last-failures.txt")"
  return 1
}

main() {
  cd "$ROOT"
  git checkout "$BRANCH" 2>/dev/null || true

  local cycle=1
  while [[ "$cycle" -le "$MAX_CYCLES" ]]; do
    if cycle_once "$cycle"; then
      exit 0
    fi
    cycle=$((cycle + 1))
    [[ "$cycle" -le "$MAX_CYCLES" ]] || break
    log "Retrying after fixes…"
    sleep 5
  done

  die "exhausted $MAX_CYCLES cycles — see $REPORT_DIR/"
}

main "$@"
