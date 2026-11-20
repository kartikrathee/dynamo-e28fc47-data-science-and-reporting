#!/usr/bin/env bash
# dynamo-self-heal.sh — poll Dynamo CI, extract failures, apply fixes, push, repeat.
#
# Usage:
#   ./scripts/dynamo-self-heal.sh --loop 8          # preflight → push → poll → fix → repeat
#   ./scripts/dynamo-self-heal.sh --no-push         # poll latest run only (no commit/push)
#   HARBOR=/path/to/harbor ./scripts/dynamo-self-heal.sh --loop 8
#
# Env: BRANCH, REPO, PR_NUMBER, POLL_INTERVAL (90), POLL_TIMEOUT (7200), HARBOR

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_DIR="$ROOT/task"
REPORT_DIR="$ROOT/.dynamo-heal"
ARTIFACT_DIR="$REPORT_DIR/artifacts"
BRANCH="${BRANCH:-submission}"
PR_NUMBER="${PR_NUMBER:-3}"
REPO="${REPO:-handshake-project-dynamo/dynamo-e28fc47-data-science-and-reporting}"
PUSH_REMOTE="${PUSH_REMOTE:-origin}"
POLL_INTERVAL="${POLL_INTERVAL:-60}"
# pass@2 agent trials can exceed 1h; use 7200+ for full pipeline
POLL_TIMEOUT="${POLL_TIMEOUT:-7200}"
MAX_CYCLES="${MAX_CYCLES:-1}"
DO_PUSH=1
HARBOR="${HARBOR:-$(command -v harbor 2>/dev/null || true)}"

log() { printf '[heal] %s\n' "$*" >&2; }
warn() { printf '[heal] WARN: %s\n' "$*" >&2; }
die() { printf '[heal] ERROR: %s\n' "$*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --loop) MAX_CYCLES="${2:-8}"; shift 2 ;;
    --no-push) DO_PUSH=0; shift ;;
    --branch) BRANCH="$2"; shift 2 ;;
    --repo) REPO="$2"; shift 2 ;;
    --pr) PR_NUMBER="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,10p' "$0" >&2
      exit 0
      ;;
    *) die "unknown arg: $1 (try --help)" ;;
  esac
done

mkdir -p "$REPORT_DIR" "$ARTIFACT_DIR"
require_cmd() { command -v "$1" >/dev/null 2>&1 || die "missing: $1"; }
require_cmd gh
require_cmd git
require_cmd python3
command -v rg >/dev/null 2>&1 || die "missing: rg (ripgrep)"

instruction_tokens() {
  python3 - "$TASK_DIR/instruction.md" << 'PY'
import sys
from pathlib import Path
text = Path(sys.argv[1]).read_text()
try:
    import tiktoken
    print(len(tiktoken.get_encoding("cl100k_base").encode(text)))
except Exception:
    print(len(text) // 4)
PY
}

committed_instruction_tokens() {
  python3 - "$ROOT" << 'PY'
import subprocess, sys
from pathlib import Path
root = Path(sys.argv[1])
try:
    text = subprocess.check_output(["git", "-C", str(root), "show", "HEAD:task/instruction.md"], text=True)
except Exception:
    text = (root / "task/instruction.md").read_text()
try:
    import tiktoken
    print(len(tiktoken.get_encoding("cl100k_base").encode(text)))
except Exception:
    print(len(text) // 4)
PY
}

preflight_local() {
  log "Local preflight..."
  local tokens committed
  tokens="$(instruction_tokens)"
  committed="$(committed_instruction_tokens)"
  log "instruction.md: $tokens tokens (limit 1500; HEAD=$committed)"
  if [[ "$tokens" -gt 1500 ]]; then
    warn "instruction over token limit (working tree)"
    [[ "$committed" -le 1500 ]] || return 1
    warn "HEAD commit is within limit — push before CI if needed"
  fi

  [[ -n "$HARBOR" && -x "$HARBOR" ]] || { warn "harbor not found — skip oracle preflight"; return 0; }

  log "Running harbor oracle (cpus=2)..."
  local logf="$REPORT_DIR/last-oracle.log"
  if ! "$HARBOR" run -p "$TASK_DIR" --agent oracle \
    -o "$ROOT/jobs" --job-name "heal-oracle-$(date +%Y%m%d-%H%M%S)" \
    --ek "cpus=2" -q 2>&1 | tee "$logf"; then
    warn "harbor oracle command failed"
    return 1
  fi
  rg -q "Mean: 1\.000|1\.0.*1" "$logf" || { warn "oracle reward not 1.0 — see $logf"; return 1; }
  log "oracle preflight OK"
  return 0
}

git_push_if_needed() {
  [[ "$DO_PUSH" -eq 1 ]] || return 0
  cd "$ROOT"
  local msg="${1:-chore(heal): automated CI fix}"
  if ! git diff --quiet || ! git diff --cached --quiet || [[ -n "$(git ls-files --others --exclude-standard task/ scripts/)" ]]; then
    git add task/ scripts/dynamo-self-heal.sh 2>/dev/null || git add task/
    git diff --cached --quiet && { log "nothing to commit"; return 0; }
    git commit -m "$msg"
  fi
  log "Pushing $BRANCH..."
  git push -u "$PUSH_REMOTE" "$BRANCH"
}

latest_run_for_head() {
  local head
  head="$(git -C "$ROOT" rev-parse HEAD)"
  gh run list --repo "$REPO" --limit 15 --json databaseId,headSha \
    | python3 -c "
import json, sys
head = sys.argv[1]
runs = json.load(sys.stdin)
for r in runs:
    if r.get('headSha') == head:
        print(r['databaseId']); break
else:
    # fallback: newest run (e.g. local commit not pushed yet)
    print(runs[0]['databaseId'] if runs else '')
" "$head"
}

pending_jobs_summary() {
  local run_id="$1"
  gh run view "$run_id" --repo "$REPO" --json jobs \
    -q '[.jobs[]|select(.status!="completed")|"\(.name)(\(.status))"]|join(", ")' 2>/dev/null || true
}

poll_until_complete() {
  local run_id="$1"
  local deadline=$(( $(date +%s) + POLL_TIMEOUT ))
  local tick=0
  while [[ $(date +%s) -lt deadline ]]; do
    local line status conclusion pending
    line="$(gh run view "$run_id" --repo "$REPO" --json status,conclusion -q '[.status,.conclusion]|@tsv' 2>/dev/null || echo $'unknown\tunknown')"
    status="${line%%$'\t'*}"
    conclusion="${line#*$'\t'}"
    if [[ "$status" == "unknown" ]]; then
      warn "gh run view failed for $run_id — retrying"
      sleep "$POLL_INTERVAL"
      continue
    fi
    pending="$(pending_jobs_summary "$run_id")"
    if [[ -n "$pending" ]]; then
      log "Run $run_id: status=$status conclusion=${conclusion:-pending} — pending: $pending"
    else
      log "Run $run_id: status=$status conclusion=${conclusion:-pending}"
    fi
    if [[ "$status" == "completed" ]]; then
      printf '%s' "${conclusion:-failure}"
      return 0
    fi
    tick=$((tick + 1))
    if [[ "$tick" -eq 1 ]] || [[ $((tick % 5)) -eq 0 ]]; then
      fetch_pr_checks >/dev/null 2>&1 || true
    fi
    sleep "$POLL_INTERVAL"
  done
  die "timed out after ${POLL_TIMEOUT}s waiting for run $run_id (pass@2 may need POLL_TIMEOUT=7200+)"
}

fetch_pr_checks() {
  gh pr checks "$PR_NUMBER" --repo "$REPO" 2>&1 | tee "$REPORT_DIR/last-checks.txt" || true
}

check_status() {
  local name="$1"
  awk -F'\t' -v n="$name" '$1 == n { print $2; exit }' "$REPORT_DIR/last-checks.txt" 2>/dev/null || true
}

wait_for_gate() {
  local deadline=$(( $(date +%s) + 300 ))
  while [[ $(date +%s) -lt deadline ]]; do
    fetch_pr_checks
    local gate
    gate="$(check_status "review / gate")"
    [[ -n "$gate" && "$gate" != "pending" ]] && return 0
    sleep 15
  done
  warn "review / gate still pending after 5m — continuing with available checks"
}

download_oracle_artifact() {
  local run_id="$1"
  rm -rf "$ARTIFACT_DIR/run-$run_id"
  mkdir -p "$ARTIFACT_DIR/run-$run_id"
  gh run download "$run_id" --repo "$REPO" \
    -n harbor-output-validate -D "$ARTIFACT_DIR/run-$run_id" 2>/dev/null || return 1
  return 0
}

parse_oracle_test_failures() {
  local run_id="$1"
  local summary="$REPORT_DIR/last-failures.txt"
  : > "$summary"

  fetch_pr_checks
  if pipeline_green; then
    echo "GATE_PASS: pipeline green" >> "$summary"
    cat "$summary"
    return 0
  fi

  local pass2_status
  pass2_status="$(check_status "review / pass2")"
  if [[ "$pass2_status" == "pending" ]]; then
    echo "PASS2_PENDING: agent trials still running" >> "$summary"
  fi

  if rg -q "review / review.*fail" "$REPORT_DIR/last-checks.txt" 2>/dev/null; then
    echo "REVIEW_FAIL: static or rubric" >> "$summary"
    gh run view "$run_id" --repo "$REPO" --log-failed --job "$(gh run view "$run_id" --repo "$REPO" --json jobs -q '.jobs[]|select(.name|test("review / review"))|.databaseId' 2>/dev/null || true)" 2>/dev/null \
      | rg "FAIL|static checks|Verdict: FAIL|eval verdict" | head -25 >> "$summary" || true
  fi

  if rg -q "review / validation.*fail" "$REPORT_DIR/last-checks.txt" 2>/dev/null; then
    echo "VALIDATION_FAIL: oracle or nop" >> "$summary"
    if download_oracle_artifact "$run_id"; then
      local testout
      testout="$(find "$ARTIFACT_DIR/run-$run_id" -name test-stdout.txt 2>/dev/null | head -1)"
      if [[ -n "$testout" ]]; then
        rg "FAILED |AssertionError|assert " "$testout" >> "$summary" || true
        local passed failed
        passed="$(rg -c "^PASSED" "$testout" 2>/dev/null || echo 0)"
        failed="$(rg -c "^FAILED" "$testout" 2>/dev/null || echo 0)"
        echo "ORACLE_SCORE: ${passed:-0} passed, ${failed:-0} failed" >> "$summary"
        find "$ARTIFACT_DIR/run-$run_id" -name scaling.txt -exec echo "SCALING:" \; -exec cat {} \; >> "$summary" 2>/dev/null || true
      fi
    fi
  fi

  if rg -q "review / pass2.*fail" "$REPORT_DIR/last-checks.txt" 2>/dev/null; then
    echo "PASS2_FAIL: too easy or agent failures" >> "$summary"
    gh pr view "$PR_NUMBER" --repo "$REPO" --comments 2>/dev/null \
      | rg -i "pass@2|too easy|valid fail|2/2 passed|pass2" | head -30 >> "$summary" || true
    gh run view "$run_id" --repo "$REPO" --log-failed --job "$(gh run view "$run_id" --repo "$REPO" --json jobs -q '.jobs[]|select(.name|test("pass2"))|.databaseId' 2>/dev/null || true)" 2>/dev/null \
      | rg -i "too easy|valid fail|passed|reward|Mean:" | head -25 >> "$summary" || true
  fi

  rg -q "review / pass2.*pass" "$REPORT_DIR/last-checks.txt" 2>/dev/null && echo "PASS2_PASS" >> "$summary"

  [[ -s "$summary" ]] || echo "UNKNOWN: see $REPORT_DIR/last-checks.txt" >> "$summary"
  cat "$summary"
}

apply_known_fixes() {
  local failures="$1"
  local changed=0

  if echo "$failures" | rg -q "test_parallel_force_time|force speedup|_MIN_FORCE_SPEEDUP"; then
    local sim="$TASK_DIR/solution/simulation.cpp"
    if ! rg -q "nthreads == 1" "$sim" 2>/dev/null; then
      warn "simulation.cpp missing single-thread force fast path — needs manual fix"
    else
      log "single-thread force path present"
    fi
    if ! rg -q "schedule\\(dynamic" "$sim" 2>/dev/null; then
      warn "consider schedule(dynamic) on force loop"
    fi
  fi

  if echo "$failures" | rg -q "solution_quality|giant heredoc"; then
    if [[ ! -f "$TASK_DIR/solution/analyze.py" ]]; then
      warn "missing analyze.py — extract from solve.sh"
    fi
  fi

  if echo "$failures" | rg -q "difficulty_explanation"; then
    if ! rg -q "computational chemists|HPC engineers" "$TASK_DIR/task.toml" 2>/dev/null; then
      warn "task.toml difficulty_explanation needs who/why"
    fi
  fi

  if echo "$failures" | rg -q "STATIC_FAIL|REVIEW_FAIL.*token|1500"; then
    local tokens
    tokens="$(instruction_tokens)"
    if [[ "$tokens" -gt 1500 ]]; then
      warn "instruction $tokens tokens — must condense below 1500"
    fi
  fi

  if echo "$failures" | rg -q "PASS2_TOO_EASY|PASS2_FAIL.*too easy|2/2 passed"; then
    warn "pass@2 too easy — need >=1 valid agent failure (50-test hardening)"
  fi

  return "$changed"
}

print_report() {
  local run_id="$1" conclusion="$2"
  log "========== HEAL REPORT =========="
  log "Run: https://github.com/$REPO/actions/runs/$run_id"
  log "Conclusion: $conclusion"
  log "Checks:"
  sed 's/^/  /' "$REPORT_DIR/last-checks.txt" 2>/dev/null || true
  log "Failures:"
  sed 's/^/  /' "$REPORT_DIR/last-failures.txt" 2>/dev/null || true
  log "================================="
}

pipeline_green() {
  [[ -f "$REPORT_DIR/last-checks.txt" ]] || fetch_pr_checks
  local gate pass2 review validation similarity
  gate="$(check_status "review / gate")"
  pass2="$(check_status "review / pass2")"
  review="$(check_status "review / review")"
  validation="$(check_status "review / validation")"
  similarity="$(check_status "review / similarity")"
  [[ "$gate" == "pass" ]] && return 0
  [[ "$review" == "pass" && "$similarity" == "pass" && "$validation" == "pass" && "$pass2" == "pass" ]] && return 0
  return 1
}

cycle_once() {
  local cycle="$1"
  log "=== cycle $cycle / $MAX_CYCLES ==="

  preflight_local || warn "local preflight failed — fix before push if possible"

  if [[ "$DO_PUSH" -eq 1 ]]; then
    git_push_if_needed "chore(heal): cycle $cycle CI fix"
  fi

  local run_id
  run_id="$(latest_run_for_head)"
  [[ -n "$run_id" ]] || die "no CI run for HEAD — push first or wait for workflow"

  log "Watching run $run_id ..."
  local conclusion
  conclusion="$(poll_until_complete "$run_id")"
  wait_for_gate

  parse_oracle_test_failures "$run_id" | tee "$REPORT_DIR/last-failures.txt" >/dev/null
  print_report "$run_id" "$conclusion"

  if pipeline_green; then
    log "Pipeline GREEN (gate passed)"
    return 0
  fi

  apply_known_fixes "$(cat "$REPORT_DIR/last-failures.txt")" || true

  if [[ "${USE_CURSOR_AGENT:-0}" == "1" ]] && command -v cursor >/dev/null 2>&1; then
    log "Invoking cursor agent for remaining failures..."
    cursor agent --prompt "Fix Dynamo CI failures in $ROOT. Failures:
$(cat "$REPORT_DIR/last-failures.txt")
Minimal honest fixes. Run harbor oracle with cpus=2. Commit task/ only." || true
  fi

  return 1
}

main() {
  cd "$ROOT"
  git checkout "$BRANCH" 2>/dev/null || true

  local cycle=1
  while [[ "$cycle" -le "$MAX_CYCLES" ]]; do
    if cycle_once "$cycle"; then
      log "Done — CI gate is green."
      exit 0
    fi
    cycle=$((cycle + 1))
    if [[ "$cycle" -le "$MAX_CYCLES" ]]; then
      log "Cycle failed — retrying after local fixes (if any)..."
      sleep 10
    fi
  done

  die "exhausted $MAX_CYCLES cycles — see $REPORT_DIR/"
}

main "$@"
