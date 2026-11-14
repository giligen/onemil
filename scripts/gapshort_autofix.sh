#!/bin/bash
# GAPSHORT nightly auto-fix (owner 2026-11-14). Runs Claude Code headless on docs/autofix/gapshort_autofix_prompt.md.
# Guard rails live in the prompt (file allow-list, frozen constants, full suite, branch-only commits) and in the tool allow-list below.
set -u
cd /home/ec2-user/onemil || exit 1
# Use the Claude Code login (plan quota), not an API key: a stale ANTHROPIC_API_KEY in the environment makes headless runs fail with "Invalid API key".
unset ANTHROPIC_API_KEY
LOG=logs/gap_short_autofix.log
if pgrep -f "claude .*gapshort_autofix" >/dev/null; then echo "$(date -u +%FT%TZ) SKIP previous autofix still running" >> "$LOG"; exit 0; fi
echo "$(date -u +%FT%TZ) START" >> "$LOG"
timeout 1800 claude -p "$(cat docs/autofix/gapshort_autofix_prompt.md)" \
  --model sonnet --output-format text \
  --allowedTools "Read,Grep,Glob,Edit,Write,Bash(python3 *),Bash(git *),Bash(pytest *),Bash(tail *),Bash(head *),Bash(ls *)" \
  < /dev/null >> "$LOG" 2>&1
rc=$?
echo "$(date -u +%FT%TZ) END rc=$rc" >> "$LOG"
if [ $rc -ne 0 ]; then python3 scripts/send_telegram_alert.py "[GAPSHORT FIX] $(date +%F): autofix run failed rc=$rc (see logs/gap_short_autofix.log)"; fi
