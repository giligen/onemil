#!/bin/bash
# Telegram ping when run_consol.py finishes. No numbers: results are reviewed before they are relayed.
cd /home/ec2-user/onemil
for i in $(seq 1 360); do
  if grep -q "ALL DONE" research/hod_consol/run.log 2>/dev/null; then
    python3 scripts/send_telegram_alert.py "[RESEARCH] Base-under-the-high study (cells 1,400-1,402) finished at $(date -u +%H:%M) UTC. Results are on disk; they get reviewed before any number is relayed. Ping the Claude session." >/dev/null 2>&1
    exit 0
  fi
  if grep -qE "Traceback|Error" research/hod_consol/run.log 2>/dev/null; then
    python3 scripts/send_telegram_alert.py "[RESEARCH] Base-under-the-high study FAILED at $(date -u +%H:%M) UTC (see research/hod_consol/run.log). Ping the Claude session." >/dev/null 2>&1
    exit 1
  fi
  sleep 30
done
