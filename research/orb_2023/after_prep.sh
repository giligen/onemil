#!/bin/bash
# Waits for the 2023 data prep, then features + pipeline + score (run_all_2023.sh) under a market-hours
# SIGSTOP/SIGCONT guard, then a Telegram ping (no numbers; reviewed before relayed).
cd /home/ec2-user/onemil
until grep -q "PREP DONE" research/orb_2023/universe.log 2>/dev/null; do sleep 60; done
if ! grep -q "PREP DONE rc=0" research/orb_2023/universe.log; then
  python3 scripts/send_telegram_alert.py "[RESEARCH] ORB 2023 data prep FAILED - see research/orb_2023/universe.log / fetch.log" >/dev/null 2>&1; exit 1
fi
setsid bash research/orb_2023/run_all_2023.sh & RUN=$!
PG=$(ps -o pgid= -p $RUN | tr -d ' ')
while kill -0 $RUN 2>/dev/null; do
  h=$(date -u +%H%M)
  if [ "$h" -ge 1325 ] && [ "$h" -lt 2005 ]; then kill -STOP -$PG 2>/dev/null; else kill -CONT -$PG 2>/dev/null; fi
  sleep 60
done
python3 scripts/send_telegram_alert.py "[RESEARCH] ORB on 2023-01..2024-06 (cells 1,418-1,419) finished $(date -u +%H:%M) UTC. Reviewed before any number is relayed - ping the Claude session." >/dev/null 2>&1
