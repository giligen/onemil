#!/bin/bash
# Telegram ping when the bull-flag 2024 chain ends. No numbers: they are reviewed before they are relayed.
cd /home/ec2-user/onemil
for i in $(seq 1 2880); do
  if [ -f research/bf_2024/REPORT.md ] && grep -q "CHAIN DONE\|CHAIN END" research/bf_2024/chain.log 2>/dev/null; then
    python3 scripts/send_telegram_alert.py "[RESEARCH] Bull-flag P1 on 2024H2 (cell 1,417) finished $(date -u +%H:%M) UTC. Reviewed before any number is relayed - ping the Claude session." >/dev/null 2>&1; exit 0
  fi
  if ! pgrep -f "research/bf_2024/chain.sh" >/dev/null && ! grep -q "CHAIN DONE\|CHAIN END" research/bf_2024/chain.log 2>/dev/null; then
    python3 scripts/send_telegram_alert.py "[RESEARCH] Bull-flag 2024 chain STOPPED without finishing ($(date -u +%H:%M) UTC) - see research/bf_2024/chain.log." >/dev/null 2>&1; exit 1
  fi
  sleep 60
done
