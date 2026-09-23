#!/bin/bash
# Detached chain: bull-flag P1 on 2024H2 (PREREG.md cell 1,417; research/bf_2024/SPEC.md step 3).
# Launch: nohup setsid nice -n 19 ionice -c3 bash research/bf_2024/chain.sh &
# Guard: SIGSTOP the whole process group 13:25-20:05 UTC (live market), SIGCONT after, so this never
# fetches Alpaca or burns CPU while the owner is trading live on the shared account.
set -uo pipefail
cd /home/ec2-user/onemil
LOG=research/bf_2024/chain.log
export CHAIN_PGID=$$

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

setsid bash -c '
while true; do
  h=$(date -u +%H%M)
  if [ "$h" -ge 1325 ] && [ "$h" -lt 2005 ]; then
    kill -STOP -"$CHAIN_PGID" 2>/dev/null
    while true; do
      sleep 60
      h2=$(date -u +%H%M)
      if [ "$h2" -ge 2005 ] || [ "$h2" -lt 1325 ]; then
        kill -CONT -"$CHAIN_PGID" 2>/dev/null
        break
      fi
    done
  fi
  sleep 60
done
' > research/bf_2024/guard.log 2>&1 &
GUARD_PID=$!

{
  log "CHAIN START pid=$$ pgid=$CHAIN_PGID guard_pid=$GUARD_PID"

  log "STAGE 1 (build-cache, 2024-07-02..2024-12-31) begin"
  BF24_STAGE=1 python3 research/bf_2024/run_bf_2024.py
  log "STAGE 1 exit=$?"

  log "STAGE 2 (P1 filters from config.yaml) begin"
  BF24_STAGE=2 python3 research/bf_2024/run_bf_2024.py
  log "STAGE 2 exit=$?"

  log "SCORE begin"
  python3 research/bf_2024/score.py
  log "SCORE exit=$?"

  kill "$GUARD_PID" 2>/dev/null
  log "CHAIN DONE"
} >> "$LOG" 2>&1
