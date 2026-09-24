#!/bin/bash
# Guarded rerun of the corrected OFI pipeline (SPEC_FIX.md, run 2). nohup/setsid-safe:
# no controlling-terminal dependency, every stream redirected to rerun.log.
# Waits out the market-hours blackout [13:25, 20:05) UTC (shared pipeline.blackout_wait(),
# same rule the fetch step already uses) before the full features step, which reads all
# 1,131 raw files (~70 min) -- never run that step live during market hours.
# nice/ionice keep it off the trading process's CPU/disk.
#
# Launch (owner's call, NOT from this session):
#   nohup setsid research/hod_ofi/rerun.sh >> research/hod_ofi/rerun.log 2>&1 &
set -uo pipefail
cd /home/ec2-user/onemil
LOG=research/hod_ofi/rerun.log

echo "[$(date -u +%H:%M:%S)] rerun start (pid $$); waiting out blackout if needed" >> "$LOG"
python3 -c "
import sys; sys.path.insert(0, 'research/hod_ofi')
import pipeline
pipeline.blackout_wait()
" >> "$LOG" 2>&1

echo "[$(date -u +%H:%M:%S)] clear of blackout; features (full, all raw files)" >> "$LOG"
nice -n 19 ionice -c3 python3 research/hod_ofi/pipeline.py features >> "$LOG" 2>&1
echo "features rc=$?" >> "$LOG"

echo "[$(date -u +%H:%M:%S)] score" >> "$LOG"
nice -n 19 ionice -c3 python3 research/hod_ofi/pipeline.py score >> "$LOG" 2>&1
echo "score rc=$?" >> "$LOG"

python3 scripts/send_telegram_alert.py "[RESEARCH] HOD order-flow study (cells 1,393-1,395) RERUN finished $(date -u +%H:%M) UTC. Reviewed before any number is relayed - ping the Claude session." >/dev/null 2>&1

echo "[$(date -u +%H:%M:%S)] RERUN DONE" >> "$LOG"
