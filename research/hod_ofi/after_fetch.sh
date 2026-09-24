#!/bin/bash
# Waits for the OFI fetch to exit, then features -> score -> Telegram ping (no numbers; reviewed before relayed).
cd /home/ec2-user/onemil
while pgrep -f "research/hod_ofi/pipeline.py fetch" >/dev/null; do sleep 60; done
echo "[$(date -u +%H:%M:%S)] fetch exited; features" >> research/hod_ofi/after.log
python3 research/hod_ofi/pipeline.py features >> research/hod_ofi/after.log 2>&1; echo "features rc=$?" >> research/hod_ofi/after.log
python3 research/hod_ofi/pipeline.py score >> research/hod_ofi/after.log 2>&1; echo "score rc=$?" >> research/hod_ofi/after.log
python3 scripts/send_telegram_alert.py "[RESEARCH] HOD order-flow study (cells 1,393-1,395) finished $(date -u +%H:%M) UTC. Reviewed before any number is relayed - ping the Claude session." >/dev/null 2>&1
echo "AFTER DONE" >> research/hod_ofi/after.log
