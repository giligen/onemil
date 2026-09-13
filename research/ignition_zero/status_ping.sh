#!/bin/bash
# hourly status for the ignition-from-zero study (owner: "send a telegram update with status every hour")
cd /home/ec2-user/onemil || exit 1
f=$(tail -1 research/ignition_zero/fetch.log 2>/dev/null | cut -c1-70)
b=$(tail -1 research/ignition_zero/build.log 2>/dev/null | cut -c1-90)
n=$(tail -1 research/ignition_zero/finra.log 2>/dev/null | cut -c1-50)
rows=$( [ -f research/ignition_zero/candidates.csv ] && wc -l < research/ignition_zero/candidates.csv || echo 0 )
/usr/bin/python3 scripts/send_telegram_alert.py "[IGNITION ZERO] $(date -u +%H:%M) UTC | bars fetch: ${f:-idle} | table build: ${b:-idle} (${rows} rows) | short interest: ${n:-idle}" >/dev/null 2>&1
