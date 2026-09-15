#!/bin/bash
# parity review 2026-09-15: wait for superset_provenance.csv, smoke-fetch 2 days, then the full Alpaca-SIP refetch.
cd /home/ec2-user/onemil
L=research/bf_zero/parity_review/refetch.log
until grep -q "CHAIN_EXIT" research/bf_zero/parity_review/provenance_run.log; do sleep 15; done
tail -4 research/bf_zero/parity_review/provenance_run.log >> $L
ulimit -v 3000000
nice -n 10 python3 research/bf_zero/refetch_thin_tape.py --fetch --days-to 2025-01-03 >> $L 2>&1; echo "SMOKE_EXIT=$?" >> $L
nice -n 10 python3 research/bf_zero/refetch_thin_tape.py --fetch >> $L 2>&1; echo "FETCH_EXIT=$?" >> $L
