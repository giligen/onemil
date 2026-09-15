#!/bin/bash
# bf_zero2 chain: wait for the universe fetch → provenance gate → pass 1 (SIP store) → pass 2 → score2 → telegram. One heavy process at a time.
cd /home/ec2-user/onemil; L=research/bf_zero2/chain.log; : > $L
until grep -q "FETCH5_EXIT" research/bf_zero2/fetch_universe.log; do sleep 60; done
echo "$(date -u +%T) fetch: $(grep FETCH5_EXIT research/bf_zero2/fetch_universe.log)" >> $L
ulimit -v 5500000
nice -n 10 python3 research/bf_zero/parity_review/tape_provenance_check.py >> $L 2>&1; echo "PROVENANCE_EXIT=$?" >> $L
echo "$(date -u +%T) pass 1 start" >> $L
BFZ_SIP_STORE=/home/ec2-user/onemil/research/bf_zero/bars_sip.db nice -n 10 python3 research/bf_zero2/build_candidates2.py > research/bf_zero2/build.log 2>&1; echo "PASS1_EXIT=$?" >> $L
echo "$(date -u +%T) pass 2 start" >> $L
BFZ_DIR=research/bf_zero2 nice -n 10 python3 research/bf_zero/pass2.py > research/bf_zero2/pass2.log 2>&1; echo "PASS2_EXIT=$?" >> $L
echo "$(date -u +%T) score start" >> $L
nice -n 10 python3 research/bf_zero2/score2.py > research/bf_zero2/score.log 2>&1; echo "SCORE_EXIT=$?" >> $L
echo "$(date -u +%T) CHAIN DONE" >> $L
python3 scripts/send_telegram_alert.py "[BF ZERO2] square-one scan finished: $(grep -c . research/bf_zero2/candidates.csv 2>/dev/null) candidate rows; $(tail -1 research/bf_zero2/score.log | cut -c 1-200)" >/dev/null 2>&1
