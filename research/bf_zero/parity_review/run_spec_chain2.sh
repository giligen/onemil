#!/bin/bash
# 9/15 22:55 UTC: the first chain died at `import build_candidates` under ulimit -v 3 GB. Same steps, 5.5 GB cap, one process at a time.
cd /home/ec2-user/onemil
PR=research/bf_zero/parity_review; L=$PR/spec_chain2.log; : > $L
ulimit -v 5500000
echo "$(date -u +%T) spec_sim start (BFZ_SIP_STORE, full re-run, state already reset)" >> $L
BFZ_SIP_STORE=/home/ec2-user/onemil/research/bf_zero/bars_sip.db nice -n 10 python3 research/bf_zero/spec_sim.py > $PR/spec_sim_sip.log 2>&1; echo "SPEC_EXIT=$?" >> $L
nice -n 10 python3 research/bf_zero/capacity_8a.py > $PR/capacity_8a_sip.log 2>&1; echo "CAP_EXIT=$?" >> $L
BFZ_SIP_STORE=/home/ec2-user/onemil/research/bf_zero/bars_sip.db nice -n 10 python3 research/bf_zero/refetch_thin_tape.py --verify 20 >> $L 2>&1; echo "VERIFY_EXIT=$?" >> $L
echo "$(date -u +%T) CHAIN2 DONE" >> $L
