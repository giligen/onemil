#!/bin/bash
# Cell 1,684: chain the remaining stages into one background job so the controlling agent only
# needs a couple of check-ins instead of one per stage.
set -x
cd /home/ec2-user/onemil
export PYTHONPATH=/home/ec2-user/onemil

# 1. wait for the backfill (bars_sip.db append) to finish
while kill -0 598127 2>/dev/null; do sleep 15; done
echo "[run_all] backfill done $(date -u)"

# 2. wait for the fastpath (prod/idea1/idea2 in-regime) to finish
while kill -0 597448 2>/dev/null; do sleep 10; done
echo "[run_all] fastpath done $(date -u)"

# 3. fresh-path feature builds (idea10/11 in-regime; all-4 out-regime)
ORB1684_WINDOW=in_regime nice -n 10 python3 research/orb_freq/1684_features.py \
  > research/orb_freq/1684_features_in.log 2>&1
echo "[run_all] features in_regime rc=$? $(date -u)"

ORB1684_WINDOW=out_regime nice -n 10 python3 research/orb_freq/1684_features.py \
  > research/orb_freq/1684_features_out.log 2>&1
echo "[run_all] features out_regime rc=$? $(date -u)"

# 4. pipeline runs per pool per window
nice -n 10 python3 research/orb_freq/1684_pipeline.py --window in_regime \
  > research/orb_freq/1684_pipeline_in.log 2>&1
echo "[run_all] pipeline in_regime rc=$? $(date -u)"

nice -n 10 python3 research/orb_freq/1684_pipeline.py --window out_regime \
  > research/orb_freq/1684_pipeline_out.log 2>&1
echo "[run_all] pipeline out_regime rc=$? $(date -u)"

echo "[run_all] ALL STAGES DONE $(date -u)"
