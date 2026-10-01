#!/bin/bash
set -x
cd /home/ec2-user/onemil
export PYTHONPATH=/home/ec2-user/onemil

ORB1684_WINDOW=in_regime nice -n 10 python3 research/orb_freq/1684_features.py \
  > research/orb_freq/1684_features_in.log 2>&1
echo "[run_rest] features in_regime rc=$? $(date -u)"

nice -n 10 python3 research/orb_freq/1684_pipeline.py --window in_regime \
  > research/orb_freq/1684_pipeline_in.log 2>&1
echo "[run_rest] pipeline in_regime rc=$? $(date -u)"

nice -n 10 python3 research/orb_freq/1684_pipeline.py --window out_regime \
  > research/orb_freq/1684_pipeline_out.log 2>&1
echo "[run_rest] pipeline out_regime rc=$? $(date -u)"

echo "[run_rest] ALL DONE $(date -u)"
