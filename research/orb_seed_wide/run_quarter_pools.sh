#!/bin/bash
# Past-quarter (TEST, owner-unsealed 9/21) walks of the three pools, sequential.
cd /home/ec2-user/onemil; export PYTHONPATH=/home/ec2-user/onemil; unset ORB_BT_RESIM_CACHE
for k in prodQ gap4Q p30Q; do
  echo "=== $k start $(date -u +%H:%M:%S)"
  env ORB_BT_FEATURES_CSV=research/orb_seed_wide/out/run${k}_features.csv ORB_BT_BOOK_OUT=research/orb_seed_wide/out/run${k}_true.csv python3 study_orb_pipeline_static_lock.py > research/orb_seed_wide/out/run${k}_true.log 2>&1
  echo "=== $k done $(date -u +%H:%M:%S) rc=$?"; grep -h "Cum P&L" research/orb_seed_wide/out/run${k}_true.log
done; echo ALL DONE
