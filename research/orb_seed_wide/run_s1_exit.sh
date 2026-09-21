#!/bin/bash
# S1 exit pass (PREREG_S1_EXIT.md, cells 1,319-1,321): walk the S1 population under three exit
# variants, one pipeline run each, sequential, nice'd. Baseline = out/runS1_true.csv (already walked).
# Launch: nohup nice -n 10 bash research/orb_seed_wide/run_s1_exit.sh > research/orb_seed_wide/out/s1_exit.log 2>&1 &
set -u
cd /home/ec2-user/onemil
OUT=research/orb_seed_wide/out
export PYTHONPATH=/home/ec2-user/onemil
export ORB_BT_FEATURES_CSV=$OUT/runS1_features.csv
unset ORB_BT_RESIM_CACHE

run() {  # run <tag> <env assignments...>
  local tag=$1; shift
  echo "=== $tag start $(date -u +%H:%M:%S) env: $* ==="
  env "$@" ORB_BT_BOOK_OUT=$OUT/runS1_$tag.csv python3 study_orb_pipeline_static_lock.py > $OUT/runS1_$tag.log 2>&1
  echo "=== $tag done  $(date -u +%H:%M:%S) rc=$? ==="
  grep -h "Cum P&L\|WARNING: exit env" $OUT/runS1_$tag.log
}

run E1 ORB_TOUCHGO_ENABLED=0
run E2 ORB_BT_LOCK_ARM_R=1.0
run E3 ORB_BT_SCALE_FRAC=0.5 ORB_BT_SCALE_LEVEL_R=2.0
echo ALL DONE
