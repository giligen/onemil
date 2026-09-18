#!/bin/bash
# Anchor-dedup cells — PREREG.md §3. Same recipe as D1_orb/run_grid.sh.
ulimit -v 2000000
cd /home/ec2-user/onemil
D=research/orb_anchor_dedup/runs
export ORB_BT_FEATURES_CSV=analysis_results/orb_features_20260916_2053.csv
export ORB_BT_RESIM_CACHE=research/fuckup_audit/D1_orb/candidates_dump.csv
export ORB_BT_RISK=375

run() {  # run <tag> <N> <anchor_dedup>
  local tag=$1 n=$2 ad=$3
  local acct=$(python3 -c "print(3333.333333333333*$n)")
  (
    export ORB_BT_N=$n ORB_BT_ACCOUNT=$acct ORB_SKIP_Q1=1
    export ORB_ANCHOR_DEDUP=$ad
    export ORB_ANCHOR_DEDUP_DROPPED_OUT=$D/dropped_$tag.csv
    export ORB_BT_BOOK_OUT=$D/book_$tag.csv ORB_BT_MONTHLY_OUT=$D/monthly_$tag.csv
    nice -n 10 python3 -u study_orb_pipeline_static_lock.py
  ) > $D/log_$tag.txt 2>&1
  echo "  $tag exit=$?"
}
run "$@"
