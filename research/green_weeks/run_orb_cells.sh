#!/bin/bash
# green_weeks — the shipped ORB selector, once per exit cell, at the LIVE N=8.
# Selection is identical across cells by construction (same features CSV, same
# orb.yaml stack); only the per-candidate exit in the dump moves.
# Per-position cap held at the live $3,333.33 (account = 3333.33 x N).
ulimit -v 3000000
cd /home/ec2-user/onemil
D=research/green_weeks
export ORB_BT_FEATURES_CSV=analysis_results/orb_features_20260916_2053.csv
export ORB_BT_RISK=375

run() {  # run <cell> <N>
  local sh=$1 n=$2
  local acct=$(python3 -c "print(3333.333333333333*$n)")
  local tag="${sh}_n${n}"
  echo "=== $tag  acct=$acct"
  (
    export ORB_BT_RESIM_CACHE=$D/dump_${sh}.csv
    export ORB_BT_N=$n ORB_BT_ACCOUNT=$acct ORB_BT_RISK=375 ORB_SKIP_Q1=1
    export ORB_BT_BOOK_OUT=$D/orb_book_$tag.csv
    export ORB_BT_MONTHLY_OUT=$D/orb_monthly_$tag.csv
    nice -n 10 python3 -u study_orb_pipeline_static_lock.py
  ) > $D/orb_log_$tag.txt 2>&1
  echo "    exit=$?"
}

CELLS="${*:-X0 E1a E1b E1c E1d E2a E2b E2c E3 E4a E4b E5}"
for sh in $CELLS; do
  run $sh 8
done
echo ORBCELLSDONE
