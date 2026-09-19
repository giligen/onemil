#!/bin/bash
# Full bar-walk variants of the EXIT physics (touchgo on/off) so Rule M and
# Rule D get a separation row.  Selector untouched.  Writes only under
# research/orb_frequency/.  One process at a time.
ulimit -v 3000000
cd /home/ec2-user/onemil
D=research/orb_frequency
export ORB_BT_FEATURES_CSV=analysis_results/orb_features_20260916_2053.csv
export ORB_BT_RISK=375 ORB_BT_N=8 ORB_BT_ACCOUNT=26666.666666666664

walk() {  # walk <tag>  (env for the variant set by caller)
  local tag=$1
  export ORB_BT_DUMP_CANDIDATES=$D/dump_$tag.csv
  export ORB_BT_BOOK_OUT=$D/walkbook_$tag.csv
  export ORB_BT_MONTHLY_OUT=$D/walkmonthly_$tag.csv
  nice -n 10 python3 -u study_orb_pipeline_static_lock.py > $D/walklog_$tag.txt 2>&1
  echo "$tag exit=$?"
}

( export ORB_TOUCHGO_ENABLED=0;        walk tg_off )
( export ORB_TOUCHGO_RULE_M_ENABLED=0; walk tgM_off )
( export ORB_TOUCHGO_RULE_D_ENABLED=0; walk tgD_off )
echo WALKS_DONE
