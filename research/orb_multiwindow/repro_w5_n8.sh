#!/bin/bash
# Step 0 reproduction gate: D1_orb book_n8_q1on.csv ($14,428.62 / 215 picks)
ulimit -v 3000000
cd /home/ec2-user/onemil
D=research/orb_multiwindow
export ORB_BT_FEATURES_CSV=analysis_results/orb_features_20260916_2053.csv
export ORB_BT_RESIM_CACHE=research/fuckup_audit/D1_orb/candidates_dump.csv
export ORB_BT_RISK=375
export ORB_BT_N=8
export ORB_BT_ACCOUNT=$(python3 -c "print(3333.333333333333*8)")
export ORB_SKIP_Q1=1
export ORB_BT_BOOK_OUT=$D/repro_w5_n8_book.csv
export ORB_BT_MONTHLY_OUT=$D/repro_w5_n8_monthly.csv
nice -n 10 python3 -u study_orb_pipeline_static_lock.py > $D/repro_w5_n8.log 2>&1
echo "EXIT=$?"
tail -5 $D/repro_w5_n8.log
