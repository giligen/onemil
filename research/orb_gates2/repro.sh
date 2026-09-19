#!/bin/bash
# Reproduction gate: the as-is dump at N=8, Q1 on must reproduce
# research/fuckup_audit/D1_orb/book_n8_q1on.csv to the cent.
ulimit -v 3000000
cd /home/ec2-user/onemil
D=research/orb_gates2
export ORB_BT_FEATURES_CSV=analysis_results/orb_features_20260916_2053.csv
export ORB_BT_RESIM_CACHE=research/fuckup_audit/D1_orb/candidates_dump.csv
export ORB_BT_RISK=375 ORB_BT_N=8 ORB_BT_ACCOUNT=26666.666666666664 ORB_SKIP_Q1=1
export ORB_BT_BOOK_OUT=$D/repro_n8_q1on.csv ORB_BT_MONTHLY_OUT=$D/repro_n8_monthly.csv
nice -n 10 python3 -u study_orb_pipeline_static_lock.py
echo "EXIT=$?"
