#!/bin/bash
# W=5 identity check of the MODIFIED feature builder: recompute the most recent
# dates into a side dir and compare row-for-row with the production CSV.
# ulimit 4,500,000 KB = PLAN §1's sanctioned ceiling for a pass-1 rebuild;
# 3,000,000 is not enough for load_daily_bars_frame (5.0M daily_bars rows).
ulimit -v 4500000
cd /home/ec2-user/onemil
export ORB_FEATURES_OUT_DIR=research/orb_multiwindow/w5check
nice -n 10 python3 -u study_orb_features.py --start-date 2026-09-08 \
    > research/orb_multiwindow/w5check/spotcheck.log 2>&1
echo "EXIT=$?"
