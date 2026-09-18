#!/bin/bash
# One pipeline book run for the meta-label study.
#   run_book.sh <tag> <arm> [sidecar.csv] [rank_col] [veto_col] [veto_thr]
# Slots are FIXED at 8 (D1's dose-response optimum, the PREREG's book) and the
# per-position cap is the $10K-stage $3,333.33, so sizing is identical in every
# cell — the model may only reorder and veto.
set -u
ulimit -v 1500000
cd /home/ec2-user/onemil
M=research/meta_label
tag=$1; arm=$2; side=${3:-}; rank=${4:-}; veto=${5:-}; thr=${6:-0.5}
mkdir -p $M/books $M/logs
export ORB_BT_FEATURES_CSV=analysis_results/orb_features_20260916_2053.csv
export ORB_BT_RESIM_CACHE=research/fuckup_audit/Q_fill/dump_${arm}.csv
export ORB_BT_RISK=375 ORB_BT_N=8 ORB_SKIP_Q1=1
export ORB_BT_ACCOUNT=26666.666666666664
export ORB_BT_BOOK_OUT=$M/books/book_${tag}_${arm}.csv
export ORB_BT_MONTHLY_OUT=$M/books/monthly_${tag}_${arm}.csv
[ -n "$side" ] && export ORB_BT_SIDECAR_CSV=$side
[ -n "$rank" ] && export ORB_META_RANK_COL=$rank
[ -n "$veto" ] && export ORB_META_VETO_COL=$veto
[ -n "$veto" ] && export ORB_META_VETO_THR=$thr
nice -n 15 python3 -u study_orb_pipeline_static_lock.py > $M/logs/${tag}_${arm}.txt 2>&1
echo "$tag/$arm exit=$?"
