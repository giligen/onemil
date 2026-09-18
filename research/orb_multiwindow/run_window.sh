#!/bin/bash
# Build one opening-range window end to end (research/orb_multiwindow/PREREG.md).
#
#   bash run_window.sh <W>
#
# Stage 1  features   — full regen at ORB_RANGE_MINUTES=W into w<W>/ (never
#                       analysis_results/; the code-version stamp carries W).
# Stage 2  bar walk   — pipeline once with ORB_BT_DUMP_CANDIDATES to freeze the
#                       shipped exit physics for every candidate of that window.
# Stage 3  selector   — pipeline off the dump (seconds) with the TRAIN-refit
#                       z-params/cutoffs and the window's own range-size veto,
#                       dumping the per-day ranked list for the combiner.
#
# One python process at a time, nice -n 10, ulimit -v 3000000, no output pipes.
set -e
W=${1:?usage: run_window.sh W}
cd /home/ec2-user/onemil
D=research/orb_multiwindow
OUT=$D/w$W
mkdir -p $OUT
# 4,500,000 KB = PLAN §1's sanctioned ceiling for a pass-1 rebuild. 3,000,000
# is not enough for load_daily_bars_frame (5.0M daily_bars rows) — measured.
ulimit -v 4500000

export ORB_RANGE_MINUTES=$W
export ORB_FEATURES_OUT_DIR=$OUT

if [ "$SKIP_FEATURES" != "1" ]; then
  echo "=== [$(date -u +%H:%M:%S)] W=$W stage 1: features (full regen) ==="
  nice -n 10 python3 -u study_orb_features.py --force-full-regen \
      > $OUT/features.log 2>&1
  echo "features exit=$?"
fi
FEAT=$(ls -1 $OUT/orb_features_*.csv | grep -v corrmatrix | tail -1)
echo "features CSV: $FEAT"

export ORB_BT_FEATURES_CSV=$FEAT
export ORB_BT_RISK=375
export ORB_BT_N=8
export ORB_BT_ACCOUNT=$(python3 -c "print(3333.333333333333*8)")
export ORB_SKIP_Q1=1
# TRAIN-refit selection (PREREG): the W=5 frozen fit is meaningless on a
# different range width. Same procedure as scripts/orb_weekly_refit.py;
# threshold and adaptive_mults untouched; TRAIN = the PLAN §1 split.
export ORB_BT_REFIT_ZPARAMS=1
export ORB_BT_TRAIN_START=2025-01-01
export ORB_BT_TRAIN_END=2025-12-31
export BT_ALLOW_REFIT=1

if [ "$SKIP_WALK" != "1" ]; then
  echo "=== [$(date -u +%H:%M:%S)] W=$W stage 2: bar walk + candidate dump ==="
  ORB_BT_DUMP_CANDIDATES=$OUT/candidates_dump.csv \
  ORB_BT_BOOK_OUT=$OUT/walk_book.csv \
  ORB_BT_MONTHLY_OUT=$OUT/walk_monthly.csv \
  nice -n 10 python3 -u study_orb_pipeline_static_lock.py > $OUT/walk.log 2>&1
  echo "walk exit=$?"
fi

echo "=== [$(date -u +%H:%M:%S)] W=$W stage 3: selector + ranked dump ==="
RS_MIN=${RS_MIN:-2.221}
ORB_BT_RESIM_CACHE=$OUT/candidates_dump.csv \
ORB_RANGE_SIZE_VETO_MIN_PCT=$RS_MIN \
ORB_BT_DUMP_RANKED=$OUT/ranked.csv \
ORB_BT_BOOK_OUT=$OUT/book_alone_n8.csv \
ORB_BT_MONTHLY_OUT=$OUT/monthly_alone_n8.csv \
nice -n 10 python3 -u study_orb_pipeline_static_lock.py > $OUT/sel.log 2>&1
echo "sel exit=$?"
tail -3 $OUT/sel.log
