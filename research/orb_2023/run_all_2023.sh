#!/bin/bash
cd /home/ec2-user/onemil
export PYTHONPATH=/home/ec2-user/onemil
LOG=research/orb_2023/run_all.log
: > "$LOG"
echo "=== START $(date -u +%H:%M:%S) ===" >> "$LOG"
for CELL in 1418 1419; do
  echo "--- build $CELL start $(date -u +%H:%M:%S)" >> "$LOG"
  Y23_CELL=$CELL python3 research/orb_2023/build_features_2023.py >> "$LOG" 2>&1
  echo "--- build $CELL rc=$? done $(date -u +%H:%M:%S)" >> "$LOG"
done
for CELL in 1418 1419; do
  FCSV=$(ls research/orb_2023/out_${CELL}/*.csv 2>/dev/null | grep -v VERSION | head -1)
  echo "--- pipeline $CELL features=$FCSV start $(date -u +%H:%M:%S)" >> "$LOG"
  if [ -n "$FCSV" ]; then
    ORB_BT_FEATURES_CSV=$FCSV ORB_BT_BOOK_OUT=research/orb_2023/book_${CELL}.csv ORB_CATALYST_VETO=0 \
      python3 study_orb_pipeline_static_lock.py >> "$LOG" 2>&1
    echo "--- pipeline $CELL rc=$? done $(date -u +%H:%M:%S)" >> "$LOG"
  else
    echo "--- pipeline $CELL SKIPPED: no features csv found" >> "$LOG"
  fi
done
echo "--- score start $(date -u +%H:%M:%S)" >> "$LOG"
python3 research/orb_2023/score_2023.py >> "$LOG" 2>&1
echo "--- score rc=$? done $(date -u +%H:%M:%S)" >> "$LOG"
echo "=== ALL DONE $(date -u +%H:%M:%S) ===" >> "$LOG"
