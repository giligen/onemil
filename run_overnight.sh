#!/bin/bash
# Overnight pipeline: wait for cache builds, then run full analysis
# Run with: bash run_overnight.sh

set -e
echo "=== OVERNIGHT PIPELINE STARTED $(date) ==="

# Wait for cache builds to finish
echo "Waiting for cache builds to complete..."
while pgrep -f "batch_backtest.py --start 2025" > /dev/null 2>&1; do
    sleep 30
    echo "  Still building... $(date +%H:%M)"
done
echo "Cache builds complete at $(date)"

# Verify all cache files exist
echo ""
echo "=== CACHE FILES ==="
for t in 8 15 20 5; do
    f="data/bull_flag_cache_e50_x30_t${t}.csv"
    if [ -f "$f" ]; then
        lines=$(wc -l < "$f")
        echo "  ${t}%: $f ($lines lines)"
    else
        echo "  ${t}%: MISSING!"
    fi
done

# Restore baseline
cp data/bull_flag_cache_e50_x30_t20.csv data/bull_flag_cache_e50_x30.csv
echo "Baseline 20% cache restored"

# Run full analysis
echo ""
echo "=== RUNNING FULL ANALYSIS ==="
python3 run_full_analysis.py 2>&1 | tee analysis_results/analysis_log.txt

echo ""
echo "=== OVERNIGHT PIPELINE COMPLETE $(date) ==="
echo "Results in: analysis_results/threshold_sweep_summary.md"
