#!/bin/bash
# BT sweep over (max_pullback_candles, fvrr_strict) over Jan-Apr 2026.
#
# Each cell builds its own Stage-1 cache via env overrides on
# pattern_detector.py, then runs Stage-2 against that cache.
# Production config.yaml is NEVER touched.
set -euo pipefail

REPO=/home/ec2-user/onemil
cd "$REPO"

START=2026-01-01
END=2026-04-30
OUT=/tmp/sweep_results.csv

echo "cell,max_pullback,fvrr_strict,stage1_trades,stage1_pnl,stage2_trades,stage2_pnl,build_seconds" > "$OUT"

run_cell() {
    local cell="$1"
    local maxpb="$2"
    local fvrr="$3"
    local cache="/tmp/sweep_${cell}.csv"
    local s1log="/tmp/sweep_${cell}_s1.log"
    local s2log="/tmp/sweep_${cell}_s2.log"

    echo ""
    echo "================================================================"
    echo "CELL ${cell}: max_pullback=${maxpb}, fvrr_strict=${fvrr}"
    echo "================================================================"

    local t0=$(date +%s)
    BF_MAX_PULLBACK_CANDLES="$maxpb" BF_FVRR_STRICT="$fvrr" \
    python3 batch_backtest.py \
        --start "$START" --end "$END" \
        --build-cache --no-cache \
        --cache-file "$cache" \
        --capital 5000 --risk 60 --max-shares 15000 \
        > "$s1log" 2>&1
    local t1=$(date +%s)
    local build_s=$((t1 - t0))

    # Extract Stage-1 totals from log
    local s1_trades=$(grep -E "^\s*TOTAL\s+" "$s1log" | tail -1 | awk '{print $3}')
    local s1_pnl=$(grep -E "^\s*TOTAL\s+" "$s1log" | tail -1 | awk '{print $4}' | tr -d '$,')

    # Stage-2: re-run against this cache (config.yaml's production filters)
    BF_MAX_PULLBACK_CANDLES="$maxpb" BF_FVRR_STRICT="$fvrr" \
    python3 batch_backtest.py \
        --start "$START" --end "$END" \
        --no-cache \
        --cache-file "$cache" \
        > "$s2log" 2>&1 || true
    local s2_trades=$(grep -E "^\s*TOTAL\s+" "$s2log" | tail -1 | awk '{print $3}')
    local s2_pnl=$(grep -E "^\s*TOTAL\s+" "$s2log" | tail -1 | awk '{print $4}' | tr -d '$,')

    echo "${cell},${maxpb},${fvrr},${s1_trades},${s1_pnl},${s2_trades},${s2_pnl},${build_s}" >> "$OUT"
    echo "Done ${cell}: S1=${s1_trades} trades / \$${s1_pnl}   S2=${s2_trades} trades / \$${s2_pnl}   (${build_s}s)"
}

run_cell baseline 5  1
run_cell A_pb7    7  1
run_cell B_pb10   10 1
run_cell C_fvroff 5  0
run_cell D_both   7  0

echo ""
echo "================================================================"
echo "SWEEP COMPLETE — results at $OUT"
echo "================================================================"
column -t -s, "$OUT"
