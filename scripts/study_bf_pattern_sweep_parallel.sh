#!/bin/bash
# Parallel BT sweep — run all cells concurrently (32 cores available).
# Phase 1: build any missing Stage-1 caches in parallel
# Phase 2: run all Stage-2s in parallel against their caches
set -uo pipefail

REPO=/home/ec2-user/onemil
cd "$REPO"

START=2026-01-01
END=2026-04-30
OUT=/tmp/sweep_results.csv

build_s1() {
    local cell="$1" maxpb="$2" fvrr="$3"
    local cache="/tmp/sweep_${cell}.csv"
    local log="/tmp/sweep_${cell}_s1.log"
    if [ -f "$cache" ] && [ -s "$cache" ]; then
        echo "S1[$cell]: cache exists, skip build"
        return 0
    fi
    echo "S1[$cell]: building (maxpb=$maxpb, fvrr=$fvrr)..."
    local t0=$(date +%s)
    BF_MAX_PULLBACK_CANDLES="$maxpb" BF_FVRR_STRICT="$fvrr" \
    python3 batch_backtest.py \
        --start "$START" --end "$END" \
        --build-cache --no-cache \
        --cache-file "$cache" \
        --capital 5000 --risk 60 --max-shares 15000 \
        > "$log" 2>&1
    local rc=$?
    echo "S1[$cell]: done in $(($(date +%s)-t0))s (rc=$rc)"
}

run_s2() {
    local cell="$1" maxpb="$2" fvrr="$3"
    local cache="/tmp/sweep_${cell}.csv"
    local log="/tmp/sweep_${cell}_s2.log"
    if [ ! -f "$cache" ]; then
        echo "S2[$cell]: no cache, skip"
        return 1
    fi
    echo "S2[$cell]: running..."
    local t0=$(date +%s)
    BF_MAX_PULLBACK_CANDLES="$maxpb" BF_FVRR_STRICT="$fvrr" \
    python3 batch_backtest.py \
        --start "$START" --end "$END" \
        --no-cache \
        --cache-file "$cache" \
        > "$log" 2>&1
    local rc=$?
    echo "S2[$cell]: done in $(($(date +%s)-t0))s (rc=$rc)"
}

export -f build_s1 run_s2
export START END

echo "=========================================="
echo "PHASE 1 — parallel Stage-1 builds"
echo "=========================================="
build_s1 baseline 5  1 &
build_s1 A_pb7    7  1 &
build_s1 B_pb10   10 1 &
build_s1 C_fvroff 5  0 &
build_s1 D_both   7  0 &
wait
echo ""

echo "=========================================="
echo "PHASE 2 — parallel Stage-2 runs"
echo "=========================================="
run_s2 baseline 5  1 &
run_s2 A_pb7    7  1 &
run_s2 B_pb10   10 1 &
run_s2 C_fvroff 5  0 &
run_s2 D_both   7  0 &
wait
echo ""

echo "=========================================="
echo "RESULTS"
echo "=========================================="
echo "cell,max_pullback,fvrr_strict,s1_trades,s1_pnl,s2_trades,s2_pnl" > "$OUT"
for combo in "baseline 5 1" "A_pb7 7 1" "B_pb10 10 1" "C_fvroff 5 0" "D_both 7 0"; do
    set -- $combo
    cell=$1 maxpb=$2 fvrr=$3
    s1=$(grep -E "^\s*TOTAL\s+" /tmp/sweep_${cell}_s1.log 2>/dev/null | tail -1)
    s2=$(grep -E "^\s*TOTAL\s+" /tmp/sweep_${cell}_s2.log 2>/dev/null | tail -1)
    s1_tr=$(echo "$s1" | awk '{print $3}')
    s1_pnl=$(echo "$s1" | awk '{print $4}' | tr -d '$,')
    s2_tr=$(echo "$s2" | awk '{print $3}')
    s2_pnl=$(echo "$s2" | awk '{print $4}' | tr -d '$,')
    echo "${cell},${maxpb},${fvrr},${s1_tr},${s1_pnl},${s2_tr},${s2_pnl}" >> "$OUT"
done
column -t -s, "$OUT"
