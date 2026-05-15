#!/bin/bash
# OOS validation: baseline (pb=5) vs B_pb10 (pb=10) on Jan-Dec 2025.
# Per-month parallelism: 12 months × 2 cells = 24 tasks, run 16 at a time.
set -uo pipefail

REPO=/home/ec2-user/onemil
cd "$REPO"

# Per-month build (Stage-1) for one (cell, month). Writes per-month cache.
build_s1_month() {
    local cell="$1" maxpb="$2" month="$3"
    local start="${month}-01"
    local end=$(date -d "${start} +1 month -1 day" +%Y-%m-%d)
    local cache="/tmp/oos2025_${cell}_${month}.csv"
    local log="/tmp/oos2025_${cell}_${month}_s1.log"
    local t0=$(date +%s)
    BF_MAX_PULLBACK_CANDLES="$maxpb" BF_FVRR_STRICT=1 \
    python3 batch_backtest.py \
        --start "$start" --end "$end" \
        --build-cache --no-cache \
        --cache-file "$cache" \
        --capital 5000 --risk 60 --max-shares 15000 \
        > "$log" 2>&1
    local rc=$?
    echo "S1[${cell}|${month}]: $(($(date +%s)-t0))s rc=$rc"
}

run_s2_month() {
    local cell="$1" maxpb="$2" month="$3"
    local start="${month}-01"
    local end=$(date -d "${start} +1 month -1 day" +%Y-%m-%d)
    local cache="/tmp/oos2025_${cell}_${month}.csv"
    local log="/tmp/oos2025_${cell}_${month}_s2.log"
    if [ ! -f "$cache" ] || [ ! -s "$cache" ]; then
        echo "S2[${cell}|${month}]: SKIP (no cache)"
        return
    fi
    local t0=$(date +%s)
    BF_MAX_PULLBACK_CANDLES="$maxpb" BF_FVRR_STRICT=1 \
    python3 batch_backtest.py \
        --start "$start" --end "$end" \
        --no-cache \
        --cache-file "$cache" \
        > "$log" 2>&1
    local rc=$?
    echo "S2[${cell}|${month}]: $(($(date +%s)-t0))s rc=$rc"
}
export -f build_s1_month run_s2_month

# Build the work list: cells × months
MONTHS="01 02 03 04 05 06 07 08 09 10 11 12"

echo "=========================================="
echo "PHASE 1 — parallel per-month Stage-1 (24 tasks, 16 at a time)"
echo "=========================================="
work_list=""
for cell in baseline B_pb10; do
    maxpb=$([ "$cell" = "baseline" ] && echo 5 || echo 10)
    for mm in $MONTHS; do
        work_list+="${cell}|${maxpb}|2025-${mm}\n"
    done
done
printf "$work_list" | xargs -I {} -P 16 bash -c '
    IFS="|" read -r cell maxpb month <<< "$1"
    build_s1_month "$cell" "$maxpb" "$month"
' _ {}

echo ""
echo "=========================================="
echo "PHASE 2 — parallel per-month Stage-2 (24 tasks, 16 at a time)"
echo "=========================================="
printf "$work_list" | xargs -I {} -P 16 bash -c '
    IFS="|" read -r cell maxpb month <<< "$1"
    run_s2_month "$cell" "$maxpb" "$month"
' _ {}

echo ""
echo "=========================================="
echo "OOS 2025 RESULTS — aggregated"
echo "=========================================="
python3 << 'PYEOF'
import os, re, glob
import collections

for cell in ['baseline', 'B_pb10']:
    months = sorted(glob.glob(f'/tmp/oos2025_{cell}_2025-*_s2.log'))
    monthly = []
    total_trades = 0
    total_pnl = 0.0
    total_wins = 0
    total_losses = 0
    for log in months:
        m = re.search(r'2025-(\d{2})_s2\.log$', log)
        month = m.group(1)
        with open(log) as f:
            txt = f.read()
        def grab(pattern, cast=int):
            mo = re.search(pattern, txt)
            return cast(mo.group(1)) if mo else None
        tr = grab(r'Total trades taken:\s+(\d+)') or 0
        pnl_raw = re.search(r'Total P&L:\s+\$?([+-]?[\d,\.]+)', txt)
        pnl = float(pnl_raw.group(1).replace(',','')) if pnl_raw else 0.0
        wins = grab(r'Winning trades:\s+(\d+)') or 0
        losses = grab(r'Losing trades:\s+(\d+)') or 0
        monthly.append((month, tr, pnl, wins, losses))
        total_trades += tr
        total_pnl += pnl
        total_wins += wins
        total_losses += losses
    wr = 100.0 * total_wins / max(total_trades, 1)
    print(f'\n=== {cell} (pb={5 if cell=="baseline" else 10}) ===')
    print(f'  Total: {total_trades} trades  ${total_pnl:+,.2f}  WR {wr:.1f}%')
    print(f'  Monthly P&L:')
    for mo, tr, pnl, w, l in monthly:
        print(f'    2025-{mo}: {tr:3d} trades  ${pnl:+10,.2f}  W{w}/L{l}')
PYEOF
