#!/bin/bash
# Wide detector-knob sweep on Jan-Dec 2025.
# 6 cells × 12 months × 2 phases = 144 tasks.
# Single-threaded BLAS to prevent thread explosion → -P 8 parallel.
set -uo pipefail

REPO=/home/ec2-user/onemil
cd "$REPO"

# Force BLAS / numpy / pandas to single-thread per process. Without this,
# each process spawns ~32 threads → load avg explodes under -P 24.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Deterministic ordering — cell:envs pairs, in order.
CELL_ORDER=(
  "A_pole2|BF_MIN_POLE_CANDLES=2"
  "B_polegain2|BF_MIN_POLE_GAIN_PCT=2.0"
  "C_retr70|BF_MAX_RETRACEMENT_PCT=70"
  "D_green3|BF_MAX_GREEN_IN_FLAG=3"
  "E_vol1|BF_MIN_BREAKOUT_VOLUME_RATIO=1.0"
  "F_combo|BF_MAX_PULLBACK_CANDLES=10 BF_MIN_POLE_CANDLES=2 BF_MAX_RETRACEMENT_PCT=70 BF_MAX_GREEN_IN_FLAG=3 BF_MIN_BREAKOUT_VOLUME_RATIO=1.0"
)

build_s1_month() {
    local cell="$1" envs="$2" month="$3"
    local start="${month}-01"
    local end=$(date -d "${start} +1 month -1 day" +%Y-%m-%d)
    local cache="/tmp/wide_${cell}_${month}.csv"
    local log="/tmp/wide_${cell}_${month}_s1.log"
    if [ -f "$cache" ] && [ -s "$cache" ]; then
        echo "S1[${cell}|${month}]: cache exists, skip"
        return
    fi
    local t0=$(date +%s)
    env $envs OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
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
    local cell="$1" envs="$2" month="$3"
    local start="${month}-01"
    local end=$(date -d "${start} +1 month -1 day" +%Y-%m-%d)
    local cache="/tmp/wide_${cell}_${month}.csv"
    local log="/tmp/wide_${cell}_${month}_s2.log"
    if [ ! -f "$cache" ] || [ ! -s "$cache" ]; then
        echo "S2[${cell}|${month}]: SKIP (no cache)"
        return
    fi
    local t0=$(date +%s)
    env $envs OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    python3 batch_backtest.py \
        --start "$start" --end "$end" \
        --no-cache \
        --cache-file "$cache" \
        > "$log" 2>&1
    local rc=$?
    echo "S2[${cell}|${month}]: $(($(date +%s)-t0))s rc=$rc"
}
export -f build_s1_month run_s2_month

MONTHS="01 02 03 04 05 06 07 08 09 10 11 12"

# Build deterministic work list
work=""
for cell_pair in "${CELL_ORDER[@]}"; do
    IFS='|' read -r cell envs <<< "$cell_pair"
    for mm in $MONTHS; do
        work+="${cell}|${envs}|2025-${mm}\n"
    done
done

echo "=========================================="
echo "PHASE 1 — Stage-1 builds (72 tasks, 8 parallel, BLAS=1)"
echo "=========================================="
printf "$work" | xargs -d '\n' -I {} -P 8 bash -c '
    IFS="|" read -r cell envs month <<< "$1"
    build_s1_month "$cell" "$envs" "$month"
' _ {}

echo ""
echo "=========================================="
echo "PHASE 2 — Stage-2 runs (72 tasks, 8 parallel)"
echo "=========================================="
printf "$work" | xargs -d '\n' -I {} -P 8 bash -c '
    IFS="|" read -r cell envs month <<< "$1"
    run_s2_month "$cell" "$envs" "$month"
' _ {}

echo ""
echo "=========================================="
echo "RESULTS — 2025 OOS, all cells"
echo "=========================================="
python3 << 'PYEOF'
import re, glob

cells_to_include = ['baseline', 'A_pole2', 'B_polegain2', 'C_retr70', 'D_green3', 'E_vol1', 'F_combo']

def aggregate(cell):
    if cell == 'baseline':
        logs = sorted(glob.glob(f'/tmp/oos2025_baseline_2025-*_s2.log'))
    else:
        logs = sorted(glob.glob(f'/tmp/wide_{cell}_2025-*_s2.log'))
    total_trades = 0
    total_pnl = 0.0
    total_wins = 0
    for log in logs:
        try:
            with open(log) as f:
                txt = f.read()
            tr = re.search(r'Total trades taken:\s+(\d+)', txt)
            tr = int(tr.group(1)) if tr else 0
            pnl_m = re.search(r'Total P&L:\s+\$?([+-]?[\d,\.]+)', txt)
            pnl = float(pnl_m.group(1).replace(',','')) if pnl_m else 0.0
            wins = re.search(r'Winning trades:\s+(\d+)', txt)
            wins = int(wins.group(1)) if wins else 0
            total_trades += tr
            total_pnl += pnl
            total_wins += wins
        except FileNotFoundError:
            continue
    wr = 100.0 * total_wins / max(total_trades, 1)
    return total_trades, total_pnl, wr

baseline_t, baseline_p, baseline_wr = aggregate('baseline')
print(f'\n{"cell":<14} {"trades":>7} {"P&L":>14} {"WR":>6}   delta_pnl   delta_trades')
print('-' * 72)
print(f'{"baseline":<14} {baseline_t:>7} ${baseline_p:>+12,.0f} {baseline_wr:>5.1f}%       —            —')
for c in cells_to_include[1:]:
    t, p, wr = aggregate(c)
    if t == 0:
        print(f'{c:<14} (no data)')
        continue
    dpnl = p - baseline_p
    dtr = t - baseline_t
    print(f'{c:<14} {t:>7} ${p:>+12,.0f} {wr:>5.1f}% ${dpnl:>+10,.0f}    {dtr:>+5d}')
PYEOF
