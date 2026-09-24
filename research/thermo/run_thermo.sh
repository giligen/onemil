#!/usr/bin/env bash
# research/thermo/run_thermo.sh -- unattended orchestration for PREREG.md cells 1,420-1,422.
# Launch: nohup setsid research/thermo/run_thermo.sh >> research/thermo/run_thermo.log 2>&1 < /dev/null &
#
# Waits out the market-hours data/cache.db blackout [13:25, 20:05) UTC AND any running
# research/hod_ofi/rerun.sh (one heavy DB job at a time), rebuilds the 2025-26 ORB book exactly as
# PREREG.md specifies, then scores all three cells (thermo.py runs the consistency check and STOPs
# into REPORT.md itself on a mismatch -- SPEC.md step 3). Never raises past a failed step: every
# step is logged and the run always ends with a Telegram ping (no numbers in it, ever).
set -uo pipefail  # deliberately NOT -e: a failed step must still reach the final Telegram ping

cd /home/ec2-user/onemil || exit 1

REPORT_OUT=research/thermo/REPORT.md
BOOK_2025_26=research/thermo/book_2025_26.csv
FEATURES_2025_26=analysis_results/orb_features_20260923_2049.csv
LOG_TAG="[run_thermo]"

ts() { date -u '+%Y-%m-%d %H:%M:%S'; }

echo "$LOG_TAG $(ts) starting, pid $$"

# --- wait for the DB blackout to clear AND for the OFI heavy job to finish ---
while true; do
    now_hhmm=$(date -u +%H%M | sed 's/^0*//')
    [ -z "$now_hhmm" ] && now_hhmm=0
    in_blackout=0
    if [ "$now_hhmm" -ge 1325 ] && [ "$now_hhmm" -lt 2005 ]; then
        in_blackout=1
    fi
    ofi_running=0
    if pgrep -f research/hod_ofi/rerun.sh > /dev/null 2>&1; then
        ofi_running=1
    fi
    if [ "$in_blackout" -eq 0 ] && [ "$ofi_running" -eq 0 ]; then
        break
    fi
    echo "$LOG_TAG $(ts) waiting (blackout=$in_blackout ofi_running=$ofi_running)"
    sleep 120
done
echo "$LOG_TAG $(ts) window clear -- rebuilding the 2025-26 ORB book"

# --- rebuild the 2025-26 book exactly as PREREG.md's Data section specifies ---
ORB_BT_FEATURES_CSV="$FEATURES_2025_26" ORB_BT_BOOK_OUT="$BOOK_2025_26" ORB_BT_MONTHLY_OUT=research/thermo/monthly_2025_26.csv ORB_CATALYST_VETO=0 \
    nice -n 19 ionice -c3 python3 study_orb_pipeline_static_lock.py
rebuild_rc=$?
echo "$LOG_TAG $(ts) book rebuild exit=$rebuild_rc"

if [ $rebuild_rc -ne 0 ] || [ ! -f "$BOOK_2025_26" ]; then
    echo "$LOG_TAG ERROR: book rebuild failed or produced no file -- see log above"
    python3 scripts/send_telegram_alert.py "[thermo] run_thermo.sh FAILED at the 2025-26 book rebuild step -- see run_thermo.log" || true
    exit 1
fi

# --- score all three cells; thermo.py itself runs the consistency check and STOPs into REPORT_OUT on mismatch ---
echo "$LOG_TAG $(ts) scoring cells 1420-1422 -> $REPORT_OUT"
python3 research/thermo/thermo.py --out "$REPORT_OUT" --orb-book-2025-26 "$BOOK_2025_26"
score_rc=$?
echo "$LOG_TAG $(ts) scoring exit=$score_rc"

if grep -q "^# REPORT -- STOPPED" "$REPORT_OUT" 2>/dev/null; then
    python3 scripts/send_telegram_alert.py "[thermo] run_thermo.sh STOPPED -- 2025-26 book consistency check failed, see research/thermo/REPORT.md" || true
elif [ $score_rc -eq 0 ]; then
    python3 scripts/send_telegram_alert.py "[thermo] run_thermo.sh done -- REPORT.md written (research/thermo/REPORT.md)" || true
else
    python3 scripts/send_telegram_alert.py "[thermo] run_thermo.sh scoring step FAILED (exit=$score_rc) -- see run_thermo.log" || true
fi

echo "$LOG_TAG $(ts) done, exit=$score_rc"
exit $score_rc
