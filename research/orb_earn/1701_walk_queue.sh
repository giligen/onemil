#!/bin/bash
# Cell 1,701 step 3 queue: wait for the SIP appender (research/orb_earn/1701_backfill_run.py,
# PID 890797) to exit, print the PREREG Amendment-2 coverage line via a READ-ONLY query, then
# run the full SIP walk. This script never writes to bars_sip.db itself and never disturbs the
# appender -- it only polls `kill -0` on the PID and, once truly gone (checked against
# /proc/<pid>/cmdline so a recycled PID number can't fool the wait), opens the DB `?mode=ro`.
#
# Usage (launched detached, per PREREG_1701.md step 3):
#   setsid nohup bash research/orb_earn/1701_walk_queue.sh \
#       > research/orb_earn/1701_walk_queue.log 2>&1 < /dev/null &
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root
POOLDIR="research/orb_earn"
APPENDER_PID=890797

log() { echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') $*"; }

# ---- 1. wait for the appender to exit, polling by PID every 120s -----------------------------
pid_is_our_appender() {
    local pid="$1"
    [ -r "/proc/$pid/cmdline" ] 2>/dev/null \
        && tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q "1701_backfill_run"
}

log "checking appender PID ${APPENDER_PID} (research/orb_earn/1701_backfill_run.py) ..."
while kill -0 "$APPENDER_PID" 2>/dev/null && pid_is_our_appender "$APPENDER_PID"; do
    log "PID ${APPENDER_PID} still running -- sleeping 120s before re-checking"
    sleep 120
done
log "PID ${APPENDER_PID} is not running (or is not the appender any more) -- proceeding"

# ---- 2. coverage line, read-only, against whatever candidate/fetch list is on disk -----------
log "coverage line (read-only select against bars_sip.db) ..."
python3 - <<'PYEOF'
import sqlite3
import pandas as pd
from pathlib import Path

pooldir = Path("research/orb_earn")
fetch_csv = pooldir / "1701_fetch_list.csv"
if fetch_csv.exists():
    fetch = pd.read_csv(fetch_csv, dtype={"symbol": str, "day": str})
else:
    cand = pd.read_csv(pooldir / "1701_candidates.csv", dtype={"symbol": str, "session": str})
    fetch = cand[["symbol", "session"]].rename(columns={"session": "day"}).drop_duplicates()
requested = len(fetch)
con = sqlite3.connect("file:research/bf_zero/bars_sip.db?mode=ro", uri=True, timeout=30)
try:
    have = pd.read_sql("select symbol, day, count(*) as n from bars group by symbol, day", con)
finally:
    con.close()
merged = fetch.merge(have, on=["symbol", "day"], how="left")
merged["n"] = merged["n"].fillna(0)
covered = int((merged["n"] >= 300).sum())
pct = 100.0 * covered / requested if requested else 0.0
gate = "MET" if pct >= 95.0 else "NOT met (walk proceeds; RESULT.md will VOID per Amendment 2)"
print(f"COVERAGE: {covered}/{requested} symbol-sessions have >=300 bars ({pct:.1f}%) -- gate 95% {gate}")
PYEOF

# ---- 3. the full SIP walk (this is the only step that writes 1701_walk.log) -------------------
log "launching full SIP walk: nice -n 10 python3 ${POOLDIR}/1701_walk.py --source sip --full"
nice -n 10 python3 "${POOLDIR}/1701_walk.py" --source sip --full > "${POOLDIR}/1701_walk.log" 2>&1
rc=$?
log "1701_walk.py --source sip --full exited rc=${rc}"
log "1701_walk_queue.sh DONE"
