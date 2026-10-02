#!/bin/bash
# Cell 1,701 step 2: ONE-process, off-hours backfill of minute bars for the E1 (earnings-
# session ORB) candidate sessions into the shared bars_sip.db appender
# (research/bf_zero/backfill_bars_sip.py). Mirrors the exact wrapper pattern
# research/orb_freq/1689b_pools.py used for cells 1,689b/1,693b: override the appender's
# FEATURES and STATE module globals only, leave DB at its default -- every ORB research cell
# shares one bar store (research/bf_zero/bars_sip.db) without clobbering another cell's resume
# checkpoint (1689b_backfill_state.json, 1693b's own, this file's 1701_backfill_state.json).
#
# Do NOT launch this manually: it must run after 20:05 UTC (clear of the trading day; the
# onemil-trader service boots at 12:30 UTC and owns cache.db + the SIP feed all session) and
# only when disk has headroom. This script enforces both waits itself; it is written, not run,
# by cell 1,701 step 1 (research/orb_earn/1701_calendar.py / PREREG_1701.md).
#
# Usage (run later, manually, after reading this file):
#   nohup bash research/orb_earn/1701_backfill_queue.sh > /dev/null 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root
POOLDIR="research/orb_earn"
LOG="$POOLDIR/1701_backfill.log"

log() { echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') $*" | tee -a "$LOG"; }

# ---- 1. wait until 20:05 UTC today (epoch compare, then one bounded sleep -- never a blind
#          poll loop) ----
TARGET_EPOCH=$(date -u -d "today 20:05:00" +%s)
NOW_EPOCH=$(date -u +%s)
if [ "$NOW_EPOCH" -lt "$TARGET_EPOCH" ]; then
    WAIT=$((TARGET_EPOCH - NOW_EPOCH))
    log "waiting ${WAIT}s for 20:05 UTC before touching bars_sip.db (live service owns the day session)"
    sleep "$WAIT"
else
    log "already past 20:05 UTC (now $(date -u +%H:%M:%SZ)) -- proceeding immediately"
fi

# ---- 2. disk floor: >= 5 GB available on / ----
AVAIL_GB=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
log "disk check: ${AVAIL_GB} GB available on /"
if [ "$AVAIL_GB" -lt 5 ]; then
    log "ABORT: only ${AVAIL_GB} GB available, below the 5 GB floor -- not fetching"
    exit 1
fi

# ---- 3. build the (symbol,day) fetch list from step 1's candidates ----
python3 - <<'PYEOF'
import pandas as pd
from pathlib import Path
pooldir = Path("research/orb_earn")
cand = pd.read_csv(pooldir / "1701_candidates.csv")
fetch = cand[["symbol", "session"]].rename(columns={"session": "day"}).drop_duplicates()
fetch.to_csv(pooldir / "1701_fetch_list.csv", index=False)
print(f"fetch list: {len(fetch)} distinct (symbol,day) pairs -> {pooldir / '1701_fetch_list.csv'}")
PYEOF

# ---- 4. ONE process, nice + ionice, through the shared appender. DB is left at its default
#          (research/bf_zero/bars_sip.db); only FEATURES and STATE are redirected, exactly as
#          research/orb_freq/1689b_pools.py did for 1689b/1693b. ----
log "launching backfill_bars_sip (nice -n 10, ionice -c3, single process) ..."
nice -n 10 ionice -c3 python3 - <<'PYEOF' >> "$LOG" 2>&1
import sys
from pathlib import Path
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "research" / "bf_zero"))
import backfill_bars_sip as bf  # noqa: E402

POOLDIR = ROOT / "research" / "orb_earn"
bf.FEATURES = POOLDIR / "1701_fetch_list.csv"
bf.STATE = POOLDIR / "1701_backfill_state.json"
print(f"backfill wrapper: FEATURES->{bf.FEATURES} STATE->{bf.STATE} DB->{bf.DB} "
      "(shared bars_sip.db; never touches 1689b/1693b or the HOD causal_filter state files)")
sys.argv = ["backfill_bars_sip.py"]
rc = bf.main()
print(f"backfill rc={rc}")
PYEOF
log "backfill process exited"

# ---- 5. coverage line: symbol-sessions with >=300 bars / requested ----
python3 - <<'PYEOF' | tee -a "$LOG"
import sqlite3
import pandas as pd
from pathlib import Path

pooldir = Path("research/orb_earn")
cand = pd.read_csv(pooldir / "1701_fetch_list.csv")
requested = len(cand)
con = sqlite3.connect("research/bf_zero/bars_sip.db")
have = pd.read_sql("select symbol, day, count(*) as n from bars group by symbol, day", con)
con.close()
merged = cand.merge(have, on=["symbol", "day"], how="left")
merged["n"] = merged["n"].fillna(0)
covered = int((merged["n"] >= 300).sum())
pct = 100.0 * covered / requested if requested else 0.0
print(f"COVERAGE: {covered}/{requested} symbol-sessions have >=300 bars ({pct:.1f}%)")
PYEOF
log "1701_backfill_queue.sh DONE"
