#!/usr/bin/env python3
"""Cell 1,701 step 2 — minute bars for every earnings-session candidate through the designed appender.

Owner 2026-10-02 10:35 UTC: "run the 19,199 now". Runs as ONE nice'd process (the sole writer on
research/bf_zero/bars_sip.db); the cmdline carries this file's path so the session's research pause
window (SIGSTOP 13:27-13:47 UTC) catches it. FEATURES/STATE are redirected to research/orb_earn/ exactly as
1689b_pools.py did; the DB stays at the appender's shared default. Ends with the coverage line the PREREG
requires (symbol-sessions with >= 300 bars / requested; gate 95 % else VOID).

    nice -n 10 ionice -c3 python3 research/orb_earn/1701_backfill_run.py
"""
from __future__ import annotations

import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
POOLDIR = ROOT / "research" / "orb_earn"


def build_fetch_list() -> Path:
    """Distinct (symbol, day) pairs from the frozen candidate list -> 1701_fetch_list.csv."""
    cand = pd.read_csv(POOLDIR / "1701_candidates.csv")
    fetch = cand[["symbol", "session"]].rename(columns={"session": "day"}).drop_duplicates()
    out = POOLDIR / "1701_fetch_list.csv"
    fetch.to_csv(out, index=False)
    print(f"fetch list: {len(fetch)} distinct (symbol,day) pairs -> {out}", flush=True)
    return out


def run_appender(fetch_list: Path) -> int:
    """The shared appender with FEATURES/STATE redirected; DB left at its default (bars_sip.db)."""
    sys.path.insert(0, str(ROOT / "research" / "bf_zero"))
    import backfill_bars_sip as bf  # noqa: E402

    bf.FEATURES = fetch_list
    bf.STATE = POOLDIR / "1701_backfill_state.json"
    print(f"backfill wrapper: FEATURES->{bf.FEATURES} STATE->{bf.STATE} DB->{bf.DB}", flush=True)
    sys.argv = ["backfill_bars_sip.py"]
    t0 = time.time()
    rc = bf.main()
    print(f"backfill rc={rc} elapsed={time.time() - t0:.0f}s", flush=True)
    return int(rc or 0)


def coverage_line(fetch_list: Path) -> None:
    """Symbol-sessions with >= 300 bars / requested, read-only on the bar store."""
    cand = pd.read_csv(fetch_list)
    requested = len(cand)
    con = sqlite3.connect(f"file:{ROOT / 'research' / 'bf_zero' / 'bars_sip.db'}?mode=ro", uri=True)
    have = pd.read_sql("select symbol, day, count(*) as n from bars group by symbol, day", con)
    con.close()
    merged = cand.merge(have, on=["symbol", "day"], how="left")
    merged["n"] = merged["n"].fillna(0)
    covered = int((merged["n"] >= 300).sum())
    pct = 100.0 * covered / requested if requested else 0.0
    print(f"COVERAGE: {covered}/{requested} symbol-sessions have >=300 bars ({pct:.1f}%)"
          f" -- gate 95 % {'MET' if pct >= 95 else 'NOT met (VOID unless completed)'}", flush=True)


def main() -> int:
    """Fetch list -> appender -> coverage; verbose, one process."""
    fetch_list = build_fetch_list()
    rc = run_appender(fetch_list)
    coverage_line(fetch_list)
    return rc


if __name__ == "__main__":
    sys.exit(main())
