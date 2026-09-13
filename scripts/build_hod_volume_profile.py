#!/usr/bin/env python3
"""Nightly: same-clock cumulative-volume checkpoints per symbol-day → cache.db hod_volume_profile.

Owner 2026-09-13: "for average volume we can build our own DB for past x days volume at 09:35 etc."
ONE definition with the research table (research/bf_zero/build_candidates.py VP_MIN) and the live
spec (trading.hod_break.VP_CHECKPOINTS): cumulative RTH volume at 09:35 09:45 10:00 10:30 11:00
12:00 13:00 14:00 15:00 ET, plus the full-day RTH volume, for every universe symbol whose 1-min bars
are in cache.db for that date (the nightly bar refresh runs first).

Usage: python3 scripts/build_hod_volume_profile.py [YYYY-MM-DD]   (default: today ET)
Never touches any table but hod_volume_profile. Verbose; exits non-zero on a DB failure.
"""
import os
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT); os.chdir(ROOT)

from persistence.database import Database                       # noqa: E402
from trading.hod_break import VP_CHECKPOINTS, OPEN_MINUTE       # noqa: E402

ET = ZoneInfo('America/New_York')


def checkpoints_for(bars: list) -> dict:
    """bars: dicts with timestamp/volume (any order). Returns {cut_minute: cum_volume, 'day': total}."""
    mv = []
    for b in bars:
        ts = b.get('timestamp')
        t = ts if isinstance(ts, datetime) else datetime.fromisoformat(str(ts).replace('Z', '+00:00'))
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        et = t.astimezone(ET); m = et.hour * 60 + et.minute
        if OPEN_MINUTE <= m < 960:
            mv.append((m, float(b.get('volume') or 0.0)))
    mv.sort()
    out = {}; cum = 0.0; i = 0
    for cut in VP_CHECKPOINTS:
        while i < len(mv) and mv[i][0] <= cut:
            cum += mv[i][1]; i += 1
        out[cut] = cum
    out['day'] = cum + sum(v for _, v in mv[i:])
    return out


def main() -> int:
    day = sys.argv[1] if len(sys.argv) > 1 else datetime.now(timezone.utc).astimezone(ET).strftime('%Y-%m-%d')
    db = Database()
    symbols = [r['symbol'] for r in db.get_active_universe()]
    print(f"[hod_volume_profile] {day}: {len(symbols)} universe symbols", flush=True)
    bars_by = db.get_intraday_bars_bulk([(s, day) for s in symbols])
    rows = []; n_sym = 0
    for (sym, d), bars in bars_by.items():
        if not bars:
            continue
        cp = checkpoints_for(bars); n_sym += 1
        for cut in VP_CHECKPOINTS:
            rows.append({'symbol': sym, 'bar_date': d, 'cut_minute': int(cut), 'cum_volume': cp[cut], 'day_volume': cp['day']})
    n = db.upsert_hod_volume_profile(rows)
    print(f"[hod_volume_profile] {day}: {n_sym} symbols with bars → {n} rows upserted", flush=True)
    if n_sym == 0:
        print(f"[hod_volume_profile] WARNING: no intraday bars in cache for {day} — did the nightly bar refresh run?", flush=True)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
