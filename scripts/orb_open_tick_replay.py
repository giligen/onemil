#!/usr/bin/env python3
"""Replay the ORB open-tick universe build for one date, READ-ONLY.

docs/orb_open_tick_stall_20261002.md: measures how long
`ORBEngine.build_orb_universe_from_snapshots` takes at the open for ~3,000
candidates against the real cache.db (opened `mode=ro`, never written).

Network is stubbed: the Alpaca snapshot call returns snapshots rebuilt from
`daily_bars` (open / prev close / prev volume of the replay date) in the exact
flat shape of `AlpacaClient.get_snapshots`, after sleeping a recorded REST
latency (5.53 s for 3,041 symbols on 2026-10-02, scaled by symbol count).

Reports three timings:
  before  - the legacy per-symbol query (no INDEXED BY, scans each symbol's
            history), sampled on --sample symbols and extrapolated to N
  after   - the per-symbol query with the INDEXED BY hint, same sample
  tick    - the real engine tick end-to-end (batched lookup + budget)

Usage:  python scripts/orb_open_tick_replay.py --date 2026-10-02 [--n 3041]
"""
import argparse
import logging
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from persistence.database import Database  # noqa: E402

logger = logging.getLogger("orb_open_tick_replay")
RECORDED_REST_SEC_PER_3041 = 5.53   # journal 2026-10-02 13:30:46
PER_QUERY_ABORT_SEC = 20.0


def open_readonly_db(cache_path: Path) -> Database:
    """Return a Database whose cache connection is read-only (no schema writes, no WAL pragma)."""
    uri = f"file:{cache_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=30, check_same_thread=False,
                           detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
    conn.row_factory = sqlite3.Row
    db = Database.__new__(Database)          # bypass __init__ (it creates tables / sets WAL)
    db._cache_conn = conn
    db._trades_conn = conn
    db._cache_path = cache_path
    db._trades_path = cache_path
    db.db_path = cache_path
    db._split = False
    return db


def load_snapshots(db: Database, date: str, n: int) -> dict:
    """Rebuild flat snapshots from daily_bars: today's open vs the prior session's close/volume."""
    conn = db._cache_conn
    prev = conn.execute("SELECT MAX(bar_date) FROM daily_bars WHERE bar_date < ?", (date,)).fetchone()[0]
    if prev is None:
        raise SystemExit(f"no daily_bars before {date}")
    today = {r['symbol']: r for r in conn.execute(
        "SELECT symbol, open FROM daily_bars WHERE bar_date = ?", (date,))}
    snaps = {}
    for r in conn.execute("SELECT symbol, open, close, volume FROM daily_bars WHERE bar_date = ? "
                          "ORDER BY volume DESC LIMIT ?", (prev, n)):
        sym = r['symbol']
        o = today[sym]['open'] if sym in today else r['close']
        snaps[sym] = {'open': float(o), 'prev_close': float(r['close']),
                      'prev_volume': int(r['volume']), 'latest_price': float(o),
                      'volume': int(r['volume']), 'close': float(r['close']),
                      'daily_bar_date': date}
    logger.info("snapshots: %d symbols (prev session %s, %d with a %s daily bar)",
                len(snaps), prev, sum(1 for s in snaps if s in today), date)
    return snaps


def time_per_symbol(db: Database, symbols: list, date: str, hinted: bool) -> float:
    """Mean seconds of the per-symbol 09:30 lookup, hinted or legacy (unhinted)."""
    hint = "INDEXED BY idx_intraday_bars_symbol_date" if hinted else ""
    sql = (f"SELECT timestamp, open FROM intraday_bars_1min {hint} "
           "WHERE symbol = ? AND bar_date = ? ORDER BY timestamp")
    t0 = time.time()
    for s in symbols:
        db._cache_conn.execute(sql, (s, date)).fetchall()
    return (time.time() - t0) / max(len(symbols), 1)


class StubAlpaca:
    """Alpaca stand-in: sleeps the recorded REST latency, returns the rebuilt snapshots."""

    def __init__(self, snaps: dict):
        self.snaps = snaps
        self.calls = 0

    def get_snapshots(self, symbols):
        self.calls += 1
        time.sleep(RECORDED_REST_SEC_PER_3041 * len(symbols) / 3041.0)
        return {s: self.snaps[s] for s in symbols if s in self.snaps}


def build_engine(alpaca, db):
    """Construct the real ORBEngine from orb.yaml with prewarm_seed on and trading stubs."""
    from unittest.mock import MagicMock
    from trading.orb_engine import ORBEngine
    from trading.stop_monitor import StopMonitor
    cfg = yaml.safe_load((ROOT / 'orb.yaml').read_text())
    cfg['strategy']['enabled'] = True
    cfg.setdefault('execution', {})['prewarm_seed'] = True
    return ORBEngine(alpaca_client=alpaca, db=db, stop_monitor=MagicMock(spec=StopMonitor), config=cfg)


class LineCounter(logging.Handler):
    """Counts log records by level (a cycle's journal footprint); DEBUG included."""

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.by_level = {}

    def emit(self, record):
        self.by_level[record.levelname] = self.by_level.get(record.levelname, 0) + 1


def replay_cycle(eng, syms, restart_at_et=None):
    """One scanner cycle as `_orb_tick` runs it: gate on universe_build_due, else build.

    Returns (seconds, log-lines-by-level, built?). Never submits an order: only
    build_orb_universe_from_snapshots / universe_build_due are called.
    """
    counter = LineCounter()
    root = logging.getLogger()
    old_level = root.level
    root.addHandler(counter)
    root.setLevel(logging.DEBUG)
    t0 = time.time()
    built = True
    try:
        if hasattr(eng, 'universe_build_due') and not eng.universe_build_due():
            built = False
        else:
            eng.build_orb_universe_from_snapshots(syms)
    finally:
        root.removeHandler(counter)
        root.setLevel(old_level)
    return time.time() - t0, counter.by_level, built


LIVE_STALE = {"2026-09-30": ["ASTX", "AEHG"]}   # snapshots whose daily bar was the prior session at 09:35


def _admit_class(gap, price, cfg):
    """BT admission class ('prod' / 'P1' / '') from orb.yaml thresholds (daily_bars gap)."""
    u = cfg['universe']
    pools = (u.get('addon_pools') or {}).get('pools') or []
    if u['min_price'] <= price <= u['max_price'] and gap >= u['min_gap_pct']:
        return 'prod'
    for p in pools:
        if p['min_price'] <= price <= p['max_price'] and p['min_gap_pct'] <= gap <= p['max_gap_pct']:
            return 'P1'
    return ''


def parity_replay(db, date: str, stale_syms: list) -> int:
    """Replay the 09:35:30 ET universe build for `date`: engine admissions vs the BT's (daily_bars).

    Snapshots are rebuilt from daily_bars (fresh, dated `date`); `stale_syms` get the PRIOR session's
    daily bar instead (the 9/30 ASTX/AEHG live condition). The 09:30 minute bars come from the real
    Alpaca historical REST (read-only market data). The clock is frozen at 09:35:30 ET of `date`.
    Never submits an order. Prints admitted-vs-BT symbol lists and the build's seconds.
    """
    import os
    import pandas as pd  # noqa: F401
    from dotenv import load_dotenv
    from data_sources.alpaca_client import AlpacaClient
    import trading.orb_engine as oe
    from zoneinfo import ZoneInfo
    load_dotenv()
    snaps = load_snapshots(db, date, 5000)
    conn = db._cache_conn
    prev = conn.execute("SELECT MAX(bar_date) FROM daily_bars WHERE bar_date < ?", (date,)).fetchone()[0]
    prev2 = conn.execute("SELECT MAX(bar_date) FROM daily_bars WHERE bar_date < ?", (prev,)).fetchone()[0]
    real = AlpacaClient(os.getenv('ALPACA_API_KEY'), os.getenv('ALPACA_API_SECRET'))
    if not conn.execute("SELECT 1 FROM daily_bars WHERE bar_date = ? LIMIT 1", (date,)).fetchone():
        d = datetime.fromisoformat(date).date()
        syms_all = list(snaps)
        for i in range(0, len(syms_all), 200):
            for sy, bars in real.get_daily_bars_range(syms_all[i:i + 200], d, d).items():
                if bars:
                    snaps[sy]['open'] = snaps[sy]['latest_price'] = float(bars[0]['open'])
    cfg = yaml.safe_load((ROOT / 'orb.yaml').read_text())
    bt = {}
    for sy, sn in snaps.items():
        c = _admit_class((sn['open'] - sn['prev_close']) / sn['prev_close'] * 100, sn['open'], cfg)
        if c and sn['prev_volume'] >= cfg['universe']['min_prev_volume']:
            bt[sy] = c
    for sy in stale_syms:       # prior session's bar as the snapshot's daily bar (live 9/30 condition)
        r = conn.execute("SELECT open, close, volume FROM daily_bars WHERE symbol=? AND bar_date=?",
                         (sy, prev)).fetchone()
        r2 = conn.execute("SELECT close FROM daily_bars WHERE symbol=? AND bar_date=?", (sy, prev2)).fetchone()
        if r and r2:
            snaps[sy] = {'open': float(r['open']), 'close': float(r['close']), 'volume': int(r['volume']),
                         'prev_close': float(r2['close']), 'prev_volume': 1_000_000,
                         'latest_price': float(r['close']), 'daily_bar_date': prev}

    class Rest(StubAlpaca):
        """Stub snapshots + the REAL historical 09:30 bars."""
        def get_1min_bars_range_multi(self, syms, s0, s1):
            self.bar_calls = getattr(self, 'bar_calls', 0) + 1
            return real.get_1min_bars_range_multi(syms, s0, s1)

    fixed = datetime.fromisoformat(f"{date}T09:35:30").replace(tzinfo=ZoneInfo('America/New_York'))

    class _Frozen:
        def now(self, tz=None):
            return fixed.astimezone(tz) if tz else fixed.astimezone(timezone.utc)

        def __getattr__(self, name):
            return getattr(datetime, name)
    real_dt = oe.datetime
    oe.datetime = _Frozen()
    try:
        alpaca = Rest(snaps)
        eng = build_engine(alpaca, db)
        t0 = time.time()
        keep = eng.build_orb_universe_from_snapshots(list(snaps))
        secs = time.time() - t0
    finally:
        oe.datetime = real_dt
    eng_set, bt_set = set(keep), set(bt)
    print(f"PARITY {date}: engine admitted {len(eng_set)}, BT admitted {len(bt_set)}, build {secs:.1f} s "
          f"(stub snapshot REST excluded), 09:30 REST calls {getattr(alpaca, 'bar_calls', 0)}, stale injected {stale_syms}")
    print(f"  engine-only (not in BT): {sorted(eng_set - bt_set)}")
    print(f"  BT-only (not in engine): {sorted(bt_set - eng_set)}")
    for sy in stale_syms:
        print(f"  {sy} admitted by engine: {sy in eng_set}; in BT: {sy in bt_set}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--date', required=True)
    ap.add_argument('--n', type=int, default=3041)
    ap.add_argument('--sample', type=int, default=50)
    ap.add_argument('--restart-at-et', default=None,
                    help="HH:MM ET: also replay one cycle after a restart at that time "
                         "(e.g. 10:03 or 16:01); the clock is frozen inside trading.orb_engine")
    ap.add_argument('--parity', action='store_true',
                    help="replay the 09:35:30 ET build for --date: admitted vs the BT's (daily_bars); "
                         "stale-snapshot symbols default to LIVE_STALE[date]; uses real read-only Alpaca bars")
    ap.add_argument('--stale', default=None, help="comma list overriding LIVE_STALE for --parity")
    ap.add_argument('--cache', default=str(ROOT / 'data' / 'cache.db'))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    db = open_readonly_db(Path(args.cache))
    if args.parity:
        stale = args.stale.split(',') if args.stale else LIVE_STALE.get(args.date, [])
        return parity_replay(db, args.date, stale)
    snaps = load_snapshots(db, args.date, args.n)
    syms = list(snaps)
    sample = syms[:args.sample]
    # Abort any single runaway statement instead of hanging the node.
    deadline = [time.time() + PER_QUERY_ABORT_SEC * 20]
    db._cache_conn.set_progress_handler(lambda: 1 if time.time() > deadline[0] else 0, 100000)
    try:
        legacy = time_per_symbol(db, sample, args.date, hinted=False)
        hinted = time_per_symbol(db, sample, args.date, hinted=True)
    except sqlite3.OperationalError as e:
        logger.error("per-symbol sample aborted by the read-only time guard: %s", e)
        return 2
    print(f"per-symbol lookup, {len(sample)}-symbol sample: legacy {legacy*1000:.1f} ms, "
          f"INDEXED BY {hinted*1000:.1f} ms")
    print(f"BEFORE (legacy per-symbol x {len(syms)}): {legacy*len(syms):.1f} s of SQL alone")
    print(f"AFTER-A (hinted per-symbol x {len(syms)}): {hinted*len(syms):.1f} s")

    deadline[0] = time.time() + PER_QUERY_ABORT_SEC * 20
    alpaca = StubAlpaca(snaps)
    eng = build_engine(alpaca, db)
    t0 = time.time()
    keep = eng.build_orb_universe_from_snapshots(syms)
    tick = time.time() - t0
    print(f"AFTER (real engine tick: stub REST {RECORDED_REST_SEC_PER_3041*len(syms)/3041:.1f} s + "
          f"batched lookup + gate): {tick:.1f} s, admitted {len(keep)} of {len(syms)}, "
          f"REST calls {alpaca.calls}, budget {eng.open_tick_budget_sec} s")
    if args.restart_at_et:
        import trading.orb_engine as oe
        from zoneinfo import ZoneInfo
        hh, mm = (int(x) for x in args.restart_at_et.split(':'))
        fixed = datetime.fromisoformat(f"{args.date}T{hh:02d}:{mm:02d}:00").replace(
            tzinfo=ZoneInfo('America/New_York')).astimezone(timezone.utc)

        class _Frozen:
            """datetime stand-in whose now() is the replayed restart instant."""
            def now(self, tz=None):
                return fixed.astimezone(tz) if tz else fixed

            def __getattr__(self, name):
                return getattr(datetime, name)
        real_dt = oe.datetime
        oe.datetime = _Frozen()
        try:
            eng2 = build_engine(StubAlpaca(snaps), db)
            deadline[0] = time.time() + PER_QUERY_ABORT_SEC * 20
            secs, lines, built = replay_cycle(eng2, syms)
        finally:
            oe.datetime = real_dt
        print(f"RESTART {args.restart_at_et} ET cycle: built={built} {secs:.2f} s, "
              f"log lines by level {lines} (total {sum(lines.values())})")
    return 0


if __name__ == '__main__':
    sys.exit(main())
