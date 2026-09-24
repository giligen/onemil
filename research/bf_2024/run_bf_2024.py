"""Bull-flag P1 on 2024H2, survivorship-free (PREREG.md, cell 1,417). Research only (2026-09-23).

`--build-cache` Stage-1 does NOT call batch_backtest.{fetch_daily_bars_cached,get_1min_bars_cached}
directly -- it goes through batch.monthly_runner.MonthlyBacktestRunner, which imported those two names
BY VALUE (`from batch_backtest import ...`) plus a hardcoded `Database()` (-> data/cache.db) and a
hardcoded `output_dir="backtest_results"`. Patching batch_backtest's own attributes would not reach any
of that, so all four seams are patched on `batch.monthly_runner` / its `MonthlyBacktestRunner` class
BEFORE batch_backtest.main() is called. Pattern: research/orb_2024/build_features_2024.py.

Never writes data/cache.db, config.yaml, orb.yaml, trading/, or the production bull-flag cache.
Side DB (persistence.database.Database schema): research/bf_2024/cache_2024.db
Raw 1-min store (schema of research/day_breadth/y2024/bars.db): research/bf_2024/bars.db
Stage-1 cache CSV: research/bf_2024/cache_2024.csv (via BT_CACHE_PATH_OVERRIDE)
Stage-2 trades CSV: research/bf_2024/stage2_2024.csv

Usage:
  BF24_SMOKE=1 python3 research/bf_2024/run_bf_2024.py    # 2024-07-02..2024-07-04 smoke, both stages
  python3 research/bf_2024/run_bf_2024.py                 # full 2024-07-02..2024-12-31, both stages
  BF24_STAGE=1 python3 research/bf_2024/run_bf_2024.py    # Stage 1 only
  BF24_STAGE=2 python3 research/bf_2024/run_bf_2024.py    # Stage 2 only (Stage 1 cache must exist)
"""
import os
import sqlite3
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path('/home/ec2-user/onemil')
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

BF_DIR = ROOT / 'research/bf_2024'
os.makedirs(BF_DIR, exist_ok=True)
SIDE_DB = BF_DIR / 'cache_2024.db'
BARS_DB = BF_DIR / 'bars.db'
Y2024_BARS_DB = ROOT / 'research/day_breadth/y2024/bars.db'
DAILY_PARQUET = ROOT / 'data/research/databento/equs_daily_2024H2.parquet'
JUNE_PARQUET = BF_DIR / 'daily_june.parquet'
CACHE_CSV = os.environ.get('BF24_CACHE', str(BF_DIR / 'cache_2024.csv'))
STAGE2_CSV = os.environ.get('BF24_STAGE2_OUT', str(BF_DIR / 'stage2_2024.csv'))
OUT_MONTHLY_DIR = os.environ.get('BF24_OUT_MONTHLY', str(BF_DIR / 'backtest_results_2024'))

SMOKE = os.environ.get('BF24_SMOKE') == '1'
STAGE = os.environ.get('BF24_STAGE', 'both')
START = date.fromisoformat(os.environ['BF24_START']) if os.environ.get('BF24_START') else date(2024, 7, 2)
END = (date.fromisoformat(os.environ['BF24_END']) if os.environ.get('BF24_END')
       else date(2024, 7, 4) if SMOKE else date(2024, 12, 31))

# 2026-09-24 repair (main session): months 2-3 (Aug, Sep) failed in the first chain with "cannot commit - no
# transaction is active" — this harness shares one SQLite connection across the runner's parallel MONTH threads.
# BF24_SEQ_MONTHS=1 forces one month at a time (production is unaffected: it opens a connection per month).
if os.environ.get('BF24_SEQ_MONTHS') == '1':
    import batch.monthly_runner as _mr
    _orig_init = _mr.MonthlyBacktestRunner.__init__

    def _seq_init(self, *a, **kw):
        kw['max_workers'] = 1
        _orig_init(self, *a, **kw)
    _mr.MonthlyBacktestRunner.__init__ = _seq_init
    print('[bf2024] WARNING month-level parallelism forced to 1 (BF24_SEQ_MONTHS=1)', flush=True)

# PREREG 1,417 froze "config.yaml as it configures it on 2026-09-23" — risk_cap OFF. The owner turned the live cap on
# 2026-09-24; Stage 2 here must keep the frozen setting unless BF24_RISK_CAP=live is set explicitly.
if os.environ.get('BF24_RISK_CAP', 'prereg') != 'live':
    import trading.bf_risk_cap as _rcap

    def _prereg_cap(_trading_cfg):
        return _rcap.RiskCapConfig(enabled=False, max_risk_mult=2.0)
    _rcap.load_risk_cap_config = _prereg_cap
    print('[bf2024] WARNING risk_cap forced OFF for this run (PREREG 1,417 froze the 2026-09-23 config); '
          'set BF24_RISK_CAP=live for the capped diagnostic', flush=True)


def _map_sym(s: str) -> str:
    """'+' warrants -> '.WS' (matches research/orb_2024's mapping); preferreds ('-' suffix) dropped by caller."""
    return s[:-1] + '.WS' if s.endswith('+') else s


_DAILY_CACHE = {}


def _load_daily_frame() -> pd.DataFrame:
    """EQUS 2024H2 daily (delisted included), symbols mapped, preferreds dropped. Cached in-process."""
    if 'df' in _DAILY_CACHE:
        return _DAILY_CACHE['df']
    daily = pd.read_parquet(DAILY_PARQUET,
                             columns=['bar_date', 'symbol', 'open', 'high', 'low', 'close', 'volume'])
    daily = daily[~daily['symbol'].astype(str).str.endswith('-')].copy()
    daily['symbol'] = daily['symbol'].astype(str).map(_map_sym)
    daily['bar_date'] = daily['bar_date'].astype(str)
    if JUNE_PARQUET.exists():
        june = pd.read_parquet(JUNE_PARQUET)
        june['bar_date'] = june['bar_date'].astype(str)
        daily = pd.concat([daily, june], ignore_index=True)
    _DAILY_CACHE['df'] = daily
    print(f"[bf2024] daily frame loaded: {len(daily):,} rows, {daily['symbol'].nunique():,} symbols "
          f"({daily['bar_date'].min()}..{daily['bar_date'].max()})", flush=True)
    return daily


import batch.monthly_runner as runner_mod  # noqa: E402
from persistence.database import Database as _RealDatabase  # noqa: E402
from batch_backtest import _market_hours_utc  # noqa: E402


def _patched_database(*_a, **_k):
    """Every `Database()` call inside batch.monthly_runner lands on the side DB, never data/cache.db."""
    return _RealDatabase(db_path=str(SIDE_DB))


runner_mod.Database = _patched_database

_ALL_SYMS_CACHE = {}


def _patched_get_active_universe(self):
    """Fresh side DB has no `universe` table rows ('No active symbols in universe' bailout) --
    survivorship-free universe IS every symbol ever seen in the 2024H2 parquet; find_big_movers'
    own price/dollar-volume/move filters narrow it, same as production's active-universe list does."""
    if 'syms' not in _ALL_SYMS_CACHE:
        syms = sorted(_load_daily_frame()['symbol'].dropna().unique().tolist())
        _ALL_SYMS_CACHE['syms'] = [{'symbol': s} for s in syms]
        print(f"[bf2024] active universe patched: {len(syms):,} symbols (all of 2024H2 parquet)", flush=True)
    return _ALL_SYMS_CACHE['syms']


_RealDatabase.get_active_universe = _patched_get_active_universe


def _patched_fetch_daily_bars_cached(symbols, start_date, end_date, client, db):
    """Daily bars from the EQUS 2024H2 parquet instead of Alpaca/cache.db; still writes through `db`
    (the side DB) so downstream point-in-time avg-volume lookups (_backtest_worker) find rows."""
    daily = _load_daily_frame()
    sub = daily[daily['symbol'].isin(symbols)
                & (daily['bar_date'] >= start_date.isoformat())
                & (daily['bar_date'] <= end_date.isoformat())]
    flat, out = [], {}
    for row in sub.itertuples(index=False):
        d = {'symbol': row.symbol, 'date': row.bar_date, 'open': float(row.open), 'high': float(row.high),
             'low': float(row.low), 'close': float(row.close), 'volume': int(row.volume)}
        flat.append(d)
        out.setdefault(row.symbol, []).append(d)
    if flat:
        db.save_daily_bars(flat)
    print(f"[bf2024] daily bars: {len(flat)} rows for {len(symbols)} requested symbols "
          f"{start_date}..{end_date}", flush=True)
    return out


runner_mod.fetch_daily_bars_cached = _patched_fetch_daily_bars_cached

# check_same_thread=False: run_month() executes inside monthly_runner's ThreadPoolExecutor (month-level
# parallelism), so these module-level connections cross threads. Discovered in smoke (silent "Month 1
# failed: SQLite objects created in a thread can only be used in that same thread" -> 0 movers/trades,
# swallowed by run_month's own try/except). No concurrent-write hazard: each thread handles disjoint
# (symbol, date) keys and every write is executemany+commit.
_Y24CONN = (sqlite3.connect(f'file:{Y2024_BARS_DB}?mode=ro', uri=True, check_same_thread=False)
            if Y2024_BARS_DB.exists() else None)
_OWN_CONN = sqlite3.connect(str(BARS_DB), check_same_thread=False)
_OWN_CONN.execute("CREATE TABLE IF NOT EXISTS bars "
                   "(symbol TEXT, day TEXT, t TEXT, o REAL, h REAL, l REAL, c REAL, v INTEGER)")
_OWN_CONN.execute("CREATE INDEX IF NOT EXISTS idx_bars_sym_day ON bars(symbol, day)")
_OWN_CONN.commit()
_FETCHED = [0]
_REUSED = [0]


def _rows_for(symbol, date_str):
    cur = _OWN_CONN.execute("SELECT t,o,h,l,c,v FROM bars WHERE symbol=? AND day=?", (symbol, date_str))
    rows = cur.fetchall()
    if rows:
        return rows
    if _Y24CONN is not None:
        cur2 = _Y24CONN.execute("SELECT t,o,h,l,c,v FROM bars WHERE symbol=? AND day=?", (symbol, date_str))
        rows2 = cur2.fetchall()
        if rows2:
            _OWN_CONN.executemany(
                "INSERT INTO bars (symbol,day,t,o,h,l,c,v) VALUES (?,?,?,?,?,?,?,?)",
                [(symbol, date_str, *r) for r in rows2])
            _OWN_CONN.commit()
            _REUSED[0] += 1
            return rows2
    return []


def _patched_get_1min_bars_cached(symbol, trade_date, client, db):
    """1-min bars from research/bf_2024/bars.db, reusing research/day_breadth/y2024/bars.db rows when
    present, else Alpaca SIP fetch (cached into bars.db AND the side DB)."""
    date_str = trade_date.isoformat()
    cached = db.get_intraday_bars_cached(symbol, date_str)
    if cached:
        return pd.DataFrame(cached)
    rows = _rows_for(symbol, date_str)
    if not rows:
        market_open, market_close = _market_hours_utc(trade_date)
        try:
            bars = client.get_historical_1min_bars(symbol, market_open, market_close)
        except Exception as e:
            print(f"[bf2024] WARNING Alpaca 1-min fetch failed {symbol} {date_str}: {e}", flush=True)
            return pd.DataFrame()
        if bars.empty:
            return bars
        recs = bars.to_dict('records')
        _OWN_CONN.executemany(
            "INSERT INTO bars (symbol,day,t,o,h,l,c,v) VALUES (?,?,?,?,?,?,?,?)",
            [(symbol, date_str, str(r['timestamp']), float(r['open']), float(r['high']), float(r['low']),
              float(r['close']), int(r['volume'])) for r in recs])
        _OWN_CONN.commit()
        _FETCHED[0] += 1
        rows = _rows_for(symbol, date_str)
    bar_records = [{'timestamp': pd.Timestamp(t, tz='UTC'), 'open': o, 'high': h, 'low': l, 'close': c,
                     'volume': v} for (t, o, h, l, c, v) in rows]
    db.save_intraday_bars(symbol, date_str, bar_records)
    return pd.DataFrame(bar_records)


runner_mod.get_1min_bars_cached = _patched_get_1min_bars_cached

_orig_run_all = runner_mod.MonthlyBacktestRunner.run_all


def _patched_run_all(self, start, end, output_dir="backtest_results"):
    """Hardcoded output_dir="backtest_results" in batch_backtest.main() -> redirect under research/bf_2024/."""
    os.makedirs(OUT_MONTHLY_DIR, exist_ok=True)
    return _orig_run_all(self, start, end, output_dir=OUT_MONTHLY_DIR)


runner_mod.MonthlyBacktestRunner.run_all = _patched_run_all


def _run():
    import batch_backtest as bb
    os.environ['BT_CACHE_PATH_OVERRIDE'] = CACHE_CSV
    common = ['--start', START.isoformat(), '--end', END.isoformat(),
              '--capital', '50000', '--risk', '2000', '--max-shares', '10000',
              '--scan-workers', '4']  # force the pre-fetch (parallel) path in monthly_runner.run_month;
              # with scan_workers<=1 it takes a sequential branch that never calls get_1min_bars_cached
    if STAGE in ('1', 'both'):
        sys.argv = ['batch_backtest.py', '--build-cache'] + common
        print(f"[bf2024] STAGE 1 start {START}..{END} smoke={SMOKE}", flush=True)
        bb.main()
        print(f"[bf2024] STAGE 1 DONE -> {CACHE_CSV}  (1min fetched={_FETCHED[0]} reused_y24={_REUSED[0]})",
              flush=True)
    if STAGE in ('2', 'both'):
        sys.argv = ['batch_backtest.py', '--output', STAGE2_CSV] + common
        print(f"[bf2024] STAGE 2 start", flush=True)
        bb.main()
        print(f"[bf2024] STAGE 2 DONE -> {STAGE2_CSV}", flush=True)


if __name__ == '__main__':
    _run()
