#!/usr/bin/env python3
"""ORB backtest — single command with automatic data-gap fill.

Runs the authoritative pipeline (study_orb_pipeline_static_lock) with the
shipped static_lock_1R exit + live yaml filter/quintile/adaptive-mult params.
Handles the data pipeline automatically:

  1. Discovers the last date in the latest `analysis_results/orb_features_*.csv`.
  2. Finds trading days between (last_date+1) and --end (inclusive).
  3. For each missing day: fetches daily bars for all symbols currently in
     `daily_bars` cache; identifies gap-up qualifying pairs via
     `load_broad_universe` logic; fetches 1-min bars for those pairs that
     don't have cached intraday bars yet.
  4. Re-runs `study_orb_features` to regenerate the features CSV (new
     timestamp; old CSV kept as historical record).
  5. Runs `study_orb_pipeline_static_lock` — full timeline + monthly
     summary. Prints a day-by-day slice for `--slice` if provided.

Usage:
    python3 orb_backtest.py                              # fill through today, run pipeline
    python3 orb_backtest.py --end 2026-04-20             # fill through 4/20
    python3 orb_backtest.py --slice 2026-04-20           # BT through today, show 4/20 detail
    python3 orb_backtest.py --end 2026-04-21 --slice 2026-04-20 --slice 2026-04-21
    python3 orb_backtest.py --no-fill                    # skip data-fill, just run pipeline
    python3 orb_backtest.py --force-features             # force features CSV regen

Idempotent: safe to re-run. Existing DB rows are INSERT OR REPLACE;
feature extraction re-reads fresh DB state each run.
"""
from __future__ import annotations

import argparse
import glob
import os
import sqlite3
import subprocess
import sys
from datetime import datetime, date, timedelta, timezone
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from config import Config
from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database

CACHE_DB = os.path.join(ROOT, 'data', 'cache.db')
FEATURES_GLOB = os.path.join(ROOT, 'analysis_results', 'orb_features_*.csv')

# Universe criteria (kept in sync with study_orb_broad.py)
MIN_GAP_PCT = 5.0
MIN_PREV_DAY_VOL = 500_000
MIN_OPEN_PRICE = 3.0
MAX_OPEN_PRICE = 30.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _latest_features_csv() -> Optional[str]:
    files = [p for p in sorted(glob.glob(FEATURES_GLOB)) if 'corrmatrix' not in p]
    return files[-1] if files else None


def _last_features_date(csv_path: str) -> Optional[date]:
    """Return the max trade_date in the features CSV."""
    if not csv_path or not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path, usecols=['date'])
    df['date'] = pd.to_datetime(df['date'])
    return df['date'].max().date()


def _trading_days_between(
    alpaca: AlpacaClient, start: date, end: date
) -> List[date]:
    """Return trading dates inclusive of start/end (if they are trading days)."""
    if start > end:
        return []
    try:
        cal = alpaca.get_market_calendar(start, end)
        # Calendar entries may have date as str or date — normalize
        out = []
        for d in cal:
            v = d['date']
            if isinstance(v, str):
                try:
                    v = datetime.strptime(v, '%Y-%m-%d').date()
                except ValueError:
                    continue
            elif hasattr(v, 'date') and not isinstance(v, date):
                v = v.date()
            out.append(v)
        return sorted(out)
    except Exception as e:
        _log(f"market_calendar failed, falling back to weekday filter: {e}")
        cur = start
        days: List[date] = []
        while cur <= end:
            if cur.weekday() < 5:  # Mon-Fri
                days.append(cur)
            cur += timedelta(days=1)
        return days


def _symbols_in_daily_cache(db_path: str = CACHE_DB) -> List[str]:
    """All symbols ever tracked in daily_bars — used as the universe to refresh."""
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute("SELECT DISTINCT symbol FROM daily_bars ORDER BY symbol")
        return [r[0] for r in cur.fetchall()]
    finally:
        conn.close()


def _daily_bar_dates_cached(
    db_path: str, symbols: List[str], start: date, end: date
) -> Set[Tuple[str, str]]:
    """Return set of (symbol, bar_date) pairs that already have daily bars."""
    if not symbols:
        return set()
    conn = sqlite3.connect(db_path)
    try:
        chunk = 500
        out: Set[Tuple[str, str]] = set()
        for i in range(0, len(symbols), chunk):
            batch = symbols[i:i + chunk]
            placeholders = ','.join('?' * len(batch))
            cur = conn.execute(f"""
                SELECT symbol, bar_date FROM daily_bars
                WHERE symbol IN ({placeholders})
                  AND bar_date BETWEEN ? AND ?
            """, batch + [str(start), str(end)])
            for row in cur.fetchall():
                out.add((row[0], row[1]))
        return out
    finally:
        conn.close()


def _qualifying_pairs_for_dates(
    db_path: str, dates: List[date],
    include_provisional_today: Optional[date] = None,
) -> Dict[str, List[str]]:
    """Run the gap-up filter against daily_bars for specific dates.
    Matches `study_orb_broad.load_broad_universe` SQL but scoped + no intraday
    existence check (we'll fetch missing intraday bars next).

    `include_provisional_today` (a date) adds a second pass that unions in
    provisional rows for that date only — so mid-day BT can also identify
    today's gap-up candidates without touching the main daily_bars table."""
    if not dates and not include_provisional_today:
        return {}
    conn = sqlite3.connect(db_path)
    out: Dict[str, List[str]] = {}
    try:
        if dates:
            date_set = ','.join('?' * len(dates))
            q = f"""
            WITH daily_ranked AS (
                SELECT symbol, bar_date, open,
                       LAG(close)  OVER (PARTITION BY symbol ORDER BY bar_date) AS prev_close,
                       LAG(volume) OVER (PARTITION BY symbol ORDER BY bar_date) AS prev_vol
                FROM daily_bars
            )
            SELECT symbol, bar_date FROM daily_ranked
            WHERE bar_date IN ({date_set})
              AND prev_close IS NOT NULL AND prev_close > 0
              AND (open - prev_close) / prev_close * 100 >= ?
              AND prev_vol >= ?
              AND open BETWEEN ? AND ?
            ORDER BY bar_date, symbol
            """
            params = [str(d) for d in dates] + [
                MIN_GAP_PCT, MIN_PREV_DAY_VOL, MIN_OPEN_PRICE, MAX_OPEN_PRICE
            ]
            cur = conn.execute(q, params)
            for sym, bd in cur.fetchall():
                out.setdefault(str(bd), []).append(sym)
        # Mid-day overlay: use today's provisional row as today's "open" and
        # pull prev_close / prev_vol from the main daily_bars table.
        if include_provisional_today is not None:
            today_str = str(include_provisional_today)
            q2 = """
            WITH prior AS (
                SELECT symbol, close AS prev_close, volume AS prev_vol,
                       ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY bar_date DESC) AS rn
                FROM daily_bars
                WHERE bar_date < ?
            )
            SELECT p.symbol FROM daily_bars_provisional p
            JOIN prior r ON r.symbol = p.symbol AND r.rn = 1
            WHERE p.bar_date = ?
              AND r.prev_close > 0
              AND (p.open - r.prev_close) / r.prev_close * 100 >= ?
              AND r.prev_vol >= ?
              AND p.open BETWEEN ? AND ?
            ORDER BY p.symbol
            """
            params2 = [
                today_str, today_str,
                MIN_GAP_PCT, MIN_PREV_DAY_VOL, MIN_OPEN_PRICE, MAX_OPEN_PRICE,
            ]
            cur = conn.execute(q2, params2)
            for (sym,) in cur.fetchall():
                out.setdefault(today_str, []).append(sym)
        return out
    finally:
        conn.close()


def _intraday_bars_cached_pairs(
    db_path: str, pairs: List[Tuple[str, str]]
) -> Set[Tuple[str, str]]:
    """Return subset of (symbol, date) pairs that already have intraday cached."""
    if not pairs:
        return set()
    conn = sqlite3.connect(db_path)
    try:
        chunk = 300
        out: Set[Tuple[str, str]] = set()
        pairs_list = list(pairs)
        for i in range(0, len(pairs_list), chunk):
            batch = pairs_list[i:i + chunk]
            placeholders = ','.join('(?,?)' for _ in batch)
            flat: List = []
            for s, d in batch:
                flat.extend([s, d])
            cur = conn.execute(f"""
                SELECT DISTINCT symbol, bar_date FROM intraday_bars_1min
                WHERE (symbol, bar_date) IN ({placeholders})
            """, flat)
            for row in cur.fetchall():
                out.add((row[0], row[1]))
        return out
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Data fill
# ---------------------------------------------------------------------------

def fill_daily_bars_for_dates(
    alpaca: AlpacaClient, db: Database, missing_dates: List[date]
) -> int:
    """For every symbol currently in daily_bars, fetch any missing dates."""
    if not missing_dates:
        return 0
    symbols = _symbols_in_daily_cache()
    _log(f"daily-fill: {len(symbols)} symbols × {len(missing_dates)} dates")
    start = missing_dates[0]
    # Fetch 2 days of lead-in so the LAG prev_close works for the earliest date
    fetch_start = start - timedelta(days=5)
    end = missing_dates[-1]
    total = 0
    chunk = 200
    for i in range(0, len(symbols), chunk):
        batch = symbols[i:i + chunk]
        try:
            res = alpaca.get_daily_bars_range(batch, fetch_start, end)
        except Exception as e:
            _log(f"daily-fill chunk {i // chunk} failed: {e}")
            continue
        rows = []
        for sym, bars in (res or {}).items():
            for b in bars:
                bdate = b['date']
                if isinstance(bdate, (datetime, )):
                    bdate = bdate.date()
                # Only write bars in our target window (leaves older data alone)
                if bdate < fetch_start or bdate > end:
                    continue
                rows.append({
                    'symbol': sym, 'date': str(bdate),
                    'open': float(b['open']), 'high': float(b['high']),
                    'low': float(b['low']), 'close': float(b['close']),
                    'volume': int(b['volume']),
                })
        if rows:
            db.save_daily_bars(rows)
            total += len(rows)
        _log(f"  progress: {min(i+chunk, len(symbols))}/{len(symbols)} symbols, +{len(rows)} rows")
    _log(f"daily-fill: {total} total rows written")
    return total


def fill_intraday_for_pairs(
    alpaca: AlpacaClient, db: Database, pairs_by_date: Dict[str, List[str]]
) -> int:
    """Fetch 1-min bars for each (date, [symbols]) pair that doesn't have them."""
    if not pairs_by_date:
        return 0
    # Check which pairs already have bars
    all_pairs = [(s, d) for d, syms in pairs_by_date.items() for s in syms]
    have = _intraday_bars_cached_pairs(CACHE_DB, all_pairs)
    missing = [p for p in all_pairs if p not in have]
    if not missing:
        _log(f"intraday-fill: all {len(all_pairs)} pairs already cached")
        return 0
    _log(f"intraday-fill: {len(missing)} missing pairs (of {len(all_pairs)} total)")
    total = 0
    # Group missing by date for batch fetch per day
    by_date: Dict[str, List[str]] = {}
    for s, d in missing:
        by_date.setdefault(d, []).append(s)
    for d, syms in sorted(by_date.items()):
        day = datetime.strptime(d, '%Y-%m-%d').date()
        start = datetime(day.year, day.month, day.day, 13, 0, tzinfo=timezone.utc)
        end = datetime(day.year, day.month, day.day, 20, 30, tzinfo=timezone.utc)
        chunk = 100
        for i in range(0, len(syms), chunk):
            batch = syms[i:i + chunk]
            try:
                # Use get_historical_1min_bars per-symbol since multi batches
                # on a specific time window aren't exposed cleanly
                for sym in batch:
                    try:
                        df = alpaca.get_historical_1min_bars(sym, start, end)
                        if df is None or df.empty:
                            continue
                        bars_list = df.to_dict('records')
                        # Need timestamp as pd.Timestamp
                        for b in bars_list:
                            b['timestamp'] = pd.to_datetime(b['timestamp'], utc=True)
                        db.save_intraday_bars(sym, d, bars_list)
                        total += len(bars_list)
                    except Exception as e:
                        continue
            except Exception as e:
                _log(f"  intraday chunk {d} failed: {e}")
        _log(f"  {d}: {len(syms)} symbols, running total {total} bars")
    _log(f"intraday-fill: {total} total bars written")
    return total


# ---------------------------------------------------------------------------
# Pipeline runners (via subprocess — keeps imports clean)
# ---------------------------------------------------------------------------

def regen_features(
    include_provisional: bool = False,
    force_full: bool = False,
) -> Optional[str]:
    """Run study_orb_features.py. Returns the path to the NEW CSV.

    `include_provisional=True` sets ORB_INCLUDE_PROVISIONAL_DAILY=1 in the
    subprocess env so `study_orb_features.load_daily_bars_frame` unions
    the `daily_bars_provisional` sidecar into its working frame — giving
    BT visibility into today's mid-day snapshot.

    `force_full=True` passes --force-full-regen, recomputing every trade
    from scratch instead of using the default incremental path. Use when
    feature-extraction logic changes (bugfixes in extract_features, new
    features added/removed, etc.).
    """
    before = set(glob.glob(FEATURES_GLOB))
    mode = "FULL REGEN" if force_full else "incremental"
    _log(f"regenerating features CSV (study_orb_features.py) — mode={mode}...")
    t0 = datetime.now()
    sub_env = os.environ.copy()
    if include_provisional:
        sub_env['ORB_INCLUDE_PROVISIONAL_DAILY'] = '1'
    cmd = [sys.executable, 'study_orb_features.py']
    if force_full:
        cmd.append('--force-full-regen')
    result = subprocess.run(
        cmd,
        cwd=ROOT, capture_output=True, text=True, timeout=900,
        env=sub_env,
    )
    if result.returncode != 0:
        _log(f"features regen failed: {result.stderr[-500:]}")
        return None
    elapsed = (datetime.now() - t0).total_seconds()
    _log(f"features regen: done in {elapsed:.0f}s")
    after = set(glob.glob(FEATURES_GLOB))
    new = sorted(after - before)
    return new[-1] if new else _latest_features_csv()


def run_pipeline_bt(slice_dates: List[str]) -> None:
    """Run study_orb_pipeline_static_lock on the latest features CSV.
    If slice_dates is non-empty, also print a day-by-day table for those dates."""
    _log("running pipeline BT (study_orb_pipeline_static_lock.py)...")
    result = subprocess.run(
        [sys.executable, 'study_orb_pipeline_static_lock.py'],
        cwd=ROOT, capture_output=True, text=True, timeout=900,
    )
    print(result.stdout)
    if result.stderr:
        # Most is progress; only echo if non-empty
        tail = '\n'.join(result.stderr.strip().splitlines()[-20:])
        if tail:
            print(f"[stderr tail] {tail}")
    if result.returncode != 0:
        return
    # Optional date-slice day-by-day
    trades_csv = os.path.join(ROOT, 'analysis_results', 'orb_static_lock_trades.csv')
    if not slice_dates or not os.path.exists(trades_csv):
        return
    t = pd.read_csv(trades_csv)
    t['date'] = pd.to_datetime(t['date']).dt.date.astype(str)
    for sd in slice_dates:
        sub = t[t['date'] == sd].copy()
        print(f"\n{'='*80}\nDay slice — {sd}\n{'='*80}")
        if sub.empty:
            print("  (no trades picked by defended pipeline that day)")
            continue
        sub = sub.sort_values('_sized_pnl')
        cols = ['symbol', '_quintile', '_composite', 'entry_price', 'pnl',
                'exit_reason', '_rp_position', '_sized_pnl']
        cols = [c for c in cols if c in sub.columns]
        print(sub[cols].to_string(index=False, float_format=lambda v: f'{v:.2f}'))
        print(f"\n  Day total: ${sub['_sized_pnl'].sum():+,.2f}  "
              f"({len(sub)} picks, "
              f"{(sub['_sized_pnl'] > 0).sum()}W / {(sub['_sized_pnl'] <= 0).sum()}L)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="ORB backtest — fill data + run pipeline")
    p.add_argument('--end', type=str, default=None,
                   help="End date YYYY-MM-DD (default: today)")
    p.add_argument('--slice', type=str, action='append', default=[],
                   help="Print day-by-day for this date (can repeat)")
    p.add_argument('--no-fill', action='store_true',
                   help="Skip data-fill, only run the pipeline")
    p.add_argument('--force-features', action='store_true',
                   help="Run features CSV regen even if no missing dates "
                        "(still incremental — just don't skip the regen step)")
    p.add_argument('--force-full-regen', action='store_true',
                   help="Tell study_orb_features.py to recompute EVERY trade "
                        "from scratch (disable incremental). Use when you "
                        "change extract_features logic or add/remove a feature. "
                        "Implies --force-features.")
    p.add_argument('--include-today-provisional', action='store_true',
                   help="(Mid-day use) Fetch today's still-open bar into the "
                        "daily_bars_provisional sidecar so BT can see today's "
                        "trades. Never touches the main daily_bars table that "
                        "live reads. Provisional rows are cleared at the start "
                        "of each run so stale mid-day values don't accumulate.")
    args = p.parse_args()

    end_date = (
        datetime.strptime(args.end, '%Y-%m-%d').date() if args.end
        else date.today()
    )

    cfg = Config()
    alpaca = AlpacaClient(cfg.alpaca_api_key, cfg.alpaca_api_secret, paper=True)

    csv_path = _latest_features_csv()
    last_date = _last_features_date(csv_path) if csv_path else None

    _log(f"features CSV: {csv_path or '(none)'}")
    _log(f"last features date: {last_date or '(empty)'}")
    _log(f"end_date: {end_date}")

    need_regen = args.force_features
    if not args.no_fill:
        # Compute missing trading days
        start = (last_date + timedelta(days=1)) if last_date else date(2025, 1, 1)
        if start <= end_date:
            missing = _trading_days_between(alpaca, start, end_date)
            # Exclude any dates already covered (defensive — calendar includes partial overlap)
            if last_date:
                missing = [d for d in missing if d > last_date]
            _log(f"missing trading days: {len(missing)}"
                 + (f"  [{missing[0]}..{missing[-1]}]" if missing else ""))
            if missing:
                db = Database(db_path=CACHE_DB)
                fill_daily_bars_for_dates(alpaca, db, missing)
                pairs = _qualifying_pairs_for_dates(
                    CACHE_DB, missing
                )
                n_pairs = sum(len(v) for v in pairs.values())
                _log(f"qualifying gap-up pairs to fetch 1-min bars: {n_pairs}")
                if n_pairs:
                    fill_intraday_for_pairs(alpaca, db, pairs)
                db.close()
                need_regen = True
        else:
            _log("no missing dates — features CSV is up to date")

    # Optional mid-day provisional overlay. Keeps the main daily_bars cache
    # clean (save_daily_bars drops today-rows during market hours) while
    # still letting BT see today's trades. Sidecar is cleared here so stale
    # rows from a prior run can't leak in.
    if args.include_today_provisional:
        today = date.today()
        db = Database(db_path=CACHE_DB)
        db.clear_provisional_daily_bars()
        try:
            all_syms = _symbols_in_daily_cache()
            _log(f"provisional fetch: {len(all_syms)} symbols × 1 date ({today})")
            chunk = 200
            written = 0
            for i in range(0, len(all_syms), chunk):
                batch = all_syms[i:i + chunk]
                try:
                    res = alpaca.get_daily_bars_range(batch, today, today)
                except Exception as e:
                    _log(f"provisional fetch chunk {i // chunk} failed: {e}")
                    continue
                rows = []
                for sym, bars in (res or {}).items():
                    for b in bars:
                        bdate = b['date']
                        if isinstance(bdate, datetime):
                            bdate = bdate.date()
                        if bdate != today:
                            continue
                        rows.append({
                            'symbol': sym, 'date': str(bdate),
                            'open': float(b['open']), 'high': float(b['high']),
                            'low': float(b['low']), 'close': float(b['close']),
                            'volume': int(b['volume']),
                        })
                if rows:
                    db.save_daily_bars_provisional(rows)
                    written += len(rows)
                _log(f"  provisional progress: "
                     f"{min(i+chunk, len(all_syms))}/{len(all_syms)} symbols, "
                     f"+{len(rows)} rows")
            _log(f"provisional: {written} rows for {today}")

            # Derive today's qualifying gap-up pairs from the provisional
            # overlay and fetch intraday 1-min bars so features extraction
            # has the 9:30-9:34 range bars. Intraday bars for a closed
            # minute are truly final — no pollution concern.
            pairs = _qualifying_pairs_for_dates(
                CACHE_DB, dates=[], include_provisional_today=today,
            )
            n_pairs = sum(len(v) for v in pairs.values())
            _log(f"provisional qualifying pairs today: {n_pairs}")
            if n_pairs:
                fill_intraday_for_pairs(alpaca, db, pairs)
            need_regen = True
        finally:
            db.close()

    # --force-full-regen implies we need to regen (it's the whole point
    # of the flag) — set need_regen so we don't skip based on "no missing
    # dates" upstream.
    if args.force_full_regen:
        need_regen = True

    if need_regen:
        new_csv = regen_features(
            include_provisional=args.include_today_provisional,
            force_full=args.force_full_regen,
        )
        if new_csv:
            _log(f"new features CSV: {new_csv}")

    run_pipeline_bt(args.slice)


if __name__ == '__main__':
    sys.exit(main())
