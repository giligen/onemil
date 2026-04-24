#!/usr/bin/env python3
"""ORB feature extraction + correlation analysis.

Goal: find features known at entry time that discriminate winners from losers.
Used to design a smart filter (later, maybe a sizing model).

Method:
  1. Run ORB_5_vanilla across broad universe (Phase B). For each trade,
     extract 23 features known STRICTLY at end of 5-min range (no look-ahead).
  2. Attach pnl_pct + win label. Write combined CSV.
  3. Run analysis: Pearson correlation with pnl_pct, decile bucket P&L / WR,
     feature-feature correlation matrix (redundancy), top-discriminator ranking.

Look-ahead audit: every feature pulls from bars 0-4 of today OR daily bars
from prior days. NEVER from entry bar (bar 5+) or anywhere later in the day.
SPY features use SPY's bars 0-4 or prior SPY daily bars.

Usage:
    python3 study_orb_features.py
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sqlite3
import sys
from dataclasses import asdict
from datetime import datetime, timedelta, timezone, date as _date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from persistence.database import Database

from study_orb import (
    ENTRY_SLIP_BPS_DEFAULT, EXIT_SLIP_BPS_DEFAULT, POSITION_SIZE_USD,
    OUT_DIR, OrbTrade, simulate_orb_trade, _bars_to_df,
    _session_open_timestamp,
)
from study_orb_broad import load_broad_universe
from trading.trading_hours import today_et


CACHE_DB = 'data/cache.db'
RANGE_MINUTES = 5  # locked to 5-min ORB for the feature study
FEATURES_GLOB = os.path.join(OUT_DIR, 'orb_features_*.csv')


# ---------------------------------------------------------------------------
# Incremental-regen helpers
# ---------------------------------------------------------------------------

def _latest_features_csv() -> Optional[str]:
    """Most recent orb_features_*.csv, excluding the corrmatrix sidecar."""
    files = [p for p in sorted(glob.glob(FEATURES_GLOB)) if 'corrmatrix' not in p]
    return files[-1] if files else None


def _load_existing_features(path: Optional[str]) -> pd.DataFrame:
    """Read a prior features CSV and normalize the date column to
    datetime.date. Empty DataFrame if no path or unreadable."""
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[features] WARN: couldn't read {path}: {e} — treating as empty")
        return pd.DataFrame()
    if 'date' not in df.columns:
        print(f"[features] WARN: {path} missing 'date' — ignoring for incremental")
        return pd.DataFrame()
    df['date'] = pd.to_datetime(df['date']).dt.date
    return df


class IncrementalSchemaDrift(RuntimeError):
    """Raised when the existing features CSV's column set doesn't match
    the new rows being produced. A silent `pd.concat` would NaN-fill
    the missing columns, corrupting downstream pipeline stats.

    Users hitting this must pass `--force-full-regen` to rebuild the CSV
    from scratch under the new feature schema.
    """


def _merge_new_with_existing(
    existing_df: pd.DataFrame, new_rows: List[Dict],
) -> pd.DataFrame:
    """Return the merged DataFrame for the new features CSV.

    Rule: for every date we PROCESSED this run (set of dates in new_rows),
    drop existing rows on those dates — the new values replace them. For
    every other date, existing rows pass through untouched. This means:
      - Full regen (dates_processed covers everything) → existing dropped,
        new takes over.
      - Incremental adding dates (no overlap) → existing untouched, new
        appended.
      - Post-close refresh of a date computed mid-day as provisional →
        that date's old row is dropped; final row replaces it.
      - Mid-day run with no provisional overlay where today's row is
        already in the CSV → today not processed, today's row preserved
        (no silent data loss).

    Safety: raises IncrementalSchemaDrift if the existing CSV's columns
    don't match the new_rows' columns. A silent `pd.concat` would
    NaN-fill the missing columns and downstream `dropna` would silently
    delete rows, skewing stats. Fail fast instead.
    """
    new_df = pd.DataFrame(new_rows) if new_rows else pd.DataFrame()
    if new_df.empty:
        return existing_df.reset_index(drop=True) if not existing_df.empty else new_df
    if 'date' in new_df.columns:
        new_df['date'] = pd.to_datetime(new_df['date']).dt.date
    if existing_df.empty:
        return new_df.sort_values(['date', 'symbol']).reset_index(drop=True)

    # Column-drift guard: both frames must have the same column set.
    # Missing columns in either direction would silently NaN-fill on concat.
    ex_cols = set(existing_df.columns)
    new_cols = set(new_df.columns)
    if ex_cols != new_cols:
        missing_in_new = ex_cols - new_cols
        missing_in_existing = new_cols - ex_cols
        raise IncrementalSchemaDrift(
            f"feature-schema mismatch between existing CSV and freshly "
            f"extracted rows. Run with --force-full-regen to rebuild from "
            f"scratch under the new schema.\n"
            f"  missing in new_rows      : {sorted(missing_in_new) or '(none)'}\n"
            f"  missing in existing_df   : {sorted(missing_in_existing) or '(none)'}\n"
            f"  (existing={len(ex_cols)} cols, new={len(new_cols)} cols)"
        )

    dates_processed = set(new_df['date'].unique())
    kept = existing_df[~existing_df['date'].isin(dates_processed)]
    merged = pd.concat([kept, new_df], ignore_index=True)
    return merged.sort_values(['date', 'symbol']).reset_index(drop=True)


def _atomic_write_csv(df: pd.DataFrame, path: str) -> None:
    """Write CSV via .tmp + os.replace so a crashed run can't leave a
    half-written file that a subsequent incremental run would misread."""
    tmp = path + '.tmp'
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Daily-bar context loader (loaded once per run, indexed for fast lookup)
# ---------------------------------------------------------------------------

def load_daily_bars_frame(db_path: str = CACHE_DB) -> pd.DataFrame:
    """All daily bars for all symbols — one query, used for prev-day + 20d stats.

    If the env var ORB_INCLUDE_PROVISIONAL_DAILY=1 is set, also reads mid-day
    provisional rows from `daily_bars_provisional` (written by
    `orb_backtest.py --include-today-provisional`). FINAL rows win for
    any (symbol, bar_date) collision; provisional rows only fill gaps.
    Live engines never set this var.
    """
    include_prov = os.environ.get('ORB_INCLUDE_PROVISIONAL_DAILY') == '1'
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT symbol, bar_date, open, high, low, close, volume FROM daily_bars",
        conn,
    )
    if include_prov:
        prov = pd.read_sql_query(
            "SELECT symbol, bar_date, open, high, low, close, volume "
            "FROM daily_bars_provisional",
            conn,
        )
        if not prov.empty:
            # Drop provisional rows that collide with an existing FINAL row.
            prov['bar_date'] = pd.to_datetime(prov['bar_date'])
            df_norm = df.copy()
            df_norm['bar_date'] = pd.to_datetime(df_norm['bar_date'])
            final_keys = set(zip(df_norm['symbol'], df_norm['bar_date']))
            mask = ~prov.apply(
                lambda r: (r['symbol'], r['bar_date']) in final_keys, axis=1
            )
            kept = prov[mask]
            if len(kept):
                df = pd.concat([df, kept], ignore_index=True)
                print(f"[features] provisional overlay: +{len(kept)} rows "
                      f"({len(prov) - len(kept)} shadowed by FINAL)",
                      flush=True)
    conn.close()
    df['bar_date'] = pd.to_datetime(df['bar_date'])
    return df.sort_values(['symbol', 'bar_date']).reset_index(drop=True)


def load_spy_intraday(db_path: str = CACHE_DB) -> pd.DataFrame:
    """SPY 1-min bars for the study period."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT timestamp, open, high, low, close, volume FROM intraday_bars_1min "
        "WHERE symbol='SPY' ORDER BY timestamp",
        conn,
    )
    conn.close()
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    return df.sort_values('timestamp').reset_index(drop=True)


# ---------------------------------------------------------------------------
# Per-trade feature extractor (NO LOOK-AHEAD)
# ---------------------------------------------------------------------------

def _safe_div(a, b, default=0.0):
    try:
        return a / b if b else default
    except Exception:
        return default


def extract_features(
    bars_df: pd.DataFrame,
    symbol: str,
    date_str: str,
    daily_by_sym: Dict[str, pd.DataFrame],
    spy_intraday: pd.DataFrame,
    spy_daily: pd.DataFrame,
) -> Optional[Dict[str, float]]:
    """Compute features at end of 5-min range. Returns None if insufficient data."""
    if bars_df.empty:
        return None
    open_ts = _session_open_timestamp(bars_df)
    if open_ts is None:
        return None
    range_end_ts = open_ts + timedelta(minutes=RANGE_MINUTES)
    range_mask = (bars_df['timestamp'] >= open_ts) & (bars_df['timestamp'] < range_end_ts)
    rb = bars_df.loc[range_mask].reset_index(drop=True)
    if len(rb) < RANGE_MINUTES:
        return None

    feat: Dict[str, float] = {}

    # --- Range features (bars 0..4 only) ---
    range_high = float(rb['high'].max())
    range_low = float(rb['low'].min())
    open_p = float(rb['open'].iloc[0])
    close_p = float(rb['close'].iloc[-1])
    range_size = range_high - range_low

    feat['range_size_pct'] = _safe_div(range_size, open_p) * 100
    feat['range_total_volume'] = float(rb['volume'].sum())
    bar_ranges = (rb['high'] - rb['low']) / rb['close'].replace(0, np.nan)
    feat['range_avg_bar_range_pct'] = float(bar_ranges.mean(skipna=True) * 100) \
        if not bar_ranges.isna().all() else 0.0
    vol_vals = rb['volume'].to_numpy(dtype=float)
    feat['range_volume_stddev_pct'] = (
        float(vol_vals.std() / max(vol_vals.mean(), 1e-9) * 100) if vol_vals.mean() > 0 else 0.0
    )
    feat['bars_green_in_range'] = float(int((rb['close'] > rb['open']).sum()))
    feat['range_close_position'] = _safe_div(close_p - range_low, range_size) if range_size > 0 else 0.5
    feat['range_return_pct'] = _safe_div(close_p - open_p, open_p) * 100
    feat['last_bar_green'] = float(1 if rb['close'].iloc[-1] > rb['open'].iloc[-1] else 0)
    # VWAP of bars 0-4
    typ_price = (rb['high'] + rb['low'] + rb['close']) / 3.0
    total_vol = float(rb['volume'].sum())
    if total_vol > 0:
        vwap = float((typ_price * rb['volume']).sum() / total_vol)
        feat['range_vwap_distance_pct'] = _safe_div(close_p - vwap, vwap) * 100
    else:
        feat['range_vwap_distance_pct'] = 0.0

    # --- Gap + prev-day features (from daily_bars for this symbol) ---
    dtx = pd.Timestamp(date_str)
    sym_daily = daily_by_sym.get(symbol)
    if sym_daily is None or sym_daily.empty:
        return None
    # Previous trading day = the daily bar strictly before dtx
    prev_mask = sym_daily['bar_date'] < dtx
    if not prev_mask.any():
        return None
    prev_row = sym_daily.loc[prev_mask].iloc[-1]
    prev_close = float(prev_row['close'])
    prev_high = float(prev_row['high'])
    prev_low = float(prev_row['low'])
    prev_volume = float(prev_row['volume'])

    feat['gap_pct'] = _safe_div(open_p - prev_close, prev_close) * 100
    feat['prev_day_range_pct'] = _safe_div(prev_high - prev_low, prev_close) * 100
    prev_range = prev_high - prev_low
    feat['prev_day_close_position'] = _safe_div(prev_close - prev_low, prev_range) if prev_range > 0 else 0.5

    # 20-day context: prior 20 days strictly BEFORE today (so use days from
    # prev_row going back 20). Exclude today itself.
    prior_df = sym_daily.loc[prev_mask].tail(20)
    if len(prior_df) < 5:
        # Too little history; still continue but flag via NaN
        feat['avg_daily_volume_20d'] = 0.0
        feat['avg_daily_range_pct_20d'] = 0.0
        feat['price_vs_20d_high_pct'] = 0.0
        feat['return_volatility_20d'] = 0.0
        feat['prev_day_volume_vs_20d'] = 0.0
    else:
        avg_vol_20d = float(prior_df['volume'].mean())
        feat['avg_daily_volume_20d'] = avg_vol_20d
        prior_ranges = (prior_df['high'] - prior_df['low']) / prior_df['close'].replace(0, np.nan)
        feat['avg_daily_range_pct_20d'] = float(prior_ranges.mean(skipna=True) * 100)
        high_20d = float(prior_df['high'].max())
        feat['price_vs_20d_high_pct'] = _safe_div(open_p - high_20d, high_20d) * 100
        # Daily returns from prior_df closes
        closes = prior_df['close'].to_numpy(dtype=float)
        if len(closes) > 1:
            rets = np.diff(closes) / closes[:-1]
            feat['return_volatility_20d'] = float(rets.std() * 100)
        else:
            feat['return_volatility_20d'] = 0.0
        feat['prev_day_volume_vs_20d'] = _safe_div(prev_volume, avg_vol_20d)

    # --- SPY features ---
    spy_open_ts = None
    # Match SPY 9:30 ET open on same date
    dtx_date = dtx.date()
    spy_day_mask = (
        (spy_intraday['timestamp'].dt.date == dtx_date) &
        (spy_intraday['timestamp'].dt.minute == 30) &
        (spy_intraday['timestamp'].dt.hour.isin([13, 14]))
    )
    if spy_day_mask.any():
        spy_open_ts = spy_intraday.loc[spy_day_mask, 'timestamp'].iloc[0]
        spy_end_ts = spy_open_ts + timedelta(minutes=RANGE_MINUTES)
        spy_range_mask = (
            (spy_intraday['timestamp'] >= spy_open_ts) &
            (spy_intraday['timestamp'] < spy_end_ts)
        )
        spy_rb = spy_intraday.loc[spy_range_mask]
        if len(spy_rb) >= RANGE_MINUTES:
            spy_hi = float(spy_rb['high'].max())
            spy_lo = float(spy_rb['low'].min())
            spy_op = float(spy_rb['open'].iloc[0])
            spy_cl = float(spy_rb['close'].iloc[-1])
            feat['spy_range_pct_5min'] = _safe_div(spy_hi - spy_lo, spy_op) * 100
            feat['spy_return_5min_pct'] = _safe_div(spy_cl - spy_op, spy_op) * 100
        else:
            feat['spy_range_pct_5min'] = 0.0
            feat['spy_return_5min_pct'] = 0.0
    else:
        feat['spy_range_pct_5min'] = 0.0
        feat['spy_return_5min_pct'] = 0.0

    # SPY daily (for gap + 3d range)
    spy_prev = spy_daily.loc[spy_daily['bar_date'] < dtx]
    spy_today = spy_daily.loc[spy_daily['bar_date'] == dtx]
    if not spy_prev.empty and not spy_today.empty:
        spy_prev_close = float(spy_prev.iloc[-1]['close'])
        spy_today_open = float(spy_today.iloc[0]['open'])
        feat['spy_gap_pct'] = _safe_div(spy_today_open - spy_prev_close, spy_prev_close) * 100
    else:
        feat['spy_gap_pct'] = 0.0
    # SPY 3-day range % using prior 3 daily bars
    spy_prior3 = spy_prev.tail(3)
    if len(spy_prior3) == 3:
        ranges = (spy_prior3['high'] - spy_prior3['low']) / spy_prior3['close'].replace(0, np.nan)
        feat['spy_3d_range_pct'] = float(ranges.mean(skipna=True) * 100)
    else:
        feat['spy_3d_range_pct'] = 0.0

    # --- Time features ---
    feat['day_of_week'] = float(dtx.dayofweek)
    feat['days_since_month_start'] = float(dtx.day)

    return feat


# ---------------------------------------------------------------------------
# Main: run ORB + extract features + analyze
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="ORB per-trade feature extraction (default: incremental)"
    )
    p.add_argument(
        '--force-full-regen', action='store_true',
        help="Ignore any existing features CSV and recompute every trade "
             "from scratch. Use when feature-extraction logic changes.",
    )
    p.add_argument(
        '--start-date', type=str, default=None,
        help="YYYY-MM-DD — explicit start date for the recompute window. "
             "Overrides the incremental auto-detect.",
    )
    return p.parse_args()


def _resolve_incremental_plan(
    args,
) -> Tuple[pd.DataFrame, Optional[_date]]:
    """Decide which existing rows to preserve and which start_date to use.

    Returns (existing_df, start_date). `start_date` is inclusive — pairs
    with date >= start_date are reprocessed; existing rows on those dates
    are dropped and replaced in the merge step. `start_date` may be None,
    meaning "no filter — full regen".
    """
    if args.force_full_regen:
        print("[features] FORCE FULL REGEN — ignoring any existing CSV")
        return pd.DataFrame(), None
    if args.start_date:
        try:
            explicit = datetime.strptime(args.start_date, '%Y-%m-%d').date()
        except Exception:
            raise SystemExit(f"--start-date: invalid YYYY-MM-DD: {args.start_date}")
        existing = _load_existing_features(_latest_features_csv())
        print(f"[features] explicit start_date={explicit}")
        return existing, explicit

    # Default: incremental, anchored at last date of most-recent CSV.
    latest = _latest_features_csv()
    if latest is None:
        print("[features] no prior CSV — full regen")
        return pd.DataFrame(), None
    existing = _load_existing_features(latest)
    if existing.empty:
        print(f"[features] prior CSV unreadable ({latest}) — full regen")
        return pd.DataFrame(), None
    last_date = existing['date'].max()
    # Recompute the LAST date in the CSV too — this covers two edge cases:
    #   1. A post-close run refreshes a date that was computed mid-day
    #      against a provisional daily bar.
    #   2. If today's already in the CSV from a prior run + the daily bar
    #      has since been updated (provisional → final), today's row gets
    #      refreshed.
    # Cost: N symbols × 1 day ≈ a second or two — negligible.
    print(f"[features] incremental — existing={latest} last_date={last_date} "
          f"(recomputing {last_date} onward)")
    return existing, last_date


def main() -> None:
    args = _parse_args()
    t0 = datetime.now()
    print(f"[{t0.isoformat(timespec='seconds')}] ORB feature study — ORB_5_vanilla")

    existing_df, start_date = _resolve_incremental_plan(args)

    print("\nLoading broad universe...")
    # Env-var toggle: include today's provisional row from daily_bars_provisional.
    # Set by orb_backtest.py --include-today-provisional via subprocess env.
    provisional_today = (
        today_et() if os.environ.get('ORB_INCLUDE_PROVISIONAL_DAILY') == '1'
        else None
    )
    universe = load_broad_universe(include_provisional_today=provisional_today)
    n_pairs_full = sum(len(v) for v in universe.values())

    if start_date is not None:
        start_str = start_date.isoformat()
        universe = {d: syms for d, syms in universe.items() if d >= start_str}
    n_pairs = sum(len(v) for v in universe.values())
    print(f"  {n_pairs:,} (symbol, date) pairs across {len(universe)} days "
          f"[filtered from {n_pairs_full:,} by start_date={start_date}]")

    if not universe and existing_df.empty:
        print("No pairs to process and no existing CSV — nothing to do.")
        return

    print("Loading daily bars + SPY context...")
    daily = load_daily_bars_frame()
    daily_by_sym: Dict[str, pd.DataFrame] = {
        s: g.reset_index(drop=True) for s, g in daily.groupby('symbol')
    }
    spy_daily = daily.loc[daily['symbol'] == 'SPY'].reset_index(drop=True)
    print(f"  daily_bars: {daily['symbol'].nunique():,} symbols, SPY daily: {len(spy_daily)} rows")

    spy_intraday = load_spy_intraday()
    print(f"  SPY intraday bars: {len(spy_intraday):,}")

    # Bulk-fetch intraday bars for the filtered pair set only.
    print("\nBulk-fetching 1-min bars for universe...")
    db = Database(db_path=CACHE_DB)
    pair_list: List[Tuple[str, str]] = [
        (s, d) for d, syms in universe.items() for s in syms
    ]
    raw = db.get_intraday_bars_bulk(pair_list) if pair_list else {}
    print(f"  Got {len(raw):,} bar sets")
    db.close()

    bars_cache: Dict[Tuple[str, str], pd.DataFrame] = {
        k: _bars_to_df(v) for k, v in raw.items()
    }

    # Simulate + extract features
    print("\nSimulating ORB_5_vanilla + extracting features per trade...")
    rows: List[Dict] = []
    n_simulated = 0
    n_extracted = 0
    for date_str in sorted(universe.keys()):
        for symbol in universe[date_str]:
            bars_df = bars_cache.get((symbol, date_str))
            if bars_df is None or bars_df.empty:
                continue
            # Extract features FIRST (independent of entry firing)
            feats = extract_features(
                bars_df, symbol, date_str,
                daily_by_sym, spy_intraday, spy_daily,
            )
            if feats is None:
                continue
            # Run simulator
            trade = simulate_orb_trade(
                bars_df, symbol, date_str, 'ORB_5_vanilla',
                range_minutes=5, entry_mode='touch', stop_mode='range_low',
                target_mult=2.0, time_stop_minutes=60,
            )
            n_simulated += 1
            if not trade.entered:
                continue  # only analyze trades where entry actually fired

            row = {
                'symbol': symbol, 'date': date_str,
                'entry_price': trade.entry_price,
                'pnl': trade.pnl, 'pnl_pct': trade.pnl_pct,
                'exit_reason': trade.exit_reason,
                'win': 1 if trade.pnl > 0 else 0,
                **feats,
            }
            rows.append(row)
            n_extracted += 1

    print(f"  Simulated: {n_simulated:,} | Entered: {n_extracted:,} "
          f"({n_extracted/max(n_simulated,1)*100:.1f}%) "
          f"| existing preserved: {len(existing_df)}")

    df = _merge_new_with_existing(existing_df, rows)
    if df.empty:
        print("No trades in merged dataset — nothing to analyze.")
        return

    # --- Analysis ---
    print("\n=== Pearson correlation with pnl_pct ===")
    feature_cols = [c for c in df.columns if c not in (
        'symbol', 'date', 'entry_price', 'pnl', 'pnl_pct', 'exit_reason', 'win'
    )]
    corrs = []
    for c in feature_cols:
        try:
            corr_pnl = df[c].corr(df['pnl_pct'])
            corr_win = df[c].corr(df['win'])
        except Exception:
            corr_pnl = corr_win = float('nan')
        corrs.append({'feature': c, 'corr_pnl_pct': corr_pnl, 'corr_win': corr_win})
    corr_df = pd.DataFrame(corrs).sort_values(
        'corr_pnl_pct', key=lambda s: s.abs(), ascending=False
    )
    print(corr_df.to_string(index=False))

    # --- Bucket (quintile) analysis ---
    print("\n=== Quintile bucket analysis (winner variant by feature) ===")
    buckets_data = []
    for c in feature_cols:
        try:
            df['_b'] = pd.qcut(df[c], 5, labels=['Q1_low','Q2','Q3','Q4','Q5_high'], duplicates='drop')
        except Exception:
            continue
        g = df.groupby('_b', observed=True).agg(
            n=('pnl', 'count'),
            wr=('win', 'mean'),
            avg_pnl=('pnl', 'mean'),
            total_pnl=('pnl', 'sum'),
        ).reset_index()
        # Compute Q5 − Q1 spread as discriminator magnitude
        if len(g) >= 2:
            q_spread = float(g['avg_pnl'].iloc[-1] - g['avg_pnl'].iloc[0])
            q5_wr = float(g['wr'].iloc[-1])
            q1_wr = float(g['wr'].iloc[0])
            wr_spread = q5_wr - q1_wr
        else:
            q_spread = wr_spread = 0.0
        buckets_data.append({
            'feature': c,
            'q5_avg_pnl': g['avg_pnl'].iloc[-1] if len(g) else 0,
            'q1_avg_pnl': g['avg_pnl'].iloc[0] if len(g) else 0,
            'q5_minus_q1': q_spread,
            'q5_wr': q5_wr if len(g) else 0,
            'q1_wr': q1_wr if len(g) else 0,
            'wr_spread': wr_spread,
        })
    buckets_df = pd.DataFrame(buckets_data).sort_values(
        'q5_minus_q1', key=lambda s: s.abs(), ascending=False
    )
    print(buckets_df.to_string(index=False, float_format='%.3f'))
    df = df.drop(columns=['_b'], errors='ignore')

    # --- Feature-feature correlation matrix (identify redundancy) ---
    feat_corr = df[feature_cols].corr().round(2)

    # --- Write artifacts ---
    os.makedirs(OUT_DIR, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    csv_path = f"{OUT_DIR}/orb_features_{ts}.csv"
    md_path = f"{OUT_DIR}/orb_features_{ts}.md"
    mat_path = f"{OUT_DIR}/orb_features_corrmatrix_{ts}.csv"
    # Atomic write — .tmp + os.replace. A crashed run cannot leave a
    # half-written CSV that the next incremental run would misread.
    _atomic_write_csv(df, csv_path)
    feat_corr.to_csv(mat_path)
    print(f"\nPer-trade features CSV: {csv_path} ({len(df)} rows)")
    print(f"Feature correlation matrix CSV: {mat_path}")

    with open(md_path, 'w') as f:
        f.write(f"# ORB Feature Correlation Study\n\n")
        f.write(f"_Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}_\n\n")
        f.write(f"**Trades analyzed**: {len(df):,} (ORB_5_vanilla, broad universe)\n\n")
        f.write(f"**Feature count**: {len(feature_cols)}\n\n")
        f.write(f"**Look-ahead audit**: all features computed from bars 0-4 or prior "
                f"daily bars only. Entry bar (bar 5+) data is NOT used.\n\n")

        wins = int(df['win'].sum())
        f.write(f"## Sample\n\n")
        f.write(f"- Wins: {wins} / {len(df)} ({wins/len(df)*100:.1f}%)\n")
        f.write(f"- Avg pnl_pct: {df['pnl_pct'].mean():.3f}%\n")
        f.write(f"- Median pnl_pct: {df['pnl_pct'].median():.3f}%\n\n")

        f.write("## Pearson correlation (sorted by |corr with pnl_pct|)\n\n")
        f.write("| feature | corr(pnl_pct) | corr(win) |\n|---|---:|---:|\n")
        for _, r in corr_df.iterrows():
            f.write(f"| {r['feature']} | {r['corr_pnl_pct']:+.3f} | {r['corr_win']:+.3f} |\n")

        f.write("\n## Quintile analysis (Q5-top bucket vs Q1-bottom avg P&L)\n\n")
        f.write("| feature | Q5 avg_pnl | Q1 avg_pnl | Q5−Q1 | Q5 WR | Q1 WR | WR spread |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for _, r in buckets_df.iterrows():
            f.write(f"| {r['feature']} | ${r['q5_avg_pnl']:+,.0f} | ${r['q1_avg_pnl']:+,.0f} | "
                    f"${r['q5_minus_q1']:+,.0f} | {r['q5_wr']*100:.0f}% | {r['q1_wr']*100:.0f}% | "
                    f"{r['wr_spread']*100:+.0f}pp |\n")

        f.write(f"\n## Feature-feature correlation matrix\n\n")
        f.write(f"Written to: `{os.path.basename(mat_path)}`. Inspect for redundant feature pairs "
                f"(|corr| > 0.8 suggests one feature is informationally duplicate of another).\n\n")
        # Print abbreviated top-redundancy pairs in the report
        pairs = []
        for i, f1 in enumerate(feature_cols):
            for f2 in feature_cols[i+1:]:
                c = feat_corr.loc[f1, f2]
                if abs(c) >= 0.7:
                    pairs.append((f1, f2, c))
        pairs.sort(key=lambda t: abs(t[2]), reverse=True)
        if pairs:
            f.write("### High-correlation pairs (|corr| ≥ 0.7)\n\n")
            f.write("| feat A | feat B | corr |\n|---|---|---:|\n")
            for f1, f2, c in pairs[:20]:
                f.write(f"| {f1} | {f2} | {c:+.2f} |\n")
        else:
            f.write("_No pairs above |0.7| — features are largely independent._\n")

        f.write("\n## Top filter candidates (high |Q5-Q1| and high |corr|)\n\n")
        # Simple rank: abs correlation + normalized Q5-Q1 spread
        top_rank = corr_df.merge(
            buckets_df[['feature', 'q5_minus_q1']], on='feature'
        )
        top_rank['abs_corr'] = top_rank['corr_pnl_pct'].abs()
        top_rank['abs_spread'] = top_rank['q5_minus_q1'].abs()
        # Normalize each to [0, 1] and sum
        for col in ['abs_corr', 'abs_spread']:
            m = top_rank[col].max() or 1.0
            top_rank[col + '_norm'] = top_rank[col] / m
        top_rank['score'] = top_rank['abs_corr_norm'] + top_rank['abs_spread_norm']
        top_rank = top_rank.sort_values('score', ascending=False).head(8)
        f.write("| rank | feature | corr(pnl_pct) | Q5−Q1 | combined score |\n|---:|---|---:|---:|---:|\n")
        for i, (_, r) in enumerate(top_rank.iterrows(), 1):
            f.write(f"| {i} | {r['feature']} | {r['corr_pnl_pct']:+.3f} | "
                    f"${r['q5_minus_q1']:+,.0f} | {r['score']:.2f} |\n")

    print(f"Markdown report: {md_path}")
    print(f"\nElapsed: {(datetime.now() - t0).total_seconds():.1f}s")


if __name__ == '__main__':
    main()
