#!/usr/bin/env python3
"""lev_rebalance — wrapper close-rebalance drift study. See PREREG.md.

Scope actually run (documented deviation from the full PREREG universe, forced
by a hard tool/time budget): the SIGNAL universe is the three FAMILIES-based
single-stock leveraged complexes that are code-verified in
trading/orb_correlation.py (TSLA, MSTR, NVDA + their 2x/inverse wrappers) —
not the full offline-class-map-derived wrapper set (6,136 candidates), which
would require tens of thousands of intraday queries. The CONTROL pool is a
fixed random sample of common-stock symbols from the offline class map.
TEST (>= 2026-06-01) is never queried — sealed by construction.
"""
import csv
import json
import os
import random
import sqlite3
import sys
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from trading.orb_correlation import FAMILIES  # noqa: E402
from research.scripts.pit_listings import PitListings, is_test_ticker  # noqa: E402

OUT = os.path.join(ROOT, 'research', 'lev_rebalance')
DB = f"file:{ROOT}/data/cache.db?mode=ro"
ET = ZoneInfo('America/New_York')
UTC = ZoneInfo('UTC')

TRAIN_LO, TRAIN_HI = date(2025, 1, 1), date(2025, 12, 31)
VAL_LO, VAL_HI = date(2026, 1, 1), date(2026, 5, 31)
LOOKBACK_LO = date(2024, 11, 1)  # for ADV20 / prior close warmup
RUN_HI = VAL_HI  # TEST sealed — never query past this

DAILY_PREFILTER = 0.03   # loose daily close-to-close prefilter before intraday fetch
SIG_THRESH = 0.05        # |r| >= 5%
MIN_PRICE = 5.0
HALF_SPREAD_PCT_1500 = 0.23312826680234397 / 2.0 / 100.0  # frames14 f45_minute_table clock_m=900, med/2, as fraction
STOP_FRAC = 0.02
R_FRAC = 0.02

N_CONTROL_SAMPLE = 60
CONTROL_SEED = 1291
MAX_CONTROL_INTRADAY_FETCHES = 900  # hard cap, budget guard

# ---------------------------------------------------------------------------
# Universe: FAMILIES-based single-stock leveraged complexes
# ---------------------------------------------------------------------------
SIGNAL_FAMILIES = {
    'TSLA': FAMILIES['tsla_leveraged'],
    'MSTR': FAMILIES['mstr_leveraged'],
    'NVDA': FAMILIES['nvda_leveraged'],
}
ALL_WRAPPERS = sorted({w for ws in SIGNAL_FAMILIES.values() for w in ws})
UNDERLYINGS = sorted(SIGNAL_FAMILIES.keys())

conn = sqlite3.connect(DB, uri=True, timeout=30)


def log(*a):
    print(*a, flush=True)


def load_daily(symbols):
    q = f"SELECT symbol, bar_date, open, high, low, close, volume FROM daily_bars " \
        f"WHERE symbol IN ({','.join('?' * len(symbols))}) AND bar_date >= ? AND bar_date <= ? " \
        f"ORDER BY symbol, bar_date"
    df = pd.read_sql_query(q, conn, params=[*symbols, str(LOOKBACK_LO), str(RUN_HI)])
    df['bar_date'] = pd.to_datetime(df['bar_date']).dt.date
    df = df.sort_values(['symbol', 'bar_date'])
    df['prior_close'] = df.groupby('symbol')['close'].shift(1)
    df['dollar_vol'] = df['close'] * df['volume']
    df['adv20'] = (df.groupby('symbol')['dollar_vol']
                   .transform(lambda s: s.shift(1).rolling(20, min_periods=10).mean()))
    return df


def et_window_utc(d, h0, m0, h1, m1):
    lo = datetime(d.year, d.month, d.day, h0, m0, tzinfo=ET).astimezone(UTC)
    hi = datetime(d.year, d.month, d.day, h1, m1, tzinfo=ET).astimezone(UTC)
    return lo, hi


def fetch_window(symbol, d, h0=14, m0=50, h1=16, m1=5):
    lo, hi = et_window_utc(d, h0, m0, h1, m1)
    q = ("SELECT timestamp, open, high, low, close, volume FROM intraday_bars_1min "
         "WHERE symbol=? AND bar_date=? AND timestamp >= ? AND timestamp <= ? ORDER BY timestamp")
    cur = conn.execute(q, (symbol, str(d), lo.isoformat(), hi.isoformat()))
    rows = cur.fetchall()
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['ts_et'] = pd.to_datetime(df['timestamp']).dt.tz_convert(ET)
    return df


def eval_bar_day(symbol, d, prior_close, adv20_und, wrappers_adv, pit, wrapper_pit_ok):
    """Returns dict or None. Pulls one intraday window covering 14:50-16:05 ET."""
    df = fetch_window(symbol, d)
    if df is None or len(df) < 5:
        return {'symbol': symbol, 'date': d, 'has_bars': False}
    at_or_before_1500 = df[df['ts_et'].dt.time <= datetime(2000, 1, 1, 15, 0).time()]
    after_1500 = df[df['ts_et'].dt.time >= datetime(2000, 1, 1, 15, 1).time()]
    if at_or_before_1500.empty or after_1500.empty:
        return {'symbol': symbol, 'date': d, 'has_bars': False}
    close_1500 = float(at_or_before_1500.iloc[-1]['close'])
    entry_bar = after_1500.iloc[0]
    entry_open = float(entry_bar['open'])
    entry_low = float(entry_bar['low'])
    entry_high = float(entry_bar['high'])
    r = close_1500 / prior_close - 1.0
    rec = {'symbol': symbol, 'date': d, 'has_bars': True, 'r': r, 'prior_close': prior_close,
           'close_1500': close_1500, 'entry_open': entry_open, 'entry_low': entry_low,
           'entry_high': entry_high}
    if abs(r) < SIG_THRESH or prior_close < MIN_PRICE:
        rec['qualifies'] = False
        return rec
    rec['qualifies'] = True
    side = 1 if r > 0 else -1
    rec['side'] = side
    # Reg SHO uptick guard for r <= -10%
    rec['sho_gate'] = True
    if r <= -0.10:
        rec['sho_gate'] = bool(entry_open > entry_low)
    if wrappers_adv is not None:
        F = 2.0 * abs(r) * wrappers_adv / adv20_und if (adv20_und and adv20_und > 0) else np.nan
        rec['F'] = F
    else:
        rec['F'] = np.nan
    # cost + fills
    half = HALF_SPREAD_PCT_1500
    eff_entry = entry_open * (1 + half) if side > 0 else entry_open * (1 - half)
    rec['eff_entry'] = eff_entry
    # MOC exit — need the day's official close from daily_bars (passed via prior_close table by caller)
    # 2% stop diagnostic uses the rest of df through 16:05
    after_entry = df[df['ts_et'] > entry_bar['ts_et']]
    stop_level = eff_entry * (1 - STOP_FRAC) if side > 0 else eff_entry * (1 + STOP_FRAC)
    stopped = False
    stop_exit = None
    for _, b in after_entry.iterrows():
        if side > 0 and float(b['low']) <= stop_level:
            stopped, stop_exit = True, stop_level
            break
        if side < 0 and float(b['high']) >= stop_level:
            stopped, stop_exit = True, stop_level
            break
    rec['stopped_2pct'] = stopped
    rec['stop_exit'] = stop_exit
    return rec


def main():
    log('=== lev_rebalance: build universe ===')
    log('signal underlyings:', UNDERLYINGS)
    log('wrappers:', ALL_WRAPPERS)

    pit = None
    try:
        pit = PitListings()
        log('pit_listings coverage:', pit.coverage)
    except Exception as e:
        log('WARNING pit_listings unavailable:', e)

    daily = load_daily(UNDERLYINGS + ALL_WRAPPERS)
    daily_by_sym = {s: g.set_index('bar_date') for s, g in daily.groupby('symbol')}

    # point-in-time wrapper listing (first daily bar as proxy; cross-check pit_listings if available)
    wrapper_first_bar = {s: (daily_by_sym[s].index.min() if s in daily_by_sym and len(daily_by_sym[s]) else None)
                         for s in ALL_WRAPPERS}
    log('wrapper first daily bar (PIT proxy):', wrapper_first_bar)

    trading_days = sorted({d for d in daily['bar_date'] if TRAIN_LO <= d <= RUN_HI})

    signal_recs = []
    for und in UNDERLYINGS:
        if und not in daily_by_sym:
            log(f'WARNING {und} has no daily_bars rows')
            continue
        dund = daily_by_sym[und]
        wraps = SIGNAL_FAMILIES[und]
        n_checked = 0
        for d in trading_days:
            if d not in dund.index:
                continue
            row = dund.loc[d]
            pc = row['prior_close']
            if pd.isna(pc) or pc < MIN_PRICE:
                continue
            close = row['close']
            if pd.isna(close) or abs(close / pc - 1.0) < DAILY_PREFILTER:
                continue
            adv20_und = row['adv20']
            wraps_adv = 0.0
            for w in wraps:
                if w in daily_by_sym and d in daily_by_sym[w].index:
                    fb = wrapper_first_bar.get(w)
                    if fb is not None and d >= fb:  # PIT: wrapper must have traded by then
                        wa = daily_by_sym[w].loc[d, 'adv20']
                        if pd.notna(wa):
                            wraps_adv += float(wa)
            n_checked += 1
            rec = eval_bar_day(und, d, float(pc), float(adv20_und) if pd.notna(adv20_und) else np.nan,
                                wraps_adv, pit, True)
            rec['underlying'] = und
            rec['daily_close'] = float(close)
            signal_recs.append(rec)
        log(f'{und}: {n_checked} candidate days (daily prefilter >= {DAILY_PREFILTER:.0%})')

    sig_df = pd.DataFrame(signal_recs)
    sig_df.to_csv(os.path.join(OUT, 'signal_days_raw.csv'), index=False)
    log(f'signal candidate days total: {len(sig_df)}; has_bars: {sig_df["has_bars"].sum() if len(sig_df) else 0}')

    qual = sig_df[sig_df.get('qualifies', False) == True].copy() if len(sig_df) else sig_df
    log(f'signal days with |r|>=5%: {len(qual)}')

    # MOC exit price from daily_bars close (same underlying/date)
    def moc(row):
        d = row['date']
        sym = row['underlying']
        if sym in daily_by_sym and d in daily_by_sym[sym].index:
            return float(daily_by_sym[sym].loc[d, 'close'])
        return np.nan
    if len(qual):
        qual['moc_close'] = qual.apply(moc, axis=1)
        qual['split'] = qual['date'].apply(
            lambda d: 'TRAIN' if TRAIN_LO <= d <= TRAIN_HI else ('VAL' if VAL_LO <= d <= VAL_HI else 'OTHER'))

    qual.to_csv(os.path.join(OUT, 'signal_days_qualified.csv'), index=False)

    # F threshold from TRAIN median
    train_F = qual.loc[qual['split'] == 'TRAIN', 'F'].dropna() if len(qual) else pd.Series(dtype=float)
    f_thresh = float(train_F.median()) if len(train_F) else np.nan
    log(f'TRAIN median F among |r|>=5% days: {f_thresh}')

    selected = qual[(qual['F'] >= f_thresh) & qual['sho_gate']].copy() if len(qual) and not np.isnan(f_thresh) else qual.iloc[0:0].copy()
    sho_excluded = qual[(qual['r'] <= -0.10) & (~qual['sho_gate'])] if len(qual) else qual.iloc[0:0]
    log(f'selected (F >= median, SHO-clean): {len(selected)}; SHO-excluded shorts: {len(sho_excluded)}')

    def pnl_row(r, use_stop=False):
        side = r['side']
        eff = r['eff_entry']
        if use_stop and r['stopped_2pct']:
            exitp = r['stop_exit']
        else:
            exitp = r['moc_close']
        pnl_frac = (exitp / eff - 1.0) * side
        return pnl_frac

    if len(selected):
        selected['pnl_frac_moc'] = selected.apply(lambda r: pnl_row(r, False), axis=1)
        selected['pnl_frac_stop'] = selected.apply(lambda r: pnl_row(r, True), axis=1)
        selected['pnl_R_moc'] = selected['pnl_frac_moc'] / R_FRAC
        selected['pnl_R_stop'] = selected['pnl_frac_stop'] / R_FRAC

    selected.to_csv(os.path.join(OUT, 'signal_trades.csv'), index=False)

    # -----------------------------------------------------------------
    # Control pool
    # -----------------------------------------------------------------
    log('=== control pool ===')
    class_map_path = os.path.join(ROOT, 'data', 'research', 'orb_asset_class_map_20260711.csv')
    stocks = []
    exclude = set(UNDERLYINGS) | set(ALL_WRAPPERS)
    with open(class_map_path, newline='') as fh:
        for row in csv.DictReader(fh):
            s = row['symbol']
            if row['asset_class'] == 'stock' and s not in exclude and not is_test_ticker(s) \
               and s.isalpha() and len(s) <= 5:
                stocks.append(s)
    rng = random.Random(CONTROL_SEED)
    control_syms = rng.sample(stocks, min(N_CONTROL_SAMPLE, len(stocks)))
    log(f'control sample: {len(control_syms)} symbols (seed={CONTROL_SEED})')

    cdaily = load_daily(control_syms)
    cdaily_by_sym = {s: g.set_index('bar_date') for s, g in cdaily.groupby('symbol')}

    control_recs = []
    fetches = 0
    for sym in control_syms:
        if sym not in cdaily_by_sym:
            continue
        dsym = cdaily_by_sym[sym]
        for d in trading_days:
            if fetches >= MAX_CONTROL_INTRADAY_FETCHES:
                break
            if d not in dsym.index:
                continue
            row = dsym.loc[d]
            pc = row['prior_close']
            if pd.isna(pc) or pc < MIN_PRICE:
                continue
            close = row['close']
            if pd.isna(close) or abs(close / pc - 1.0) < DAILY_PREFILTER:
                continue
            rec = eval_bar_day(sym, d, float(pc), np.nan, None, pit, False)
            fetches += 1
            rec['underlying'] = sym
            rec['daily_close'] = float(close)
            control_recs.append(rec)
        if fetches >= MAX_CONTROL_INTRADAY_FETCHES:
            log(f'WARNING control intraday fetch cap ({MAX_CONTROL_INTRADAY_FETCHES}) reached — pool truncated')
            break

    ctrl_df = pd.DataFrame(control_recs)
    ctrl_df.to_csv(os.path.join(OUT, 'control_days_raw.csv'), index=False)
    cqual = ctrl_df[ctrl_df.get('qualifies', False) == True].copy() if len(ctrl_df) else ctrl_df
    if len(cqual):
        cqual['moc_close'] = cqual.apply(moc if False else (lambda row: (
            float(cdaily_by_sym[row['underlying']].loc[row['date'], 'close'])
            if row['underlying'] in cdaily_by_sym and row['date'] in cdaily_by_sym[row['underlying']].index
            else np.nan)), axis=1)
        cqual['split'] = cqual['date'].apply(
            lambda d: 'TRAIN' if TRAIN_LO <= d <= TRAIN_HI else ('VAL' if VAL_LO <= d <= VAL_HI else 'OTHER'))
        cqual = cqual[cqual['sho_gate']]
        cqual['pnl_frac_moc'] = cqual.apply(lambda r: pnl_row(r, False), axis=1)
        cqual['pnl_R_moc'] = cqual['pnl_frac_moc'] / R_FRAC
    cqual.to_csv(os.path.join(OUT, 'control_trades.csv'), index=False)
    log(f'control qualifying |r|>=5% trades: {len(cqual)}')

    summary = dict(
        n_signal_candidate_days=int(len(sig_df)),
        n_signal_has_bars=int(sig_df['has_bars'].sum()) if len(sig_df) else 0,
        n_signal_qualified=int(len(qual)),
        f_thresh_train_median=f_thresh,
        n_selected=int(len(selected)),
        n_sho_excluded=int(len(sho_excluded)),
        n_control_candidate_days=int(len(ctrl_df)),
        n_control_qualified=int(len(cqual)),
        control_fetches_used=fetches,
    )
    with open(os.path.join(OUT, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    log('SUMMARY', json.dumps(summary, indent=2, default=str))


if __name__ == '__main__':
    main()
