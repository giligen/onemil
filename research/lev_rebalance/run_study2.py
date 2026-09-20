#!/usr/bin/env python3
"""lev_rebalance PASS 2 — full-population rescore. See PREREG.md Pass 2
addendum + FREEZE.md.

ONLY CHANGE FROM run_study.py: the bars source and the universe.
- Universe: every underlying resolved from the full wrapper class map via
  trading.orb_asset_class.underlying_anchor (build_universe.py), not just
  TSLA/MSTR/NVDA.
- Bars source: MERGED — research/lev_rebalance/bars_1500_1600.db (freshly
  pulled from Alpaca SIP for the candidate days cache.db didn't have) UNION
  cache.db intraday_bars_1min (read-only, used where it already had the
  window). Same 14:59/15:01 ET slicing logic, same F formula, same SHO
  gate, same cost model, same MOC exit, same 2% stop diagnostic, same
  splits, same book. Control pool sized up from 60->400 symbols (cheap;
  more power) via the same build_control.py + pull_bars.py path.
"""
import json
import os
import sqlite3
import sys
from datetime import date, datetime

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

ROOT = '/home/ec2-user/onemil'
sys.path.insert(0, ROOT)
os.chdir(ROOT)

OUT = os.path.join(ROOT, 'research', 'lev_rebalance')
CACHE_DB = f"file:{ROOT}/data/cache.db?mode=ro"
BARS_DB = os.path.join(OUT, 'bars_1500_1600.db')
ET = ZoneInfo('America/New_York')
UTC = ZoneInfo('UTC')

TRAIN_LO, TRAIN_HI = date(2025, 1, 1), date(2025, 12, 31)
VAL_LO, VAL_HI = date(2026, 1, 1), date(2026, 5, 31)
LOOKBACK_LO = date(2024, 11, 1)
RUN_HI = VAL_HI

DAILY_PREFILTER_HL = 0.05  # pull-filter equals the signal threshold on high/low (superset of close-based r)
SIG_THRESH = 0.05
MIN_PRICE = 5.0
HALF_SPREAD_PCT_1500 = 0.23312826680234397 / 2.0 / 100.0
STOP_FRAC = 0.02
R_FRAC = 0.02

cache_conn = sqlite3.connect(CACHE_DB, uri=True, timeout=30)
bars_conn = sqlite3.connect(BARS_DB)


def log(*a):
    print(*a, flush=True)


def load_daily(symbols):
    if not symbols:
        return pd.DataFrame(columns=['symbol', 'bar_date', 'open', 'high', 'low', 'close', 'volume',
                                      'prior_close', 'dollar_vol', 'adv20'])
    parts = []
    CH = 400
    symbols = sorted(set(symbols))
    for i in range(0, len(symbols), CH):
        chunk = symbols[i:i + CH]
        q = (f"SELECT symbol, bar_date, open, high, low, close, volume FROM daily_bars "
             f"WHERE symbol IN ({','.join('?' * len(chunk))}) AND bar_date >= ? AND bar_date <= ? "
             f"ORDER BY symbol, bar_date")
        df = pd.read_sql_query(q, cache_conn, params=[*chunk, str(LOOKBACK_LO), str(RUN_HI)])
        parts.append(df)
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if len(df) == 0:
        return df
    df['bar_date'] = pd.to_datetime(df['bar_date']).dt.date
    df = df.sort_values(['symbol', 'bar_date'])
    df['prior_close'] = df.groupby('symbol')['close'].shift(1)
    df['dollar_vol'] = df['close'] * df['volume']
    df['adv20'] = (df.groupby('symbol')['dollar_vol']
                   .transform(lambda s: s.shift(1).rolling(20, min_periods=10).mean()))
    return df


def fetch_window_merged(symbol, d):
    """Bars 14:50-16:05 ET for (symbol, d): prefer our pulled db, fall back
    to cache.db (read-only). Never mixes rows from both for the same
    (symbol, date) — one source per symbol-day, whichever has data."""
    cur = bars_conn.execute(
        "SELECT timestamp, open, high, low, close, volume FROM bars_1500_1600 "
        "WHERE symbol=? AND bar_date=? ORDER BY timestamp", (symbol, str(d)))
    rows = cur.fetchall()
    if rows:
        df = pd.DataFrame(rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['ts_et'] = pd.to_datetime(df['timestamp']).dt.tz_convert(ET)
        return df
    lo = datetime(d.year, d.month, d.day, 14, 50, tzinfo=ET).astimezone(UTC)
    hi = datetime(d.year, d.month, d.day, 16, 5, tzinfo=ET).astimezone(UTC)
    cur = cache_conn.execute(
        "SELECT timestamp, open, high, low, close, volume FROM intraday_bars_1min "
        "WHERE symbol=? AND bar_date=? AND timestamp >= ? AND timestamp <= ? ORDER BY timestamp",
        (symbol, str(d), lo.isoformat(), hi.isoformat()))
    rows = cur.fetchall()
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['ts_et'] = pd.to_datetime(df['timestamp']).dt.tz_convert(ET)
    return df


def eval_bar_day(symbol, d, prior_close, adv20_und, wrappers_adv):
    df = fetch_window_merged(symbol, d)
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
    rec['sho_gate'] = True
    if r <= -0.10:
        rec['sho_gate'] = bool(entry_open > entry_low)
    if wrappers_adv is not None:
        F = 2.0 * abs(r) * wrappers_adv / adv20_und if (adv20_und and adv20_und > 0) else np.nan
        rec['F'] = F
    else:
        rec['F'] = np.nan
    half = HALF_SPREAD_PCT_1500
    eff_entry = entry_open * (1 + half) if side > 0 else entry_open * (1 - half)
    rec['eff_entry'] = eff_entry
    after_entry = df[df['ts_et'] > entry_bar['ts_et']]
    stop_level = eff_entry * (1 - STOP_FRAC) if side > 0 else eff_entry * (1 + STOP_FRAC)
    stopped, stop_exit = False, None
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


def pnl_row(r, use_stop=False):
    side = r['side']
    eff = r['eff_entry']
    exitp = r['stop_exit'] if (use_stop and r['stopped_2pct']) else r['moc_close']
    return (exitp / eff - 1.0) * side


def main():
    universe_map = json.load(open(os.path.join(OUT, 'universe_map.json')))
    cand = pd.read_csv(os.path.join(OUT, 'candidate_underlying_days.csv'))
    cand['bar_date'] = pd.to_datetime(cand['bar_date']).dt.date
    underlyings = sorted(cand['symbol'].unique().tolist())
    log(f'underlyings with >=1 candidate day: {len(underlyings)}')

    wrappers_needed = sorted({w for u in underlyings for w in universe_map.get(u, [])})
    log(f'wrappers needed for F (union of these underlyings\' complexes): {len(wrappers_needed)}')

    daily_und = load_daily(underlyings)
    daily_wrap = load_daily(wrappers_needed)
    daily_by_sym = {s: g.set_index('bar_date') for s, g in pd.concat([daily_und, daily_wrap]).groupby('symbol')}

    wrapper_first_bar = {s: (daily_by_sym[s].index.min() if s in daily_by_sym and len(daily_by_sym[s]) else None)
                         for s in wrappers_needed}

    trading_days = sorted(cand['bar_date'].unique().tolist())
    signal_recs = []
    for und in underlyings:
        if und not in daily_by_sym:
            continue
        dund = daily_by_sym[und]
        wraps = universe_map.get(und, [])
        udays = cand.loc[cand['symbol'] == und, 'bar_date'].tolist()
        for d in udays:
            if d not in dund.index:
                continue
            row = dund.loc[d]
            pc = row['prior_close']
            if pd.isna(pc) or pc < MIN_PRICE:
                continue
            adv20_und = row['adv20']
            wraps_adv = 0.0
            for w in wraps:
                if w in daily_by_sym and d in daily_by_sym[w].index:
                    fb = wrapper_first_bar.get(w)
                    if fb is not None and d >= fb:
                        wa = daily_by_sym[w].loc[d, 'adv20']
                        if pd.notna(wa):
                            wraps_adv += float(wa)
            rec = eval_bar_day(und, d, float(pc), float(adv20_und) if pd.notna(adv20_und) else np.nan, wraps_adv)
            rec['underlying'] = und
            signal_recs.append(rec)

    sig_df = pd.DataFrame(signal_recs)
    sig_df.to_csv(os.path.join(OUT, 'p2_signal_days_raw.csv'), index=False)
    n_has_bars = int(sig_df['has_bars'].sum()) if len(sig_df) else 0
    log(f'signal candidate underlying-days scored: {len(sig_df)}; has_bars (both sides of 15:00): {n_has_bars} '
        f'({n_has_bars/max(len(sig_df),1):.1%})')

    qual = sig_df[sig_df.get('qualifies', False) == True].copy() if len(sig_df) else sig_df
    log(f'signal days with |r|>=5% AND usable bars: {len(qual)}')

    def moc(row):
        d, sym = row['date'], row['underlying']
        if sym in daily_by_sym and d in daily_by_sym[sym].index:
            return float(daily_by_sym[sym].loc[d, 'close'])
        return np.nan
    if len(qual):
        qual['moc_close'] = qual.apply(moc, axis=1)
        qual['split'] = qual['date'].apply(
            lambda d: 'TRAIN' if TRAIN_LO <= d <= TRAIN_HI else ('VAL' if VAL_LO <= d <= VAL_HI else 'OTHER'))
    qual.to_csv(os.path.join(OUT, 'p2_signal_days_qualified.csv'), index=False)

    train_F = qual.loc[qual['split'] == 'TRAIN', 'F'].dropna() if len(qual) else pd.Series(dtype=float)
    f_thresh = float(train_F.median()) if len(train_F) else np.nan
    log(f'TRAIN median F among |r|>=5% days: {f_thresh} (n={len(train_F)})')

    selected = qual[(qual['F'] >= f_thresh) & qual['sho_gate']].copy() if len(qual) and not np.isnan(f_thresh) else qual.iloc[0:0].copy()
    sho_excluded = qual[(qual['r'] <= -0.10) & (~qual['sho_gate'])] if len(qual) else qual.iloc[0:0]
    log(f'selected (F >= TRAIN median, SHO-clean): {len(selected)}; SHO-excluded shorts (pool): {len(sho_excluded)}')

    if len(selected):
        selected['pnl_frac_moc'] = selected.apply(lambda r: pnl_row(r, False), axis=1)
        selected['pnl_frac_stop'] = selected.apply(lambda r: pnl_row(r, True), axis=1)
        selected['pnl_R_moc'] = selected['pnl_frac_moc'] / R_FRAC
        selected['pnl_R_stop'] = selected['pnl_frac_stop'] / R_FRAC
    selected.to_csv(os.path.join(OUT, 'p2_signal_trades.csv'), index=False)

    # ---------------- control ----------------
    log('=== control ===')
    ccand = pd.read_csv(os.path.join(OUT, 'control_candidate_days.csv'))
    ccand['bar_date'] = pd.to_datetime(ccand['bar_date']).dt.date
    control_syms = sorted(ccand['symbol'].unique().tolist())
    daily_ctrl = load_daily(control_syms)
    cdaily_by_sym = {s: g.set_index('bar_date') for s, g in daily_ctrl.groupby('symbol')}

    control_recs = []
    for sym in control_syms:
        if sym not in cdaily_by_sym:
            continue
        dsym = cdaily_by_sym[sym]
        sdays = ccand.loc[ccand['symbol'] == sym, 'bar_date'].tolist()
        for d in sdays:
            if d not in dsym.index:
                continue
            row = dsym.loc[d]
            pc = row['prior_close']
            if pd.isna(pc) or pc < MIN_PRICE:
                continue
            rec = eval_bar_day(sym, d, float(pc), np.nan, None)
            rec['underlying'] = sym
            control_recs.append(rec)

    ctrl_df = pd.DataFrame(control_recs)
    ctrl_df.to_csv(os.path.join(OUT, 'p2_control_days_raw.csv'), index=False)
    n_ctrl_has_bars = int(ctrl_df['has_bars'].sum()) if len(ctrl_df) else 0
    log(f'control candidate days scored: {len(ctrl_df)}; has_bars: {n_ctrl_has_bars} '
        f'({n_ctrl_has_bars/max(len(ctrl_df),1):.1%})')

    cqual = ctrl_df[ctrl_df.get('qualifies', False) == True].copy() if len(ctrl_df) else ctrl_df

    def cmoc(row):
        sym, d = row['underlying'], row['date']
        if sym in cdaily_by_sym and d in cdaily_by_sym[sym].index:
            return float(cdaily_by_sym[sym].loc[d, 'close'])
        return np.nan
    if len(cqual):
        cqual['moc_close'] = cqual.apply(cmoc, axis=1)
        cqual['split'] = cqual['date'].apply(
            lambda d: 'TRAIN' if TRAIN_LO <= d <= TRAIN_HI else ('VAL' if VAL_LO <= d <= VAL_HI else 'OTHER'))
        cqual = cqual[cqual['sho_gate']]
        cqual['pnl_frac_moc'] = cqual.apply(lambda r: pnl_row(r, False), axis=1)
        cqual['pnl_R_moc'] = cqual['pnl_frac_moc'] / R_FRAC
    cqual.to_csv(os.path.join(OUT, 'p2_control_trades.csv'), index=False)
    log(f'control qualifying |r|>=5% SHO-clean trades: {len(cqual)}')

    coverage_signal = n_has_bars / max(len(sig_df), 1)
    coverage_control = n_ctrl_has_bars / max(len(ctrl_df), 1)
    summary = dict(
        n_signal_candidate_days=int(len(sig_df)), n_signal_has_bars=n_has_bars,
        coverage_signal=coverage_signal,
        n_signal_qualified=int(len(qual)), f_thresh_train_median=f_thresh,
        n_selected=int(len(selected)), n_sho_excluded=int(len(sho_excluded)),
        n_control_candidate_days=int(len(ctrl_df)), n_control_has_bars=n_ctrl_has_bars,
        coverage_control=coverage_control,
        n_control_qualified=int(len(cqual)),
    )
    with open(os.path.join(OUT, 'p2_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    log('SUMMARY', json.dumps(summary, indent=2, default=str))


if __name__ == '__main__':
    main()
