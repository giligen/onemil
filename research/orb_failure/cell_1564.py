#!/usr/bin/env python3
"""Cells 1,564-1,566 -- research/orb_failure/PREREG_1564.md (FROZEN 2026-09-27 07:40 UTC).

The FAILED opening-range breakout, declared at 10:30 ET, as its own signal on every ORB
candidate symbol-day (not only HOD fills): a gap-up name that breaks its opening range and is
then stopped back through the range low has trapped the morning's buyers; the deduction from
`research/hod_entry/review/orb_x_hod_diagnostic.md` is that its LATER breaks fail (HOD-break
fills after an ORB loser on the same symbol-day earn -0.90/-0.54 R). This cell tests the signal
on its own population, with a declaration hour that makes it causal: only bars through 10:30:00
ET decide BREAK / FAILURE / SUCCESS, and the trade enters at the 10:30 bar open.

Population: analysis_results/orb_features_20260925_2054.csv (13,316 candidate symbol-days,
2025-01-02..2026-09-25), read through trading/orb_csv.read_orb_csv. Minute bars: data/cache.db,
table intraday_bars_1min, READ ONLY (opened via the sqlite URI file:...?mode=ro -- this module
never writes to it). Opening range = bars in [09:30, 09:35) ET.

Events (bars through 10:30:00 ET only):
  * BREAK    -- a bar high >= range_high + $0.01 in [09:35, 10:30).
  * FAILURE  -- a BREAK followed by a bar low <= range_low - $0.01, before 10:30 (cell 1,564).
  * SUCCESS  -- a BREAK with no low <= range_low - $0.01 before 10:30 AND close(10:29) >=
                range_high (cell 1,566, the mirror).
Report-only declaration hours 10:00 and 11:00 use the identical rule with the window end moved;
they are counted, never traded, and never selected among (PREREG "Not allowed").

Trades, entered at the 10:30 bar OPEN (the first obtainable price after the declaration):
  * 1,564 FAILED-BREAK SHORT: entry = 10:30 open - half_spread; stop = max(high, bars with
    m < 630) + $0.01; target = entry - 2*R (R = stop - entry); cover 15:55 ET at the ask.
  * 1,565 FAILED-BREAK SHORT, VWAP TARGET: as 1,564, target = causal session VWAP at 10:30
    (bars from open through 10:29); report-only if the VWAP is within 0.5% of the entry (too
    close to be a usable target).
  * 1,566 HELD-BREAK LONG (mirror): entry = 10:30 open + half_spread; stop = range_low - $0.01;
    target = entry + 2*R (R = entry - stop); exit 15:55 ET at the bid.
Exclusions (short cells only): shortable == False (or missing from borrow_flags.csv, logged as a
WARNING and excluded conservatively) and SSR (a bar low through 10:30 <= 0.90 * the prior
trading day's daily-bar close, from data/cache.db's `daily_bars` table). All cells: price < $5
excluded.

Cost model (documented conventions; every one is a deviation the RESULT states in full):
  * Entry half-spread: the real Alpaca NBBO at 10:30:00 ET, fetched per event (resumable cache
    `quotes_1564/`, `fetch_quotes_1564.py`). Falls back, in order, to (a) the measured per-
    (day,symbol) spread in `research/bf_zero/causal_filter/nbbo.csv` (real data, not minute-
    specific -- the "minute-of-day half-spread table" cited in the task's cell_1445.py pointer
    was not found there on inspection; this is the closest measured substitute on record) and
    then (b) a flat 25 bps half-spread, logged as a WARNING. The share of events on each source
    is reported (the pass-bar independent check requires this).
  * Stop exit: cell 1,478's SLIP_STOP_BPS (TRAIN 2025 / VAL 2026), the standing stop-limit
    standard for this codebase -- charged on the stop price, no additional half-spread (the
    constant is already the all-in measured cost of that mechanism).
  * Target (limit) exit: one half-spread (same source/fallback order as the entry) on the
    target price -- a limit order pays the spread, not slippage.
  * EOD (15:55) exit: cell 1,443/1,478's measured EOD-holdout means, EOD_BPS = {TRAIN: 11.5,
    VAL: 9.7} bps, on the exit price (the codebase-standard constant, not re-derived here).
  * Borrow (shorts only): 3%/yr on the entry notional, pro-rated by minutes held / (60*24*365).
  * All costs are converted to R units by dividing the dollar cost by R (the dollar risk/share).

Usage:
    python3 research/orb_failure/fetch_quotes_1564.py [--limit-days N]   # fill the quote cache
    python3 research/orb_failure/cell_1564.py [--limit-days N]          # classify + score

Outputs: research/orb_failure/cell_1564_events.csv (one row per candidate per cell) and
research/orb_failure/RESULT_1564.md.
"""
import argparse
import json
import logging
import os
import sqlite3
import sys
import time
from bisect import bisect_right

import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

from trading.orb_csv import read_orb_csv  # noqa: E402

CACHE_DB = os.path.join(REPO, 'data/cache.db')
CANDIDATES_CSV = os.path.join(REPO, 'analysis_results/orb_features_20260925_2054.csv')
BORROW_CSV = os.path.join(REPO, 'research/fuckup_audit/O_halt/PASSIVE/borrow_flags.csv')
NBBO_CSV = os.path.join(REPO, 'research/bf_zero/causal_filter/nbbo.csv')
MODEL_1478_CSV = os.path.join(REPO, 'research/hod_entry/model_1478_L3_predictions.csv')
QUOTE_CACHE_DIR = os.path.join(HERE, 'quotes_1564')
EVENTS_CSV = os.path.join(HERE, 'cell_1564_events.csv')
RESULT_MD = os.path.join(HERE, 'RESULT_1564.md')

# -- cost constants (see module docstring for provenance) --------------------------------------
SLIP_STOP_BPS = {'TRAIN': 0.88 * 2.9 + 0.12 * 94.0, 'VAL': 0.88 * 3.2 + 0.12 * 76.0}  # cell_1478
EOD_BPS = {'TRAIN': 11.5, 'VAL': 9.7}                                                  # cell_1445/1478
FLAT_HALF_SPREAD_BPS = 25.0     # tertiary fallback only; every use is logged as a WARNING
BORROW_ANNUAL = 0.03
MIN_PRICE = 5.0
RANGE_BUFFER = 0.01             # $0.01 break/stop buffer, per PREREG
TARGET_R = 2.0
VWAP_MIN_DIST_PCT = 0.5         # 1,565 report-only threshold

RANGE_START_M = 9 * 60 + 30     # 570
RANGE_END_M = 9 * 60 + 35       # 575 (exclusive)
DECL_1030_M = 10 * 60 + 30      # 630
DECL_1000_M = 10 * 60 + 0       # 600
DECL_1100_M = 11 * 60 + 0       # 660
EOD_M = 15 * 60 + 55            # 955

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger('cell_1564')


# ================================================================================================
# Loaders
# ================================================================================================

def load_candidates(limit_days=0):
    """Candidates from the cumulative ORB feature file, split-tagged. Logs the price-scale
    glitch (gap_pct outliers) the refuter must see, without excluding on it."""
    df = read_orb_csv(CANDIDATES_CSV)
    df = df.drop_duplicates(subset=['symbol', 'date']).reset_index(drop=True)
    df['date'] = df['date'].astype(str)
    n_glitch = int((df['gap_pct'].abs() > 200).sum())
    if n_glitch:
        log.warning('PRICE-SCALE: %d/%d candidates (%.2f%%) have |gap_pct| > 200%% -- data '
                     'glitches disclosed for the refuter, not excluded here', n_glitch, len(df),
                     100.0 * n_glitch / len(df))
    df['year'] = df['date'].str[:4].astype(int)
    df['split'] = np.where(df['year'] == 2025, 'TRAIN', np.where(df['year'] == 2026, 'VAL', None))
    df = df[df['split'].notna()].reset_index(drop=True)
    days = sorted(df['date'].unique())
    if limit_days:
        keep = set(days[:limit_days])
        df = df[df['date'].isin(keep)].reset_index(drop=True)
        days = sorted(keep)
    log.info('candidates: %d rows, %d distinct days, %.1f/day (TRAIN %d, VAL %d)',
              len(df), len(days), len(df) / max(len(days), 1),
              int((df.split == 'TRAIN').sum()), int((df.split == 'VAL').sum()))
    return df


def load_borrow_flags():
    """{symbol -> shortable bool}. Missing symbol => not shortable (conservative), WARNING at use."""
    b = read_orb_csv(BORROW_CSV)
    return {r.symbol: bool(r.shortable) for r in b.itertuples()}


def load_prior_close_index(symbols):
    """{symbol -> (sorted bar_dates list, aligned closes list)} from cache.db daily_bars, RO."""
    con = sqlite3.connect(f'file:{CACHE_DB}?mode=ro', uri=True)
    out = {}
    try:
        syms = sorted(symbols)
        for i in range(0, len(syms), 500):
            chunk = syms[i:i + 500]
            qmarks = ','.join('?' * len(chunk))
            cur = con.execute(
                f"SELECT symbol, bar_date, close FROM daily_bars WHERE symbol IN ({qmarks}) "
                f"ORDER BY symbol, bar_date", chunk)
            for sym, bd, close in cur:
                out.setdefault(sym, ([], []))
                out[sym][0].append(str(bd))
                out[sym][1].append(float(close))
    finally:
        con.close()
    return out


def prior_close(idx, symbol, date):
    """The daily-bar close strictly before `date`, or None."""
    rec = idx.get(symbol)
    if not rec:
        return None
    dates, closes = rec
    i = bisect_right(dates, date) - 1
    if i >= 0 and dates[i] < date:
        return closes[i]
    if i >= 1:
        return closes[i - 1]
    return None


def load_quote_cache():
    """{(day, symbol) -> {'bid':.., 'ask':..}} from fetch_quotes_1564.py's per-day JSON cache."""
    out = {}
    if not os.path.isdir(QUOTE_CACHE_DIR):
        return out
    for fn in os.listdir(QUOTE_CACHE_DIR):
        if not fn.endswith('.json'):
            continue
        day = fn[:-5]
        with open(os.path.join(QUOTE_CACHE_DIR, fn)) as f:
            day_data = json.load(f)
        for sym, q in day_data.items():
            out[(day, sym)] = q
    return out


def load_nbbo_fallback():
    """{(day, symbol) -> half_spread $} from the measured (not minute-of-day) causal_filter/nbbo.csv,
    the closest real substitute found on inspection of cell_1445.py's pointer (see docstring)."""
    if not os.path.exists(NBBO_CSV):
        log.warning('NBBO fallback file missing: %s -- flat %.0f bps only', NBBO_CSV,
                     FLAT_HALF_SPREAD_BPS)
        return {}
    n = read_orb_csv(NBBO_CSV)
    n = n.dropna(subset=['spread_mean'])
    return {(r.day, r.symbol): float(r.spread_mean) / 2.0 for r in n.itertuples()}


def load_bars_for_day(con, symbols, date):
    """{symbol -> DataFrame(m, o,h,l,c,v)} for one day, m = ET minute-of-day (int)."""
    qmarks = ','.join('?' * len(symbols))
    cur = con.execute(
        f"SELECT symbol, timestamp, open, high, low, close, volume FROM intraday_bars_1min "
        f"WHERE bar_date = ? AND symbol IN ({qmarks}) ORDER BY symbol, timestamp",
        [date] + list(symbols))
    rows = cur.fetchall()
    if not rows:
        return {}
    df = pd.DataFrame(rows, columns=['symbol', 'timestamp', 'o', 'h', 'l', 'c', 'v'])
    ts = pd.to_datetime(df['timestamp'], utc=True).dt.tz_convert('America/New_York')
    df['m'] = ts.dt.hour * 60 + ts.dt.minute
    out = {}
    for sym, g in df.groupby('symbol'):
        out[sym] = g.sort_values('m').reset_index(drop=True)
    return out


# ================================================================================================
# Event classification
# ================================================================================================

def classify_one(bars, decl_m):
    """One candidate's bars (DataFrame m,o,h,l,c) -> dict with range_high/low, event, and the
    fields the trade needs (break_m, high_through_decl, close_before_decl, vwap_through_decl,
    entry_bar). `decl_m` is the declaration minute (600/630/660). Returns None if the opening
    range itself is incomplete (no bars in [09:30,09:35))."""
    rng = bars[(bars.m >= RANGE_START_M) & (bars.m < RANGE_END_M)]
    if not len(rng):
        return None
    range_high = float(rng.h.max())
    range_low = float(rng.l.min())
    n_range_bars = len(rng)
    pre_decl = bars[(bars.m >= RANGE_END_M) & (bars.m < decl_m)]
    break_row = pre_decl[pre_decl.h >= range_high + RANGE_BUFFER]
    event = 'NO_BREAK'
    break_m = None
    if len(break_row):
        break_m = int(break_row.iloc[0].m)
        after_break = pre_decl[pre_decl.m >= break_m]
        stopped = after_break[after_break.l <= range_low - RANGE_BUFFER]
        if len(stopped):
            event = 'FAILURE'
        else:
            last_bar = pre_decl.iloc[-1] if len(pre_decl) else None
            held = last_bar is not None and last_bar.c >= range_high
            event = 'SUCCESS' if held else 'INDETERMINATE'
    high_through_decl = float(pre_decl.h.max()) if len(pre_decl) else range_high
    vwap = None
    if len(pre_decl) and pre_decl.v.sum() > 0:
        vwap = float((pre_decl.c * pre_decl.v).sum() / pre_decl.v.sum())
    entry_bar = bars[bars.m == decl_m]
    entry_open = float(entry_bar.iloc[0].o) if len(entry_bar) else None
    return dict(range_high=range_high, range_low=range_low, n_range_bars=n_range_bars,
                event=event, break_m=break_m, high_through_decl=high_through_decl,
                vwap_through_decl=vwap, entry_open=entry_open)


# ================================================================================================
# Cost / walk
# ================================================================================================

def half_spread_for(day, symbol, quote_cache, nbbo_fallback, price, coverage):
    """(half_spread_$, source) with the 3-tier fallback in the docstring. `coverage` is a Counter-
    like dict this mutates in place (real/nbbo_fallback/flat_fallback)."""
    q = quote_cache.get((day, symbol))
    if q is not None and q.get('bid') and q.get('ask') and q['ask'] > q['bid'] > 0:
        coverage['real'] = coverage.get('real', 0) + 1
        return (q['ask'] - q['bid']) / 2.0, 'real'
    hs = nbbo_fallback.get((day, symbol))
    if hs is not None and hs > 0:
        coverage['nbbo_fallback'] = coverage.get('nbbo_fallback', 0) + 1
        return hs, 'nbbo_fallback'
    coverage['flat_fallback'] = coverage.get('flat_fallback', 0) + 1
    return price * FLAT_HALF_SPREAD_BPS / 1e4, 'flat_fallback'


def walk_short(entry_m, entry_fill, stop, target, path):
    """Short-side walk: stop first on a bar touching both, gap-through at the open, EOD at the
    15:55 bar's open. `path` = bars with m >= entry_m, sorted. Mirrors sip_rebuild.walk_path."""
    for row in path.itertuples():
        if row.m >= EOD_M:
            return int(row.m), float(row.o), 'eod'
        if row.h >= stop:
            return int(row.m), float(row.o if row.o >= stop else stop), 'stop'
        if row.l <= target:
            return int(row.m), float(target), 'target'
    last = path.iloc[-1]
    return int(last.m), float(last.c), 'eod_fallback'


def walk_long(entry_m, entry_fill, stop, target, path):
    """Long-side walk (1,566 mirror): stop first, gap-through at open, EOD at 15:55 open."""
    for row in path.itertuples():
        if row.m >= EOD_M:
            return int(row.m), float(row.o), 'eod'
        if row.l <= stop:
            return int(row.m), float(row.o if row.o <= stop else stop), 'stop'
        if row.h >= target:
            return int(row.m), float(target), 'target'
    last = path.iloc[-1]
    return int(last.m), float(last.c), 'eod_fallback'


def cost_R(why, entry_fill, exit_price, entry_half, exit_half, split, R, is_short, minutes_held,
           borrow_applicable):
    """Dollar cost -> R units. Stop uses SLIP_STOP_BPS on the stop/exit price; target/eod use a
    half-spread (or the EOD_BPS constant); shorts add pro-rated 3%/yr borrow."""
    if why in ('stop', 'target'):
        exit_cost = (SLIP_STOP_BPS[split] / 1e4 * exit_price) if why == 'stop' else exit_half
    else:  # eod / eod_fallback
        exit_cost = EOD_BPS[split] / 1e4 * exit_price
    dollar_cost = entry_half + exit_cost
    if borrow_applicable:
        dollar_cost += entry_fill * BORROW_ANNUAL * (minutes_held / (60.0 * 24.0 * 365.0))
    return dollar_cost / R


# ================================================================================================
# Stats
# ================================================================================================

def day_clustered_t(y, day):
    y = pd.Series(y).dropna()
    if len(y) < 2:
        return np.nan
    d = pd.Series(day).loc[y.index]
    if d.nunique() < 2:
        return np.nan
    X = np.ones((len(y), 1))
    m = sm.OLS(y.to_numpy(), X).fit(cov_type='cluster', cov_kwds={'groups': d.to_numpy()})
    return float(m.tvalues[0])


def ex_topk_mean(y, frac):
    y = pd.Series(y).dropna().sort_values(ascending=False)
    n = len(y)
    if n == 0:
        return np.nan
    k = int(round(frac * n))
    return float(y.iloc[k:].mean()) if k < n else float(y.mean())


def winner_capped_mean(y, cap=3.0):
    y = pd.Series(y).dropna()
    return float(np.minimum(y, cap).mean()) if len(y) else np.nan


# ================================================================================================
# Trade construction (shared by 1,564/1,565/1,566, the universe placebo and the null control)
# ================================================================================================

def build_trade(day, symbol, split, price0, cl, mode, quote_cache, nbbo_fallback, borrow_ok,
                 ssr, bars_after, coverage, target_mode='fixed2R'):
    """One trade record for a classified candidate `cl` (classify_one's dict) at the 10:30
    declaration. `mode` = 'short' (1,564/1,565) or 'long' (1,566). Excludes on price/shortable/
    SSR for shorts; returns a dict with entered=False and `why` set to the exclusion reason when
    not tradeable, else the full walked/costed trade."""
    if cl is None or cl['entry_open'] is None:
        return dict(entered=False, why='no_1030_bar')
    price = cl['entry_open']
    if price < MIN_PRICE:
        return dict(entered=False, why='price_lt_5')
    if mode == 'short':
        if not borrow_ok:
            return dict(entered=False, why='not_shortable')
        if ssr:
            return dict(entered=False, why='ssr')
    half, half_src = half_spread_for(day, symbol, quote_cache, nbbo_fallback, price, coverage)
    if mode == 'short':
        entry_fill = price - half
        stop = cl['high_through_decl'] + RANGE_BUFFER
        R = stop - entry_fill
        if R <= 0:
            return dict(entered=False, why='non_positive_R')
        if target_mode == 'vwap':
            vwap = cl['vwap_through_decl']
            if vwap is None:
                return dict(entered=False, why='no_vwap')
            target = vwap
            dist_pct = 100.0 * (entry_fill - target) / entry_fill
            report_only = dist_pct < VWAP_MIN_DIST_PCT
        else:
            target = entry_fill - TARGET_R * R
            report_only = False
        walker = walk_short
    else:
        entry_fill = price + half
        stop = cl['range_low'] - RANGE_BUFFER
        R = entry_fill - stop
        if R <= 0:
            return dict(entered=False, why='non_positive_R')
        target = entry_fill + TARGET_R * R
        report_only = False
        walker = walk_long
    if bars_after is None or not len(bars_after):
        return dict(entered=False, why='no_path')
    exit_m, exit_price, why = walker(DECL_1030_M, entry_fill, stop, target, bars_after)
    minutes_held = exit_m - DECL_1030_M
    raw_R = (entry_fill - exit_price) / R if mode == 'short' else (exit_price - entry_fill) / R
    exit_half = half  # same-morning proxy, documented in the module docstring
    c_R = cost_R(why, entry_fill, exit_price, half, exit_half, split, R, mode == 'short',
                 minutes_held, borrow_applicable=(mode == 'short'))
    net_R = raw_R - c_R
    r_pct_price = 100.0 * R / price
    return dict(entered=True, entry=entry_fill, exit_price=exit_price, why=why, net_R=net_R,
                raw_R=raw_R, cost_R=c_R, r_pct_price=r_pct_price, half_src=half_src,
                report_only=report_only, R_dollars=R)


# ================================================================================================
# Full pipeline
# ================================================================================================

def run(limit_days=0):
    cands = load_candidates(limit_days)
    days = sorted(cands['date'].unique())
    symbols = set(cands['symbol'])
    borrow = load_borrow_flags()
    prior_idx = load_prior_close_index(symbols)
    nbbo_fallback = load_nbbo_fallback()
    quote_cache = load_quote_cache()

    con = sqlite3.connect(f'file:{CACHE_DB}?mode=ro', uri=True)
    events = []
    n_no_bars, n_no_range = 0, 0
    coverage = {}
    t0 = time.time()
    for i, day in enumerate(days):
        day_rows = cands[cands.date == day]
        bars_by_sym = load_bars_for_day(con, sorted(day_rows.symbol.unique()), day)
        for r in day_rows.itertuples():
            bars = bars_by_sym.get(r.symbol)
            if bars is None or not len(bars):
                n_no_bars += 1
                continue
            cl30 = classify_one(bars, DECL_1030_M)
            if cl30 is None:
                n_no_range += 1
                continue
            cl00 = classify_one(bars, DECL_1000_M)
            cl11 = classify_one(bars, DECL_1100_M)
            ssr_close = prior_close(prior_idx, r.symbol, day)
            through_decl = bars[bars.m < DECL_1030_M]
            ssr = (ssr_close is not None and len(through_decl)
                   and through_decl.l.min() <= 0.90 * ssr_close)
            bars_after = bars[bars.m >= DECL_1030_M].reset_index(drop=True)
            base = dict(symbol=r.symbol, date=day, split=r.split,
                        event=cl30['event'], event_1000=cl00['event'] if cl00 else 'no_range',
                        event_1100=cl11['event'] if cl11 else 'no_range',
                        n_range_bars=cl30['n_range_bars'], ssr=ssr,
                        borrow_ok=borrow.get(r.symbol, False))
            for cell, mode, tmode in (('1564', 'short', 'fixed2R'), ('1565', 'short', 'vwap'),
                                       ('1566', 'long', 'fixed2R')):
                tr = build_trade(day, r.symbol, r.split, r.entry_price, cl30, mode, quote_cache,
                                  nbbo_fallback, base['borrow_ok'], ssr, bars_after, coverage,
                                  target_mode=tmode)
                row = {**base, 'cell': cell, **tr}
                events.append(row)
        if (i + 1) % 50 == 0 or i + 1 == len(days):
            log.info('progress: %d/%d days, %d event-rows so far (%.0fs elapsed)',
                      i + 1, len(days), len(events), time.time() - t0)
    con.close()
    log.info('bars coverage: %d candidates missing bars entirely, %d missing a complete opening '
              'range (excluded)', n_no_bars, n_no_range)
    log.info('half-spread coverage: %s', coverage)
    ev = pd.DataFrame(events)
    n_eod_fallback = int((ev.get('why') == 'eod_fallback').sum()) if 'why' in ev else 0
    if n_eod_fallback:
        log.warning('%d entered trades ran out of bars before the 15:55 EOD bar and exited at '
                     'the last available close (bar-store data-sparsity fallback, "eod_fallback" '
                     'in the exit mix)', n_eod_fallback)
    ev.to_csv(EVENTS_CSV, index=False)
    log.info('wrote %s (%d rows)', EVENTS_CSV, len(ev))
    return ev, coverage, n_no_bars, n_no_range


# ================================================================================================
# Reporting
# ================================================================================================

def cell_stats(sub, day_col='date'):
    y = sub['net_R']
    n = len(sub)
    if n == 0:
        return dict(n=0)
    weeks = max(sub[day_col].nunique() / 5.0, 1e-9)
    return dict(n=n, events_wk=n / weeks, mean_net_R=float(y.mean()),
                t=day_clustered_t(y, sub[day_col]), ex_top5=ex_topk_mean(y, 0.05),
                ex_top1=ex_topk_mean(y, 0.01), winner_capped=winner_capped_mean(y),
                median_r_pct_price=float(sub['r_pct_price'].median()),
                exit_mix=sub['why'].value_counts(normalize=True).round(3).to_dict())


def null_control(ev, cell, split, n_draws=1000, seed=1564):
    """Count-matched null: for each signal day in `split`, draw as many NON-signal candidates of
    that SAME day (any event other than the cell's own signal event, same trade already built for
    `cell`) as there were signal trades that day; repeat n_draws times; return (null_mean_R_dist,
    actual_mean). `cell`'s signal event is FAILURE for 1,564/1,565 and SUCCESS for 1,566 -- using
    the wrong event here would silently compare 1,566 against its own population under the WRONG
    label (a real bug caught on first read of this file's own output, see RESULT's caveats)."""
    signal_event = 'SUCCESS' if cell == '1566' else 'FAILURE'
    fail = ev[(ev.cell == cell) & (ev.split == split) & (ev.event == signal_event) & ev.entered]
    if not len(fail):
        return None, None
    actual_mean = float(fail['net_R'].mean())
    per_day_n = fail.groupby('date').size()
    pool = ev[(ev.cell == cell) & (ev.split == split) & (ev.event != signal_event) & ev.entered]
    pool_by_day = {d: g['net_R'].to_numpy() for d, g in pool.groupby('date')}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_draws):
        picked = []
        for d, k in per_day_n.items():
            arr = pool_by_day.get(d)
            if arr is None or not len(arr):
                continue
            picked.append(rng.choice(arr, size=min(k, len(arr)), replace=len(arr) < k))
        if picked:
            draws.append(float(np.concatenate(picked).mean()))
    if not draws:
        return None, actual_mean
    draws = np.array(draws)
    pctile = float((draws < actual_mean).mean() * 100.0)
    return dict(null_mean=float(draws.mean()), null_p99=float(np.percentile(draws, 99)),
                actual_pctile=pctile), actual_mean


def universe_placebo(ev, cell, split):
    sub = ev[(ev.cell == cell) & (ev.split == split) & ev.entered]
    return float(sub['net_R'].mean()) if len(sub) else np.nan


def calibration_line(ev):
    """HOD base fills on FAILURE / SUCCESS symbol-days, fill_min >= 630, both splits."""
    if not os.path.exists(MODEL_1478_CSV):
        log.warning('calibration source missing: %s', MODEL_1478_CSV)
        return pd.DataFrame()
    hod = pd.read_csv(MODEL_1478_CSV)
    hod = hod[hod.fill_min >= DECL_1030_M]
    fail_days = set(map(tuple, ev[(ev.cell == '1564') & (ev.event == 'FAILURE')]
                        [['date', 'symbol']].drop_duplicates().to_numpy()))
    succ_days = set(map(tuple, ev[(ev.cell == '1566') & (ev.event == 'SUCCESS')]
                        [['date', 'symbol']].drop_duplicates().to_numpy()))
    rows = []
    for label, keyset in (('FAILURE', fail_days), ('SUCCESS', succ_days)):
        for split, g in hod.groupby('split'):
            m = g.apply(lambda r: (r.day, r.symbol) in keyset, axis=1)
            gg = g[m]
            if len(gg):
                rows.append(dict(label=label, split=split, n=len(gg),
                                  mean_outcome_R=float(gg.outcome_R.mean()),
                                  t=day_clustered_t(gg.outcome_R, gg.day)))
    return pd.DataFrame(rows)


def per_month_table(ev, cell, split):
    sub = ev[(ev.cell == cell) & (ev.split == split) & ev.entered & (ev.event ==
             ('FAILURE' if cell in ('1564', '1565') else 'SUCCESS'))].copy()
    if not len(sub):
        return pd.DataFrame()
    sub['month'] = sub['date'].str[:7]
    return sub.groupby('month')['net_R'].agg(['count', 'mean']).reset_index()


def pass_bar(stats, null, train_t):
    """Frozen VAL pass bar from PREREG_1564.md."""
    if not stats or stats.get('n', 0) == 0:
        return False
    ok = (stats['mean_net_R'] >= 0.15 and stats['t'] >= 2.5 and stats['ex_top5'] > 0
          and stats['winner_capped'] > 0 and stats['events_wk'] >= 3
          and (null is not None and null.get('actual_pctile', 0) >= 99)
          and (train_t is not None and train_t >= 1)
          and stats['median_r_pct_price'] >= 0.5)
    return bool(ok)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--limit-days', type=int, default=0)
    a = ap.parse_args()
    ev, coverage, n_no_bars, n_no_range = run(a.limit_days)

    lines = ['# RESULT_1564 -- cells 1,564-1,566: failed/held opening-range breakout at 10:30 ET',
              '', f'Generated {time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())}. '
              f'Bars excluded: {n_no_bars} (no bars at all), {n_no_range} (incomplete opening '
              f'range). Half-spread source coverage: {coverage}.', '']
    total = len(coverage) and sum(coverage.values())
    if total:
        for k, v in coverage.items():
            lines.append(f'- {k}: {v} ({100.0*v/total:.1f}%)')
    lines.append('')

    report_events = ev.groupby(['event']).size()
    lines.append('## Event counts at 10:30 (the only hour that trades)')
    lines.append(report_events.to_string())
    lines.append('')
    lines.append('## Report-only declaration hours (counts, no trades -- PREREG "Not allowed" '
                  'forbids selecting among these)')
    lines.append(ev.groupby('event_1000').size().rename('n_at_1000').to_string())
    lines.append(ev.groupby('event_1100').size().rename('n_at_1100').to_string())
    lines.append('')

    for cell in ('1564', '1565', '1566'):
        want_event = 'SUCCESS' if cell == '1566' else 'FAILURE'
        lines.append(f'## Cell {cell}')
        for split in ('TRAIN', 'VAL'):
            not_report_only = ~ev['report_only'].infer_objects(copy=False).fillna(False).astype(bool)
            sub = ev[(ev.cell == cell) & (ev.split == split) & ev.entered
                     & (ev.event == want_event) & not_report_only]
            stats = cell_stats(sub)
            null, actual_mean = null_control(ev, cell, split)
            placebo = universe_placebo(ev, cell, split)
            lines.append(f'### split={split}')
            lines.append(json.dumps({**stats, 'null': null, 'universe_placebo_R': placebo},
                                     default=str, indent=2))
            pm = per_month_table(ev, cell, split)
            if len(pm):
                lines.append(pm.to_string(index=False))
        train_stats = cell_stats(ev[(ev.cell == cell) & (ev.split == 'TRAIN') & ev.entered
                                     & (ev.event == want_event)])
        val_stats = cell_stats(ev[(ev.cell == cell) & (ev.split == 'VAL') & ev.entered
                                   & (ev.event == want_event)])
        val_null, _ = null_control(ev, cell, 'VAL')
        passes = pass_bar(val_stats, val_null, train_stats.get('t'))
        lines.append(f'**PASS BAR (VAL): {passes}**')
        lines.append('')

    cal = calibration_line(ev)
    lines.append('## Calibration line (report-only): HOD base fills on FAILURE/SUCCESS days, '
                  'fill_min >= 630')
    lines.append(cal.to_string(index=False) if len(cal) else '(no overlap)')
    lines.append('')
    lines.append('## Caveats (read as an adversary)')
    lines.append('- Independent reimplementation, causality trace and fill-realism review are '
                  'still owed before this ships to the owner (see the PREREG\'s "Independent '
                  'check" section) -- this run is the BUILDER only.')
    lines.append('- Half-spread fallback shares above must be inspected: a high flat_fallback '
                  'share weakens the fill-realism claim.')
    lines.append('- The count-matched null draws from the SAME split\'s non-failure population; '
                  'a small non-failure pool on thin days widens the null.')

    with open(RESULT_MD, 'w') as f:
        f.write('\n'.join(str(x) for x in lines))
    log.info('wrote %s', RESULT_MD)


if __name__ == '__main__':
    sys.exit(main() or 0)
