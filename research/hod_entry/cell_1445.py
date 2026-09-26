#!/usr/bin/env python3
"""Cells 1,445-1,456 -- research/hod_entry/PREREG_1445.md (FROZEN 2026-09-26 02:20 UTC).

Reviving the resting-order fill book (cell 1,438's population, 9,911 fills, status == 'fill',
TRAIN-H2 + VAL) by a CAUSAL restatement: does isolating the +0.24/+0.28R cohort (the fills where
data/cache.db happened to hold complete bars) with terms knowable AT THE ARM BAR -- the live
scanner's own qualification predicate, a multi-day-high, or bar-density/dollar-volume/spread --
recover a positive, separating, obtainable edge?

Every cell partitions the SAME fixed base book (no outcome is recomputed) under:
  * corrected cost -- the CSV's cost_R double-charges the entry half-spread (the CSV's `fill` is
    already an ask fill); half_entry is recovered from cost_R/R/exit_price/exit_half_src (joining
    research/bf_zero/causal_filter/nbbo.csv for exit_half_src == 'nbbo' rows) and added back.
  * measured stop slip -- cell 1,443's per-trade tape measurement (sip_cache_stopslip/*.pkl.gz),
    else the holdout-pooled mean (35 bps); a flat-30-bps variant is reported beside for continuity.

Features are computed causally: every bar/day used to build a flag has `m < fill_min` (arm bar j)
or is the PRIOR session (never the signal day) for the multi-day-high terms. bars_sip.db
(research/bf_zero/bars_sip.db) is the ONLY intraday bar source used for features -- data/cache.db
is never touched for a feature (its selection is exactly the look-ahead confound being isolated
here); data/cache.db is read ONLY for the float snapshot, which the PREREG cells explicitly source
from there and disclose as current-snapshot-dated.

Usage:
    nice -n 19 python3 research/hod_entry/cell_1445.py [--dry-run]

    --dry-run scores a fixed 300-row sample (seed 1445) instead of the full 9,911-row book, for
    fast test iteration; it still writes both output files, prefixed so they never collide with a
    full run (cell_1445_features_DRYRUN.csv / RESULT_1445_DRYRUN.md).

Outputs: research/hod_entry/cell_1445_features.csv (one row per base fill: day, symbol, fill_min,
split, every flag as 0/1 with NaN when not computable, net_R_corr, net_R_corr_flat30) and
research/hod_entry/RESULT_1445.md (one table: cell x holdout, plus caveats).
"""
import argparse
import gzip
import os
import pickle
import sqlite3
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')

import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

from research.hod_entry import causal_arming as ca          # noqa: E402  (_rth reuse, see load_sip_bars)
from research.hod_consol import run_consol as consol        # noqa: E402  (simulate_slots)

FILLS_CSV = os.path.join(HERE, 'causal_arming_causal.csv')
NBBO_CSV = os.path.join(REPO, 'research/bf_zero/causal_filter/nbbo.csv')
BARS_SIP_DB = os.path.join(REPO, 'research/bf_zero/bars_sip.db')
CACHE_DB = os.path.join(REPO, 'data/cache.db')
DAILY_PARQUET = os.path.join(REPO, 'data/research/databento/equs_daily_2025_2026.parquet')
SYMBOL_MAP_CSV = os.path.join(REPO, 'data/research/databento/equs_instrument_symbol_map.csv')
STOPSLIP_CACHE_DIR = os.path.join(HERE, 'sip_cache_stopslip')

FLAT_SLIP_FALLBACK_BPS = 35.0     # cell 1,443's pooled holdout mean -- "else the holdout mean"
FLAT_SLIP_VARIANT_BPS = 30.0      # the continuity variant the PREREG asks to keep beside it
SLIP_APPLICABLE_WHY = {'stop', 'stop_bar', 'eod'}   # matches cell 1,443's own to_measure() subset
OPEN_M = 570                       # 09:30 ET in minutes-since-midnight
COVERAGE_VOID_BAR = 0.80
NULL_SEED = 1445
NULL_DRAWS = 1000
WINNER_CAP_R = 3.0

MOVER_CELLS = ['1445', '1446', '1447', '1448']
LIQUIDITY_CELLS = ['1452', '1453', '1454']
CAUSAL_CELLS = MOVER_CELLS + ['1449', '1450', '1451'] + LIQUIDITY_CELLS   # 1,445-1,454, ten cells


def log(msg):
    """Verbose progress line, flushed immediately (print() is buffered under nohup otherwise)."""
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


# --------------------------------------------------------------------------------------------
# Step 0: base book
# --------------------------------------------------------------------------------------------

def load_base_book():
    """The 9,911 fills of cell 1,438: status == 'fill', TRAIN-H2 + VAL (TEST excluded, never read).
    Adds a `holdout` column ('TRAIN-H2' / 'VAL')."""
    df = pd.read_csv(FILLS_CSV, low_memory=False)
    f = df[df.status == 'fill'].copy()
    keep = (f.split == 'VAL') | ((f.split == 'TRAIN') & (f.half == 'H2'))
    f = f[keep].copy()
    f['holdout'] = np.where(f.split == 'VAL', 'VAL', 'TRAIN-H2')
    n = len(f)
    if n != 9911:
        log(f'WARNING base book has {n} rows, PREREG expects 9,911 -- proceeding, but flag this')
    return f.reset_index(drop=True)


# --------------------------------------------------------------------------------------------
# Step 1: corrected cost + measured stop slip
# --------------------------------------------------------------------------------------------

def load_nbbo_lookup():
    """{(day, symbol): spread_mean} from research/bf_zero/causal_filter/nbbo.csv (dollar full spread)."""
    n = pd.read_csv(NBBO_CSV)
    n = n.dropna(subset=['spread_mean'])
    dup = n.duplicated(subset=['day', 'symbol']).sum()
    if dup:
        log(f'WARNING nbbo.csv has {dup} duplicate (day,symbol) rows -- keeping the first')
        n = n.drop_duplicates(subset=['day', 'symbol'], keep='first')
    return n.set_index(['day', 'symbol'])['spread_mean'].to_dict()


def corrected_cost(fills, nbbo_lookup):
    """Recover half_entry (dollars) per the PREREG cost section and return
    (half_entry, net_R_costfix, nbbo_fallback_flag). nbbo_fallback_flag is True where an
    exit_half_src == 'nbbo' row had no (day,symbol) match in nbbo.csv and fell back to the
    fill_instant formula (counted, never silently dropped)."""
    R = fills['R'].to_numpy(float)
    cost_R = fills['cost_R'].to_numpy(float)
    exit_price = fills['exit_price'].to_numpy(float)
    src = fills['exit_half_src'].to_numpy()
    days = fills['day'].to_numpy()
    syms = fills['symbol'].to_numpy()

    half_entry = np.empty(len(fills))
    fallback = np.zeros(len(fills), dtype=bool)
    for i in range(len(fills)):
        he = None
        if src[i] == 'nbbo':
            spread_mean = nbbo_lookup.get((days[i], syms[i]))
            if spread_mean is not None:
                exit_half = spread_mean / 2.0
                he = cost_R[i] * R[i] - exit_half - 0.0002 * exit_price[i]
            else:
                fallback[i] = True
        if he is None:                                    # fill_instant src, or a failed nbbo join
            he = (cost_R[i] * R[i] - 0.0002 * exit_price[i]) / 2.0
        half_entry[i] = he

    n_fb = int(fallback.sum())
    n_nbbo = int((src == 'nbbo').sum())
    log(f'corrected_cost: {n_nbbo} nbbo-src rows, {n_fb} fell back to fill_instant formula '
        f'({n_fb / max(n_nbbo, 1):.1%} of nbbo-src)')
    net_R_costfix = fills['net_R'].to_numpy(float) + half_entry / R
    return half_entry, net_R_costfix, fallback


def row_key(symbol, exit_m, why, fill_min):
    """Same cache key as cell_1443.py's row_key -- (symbol, exit_m, why, fill_min)."""
    return f'{symbol}|{exit_m}|{why}|{fill_min}'


_STOPSLIP_CACHE_BY_DAY = {}


def load_stopslip_cache(day):
    """Per-day pickle cache written by cell_1443.py, memoized in-process."""
    if day in _STOPSLIP_CACHE_BY_DAY:
        return _STOPSLIP_CACHE_BY_DAY[day]
    path = os.path.join(STOPSLIP_CACHE_DIR, f'{day}.pkl.gz')
    if os.path.exists(path):
        with gzip.open(path, 'rb') as fh:
            cache = pickle.load(fh)
    else:
        cache = {}
    _STOPSLIP_CACHE_BY_DAY[day] = cache
    return cache


def measured_slip_bps_one(day, symbol, exit_m, why, fill_min):
    """cell 1,443's measured slip (bps) for one row, or None if uncached/unmeasured (caller applies
    the holdout-mean fallback in that case). Rows outside {stop, stop_bar, eod} never get a slip
    charge (target/eod_fallback exits are not a stop-slip question), signalled by returning 0.0
    and the caller's is_applicable check."""
    cache = load_stopslip_cache(day)
    res = cache.get(row_key(symbol, exit_m, why, fill_min))
    if res is not None and res.get('measured'):
        return float(res['slip_bps'])
    return None


def apply_stop_slip(fills, flat_bps=None):
    """slip_R per row: 0 for why not in {stop, stop_bar, eod}; else cell 1,443's per-trade measured
    bps (or `flat_bps` if given, forcing the flat variant) converted to R units on the stop price
    (or exit_price for eod exits), else the pooled holdout-mean fallback (35 bps) when uncached."""
    slip_R = np.zeros(len(fills))
    n_measured = n_fallback = n_flat = 0
    for i, row in enumerate(fills.itertuples()):
        if row.why not in SLIP_APPLICABLE_WHY:
            continue
        if flat_bps is not None:
            bps = flat_bps
            n_flat += 1
        else:
            m = measured_slip_bps_one(row.day, row.symbol, row.exit_m, row.why, row.fill_min)
            if m is None:
                bps = FLAT_SLIP_FALLBACK_BPS
                n_fallback += 1
            else:
                bps = m
                n_measured += 1
        base_price = row.exit_price if row.why == 'eod' else row.stop
        slip_R[i] = base_price * bps / 1e4 / row.R
    if flat_bps is None:
        log(f'apply_stop_slip: {n_measured} measured, {n_fallback} fell back to '
            f'{FLAT_SLIP_FALLBACK_BPS:.0f} bps (of {n_measured + n_fallback} applicable rows)')
    return slip_R


# --------------------------------------------------------------------------------------------
# Step 2: features
# --------------------------------------------------------------------------------------------

def load_sip_bars(symbol_days):
    """{(symbol, day): RTH minute bar frame (m,o,h,l,c,v)}, sourced ONLY from bars_sip.db (never
    data/cache.db -- the PREREG forbids cache.db for any feature). Reuses causal_arming._rth for
    the UTC->ET RTH-minute conversion, batched one SQL query per day."""
    con = sqlite3.connect(f'file:{BARS_SIP_DB}?mode=ro', uri=True)
    by_day = {}
    for sym, day in symbol_days:
        by_day.setdefault(day, set()).add(sym)
    out = {}
    days_sorted = sorted(by_day)
    for di, day in enumerate(days_sorted):
        syms = sorted(by_day[day])
        q = (f"select symbol, t, o, h, l, c, v from bars where day=? and symbol in "
             f"({','.join('?' * len(syms))})")
        g = pd.read_sql(q, con, params=[day] + syms)
        for s, gg in g.groupby('symbol'):
            out[(s, day)] = ca._rth(gg, 't')
        if di % 50 == 0 or di == len(days_sorted) - 1:
            log(f'load_sip_bars: day {di + 1}/{len(days_sorted)} ({day}), {len(out)} symbol-days loaded')
    con.close()
    return out


def arm_bar_features(bars, fill_min):
    """Bars strictly before fill_min (arm bar j = the last one) -> dict of causal features, or
    None if no bar exists before fill_min (should not happen for an armed fill; logged if it does)."""
    b = bars[bars.m < fill_min]
    if not len(b):
        return None
    j_m = int(b.m.iloc[-1])
    close_j = float(b.c.iloc[-1])
    running_high = float(b.h.max())
    running_low = float(b.l.min())
    dollar_vol_j = float((b.v * b.c).sum())
    n_minutes_possible = j_m - OPEN_M + 1
    bar_density = len(b) / n_minutes_possible if n_minutes_possible > 0 else np.nan
    return dict(close_j=close_j, running_high_j=running_high, running_low_j=running_low,
                dollar_vol_j=dollar_vol_j, bar_density_j=bar_density, arm_m=j_m, n_bars_j=len(b))


def load_symbol_map():
    m = pd.read_csv(SYMBOL_MAP_CSV, parse_dates=['d0', 'd1'])
    return m


def resolve_instrument_ids(pairs, map_df):
    """{(symbol, day): instrument_id} for unique (symbol, day) pairs, d0 <= day <= d1."""
    u = pd.DataFrame(pairs, columns=['symbol', 'day']).drop_duplicates()
    u['day_ts'] = pd.to_datetime(u['day'])
    out = {}
    for sym, g in u.groupby('symbol'):
        cand = map_df[map_df.symbol == sym]
        if not len(cand):
            continue
        for row in g.itertuples():
            hit = cand[(cand.d0 <= row.day_ts) & (cand.d1 >= row.day_ts)]
            if len(hit):
                out[(sym, row.day)] = int(hit.iloc[0].instrument_id)
    return out


def build_daily_panel(instrument_ids):
    """Databento daily panel restricted to the instrument_ids we need, with prev_close, prev_high
    and high20 (max high of the PRIOR 20 sessions, never including the signal day) precomputed per
    (instrument_id, bar_date) via a causal shift -- both multi-day-high terms are therefore
    automatically excluded-current-day by construction."""
    df = pd.read_parquet(DAILY_PARQUET,
                          columns=['bar_date', 'symbol', 'instrument_id', 'open', 'high', 'low',
                                   'close', 'volume'])
    df = df[df.instrument_id.isin(instrument_ids)].copy()
    df['bar_date'] = pd.to_datetime(df['bar_date'])
    df = df.sort_values(['instrument_id', 'bar_date']).reset_index(drop=True)
    g = df.groupby('instrument_id', sort=False)
    df['prev_close'] = g['close'].shift(1)
    df['prev_high'] = g['high'].shift(1)
    df['high20'] = g['high'].transform(lambda s: s.shift(1).rolling(20, min_periods=20).max())
    return df


def load_float_lookup():
    """{symbol: float_shares} from data/cache.db `universe`, read-only URI -- the ONE feature the
    PREREG allows from cache.db, sourced as a CURRENT snapshot and disclosed as such (known for
    ~51% of the base book; not treated as a coverage-VOID trigger -- see RESULT_1445.md caveats)."""
    con = sqlite3.connect(f'file:{CACHE_DB}?mode=ro', uri=True)
    df = pd.read_sql('select symbol, float_shares from universe', con)
    con.close()
    return df.set_index('symbol')['float_shares'].to_dict()


def build_features(fills):
    """Attach every causal flag + the placebo's look-ahead fields to `fills`. Returns the enriched
    frame and a coverage dict {feature_name: share_computable}."""
    log('build_features: loading SIP bars (bars_sip.db only)')
    bars_by_sd = load_sip_bars(list(zip(fills.symbol, fills.day)))

    log('build_features: resolving Databento instrument ids')
    map_df = load_symbol_map()
    instr_by_sd = resolve_instrument_ids(list(zip(fills.symbol, fills.day)), map_df)
    n_resolved = sum(1 for sd in zip(fills.symbol, fills.day) if sd in instr_by_sd)
    log(f'build_features: instrument id resolved for {n_resolved}/{len(fills)} fills')

    daily_panel = build_daily_panel(set(instr_by_sd.values()))
    daily_idx = daily_panel.set_index(['instrument_id', 'bar_date'])

    log('build_features: loading float snapshot from data/cache.db (read-only)')
    float_lookup = load_float_lookup()

    rows = []
    n_no_bars_before_j = 0
    empty_bars = pd.DataFrame(columns=['m', 'o', 'h', 'l', 'c', 'v'])
    for r in fills.itertuples():
        feat = arm_bar_features(bars_by_sd.get((r.symbol, r.day), empty_bars), r.fill_min)
        rec = dict(day=r.day, symbol=r.symbol)
        if feat is None:
            n_no_bars_before_j += 1
            feat = {}
        rec.update(feat)

        iid = instr_by_sd.get((r.symbol, r.day))
        day_ts = pd.Timestamp(r.day)
        prev_close = prev_high = high20 = day_high = day_low = day_close = np.nan
        if iid is not None and (iid, day_ts) in daily_idx.index:
            drow = daily_idx.loc[(iid, day_ts)]
            if isinstance(drow, pd.DataFrame):     # duplicate (instrument_id, day) -- take the first
                drow = drow.iloc[0]
            prev_close, prev_high, high20 = drow['prev_close'], drow['prev_high'], drow['high20']
            day_high, day_low, day_close = drow['high'], drow['low'], drow['close']
        rec['prev_close'] = prev_close
        rec['prev_high'] = prev_high
        rec['high20'] = high20
        rec['day_high'] = day_high
        rec['day_low'] = day_low
        rec['day_close'] = day_close
        rec['float_shares'] = float_lookup.get(r.symbol, np.nan)
        rows.append(rec)

    if n_no_bars_before_j:
        log(f'WARNING build_features: {n_no_bars_before_j}/{len(fills)} fills had NO bar before '
            f'fill_min in bars_sip.db (arm-bar features NaN for those)')

    feat_df = pd.DataFrame(rows)
    out = pd.concat([fills.reset_index(drop=True), feat_df.drop(columns=['day', 'symbol'])], axis=1)

    out['gap_j'] = (out.close_j - out.prev_close) / out.prev_close * 100
    out['range_j'] = (out.running_high_j - out.running_low_j) / out.running_low_j * 100
    out['mover_j'] = out[['gap_j', 'range_j']].max(axis=1)
    out['float_known'] = out.float_shares.notna()
    out['spread_frac'] = 2 * out.half_entry / out.fill

    # placebo: full-day, look-ahead terms (report-only)
    out['placebo_range'] = (out.day_high - out.day_low) / out.day_low * 100
    out['placebo_gap_close'] = (out.day_high - out.prev_close) / out.prev_close * 100

    coverage = {}
    for col in ['gap_j', 'range_j', 'mover_j', 'prev_close', 'prev_high', 'high20',
                'bar_density_j', 'dollar_vol_j', 'spread_frac', 'float_shares',
                'placebo_range', 'placebo_gap_close']:
        coverage[col] = float(out[col].notna().mean())
    return out, coverage


# --------------------------------------------------------------------------------------------
# Cell conditions (evaluated with data through arm bar j only; placebo is the sole exception)
# --------------------------------------------------------------------------------------------

def cell_conditions(df):
    """{cell_id: boolean Series (NaN treated as False -- "not computable" cannot qualify)}."""
    price_ok_prev = df.prev_close.between(1, 30)
    price_ok_close = df.close_j.between(1, 30)
    float_ok = df.float_known & (df.float_shares > 0) & (df.float_shares <= 50_000_000)
    mover15 = df.mover_j >= 15
    mover10 = df.mover_j >= 10

    cond = {}
    cond['1445'] = price_ok_prev & float_ok & price_ok_close & mover15
    cond['1446'] = float_ok & mover15
    cond['1447'] = float_ok & mover10
    cond['1448'] = mover15
    cond['1449'] = float_ok
    cond['1450'] = df.level >= df.prev_high
    cond['1451'] = df.level >= df.high20
    cond['1452'] = df.bar_density_j >= 0.90
    cond['1453'] = df.dollar_vol_j >= 1_000_000
    cond['1454'] = df.spread_frac <= 0.0010
    cond['1455'] = ((df.placebo_range >= 10) | (df.placebo_gap_close >= 10)) & \
        df.day_close.between(1, 30) & (~df.float_known | (df.float_shares <= 50_000_000))
    for k in cond:
        cond[k] = cond[k].fillna(False)
    return cond


# --------------------------------------------------------------------------------------------
# Step 3: scoring
# --------------------------------------------------------------------------------------------

def day_clustered_t(y, day):
    """statsmodels OLS on a constant, clustered by day -- the t-stat on the mean."""
    y = pd.Series(y).dropna()
    if len(y) < 2:
        return np.nan
    d = pd.Series(day).loc[y.index]
    if d.nunique() < 2:
        return np.nan
    X = np.ones((len(y), 1))
    model = sm.OLS(y.to_numpy(), X).fit(cov_type='cluster', cov_kwds={'groups': d.to_numpy()})
    return float(model.tvalues[0])


def ex_top5_mean(y):
    """Mean excluding the top 5% (by value) of a series -- tail-dependence check."""
    y = pd.Series(y).dropna().sort_values(ascending=False)
    n = len(y)
    if n == 0:
        return np.nan
    k = int(round(0.05 * n))
    return float(y.iloc[k:].mean()) if k < n else float(y.mean())


def winner_capped_mean(y, cap=WINNER_CAP_R):
    y = pd.Series(y).dropna()
    if not len(y):
        return np.nan
    return float(np.minimum(y, cap).mean())


def weeks_spanned(days):
    """Distinct ISO (year, week) count over a day-string series -- the denominator for fills/wk."""
    iso = pd.to_datetime(pd.Series(days).unique())
    wk = {(d.isocalendar()[0], d.isocalendar()[1]) for d in iso}
    return max(len(wk), 1)


def fills_per_week(kept_subset, weeks):
    """Fills/wk under first-12/day, 4-concurrent slotting (research/hod_consol/run_consol.simulate_slots)."""
    if not len(kept_subset):
        return 0.0
    trades = kept_subset.rename(columns={'fill_min': 'entry_m'})[['day', 'entry_m', 'exit_m']].copy()
    keep = consol.simulate_slots(trades)
    return float(keep.sum()) / weeks


def null_percentile_of(pool_values, n_kept, kept_mean, seed=NULL_SEED, n_draws=NULL_DRAWS):
    """Percentile of `kept_mean` within 1,000 random count-matched subsets of `pool_values`."""
    vals = pd.Series(pool_values).dropna().to_numpy()
    n = len(vals)
    if n_kept == 0 or n_kept > n or np.isnan(kept_mean):
        return np.nan
    rng = np.random.RandomState(seed)
    draws = np.empty(n_draws)
    for i in range(n_draws):
        idx = rng.choice(n, size=n_kept, replace=False)
        draws[i] = vals[idx].mean()
    return float((draws <= kept_mean).mean() * 100)


def score_one(cell_id, cond, holdout_df, holdout_name, weeks):
    """One cell x holdout row of the pass-bar table."""
    kept = holdout_df[cond]
    dropped = holdout_df[~cond]
    n_kept, n_dropped = len(kept), len(dropped)
    kept_mean = float(kept.net_R_corr.mean()) if n_kept else np.nan
    dropped_mean = float(dropped.net_R_corr.mean()) if n_dropped else np.nan
    t_kept = day_clustered_t(kept.net_R_corr, kept.day) if n_kept else np.nan
    ex5 = ex_top5_mean(kept.net_R_corr) if n_kept else np.nan
    fwk = fills_per_week(kept, weeks) if n_kept else 0.0
    wcap = winner_capped_mean(kept.net_R_corr) if n_kept else np.nan
    flat30_mean = float(kept.net_R_corr_flat30.mean()) if n_kept else np.nan
    npct = null_percentile_of(holdout_df.net_R_corr, n_kept, kept_mean)
    return dict(cell=cell_id, holdout=holdout_name, n_kept=n_kept, n_dropped=n_dropped,
                kept_mean=kept_mean, dropped_mean=dropped_mean, delta_R=kept_mean - dropped_mean
                if n_kept and n_dropped else np.nan, t_kept=t_kept, ex_top5=ex5, fills_wk=fwk,
                winner_capped_mean=wcap, kept_mean_flat30=flat30_mean, null_pctile=npct)


def evaluate_pass_bar(rows_by_cell):
    """PASS iff (on VAL): kept mean >= 0.15, t >= 2.5, ex-top-5% > 0, fills/wk >= 3, same sign on
    TRAIN-H2 with t >= 1, and dropped < kept on BOTH holdouts. Returns {cell: bool}."""
    verdict = {}
    for cell, by_h in rows_by_cell.items():
        val, th2 = by_h.get('VAL'), by_h.get('TRAIN-H2')
        if val is None or th2 is None:
            verdict[cell] = False
            continue
        ok = (val['n_kept'] > 0 and val['kept_mean'] >= 0.15 and val['t_kept'] >= 2.5
              and val['ex_top5'] > 0 and val['fills_wk'] >= 3
              and th2['n_kept'] > 0 and np.sign(th2['kept_mean']) == np.sign(val['kept_mean'])
              and th2['t_kept'] >= 1
              and val['dropped_mean'] < val['kept_mean']
              and th2['dropped_mean'] < th2['kept_mean'])
        verdict[cell] = bool(ok)
    return verdict


def run_scoring(df):
    """Score every causal cell + the placebo + the joint cell on both holdouts. Returns a list of
    row dicts (cell x holdout) and the pass/fail verdict per causal cell."""
    cond = cell_conditions(df)
    holdouts = {name: df[df.holdout == name] for name in ('TRAIN-H2', 'VAL')}
    weeks = {name: weeks_spanned(holdouts[name].day) for name in holdouts}

    rows_by_cell = {}
    for cell_id in CAUSAL_CELLS + ['1455']:
        rows_by_cell[cell_id] = {}
        for hname, hdf in holdouts.items():
            c = cond[cell_id].loc[hdf.index]
            rows_by_cell[cell_id][hname] = score_one(cell_id, c, hdf, hname, weeks[hname])

    verdict = evaluate_pass_bar({k: v for k, v in rows_by_cell.items() if k in CAUSAL_CELLS})

    # 1,456 JOINT: components chosen by TRAIN-H2 kept mean ONLY (never VAL).
    th2_means = {c: rows_by_cell[c]['TRAIN-H2']['kept_mean'] for c in LIQUIDITY_CELLS + MOVER_CELLS}
    best_liq = max(LIQUIDITY_CELLS, key=lambda c: (th2_means[c] if not np.isnan(th2_means[c]) else -1e9))
    best_mov = max(MOVER_CELLS, key=lambda c: (th2_means[c] if not np.isnan(th2_means[c]) else -1e9))
    log(f'1,456 JOINT: best liquidity cell = {best_liq} (TRAIN-H2 kept mean '
        f'{th2_means[best_liq]:.4f}), best mover cell = {best_mov} (TRAIN-H2 kept mean '
        f'{th2_means[best_mov]:.4f})')
    joint_cond = cond[best_liq] & cond[best_mov]
    rows_by_cell['1456'] = {}
    for hname, hdf in holdouts.items():
        c = joint_cond.loc[hdf.index]
        rows_by_cell['1456'][hname] = score_one('1456', c, hdf, hname, weeks[hname])
    joint_verdict = evaluate_pass_bar({'1456': rows_by_cell['1456']})
    verdict.update(joint_verdict)
    verdict['1455'] = False   # placebo is report-only, never a candidate rule

    all_rows = []
    for cell_id, by_h in rows_by_cell.items():
        for hname in ('TRAIN-H2', 'VAL'):
            r = dict(by_h[hname])
            r['passes_bar'] = verdict.get(cell_id, False)
            all_rows.append(r)
    return all_rows, verdict, dict(best_liq=best_liq, best_mov=best_mov)


# --------------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------------

def write_features_csv(df, path):
    cond = cell_conditions(df)
    out = df[['day', 'symbol', 'fill_min', 'holdout']].copy().rename(columns={'holdout': 'split'})
    for cell_id in CAUSAL_CELLS + ['1455']:
        out[f'flag_{cell_id}'] = cond[cell_id].astype(int)
    out['net_R_corr'] = df['net_R_corr']
    out['net_R_corr_flat30'] = df['net_R_corr_flat30']
    out.to_csv(path, index=False)
    log(f'wrote {path} ({len(out)} rows)')


def write_result_md(rows, coverage, verdict, joint_info, placebo_check, path):
    cols = ['cell', 'holdout', 'n_kept', 'n_dropped', 'kept_mean', 'dropped_mean', 'delta_R',
            't_kept', 'ex_top5', 'fills_wk', 'winner_capped_mean', 'kept_mean_flat30',
            'null_pctile', 'passes_bar']
    lines = ['# RESULT 1,445 -- causal restatement of the resting-order fill book\n']
    header = '| ' + ' | '.join(cols) + ' |'
    sep = '|' + '---|' * len(cols)
    lines += [header, sep]
    for r in rows:
        vals = []
        for c in cols:
            v = r.get(c)
            if isinstance(v, float):
                vals.append(f'{v:.4f}' if not np.isnan(v) else 'NaN')
            else:
                vals.append(str(v))
        lines.append('| ' + ' | '.join(vals) + ' |')
    lines.append('')
    cov_str = ', '.join(f'{k}={v:.0%}' for k, v in coverage.items())
    lines.append(f'Coverage: {cov_str}. float_shares is a CURRENT cache.db snapshot (disclosed, '
                 f'~51% known by design in the FROZEN prereg) -- not treated as a VOID trigger; '
                 f'every other feature is bar/Databento-derived and gated at {COVERAGE_VOID_BAR:.0%}.')
    voided = [k for k, v in coverage.items() if k != 'float_shares' and v < COVERAGE_VOID_BAR]
    lines.append(f'VOID features (<{COVERAGE_VOID_BAR:.0%} coverage): {voided or "none"}.')
    lines.append(f'Placebo (1,455) reproduces the +0.24/+0.28 (~+0.34/+0.38 corrected) cohort: '
                 f'{placebo_check}.')
    passing = [c for c, v in verdict.items() if v]
    lines.append(f'Cells that PASS the bar: {passing or "none"}.')
    lines.append(f'1,456 joint components (chosen on TRAIN-H2 only): liquidity={joint_info["best_liq"]}, '
                 f'mover={joint_info["best_mov"]}.')
    lines.append('Caveats: TEST is sealed and was not read. Stop-slip cache covers 230/230 days at '
                 'the file level but not every individual row -- uncached applicable rows used the '
                 '35 bps pooled fallback (counted in the log). half_entry can be negative for a small '
                 'share of rows (recovered algebraically, not clipped) -- reported, not corrected, '
                 'per the PREREG formula as written.')
    with open(path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    log(f'wrote {path}')


# --------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dry-run', action='store_true',
                     help='score a fixed 300-row sample (seed 1445) instead of the full book')
    args = ap.parse_args(argv)

    log('loading base book (causal_arming_causal.csv, status==fill, TRAIN-H2+VAL)')
    fills = load_base_book()
    if args.dry_run:
        fills = fills.sample(n=min(300, len(fills)), random_state=1445).reset_index(drop=True)
        log(f'--dry-run: scoring a {len(fills)}-row sample')

    log('Step 1: corrected cost')
    nbbo_lookup = load_nbbo_lookup()
    half_entry, net_R_costfix, nbbo_fallback = corrected_cost(fills, nbbo_lookup)
    fills = fills.copy()
    fills['half_entry'] = half_entry
    fills['net_R_costfix'] = net_R_costfix
    fills['nbbo_fallback'] = nbbo_fallback

    log('Step 1: measured stop slip')
    slip_R = apply_stop_slip(fills)
    slip_R_flat30 = apply_stop_slip(fills, flat_bps=FLAT_SLIP_VARIANT_BPS)
    fills['net_R_corr'] = fills['net_R_costfix'] - slip_R
    fills['net_R_corr_flat30'] = fills['net_R_costfix'] - slip_R_flat30

    log('Step 2: features')
    df, coverage = build_features(fills)
    for name, cov in coverage.items():
        flag = ' VOID' if (name != 'float_shares' and cov < COVERAGE_VOID_BAR) else ''
        log(f'  coverage {name}: {cov:.1%}{flag}')

    log('Step 3: scoring')
    rows, verdict, joint_info = run_scoring(df)

    placebo_th2 = next(r for r in rows if r['cell'] == '1455' and r['holdout'] == 'TRAIN-H2')
    placebo_val = next(r for r in rows if r['cell'] == '1455' and r['holdout'] == 'VAL')
    target_low = 0.5 * min(placebo_th2['kept_mean'], placebo_val['kept_mean']) if not (
        np.isnan(placebo_th2['kept_mean']) or np.isnan(placebo_val['kept_mean'])) else np.nan
    placebo_check = (f'TRAIN-H2={placebo_th2["kept_mean"]:.3f} VAL={placebo_val["kept_mean"]:.3f} '
                      f'vs target ~0.34/0.38 (half-of-placebo separating threshold {target_low:.3f})')

    suffix = '_DRYRUN' if args.dry_run else ''
    feat_path = os.path.join(HERE, f'cell_1445_features{suffix}.csv')
    result_path = os.path.join(HERE, f'RESULT_1445{suffix}.md')
    write_features_csv(df, feat_path)
    write_result_md(rows, coverage, verdict, joint_info, placebo_check, result_path)

    log('done')
    return rows


if __name__ == '__main__':
    main()
