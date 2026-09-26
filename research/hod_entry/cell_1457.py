#!/usr/bin/env python3
"""Cells 1,457-1,465 -- research/hod_entry/PREREG_1457.md (FROZEN 2026-09-26 07:05 UTC).

Last round on the resting-order HOD-break fill book (cell 1,438's population, 9,911 fills,
status == 'fill', TRAIN-H2 + VAL, TEST sealed and never read). Builds directly on cell_1445.py's
machinery (base book, corrected cost, day-clustered scoring, null percentile, fills/wk via
research/hod_consol/run_consol.simulate_slots) and cell_1440.py's paired stop re-walk.

Cost standard: the 1,445 standard (half_entry recovered and charged once via corrected_cost(),
per-trade measured stop slip from cell 1,443's sip_cache_stopslip/ cache, holdout-pooled-mean
fallback) with ONE fix named by this PREREG: unmeasured EOD exits fall back to an EOD-SPECIFIC
pooled mean (~10-11 bps, computed live from this run's own measured eod rows), never the STOP
pooled mean (35 bps) -- see apply_stop_slip_eodfix(). The flat-30-bps variant is reported beside
every number.

Cells:
  1,457 CEILING (look-ahead, report-only) -- keep fills whose FULL-DAY (high-low)/low >= 10% on the
    Databento PIT daily bar. KILL SWITCH: if 1,457's VAL kept mean net R < +0.15, every other entry
    cell (1,458-1,462, 1,465) is reported with passes_bar=False, note='kill switch' -- the book is
    closed at every filter on this population. 1,463 (a universal exit-mechanism change) and 1,464
    (a re-walk of the SAME entries) are still scored against their own ship bars; 1,464 is treated
    as part of "every filter and every exit" per the PREREG's Consequences section and IS gated by
    the kill switch, 1,463 is NOT (its ship bar is independent, per the PREREG: "regardless of the
    entry verdict, because it applies to every book").
  1,458 -- pre-market ($, 04:00-09:29 ET) >= $500K, bars_sip.db else data/cache.db (cache-only days).
  1,459 -- ATR14 as % of the prior close >= 4% (true range over the 14 PRIOR sessions, Databento).
  1,460 -- prior session's (high-low)/low >= 5% (Databento).
  1,461 -- news catalyst (orb_news_catalyst_nightly.csv, n_articles >= 1); VOID if coverage < 80%.
  1,462 -- spread (2*half_entry/fill <= 10bps) AND 1,458.
  1,463 COST -- stop-LIMIT reexecution on cell 1,443's tape windows, offsets 20bps / 50bps.
  1,464 COST -- R floor at 2.5% of price (cell_1440.py's paired re-walk, floor only, no cap),
    target 2R from the new R, measured slip re-scaled onto the new R.
  1,465 JOINT -- best of 1,458-1,461 on TRAIN-H2 kept mean, AND 1,463's better variant; one VAL read.

Usage:
    nice -n 19 python3 research/hod_entry/cell_1457.py [--dry-run]

    --dry-run scores a fixed 300-row sample (seed 1457) instead of the full 9,911-row book, and
    caps the 1,463 tape re-fetch budget tighter, for fast test iteration; it still writes both
    output files, prefixed so they never collide with a full run
    (cell_1457_features_DRYRUN.csv / RESULT_1457_DRYRUN.md).

Outputs: research/hod_entry/cell_1457_features.csv (one row per base fill: day, symbol, fill_min,
split, every flag 0/1/NaN, net_R_corr_v2, net_R_corr_flat30, and the 1,463 per-stop slips under
each variant) and research/hod_entry/RESULT_1457.md (cell x holdout table + the 1,463 cost table +
caveats).

Not allowed (PREREG): moving a threshold, choosing 1,465's components on VAL, recomputing outcomes
except in 1,464's paired re-walk, reading TEST, any feature using data after bar j except the
declared ceiling.
"""
import argparse
import os
import sqlite3
import sys
import time
from zoneinfo import ZoneInfo

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

from research.hod_entry import cell_1445 as c1445          # noqa: E402 -- base book, cost, scoring
from research.hod_entry import cell_1440 as c1440          # noqa: E402 -- paired stop re-walk
from research.hod_entry import cell_1443 as c1443          # noqa: E402 -- stop-slip tape cache
from research.hod_entry import causal_arming as ca         # noqa: E402 -- fetch_window (SIP tape)

NEWS_CSV = os.path.join(REPO, 'data/research/orb_news_catalyst_nightly.csv')
NEWS_GENERATOR = 'research/scripts/orb_pm_news_nightly_append.py'

NULL_SEED = 1457
NULL_DRAWS = c1445.NULL_DRAWS
COVERAGE_VOID_BAR = 0.80
KILL_VAL_THRESHOLD = 0.15
ET = ZoneInfo('America/New_York')

PM_START_M = 240      # 04:00 ET, minutes since midnight
PM_END_M_EXCL = 570   # 09:30 ET exclusive -> 04:00-09:29 inclusive
CEILING_RANGE_PCT = 10.0
ATR_WINDOW = 14
ATR_PCT_THRESHOLD = 4.0
PRIOR_RANGE_THRESHOLD_PCT = 5.0
PM_DOLLAR_VOL_THRESHOLD = 500_000.0
SPREAD_BPS_THRESHOLD = 10.0
STOPLIMIT_OFFSETS_BPS = (20.0, 50.0)
R_FLOOR_PCT = 0.025
ENTRY_CELLS = ['1458', '1459', '1460', '1461', '1462']

# A tape re-fetch inside stop_limit_fill() is a real Alpaca SIP market-data API call on the shared
# live account (read-only, same endpoint cell_1443.py already uses) -- bounded so this single run
# cannot fire thousands of new requests. Rows beyond the budget are counted, never silently dropped.
MAX_NEW_FETCHES = int(os.environ.get('CELL1457_MAX_FETCHES', '150'))
_new_fetch_count = 0


def log(msg):
    """Verbose progress line, flushed immediately (print() is buffered under nohup otherwise)."""
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


# ================================================================================================
# Step 1: base book + cost (1,445 standard, EOD-fallback fix)
# ================================================================================================

def apply_stop_slip_eodfix(fills, flat_bps=None):
    """As cell_1445.apply_stop_slip, with the PREREG's ONE fix: unmeasured `why == 'eod'` rows fall
    back to an EOD-SPECIFIC pooled mean bps -- computed live from THIS call's own measured eod rows
    (review/1445_cost_reconciliation.md: ~10-11bps, matching RESULT_1443.md's 11.5/9.7bps table) --
    instead of cell_1445's flat STOP pooled mean (FLAT_SLIP_FALLBACK_BPS = 35.0). Stop/stop_bar
    fallback behaviour is unchanged. Returns (slip_R array, eod_fallback_bps used)."""
    eod_measured = [c1445.measured_slip_bps_one(r.day, r.symbol, r.exit_m, r.why, r.fill_min)
                    for r in fills.itertuples() if r.why == 'eod']
    eod_measured = [m for m in eod_measured if m is not None]
    eod_fallback_bps = float(np.mean(eod_measured)) if eod_measured else c1445.FLAT_SLIP_FALLBACK_BPS
    log(f'apply_stop_slip_eodfix: eod-specific fallback = {eod_fallback_bps:.2f} bps '
        f'(from {len(eod_measured)} measured eod rows, flat_bps={flat_bps})')

    slip_R = np.zeros(len(fills))
    n_measured = n_fb_stop = n_fb_eod = n_flat = 0
    for i, row in enumerate(fills.itertuples()):
        if row.why not in c1445.SLIP_APPLICABLE_WHY:
            continue
        if flat_bps is not None:
            bps = flat_bps
            n_flat += 1
        else:
            m = c1445.measured_slip_bps_one(row.day, row.symbol, row.exit_m, row.why, row.fill_min)
            if m is None:
                if row.why == 'eod':
                    bps = eod_fallback_bps
                    n_fb_eod += 1
                else:
                    bps = c1445.FLAT_SLIP_FALLBACK_BPS
                    n_fb_stop += 1
            else:
                bps = m
                n_measured += 1
        base_price = row.exit_price if row.why == 'eod' else row.stop
        slip_R[i] = base_price * bps / 1e4 / row.R
    if flat_bps is None:
        log(f'apply_stop_slip_eodfix: {n_measured} measured, {n_fb_stop} stop-fallback (35bps), '
            f'{n_fb_eod} eod-fallback ({eod_fallback_bps:.1f}bps)')
    return slip_R, eod_fallback_bps


def build_base_cost(fills):
    """Attach half_entry, net_R_costfix, net_R_corr (1,445 standard + EOD fix), net_R_corr_flat30
    and spread_frac to `fills`. Returns (enriched df, eod_fallback_bps)."""
    nbbo_lookup = c1445.load_nbbo_lookup()
    half_entry, net_R_costfix, nbbo_fallback = c1445.corrected_cost(fills, nbbo_lookup)
    out = fills.copy()
    out['half_entry'] = half_entry
    out['net_R_costfix'] = net_R_costfix
    out['nbbo_fallback'] = nbbo_fallback
    slip_R, eod_fb_bps = apply_stop_slip_eodfix(out)
    slip_R_flat30, _ = apply_stop_slip_eodfix(out, flat_bps=30.0)
    out['net_R_corr'] = out['net_R_costfix'] - slip_R          # == net_R_corr_v2 of the PREREG
    out['net_R_corr_flat30'] = out['net_R_costfix'] - slip_R_flat30
    out['spread_frac'] = 2.0 * out['half_entry'] / out['fill']
    return out, eod_fb_bps


# ================================================================================================
# Step 2: Databento daily panel (day_high/day_low for the ceiling; prev_close/prev_high/prev_low;
# ATR14 as % of prior close; prior session's range) -- every join is a causal shift, never the
# signal day, so ATR14 and prior-range terms exclude the current day by construction.
# ================================================================================================

def build_daily_panel_ext(instrument_ids):
    """Databento daily panel restricted to instrument_ids: day_high/day_low/day_close (the signal
    day's own bar, needed ONLY for the declared look-ahead ceiling) plus prev_close/prev_high/
    prev_low (causal shift(1)) and atr14_pct (true range over the 14 PRIOR sessions as % of the
    prior close: TR_t from day t's own high/low/prev_close, then atr14_t = mean(TR_[t-14..t-1]) via
    shift(1).rolling(14) on the TR series -- no day's own TR ever contributes to its own atr14_pct,
    matching build_daily_panel's high20 pattern in cell_1445.py)."""
    df = pd.read_parquet(c1445.DAILY_PARQUET,
                          columns=['bar_date', 'symbol', 'instrument_id', 'open', 'high', 'low',
                                   'close', 'volume'])
    df = df[df.instrument_id.isin(instrument_ids)].copy()
    df['bar_date'] = pd.to_datetime(df['bar_date'])
    df = df.sort_values(['instrument_id', 'bar_date']).reset_index(drop=True)
    g = df.groupby('instrument_id', sort=False)
    df['prev_close'] = g['close'].shift(1)
    df['prev_high'] = g['high'].shift(1)
    df['prev_low'] = g['low'].shift(1)
    tr = np.maximum.reduce([
        (df['high'] - df['low']).to_numpy(),
        (df['high'] - df['prev_close']).abs().to_numpy(),
        (df['low'] - df['prev_close']).abs().to_numpy(),
    ])
    df['tr'] = tr
    df['atr14'] = df.groupby('instrument_id', sort=False)['tr'].transform(
        lambda s: s.shift(1).rolling(ATR_WINDOW, min_periods=ATR_WINDOW).mean())
    df['atr14_pct'] = df['atr14'] / df['prev_close'] * 100
    df['prior_range_pct'] = (df['prev_high'] - df['prev_low']) / df['prev_low'] * 100
    return df


def attach_daily_ext(df):
    """Join build_daily_panel_ext onto `df` by (symbol, day) -> instrument_id -> (instrument_id,
    bar_date), same resolve-then-loc pattern as cell_1445.build_features."""
    map_df = c1445.load_symbol_map()
    instr_by_sd = c1445.resolve_instrument_ids(list(zip(df.symbol, df.day)), map_df)
    n_resolved = sum(1 for sd in zip(df.symbol, df.day) if sd in instr_by_sd)
    log(f'attach_daily_ext: instrument id resolved for {n_resolved}/{len(df)} fills')
    daily = build_daily_panel_ext(set(instr_by_sd.values()))
    idx = daily.set_index(['instrument_id', 'bar_date'])
    cols = ['prev_close', 'prev_high', 'prev_low', 'atr14_pct', 'prior_range_pct',
            'high', 'low', 'close']
    recs = []
    for r in df.itertuples():
        iid = instr_by_sd.get((r.symbol, r.day))
        day_ts = pd.Timestamp(r.day)
        vals = {c: np.nan for c in cols}
        if iid is not None and (iid, day_ts) in idx.index:
            drow = idx.loc[(iid, day_ts)]
            if isinstance(drow, pd.DataFrame):
                drow = drow.iloc[0]
            vals = {c: drow[c] for c in cols}
        recs.append(vals)
    feat = pd.DataFrame(recs, index=df.index).rename(
        columns={'high': 'day_high', 'low': 'day_low', 'close': 'day_close'})
    out = pd.concat([df, feat], axis=1)
    out['full_day_range_pct'] = (out.day_high - out.day_low) / out.day_low * 100
    return out


# ================================================================================================
# Step 3: pre-market dollar volume (1,458) -- bars_sip.db when it holds ANY row for that
# (symbol, day), else data/cache.db intraday_bars_1min (the 1,928-fill cache-only population).
# ================================================================================================

def _et_minute(g, tcol):
    """Row-wise UTC -> America/New_York minute-of-day, DST-safe (each timestamp converts on its own
    calendar date, so a fall/spring DST boundary inside the base book is handled per-row)."""
    ts = pd.to_datetime(g[tcol], utc=True).dt.tz_convert(ET)
    m = (ts.dt.hour * 60 + ts.dt.minute).to_numpy()
    return g.assign(m=m)


def load_premarket_dollar_vol(symbol_days):
    """{(symbol, day): (pm_dollar_vol, source)} for 04:00-09:29 ET. Source = 'bars_sip' when
    bars_sip.db holds >=1 row for that (symbol, day) (the live default), else 'cache_db' (data/
    cache.db intraday_bars_1min, read-only URI) for the cache-only population whose bars_sip.db
    replay is empty. A (symbol, day) with NEITHER source holding any row at all is left out of the
    dict entirely (uncomputable -> NaN downstream, not a silent zero)."""
    con_sip = sqlite3.connect(f'file:{c1445.BARS_SIP_DB}?mode=ro', uri=True)
    con_cache = sqlite3.connect(f'file:{c1445.CACHE_DB}?mode=ro', uri=True)
    by_day = {}
    for sym, day in symbol_days:
        by_day.setdefault(day, set()).add(sym)
    out = {}
    n_cache_only = 0
    days_sorted = sorted(by_day)
    for di, day in enumerate(days_sorted):
        syms = sorted(by_day[day])
        ph = ','.join('?' * len(syms))
        sip = pd.read_sql(f'select symbol, t, o, h, l, c, v from bars where day=? and symbol in ({ph})',
                           con_sip, params=[day] + syms)
        sip_present = set(sip.symbol.unique())
        cache_syms = [s for s in syms if s not in sip_present]
        if cache_syms:
            n_cache_only += len(cache_syms)
            ph2 = ','.join('?' * len(cache_syms))
            cache = pd.read_sql(
                f"select symbol, timestamp as t, open as o, high as h, low as l, close as c, "
                f"volume as v from intraday_bars_1min where bar_date=? and symbol in ({ph2})",
                con_cache, params=[day] + cache_syms)
        else:
            cache = pd.DataFrame(columns=['symbol', 't', 'o', 'h', 'l', 'c', 'v'])
        for s, gg in sip.groupby('symbol'):
            b = _et_minute(gg, 't')
            pm = b[(b.m >= PM_START_M) & (b.m < PM_END_M_EXCL)]
            out[(s, day)] = (float((pm.v * pm.c).sum()), 'bars_sip')
        for s, gg in cache.groupby('symbol'):
            b = _et_minute(gg, 't')
            pm = b[(b.m >= PM_START_M) & (b.m < PM_END_M_EXCL)]
            out[(s, day)] = (float((pm.v * pm.c).sum()), 'cache_db')
        if di % 50 == 0 or di == len(days_sorted) - 1:
            log(f'load_premarket_dollar_vol: day {di + 1}/{len(days_sorted)} ({day}), '
                f'{len(out)} symbol-days resolved so far')
    con_sip.close()
    con_cache.close()
    log(f'load_premarket_dollar_vol: {n_cache_only} symbol-days sourced from cache.db '
        f'(bars_sip.db held zero rows -- the cache-only population)')
    return out


# ================================================================================================
# Step 4: news catalyst (1,461)
# ================================================================================================

def load_news(df):
    """n_articles per (symbol, day), NaN outside the nightly generator's own scanned universe
    (orb_pm_news_nightly_append.py's main(): the (symbol, day) pairs present in the LATEST ORB
    features CSV -- an ORB-scanner universe, not the HOD-break universe this book is drawn from,
    so most HOD-break symbol-days are expected to fall outside it). Returns (n_articles array,
    coverage fraction)."""
    news = pd.read_csv(NEWS_CSV, usecols=['symbol', 'day', 'n_articles'])
    news = news.drop_duplicates(subset=['symbol', 'day'])
    merged = df[['symbol', 'day']].merge(news, on=['symbol', 'day'], how='left', indicator=True)
    coverage = float((merged['_merge'] == 'both').mean())
    log(f'load_news: universe = orb_pm_news_nightly_append.py:main()\'s (symbol,day) pairs from '
        f'the latest ORB features CSV (an ORB-scanner universe) -- coverage on this HOD-break book '
        f'= {coverage:.1%}')
    return merged['n_articles'].to_numpy(), coverage


# ================================================================================================
# Cell conditions (1,458-1,462), evaluated causally
# ================================================================================================

def cell_conditions(df):
    """{cell_id: boolean Series, NaN treated as False}. 1,457's ceiling condition is separate
    (report-only, scored on its own)."""
    cond = {}
    cond['1458'] = df.pm_dollar_vol >= PM_DOLLAR_VOL_THRESHOLD
    cond['1459'] = df.atr14_pct >= ATR_PCT_THRESHOLD
    cond['1460'] = df.prior_range_pct >= PRIOR_RANGE_THRESHOLD_PCT
    cond['1461'] = df.n_articles >= 1
    cond['1462'] = (df.spread_frac <= SPREAD_BPS_THRESHOLD / 1e4) & cond['1458']
    for k in cond:
        cond[k] = cond[k].fillna(False)
    return cond


def ceiling_condition(df):
    return (df.full_day_range_pct >= CEILING_RANGE_PCT).fillna(False)


# ================================================================================================
# Scoring -- reuses cell_1445's day_clustered_t / ex_top5_mean / winner_capped_mean / weeks_spanned
# / fills_per_week / null_percentile_of / evaluate_pass_bar verbatim; score_one is a thin local copy
# so the count-matched null uses THIS study's seed (1457), not cell_1445's (1445).
# ================================================================================================

def score_one(cell_id, cond, holdout_df, holdout_name, weeks):
    """One cell x holdout row -- identical to cell_1445.score_one except null_percentile_of's seed
    (1457, per this PREREG, not cell_1445's 1445)."""
    kept = holdout_df[cond]
    dropped = holdout_df[~cond]
    n_kept, n_dropped = len(kept), len(dropped)
    kept_mean = float(kept.net_R_corr.mean()) if n_kept else np.nan
    dropped_mean = float(dropped.net_R_corr.mean()) if n_dropped else np.nan
    t_kept = c1445.day_clustered_t(kept.net_R_corr, kept.day) if n_kept else np.nan
    ex5 = c1445.ex_top5_mean(kept.net_R_corr) if n_kept else np.nan
    fwk = c1445.fills_per_week(kept, weeks) if n_kept else 0.0
    wcap = c1445.winner_capped_mean(kept.net_R_corr) if n_kept else np.nan
    flat30_mean = float(kept.net_R_corr_flat30.mean()) if n_kept else np.nan
    npct = c1445.null_percentile_of(holdout_df.net_R_corr, n_kept, kept_mean, seed=NULL_SEED,
                                     n_draws=NULL_DRAWS)
    return dict(cell=cell_id, holdout=holdout_name, n_kept=n_kept, n_dropped=n_dropped,
                kept_mean=kept_mean, dropped_mean=dropped_mean,
                delta_R=(kept_mean - dropped_mean) if n_kept and n_dropped else np.nan,
                t_kept=t_kept, ex_top5=ex5, fills_wk=fwk, winner_capped_mean=wcap,
                kept_mean_flat30=flat30_mean, null_pctile=npct, note='')


# ================================================================================================
# Cell 1,463 COST -- stop-limit reexecution on cell 1,443's tape windows
# ================================================================================================

def stop_limit_fill(symbol, day, stop, exit_m, why, fill_min, offset_bps):
    """One stop/stop_bar row's outcome under a stop-limit exit, limit = stop*(1 - offset_bps/1e4):
    filled at the bid at t0+250ms (cell 1,443's own cached measurement -- ZERO new network cost)
    if that bid >= limit; else a fresh tape re-fetch (ca.fetch_window, the SAME window cell 1,443
    used) looks for the first print after t0+250ms that is >= limit within the minute, else the
    minute's last print (the no-fill tail). Bounded by MAX_NEW_FETCHES (see module docstring).
    Returns a dict: resolved, fill_price, mechanism, slip_bps, fetched -- or resolved=False, reason.
    """
    global _new_fetch_count
    cache = c1443.load_cache(day)
    res = cache.get(c1443.row_key(symbol, exit_m, why, fill_min))
    if res is None or not res.get('measured'):
        return dict(resolved=False, reason='unmeasured_base', fetched=False)
    bid_250, t0 = res['bid_250'], res['t0']
    limit = stop * (1.0 - offset_bps / 1e4)
    if bid_250 >= limit:
        slip_bps = (stop - bid_250) / stop * 1e4
        return dict(resolved=True, fill_price=bid_250, mechanism='immediate_bid250',
                     slip_bps=slip_bps, fetched=False)
    if _new_fetch_count >= MAX_NEW_FETCHES:
        return dict(resolved=False, reason='fetch_budget_exceeded', fetched=False)
    if why == 'stop_bar':
        m = int(np.floor(fill_min)) if np.isfinite(fill_min) else int(exit_m) - 1
    else:
        m = int(exit_m)
    try:
        t, q = ca.fetch_window(symbol, day, m, m + 1)
    except Exception as e:                                        # noqa: BLE001 -- network/rate limit
        log(f'  WARNING stop_limit_fill fetch failed {symbol} {day} m={m}: {type(e).__name__}: {e}')
        return dict(resolved=False, reason=f'fetch_error:{type(e).__name__}', fetched=False)
    _new_fetch_count += 1
    if not len(t):
        return dict(resolved=False, reason='no_tape_for_reexec', fetched=True)
    t = t.sort_values('ts', kind='stable')
    after = t[t.ts > t0 + 250_000_000]
    hit = after[after.price >= limit]
    if len(hit):
        px, mech = float(hit.price.iloc[0]), 'reprint_after_limit'
    else:
        px, mech = float(t.price.iloc[-1]), 'no_fill_tail'
    slip_bps = (stop - px) / stop * 1e4
    return dict(resolved=True, fill_price=px, mechanism=mech, slip_bps=slip_bps, fetched=True)


def run_cell_1463(df):
    """Re-executes every why in {stop, stop_bar} row (the population cell 1,443 tape-measured) as a
    stop-limit under both offsets. Returns (per_row DataFrame with slip_bps/mechanism/net_R per
    variant, cost_table rows list). Unresolved rows (budget/error/unmeasured) keep the row's
    baseline net_R_corr (the current live standard), counted and disclosed, never silently dropped
    from the per-row CSV -- but excluded from the variant's own slip/no-fill-tail statistics."""
    global _new_fetch_count
    stop_rows = df[df.why.isin(['stop', 'stop_bar'])].copy()
    log(f'run_cell_1463: {len(stop_rows)} stop/stop_bar rows to re-execute per variant')
    per_variant = {}
    cost_rows = []
    for offset in STOPLIMIT_OFFSETS_BPS:
        _new_fetch_count = 0
        slip_bps_col, mech_col, net_R_col = [], [], df['net_R_corr'].to_numpy(float).copy()
        idx_map = {idx: pos for pos, idx in enumerate(df.index)}
        n_unresolved = 0
        for i, row in enumerate(stop_rows.itertuples()):
            r = stop_limit_fill(row.symbol, row.day, row.stop, row.exit_m, row.why, row.fill_min, offset)
            if r['resolved']:
                slip_R_new = row.stop * r['slip_bps'] / 1e4 / row.R
                net_R_col[idx_map[row.Index]] = row.net_R_costfix - slip_R_new
                slip_bps_col.append(r['slip_bps'])
                mech_col.append(r['mechanism'])
            else:
                n_unresolved += 1
                slip_bps_col.append(np.nan)
                mech_col.append(r['reason'])
            if i % 500 == 0 or i == len(stop_rows) - 1:
                log(f'run_cell_1463 offset={offset:.0f}bps: row {i + 1}/{len(stop_rows)}, '
                    f'{_new_fetch_count} new fetches so far, {n_unresolved} unresolved so far')
        stop_rows[f'slip_bps_1463_{offset:.0f}bps'] = slip_bps_col
        stop_rows[f'mechanism_1463_{offset:.0f}bps'] = mech_col
        book_net_R = pd.Series(net_R_col, index=df.index)
        per_variant[offset] = dict(net_R=book_net_R,
                                    slip_col=f'slip_bps_1463_{offset:.0f}bps',
                                    mech_col=f'mechanism_1463_{offset:.0f}bps',
                                    n_unresolved=n_unresolved)
        log(f'run_cell_1463 offset={offset:.0f}bps: {n_unresolved}/{len(stop_rows)} unresolved '
            f'(budget/error/unmeasured), {_new_fetch_count} new tape fetches this offset')

    base_stop_slip_bps = {}
    for hname in ('TRAIN-H2', 'VAL'):
        hdf = stop_rows[stop_rows.holdout == hname]
        base_measured = [c1445.measured_slip_bps_one(r.day, r.symbol, r.exit_m, r.why, r.fill_min)
                          for r in hdf.itertuples()]
        base_measured = [m for m in base_measured if m is not None]
        base_stop_slip_bps[hname] = float(np.mean(base_measured)) if base_measured else np.nan

    for offset in STOPLIMIT_OFFSETS_BPS:
        v = per_variant[offset]
        for hname in ('TRAIN-H2', 'VAL'):
            hmask = stop_rows.holdout == hname
            hstop = stop_rows[hmask]
            slip_s = hstop[v['slip_col']].dropna()
            nofill = hstop[hstop[v['mech_col']] == 'no_fill_tail'][v['slip_col']].dropna()
            book_mask = df.holdout == hname
            cost_rows.append(dict(
                variant=f'{offset:.0f}bps', holdout=hname,
                n_resolved=int(len(slip_s)), n_no_fill_tail=int(len(nofill)),
                n_unresolved_total=int(v['n_unresolved']),
                slip_mean=float(slip_s.mean()) if len(slip_s) else np.nan,
                slip_median=float(slip_s.median()) if len(slip_s) else np.nan,
                slip_p90=float(slip_s.quantile(.90)) if len(slip_s) else np.nan,
                no_fill_mean_slip=float(nofill.mean()) if len(nofill) else np.nan,
                base_stop_slip_mean=base_stop_slip_bps[hname],
                book_net_R=float(v['net_R'][book_mask].mean()),
                book_net_R_before=float(df.loc[book_mask, 'net_R_corr'].mean()),
            ))
    for offset in STOPLIMIT_OFFSETS_BPS:
        v = per_variant[offset]
        df[f'net_R_1463_{offset:.0f}bps'] = v['net_R']
    df = df.merge(stop_rows[['symbol', 'day', 'fill_min'] +
                             [c for c in stop_rows.columns if c.startswith('slip_bps_1463') or
                              c.startswith('mechanism_1463')]],
                   on=['symbol', 'day', 'fill_min'], how='left')
    return df, cost_rows, per_variant


def evaluate_1463_ship_bar(cost_rows):
    """Ship bar (PREREG): mean slip lower by >= 10bps AND the no-fill tail's mean slip <= 100bps, on
    BOTH holdouts, for a given variant."""
    by_variant = {}
    for r in cost_rows:
        by_variant.setdefault(r['variant'], []).append(r)
    verdict = {}
    for variant, rows in by_variant.items():
        ok = True
        for r in rows:
            lower_by = r['base_stop_slip_mean'] - r['slip_mean'] if not (
                np.isnan(r['base_stop_slip_mean']) or np.isnan(r['slip_mean'])) else np.nan
            nofill_ok = (np.isnan(r['no_fill_mean_slip']) or r['no_fill_mean_slip'] <= 100.0)
            if np.isnan(lower_by) or lower_by < 10.0 or not nofill_ok:
                ok = False
        verdict[variant] = ok
    return verdict


# ================================================================================================
# Cell 1,464 COST -- R floor at 2.5% of price, cell_1440's paired re-walk (floor only, no cap)
# ================================================================================================

def run_cell_1464(df):
    """R floor 2.5% of price via cell_1440.build_variant (new_stop_distance floors d_old at
    R_FLOOR_PCT, no cap) -- new_stop = min(consolidation low, fill*0.975) exactly (d_new =
    max(d_old, 0.025) <=> fill*(1-d_new) = min(fill*(1-d_old), fill*0.975) = min(stop, fill*0.975)).
    Cost dollar is held fixed at THIS study's corrected half_entry (not 1,438's double-counted
    cost_R) so cost_R_new = half_entry/R_new. Measured slip is RE-SCALED onto R_new using the SAME
    tape-measured bps as the row's original stop exit (cell 1,443's cache) -- the physical execution
    friction in price terms is assumed unchanged by the R floor, only its R-denominator changes;
    unmeasured rows use the pooled stop fallback (35bps); rows that re-walk to a 'target' exit carry
    zero slip, matching the base convention. Rows the floor doesn't bind (changed=False) keep
    net_R_corr exactly, ΔR=0 by construction."""
    base = df.copy()
    base['d_old'] = (base.fill - base.stop) / base.fill
    assert np.allclose(base.R, base.fill - base.stop), 'base R != fill - stop'
    old_floor = c1440.FLOOR_PCT
    c1440.FLOOR_PCT = R_FLOOR_PCT
    try:
        cell = c1440.build_variant(base.reset_index(drop=True), cap_pct=None)
    finally:
        c1440.FLOOR_PCT = old_floor
    cell = cell.reset_index(drop=True)
    base_r = base.reset_index(drop=True)

    net_R_1464 = base_r['net_R_corr'].to_numpy(float).copy()
    slip_bps_1464 = np.full(len(base_r), np.nan)
    for i in range(len(base_r)):
        if not cell.loc[i, 'changed']:
            continue
        row = base_r.iloc[i]
        R_new = float(cell.loc[i, 'R'])
        why_new = cell.loc[i, 'why']
        raw_R_new = float(cell.loc[i, 'raw_R'])
        cost_R_new = row.half_entry / R_new
        if why_new in ('stop', 'stop_infill'):
            m = c1445.measured_slip_bps_one(row.day, row.symbol, row.exit_m, row.why, row.fill_min)
            bps = m if m is not None else c1445.FLAT_SLIP_FALLBACK_BPS
            slip_bps_1464[i] = bps
            exit_price_new = float(cell.loc[i, 'exit_price'])
            slip_R_new = exit_price_new * bps / 1e4 / R_new
        else:
            slip_R_new = 0.0
        net_R_1464[i] = (raw_R_new - cost_R_new) - slip_R_new
    base_r['net_R_1464'] = net_R_1464
    base_r['slip_bps_1464'] = slip_bps_1464
    base_r['changed_1464'] = cell['changed'].to_numpy()

    weeks = {h: c1445.weeks_spanned(base_r.loc[base_r.holdout == h, 'day']) for h in ('TRAIN-H2', 'VAL')}
    rows = []
    for hname in ('TRAIN-H2', 'VAL'):
        hb = base_r[base_r.holdout == hname]
        delta = hb['net_R_1464'].to_numpy(float) - hb['net_R_corr'].to_numpy(float)
        t_delta = c1445.day_clustered_t(pd.Series(delta, index=hb.index), hb.day)
        fwk = c1445.fills_per_week(hb.rename(columns={'fill_min': 'entry_m'}), weeks[hname])
        rows.append(dict(cell='1464', holdout=hname, n_kept=len(hb), n_dropped=0,
                          kept_mean=float(hb['net_R_1464'].mean()),
                          dropped_mean=float(hb['net_R_corr'].mean()),
                          delta_R=float(np.mean(delta)), t_kept=t_delta,
                          ex_top5=c1445.ex_top5_mean(hb['net_R_1464']),
                          fills_wk=fwk, winner_capped_mean=c1445.winner_capped_mean(hb['net_R_1464']),
                          kept_mean_flat30=np.nan, null_pctile=np.nan,
                          changed_pct=float(hb['changed_1464'].mean() * 100), note=''))
    val_row = next(r for r in rows if r['holdout'] == 'VAL')
    th2_row = next(r for r in rows if r['holdout'] == 'TRAIN-H2')
    passes = (val_row['delta_R'] >= 0.05 and th2_row['delta_R'] >= 0.05 and val_row['t_kept'] >= 2.5)
    for r in rows:
        r['passes_bar'] = passes
    return base_r, rows


# ================================================================================================
# Report
# ================================================================================================

def write_features_csv(df, cond, ceiling_cond, path):
    out = df[['day', 'symbol', 'fill_min', 'holdout']].copy().rename(columns={'holdout': 'split'})
    out['flag_1457'] = ceiling_cond.astype(int)
    for cell_id in ENTRY_CELLS:
        raw = df.pipe(lambda d: cond[cell_id])
        out[f'flag_{cell_id}'] = raw.astype(float)
    # overlay coverage-aware NaN where the underlying source feature is uncomputable
    out.loc[df.pm_dollar_vol.isna(), 'flag_1458'] = np.nan
    out.loc[df.atr14_pct.isna(), 'flag_1459'] = np.nan
    out.loc[df.prior_range_pct.isna(), 'flag_1460'] = np.nan
    out.loc[df.n_articles.isna(), 'flag_1461'] = np.nan
    out.loc[df.spread_frac.isna() | df.pm_dollar_vol.isna(), 'flag_1462'] = np.nan
    out['net_R_corr_v2'] = df['net_R_corr']
    out['net_R_corr_flat30'] = df['net_R_corr_flat30']
    for offset in STOPLIMIT_OFFSETS_BPS:
        scol = f'slip_bps_1463_{offset:.0f}bps'
        if scol in df.columns:
            out[scol] = df[scol]
    if 'net_R_1464' in df.columns:
        out['net_R_1464'] = df['net_R_1464']
        out['slip_bps_1464'] = df['slip_bps_1464']
    out.to_csv(path, index=False)
    log(f'wrote {path} ({len(out)} rows)')


def write_result_md(all_rows, cost_rows, ship_1463, coverage, kill_switch, ceiling_rows, path):
    cols = ['cell', 'holdout', 'n_kept', 'n_dropped', 'kept_mean', 'dropped_mean', 'delta_R',
            't_kept', 'ex_top5', 'fills_wk', 'winner_capped_mean', 'kept_mean_flat30',
            'null_pctile', 'passes_bar', 'note']
    lines = ['# RESULT 1,457-1,465 -- perfect-foresight ceiling, causal big-day predictors, cost\n']
    header = '| ' + ' | '.join(cols) + ' |'
    sep = '|' + '---|' * len(cols)
    lines += [header, sep]
    for r in ceiling_rows + all_rows:
        vals = []
        for c in cols:
            v = r.get(c)
            if isinstance(v, float):
                vals.append(f'{v:.4f}' if not np.isnan(v) else 'NaN')
            else:
                vals.append(str(v))
        lines.append('| ' + ' | '.join(vals) + ' |')
    lines.append('')
    lines.append('## 1,463 cost table (stop-limit reexecution, bps unless noted)\n')
    cc = ['variant', 'holdout', 'n_resolved', 'n_no_fill_tail', 'n_unresolved_total', 'slip_mean',
          'slip_median', 'slip_p90', 'no_fill_mean_slip', 'base_stop_slip_mean', 'book_net_R',
          'book_net_R_before']
    lines.append('| ' + ' | '.join(cc) + ' |')
    lines.append('|' + '---|' * len(cc))
    for r in cost_rows:
        vals = [f'{r[c]:.3f}' if isinstance(r[c], float) and not np.isnan(r[c]) else
                ('NaN' if isinstance(r[c], float) else str(r[c])) for c in cc]
        lines.append('| ' + ' | '.join(vals) + ' |')
    lines.append('')
    lines.append(f'1,463 ship bar (mean slip lower by >=10bps AND no-fill-tail mean slip <=100bps, '
                 f'both holdouts): {ship_1463}')
    lines.append('')
    cov_str = ', '.join(f'{k}={v:.0%}' for k, v in coverage.items())
    lines.append(f'Coverage: {cov_str}.')
    lines.append(f'Kill switch (1,457 VAL kept mean < +0.15): {"TRIGGERED" if kill_switch else "not triggered"}.')
    lines.append('Caveats: TEST is sealed and was not read. 1,463 re-executes the cached bid-250ms '
                 'measurement with zero new network cost when it already resolves the fill; the '
                 'remainder needed a fresh Alpaca SIP tape re-fetch (cell 1,443\'s own fetch_window), '
                 f'bounded at {MAX_NEW_FETCHES} new fetches per variant per this run\'s time/token '
                 'budget -- unresolved rows keep the baseline net_R_corr and are counted, never '
                 'silently dropped (see n_unresolved_total). 1,461\'s universe is the ORB scanner\'s '
                 f'own scanned symbol-days ({NEWS_GENERATOR}:main()), not the HOD-break universe -- '
                 'low coverage is expected and reported, not treated as a defect. 1,464 rescales the '
                 'ORIGINAL tape-measured slip bps onto the new R rather than re-measuring the tape at '
                 'the new stop price (no new fetch); this is an approximation, disclosed here.')
    with open(path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    log(f'wrote {path}')


# ================================================================================================
# Main
# ================================================================================================

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dry-run', action='store_true',
                     help='score a fixed 300-row sample (seed 1457) instead of the full book, '
                          'tighter 1,463 fetch budget')
    args = ap.parse_args(argv)
    global MAX_NEW_FETCHES
    if args.dry_run:
        MAX_NEW_FETCHES = min(MAX_NEW_FETCHES, 30)

    log('loading base book (causal_arming_causal.csv, status==fill, TRAIN-H2+VAL)')
    fills = c1445.load_base_book()
    if args.dry_run:
        fills = fills.sample(n=min(300, len(fills)), random_state=1457).reset_index(drop=True)
        log(f'--dry-run: scoring a {len(fills)}-row sample')

    log('Step 1: corrected cost (1,445 standard + EOD-fallback fix)')
    df, eod_fb_bps = build_base_cost(fills)

    log('Step 2: Databento daily panel (ceiling, ATR14, prior-day range)')
    df = attach_daily_ext(df)

    log('Step 3: pre-market dollar volume')
    pm = load_premarket_dollar_vol(list(zip(df.symbol, df.day)))
    df['pm_dollar_vol'] = [pm.get((s, d), (np.nan, None))[0] for s, d in zip(df.symbol, df.day)]
    df['pm_source'] = [pm.get((s, d), (np.nan, None))[1] for s, d in zip(df.symbol, df.day)]

    log('Step 4: news catalyst')
    df['n_articles'], news_coverage = load_news(df)

    coverage = dict(
        pm_dollar_vol=float(df.pm_dollar_vol.notna().mean()),
        atr14_pct=float(df.atr14_pct.notna().mean()),
        prior_range_pct=float(df.prior_range_pct.notna().mean()),
        n_articles=news_coverage,
        full_day_range_pct=float(df.full_day_range_pct.notna().mean()),
    )
    for k, v in coverage.items():
        log(f'  coverage {k}: {v:.1%}')

    holdouts = {name: df[df.holdout == name] for name in ('TRAIN-H2', 'VAL')}
    weeks = {name: c1445.weeks_spanned(holdouts[name].day) for name in holdouts}

    log('Step 5: cell 1,457 CEILING')
    ceiling_cond = ceiling_condition(df)
    ceiling_rows = [dict(score_one('1457', ceiling_cond.loc[holdouts[h].index], holdouts[h], h,
                                    weeks[h]), passes_bar=False) for h in ('TRAIN-H2', 'VAL')]
    val_ceiling = next(r for r in ceiling_rows if r['holdout'] == 'VAL')
    kill_switch = not (val_ceiling['n_kept'] and val_ceiling['kept_mean'] >= KILL_VAL_THRESHOLD)
    log(f'KILL SWITCH: ceiling VAL kept mean = {val_ceiling["kept_mean"]:.4f} '
        f'(n={val_ceiling["n_kept"]}) vs +{KILL_VAL_THRESHOLD} -> '
        f'{"TRIGGERED, closing every entry filter" if kill_switch else "not triggered"}')

    log('Step 6: cells 1,458-1,462')
    cond = cell_conditions(df)
    rows_by_cell = {}
    for cell_id in ENTRY_CELLS:
        rows_by_cell[cell_id] = {h: score_one(cell_id, cond[cell_id].loc[holdouts[h].index],
                                               holdouts[h], h, weeks[h]) for h in holdouts}
    verdict = c1445.evaluate_pass_bar(rows_by_cell)

    log('Step 6b: cell 1,465 JOINT (component chosen on TRAIN-H2 kept mean only)')
    th2_means = {c: rows_by_cell[c]['TRAIN-H2']['kept_mean'] for c in ENTRY_CELLS}
    best_pred = max(ENTRY_CELLS, key=lambda c: th2_means[c] if not np.isnan(th2_means[c]) else -1e9)
    log(f'1,465 JOINT: best entry predictor on TRAIN-H2 = {best_pred} '
        f'(kept mean {th2_means[best_pred]:.4f})')

    log('Step 7: cell 1,463 COST (stop-limit reexecution)')
    df, cost_rows, per_variant_1463 = run_cell_1463(df)
    ship_1463 = evaluate_1463_ship_bar(cost_rows)
    log(f'1,463 ship bar: {ship_1463}')
    better_variant = None
    for offset in STOPLIMIT_OFFSETS_BPS:
        book_r = next(r['book_net_R'] for r in cost_rows
                      if r['variant'] == f'{offset:.0f}bps' and r['holdout'] == 'TRAIN-H2')
        if better_variant is None or book_r > better_variant[1]:
            better_variant = (offset, book_r)
    log(f'1,465 JOINT: better 1,463 variant on TRAIN-H2 book net R = {better_variant[0]:.0f}bps')

    log('Step 8: cell 1,464 COST (R floor 2.5%, cell_1440 machinery)')
    df, rows_1464 = run_cell_1464(df)

    log('Step 9: cell 1,465 JOINT scoring (one VAL read)')
    joint_cond = cond[best_pred]
    net_R_joint_col = f'net_R_1463_{better_variant[0]:.0f}bps'
    df_joint = df.copy()
    df_joint['net_R_corr'] = df_joint[net_R_joint_col]
    rows_1465 = [score_one('1465', joint_cond.loc[holdouts[h].index], df_joint[df_joint.holdout == h],
                            h, weeks[h]) for h in ('TRAIN-H2', 'VAL')]

    all_rows = []
    for cell_id in ENTRY_CELLS:
        for h in ('TRAIN-H2', 'VAL'):
            r = dict(rows_by_cell[cell_id][h])
            r['passes_bar'] = bool(verdict.get(cell_id, False))
            if kill_switch:
                r['passes_bar'] = False
                r['note'] = 'kill switch'
            all_rows.append(r)
    for r in rows_1465:
        r['passes_bar'] = False
        r['note'] = 'kill switch' if kill_switch else 'one VAL read, entry component ' + best_pred
    all_rows += rows_1465
    for r in rows_1464:
        if kill_switch:
            r['passes_bar'] = False
            r['note'] = 'kill switch'
    all_rows += rows_1464

    feat_suffix = '_DRYRUN' if args.dry_run else ''
    feat_path = os.path.join(HERE, f'cell_1457_features{feat_suffix}.csv')
    result_path = os.path.join(HERE, f'RESULT_1457{feat_suffix}.md')
    write_features_csv(df, cond, ceiling_cond, feat_path)
    write_result_md(all_rows, cost_rows, ship_1463, coverage, kill_switch, ceiling_rows, result_path)

    log('done')
    return dict(ceiling_rows=ceiling_rows, all_rows=all_rows, cost_rows=cost_rows,
                kill_switch=kill_switch, ship_1463=ship_1463, coverage=coverage)


if __name__ == '__main__':
    main()
