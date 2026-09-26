#!/usr/bin/env python3
"""Build FEATURE SET A for cells 1,478-1,480 (PREREG_1478.md + the 2026-09-26 amendment).

Feature Set A = the stock's own state at the close of arm bar j (PREREG items 1, 2, 5).
Every bar-derived quantity is recomputed from ONE fresh store, `bars_fills_1478.db`
(Alpaca SIP 1-minute bars fetched per fill symbol-day) -- no cache.db or bars_sip.db bar
enters any feature (the amendment's leak fix: store identity was the look-ahead cohort in
cell 1,438). The two decoy columns that measure exactly that leak (store_served_1438,
rth_bar_count_1438) are the only place this script reads cache.db / bars_sip.db, and only
for row COUNTS, read-only, batched per day (never a per-bar payload).

Reuses research/hod_entry/cell_1445.py and cell_1457.py for the non-bar joins (float
snapshot, prior-day high / 20-session high, ATR14%, prior-day range%) -- same causal
shift(1) daily panel already built and reviewed for those cells; recomputing it here from
scratch would be a second, divergent implementation of an already-passing join.

Output: research/hod_entry/features_1478_A.csv, research/hod_entry/FEATURES_A.md.
"""
from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
import time

import numpy as np
import pandas as pd

REPO = '/home/ec2-user/onemil'
sys.path.insert(0, REPO)

from research.hod_entry import cell_1445 as c1445          # noqa: E402
from research.hod_entry import cell_1457 as c1457          # noqa: E402
from trading.hod_break import HodBreakParams, rv_profile    # noqa: E402

BARS_DB = os.path.join(REPO, 'research/hod_entry/bars_fills_1478.db')
BASE_CSV = os.path.join(REPO, 'research/hod_entry/causal_arming_causal.csv')
UNIVERSE_CSV = os.path.join(REPO, 'research/bf_zero/universe.csv')
CELL_C_CSV = os.path.join(REPO, 'research/hod_entry/features_1478_C.csv')
BARS_SIP_DB = os.path.join(REPO, 'research/bf_zero/bars_sip.db')
CACHE_DB = os.path.join(REPO, 'data/cache.db')
OUT_CSV = os.path.join(REPO, 'research/hod_entry/features_1478_A.csv')
OUT_MD = os.path.join(REPO, 'research/hod_entry/FEATURES_A.md')
LOG_FILE = os.path.join(REPO, 'research/hod_entry/build_features_1478_A.log')

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                     handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()])
log = logging.getLogger(__name__).info

RTH_OPEN_M = 570            # 09:30 ET, minutes since midnight -- matches trading/hod_break.OPEN_MINUTE
RTH_CLOSE_M = 959           # 15:59 ET
PM_OPEN_M = 240             # 04:00 ET
CONSOL_K = HodBreakParams().consol_bars   # 5 -- the live rule's own K, reused for item 2's "last K bars"
LEVEL_TOUCH_PCT = 0.002     # 0.2%, PREREG item 2's touch band

# Amendment 3 (2026-09-26, PREREG_1478.md): the original arm-bar rule used `rth.m < fill_min` with
# a FRACTIONAL fill_min (minute + seconds) -- since every bar's m is an integer minute, m < fill_min
# includes the fill bar itself in 98.4% of rows (a look-ahead: bar j was the FILL bar, not the bar
# before it). ARM_BAR_CLOSED=True selects bar j = the last RTH bar with m <= floor(fill_min) - 1,
# i.e. the last bar FULLY CLOSED before the fill minute. Default False preserves the original
# (leaky) behaviour so nothing else that imports this module changes; only build_features_1478_A_v2
# (main() with --arm-bar-closed) flips it.
ARM_BAR_CLOSED = False


# ================================================================================================
# Loaders
# ================================================================================================

def load_fills():
    """The 9,911-fill base book, status=='fill' rows only."""
    df = pd.read_csv(BASE_CSV, low_memory=False)
    f = df[df.status == 'fill'].reset_index(drop=True).copy()
    log(f'load_fills: {len(f)} status==fill rows from {BASE_CSV}')
    return f


def load_bars_from_store():
    """{(symbol, day): DataFrame(m,o,h,l,c,v)} for EVERY bar in bars_fills_1478.db (04:00-20:00 ET
    fetch window), ET minute-of-day attached. One full-table read (the store was built for exactly
    this 9,911-row job, ~4.46M rows -- small enough to hold in memory once)."""
    con = sqlite3.connect(f'file:{BARS_DB}?mode=ro', uri=True)
    df = pd.read_sql('select symbol, day, t, o, h, l, c, v from bars', con)
    con.close()
    log(f'load_bars_from_store: {len(df)} rows loaded from bars_fills_1478.db')
    ts = pd.to_datetime(df['t'], utc=True).dt.tz_convert('America/New_York')
    df['m'] = ts.dt.hour * 60 + ts.dt.minute
    date_et = ts.dt.strftime('%Y-%m-%d')
    bad = int((date_et != df['day']).sum())
    if bad:
        log(f'WARNING load_bars_from_store: {bad} bars have ET calendar date != stored day column '
            f'-- dropped (guards the 04:00-20:00 ET fetch window against any date bleed)')
        df = df[date_et == df['day']]
    df = df.sort_values(['symbol', 'day', 'm']).reset_index(drop=True)
    groups = {}
    for (sym, day), g in df.groupby(['symbol', 'day'], sort=False):
        groups[(sym, day)] = g[['m', 'o', 'h', 'l', 'c', 'v']].reset_index(drop=True)
    log(f'load_bars_from_store: {len(groups)} (symbol,day) groups')
    return groups


def batched_bar_counts(db_path, table, symbol_col, day_col, symbol_days, rth_only=True):
    """{(symbol, day): row_count} for exactly the requested (symbol, day) pairs, one SQL query per
    day (day/bar_date is indexed on both stores) -- never a COUNT(*) full-table scan (both stores
    are 14-17 GB; a bare COUNT(*) was timed out and killed during this build's own reconnaissance).
    rth_only restricts to 09:30-15:59 ET by string-matching the UTC hour (both stores' timestamps
    are UTC ISO; ET RTH = UTC 13:30-19:59 on a standard session -- DST is not resolved here, so this
    count is a coarse control feature for the decoy model only, never a modelling input; disclosed
    in FEATURES_A.md)."""
    con = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
    by_day = {}
    for sym, day in symbol_days:
        by_day.setdefault(day, set()).add(sym)
    out = {}
    days_sorted = sorted(by_day)
    for di, day in enumerate(days_sorted):
        syms = sorted(by_day[day])
        ph = ','.join('?' * len(syms))
        rth_clause = " AND substr(t,12,2) BETWEEN '13' AND '19'" if rth_only and table == 'bars' else \
                     (" AND substr(timestamp,12,2) BETWEEN '13' AND '19'" if rth_only else '')
        tcol = 't' if table == 'bars' else 'timestamp'
        q = (f"select {symbol_col} as symbol, count(*) as n from {table} where {day_col}=? "
             f"and {symbol_col} in ({ph}){rth_clause} group by {symbol_col}")
        g = pd.read_sql(q, con, params=[day] + syms)
        for row in g.itertuples():
            out[(row.symbol, day)] = int(row.n)
        for sym in syms:
            out.setdefault((sym, day), 0)
        if di % 50 == 0 or di == len(days_sorted) - 1:
            log(f'batched_bar_counts[{table}]: day {di + 1}/{len(days_sorted)} ({day})')
    con.close()
    return out


# ================================================================================================
# Bar-derived features (Feature Set A items 1 + 2's bar parts) -- bars_fills_1478.db ONLY
# ================================================================================================

def bar_features_for_fill(bars, fill_min, level, arm_bar_closed=None):
    """bars: (m,o,h,l,c,v) for ONE (symbol,day) across the whole 04:00-20:00 ET fetch window.
    fill_min: ET minute of the fill bar. arm_bar_closed (defaults to the module-level
    ARM_BAR_CLOSED) selects the rule:
      False (original, same convention as cell_1445.arm_bar_features): arm bar j = the last RTH
        bar with m < fill_min. Since fill_min is FRACTIONAL (minute + seconds) and every bar's m
        is an integer minute, this includes the fill bar itself in ~98% of rows -- a look-ahead
        (Amendment 3, PREREG_1478.md).
      True (corrected): arm bar j = the last RTH bar with m <= floor(fill_min) - 1, i.e. the last
        bar FULLY CLOSED before the fill minute.
    Uses only bars 0..j -- never bar j+1 or later. Returns {} (all-NaN row, counted by the caller)
    if no RTH bar exists before fill_min under the selected rule."""
    if arm_bar_closed is None:
        arm_bar_closed = ARM_BAR_CLOSED
    rth = bars[(bars.m >= RTH_OPEN_M) & (bars.m <= RTH_CLOSE_M)].reset_index(drop=True)
    if arm_bar_closed:
        cutoff_m = int(np.floor(fill_min)) - 1
        b = rth[rth.m <= cutoff_m].reset_index(drop=True)
    else:
        b = rth[rth.m < fill_min].reset_index(drop=True)
    if not len(b):
        return {}
    j = len(b) - 1
    j_m = int(b.m.iloc[j])
    h = b.h.to_numpy(float); l = b.l.to_numpy(float); c = b.c.to_numpy(float)
    v = b.v.to_numpy(float); o = b.o.to_numpy(float); m = b.m.to_numpy(int)

    out = {}
    close_j = float(c[j])
    running_high_j = float(h[:j + 1].max())
    running_low_j = float(l[:j + 1].min())
    out['range_to_j_pct'] = (running_high_j - running_low_j) / running_low_j * 100 if running_low_j > 0 else np.nan
    out['dollar_vol_to_j'] = float((v[:j + 1] * c[:j + 1]).sum())
    n_minutes_possible = j_m - RTH_OPEN_M + 1
    out['bar_density_j'] = len(b) / n_minutes_possible if n_minutes_possible > 0 else np.nan
    cum_v = float(v[:j + 1].sum())
    out['cum_volume_j'] = cum_v
    out['dist_from_open_pct'] = (close_j - float(o[0])) / float(o[0]) * 100 if o[0] > 0 else np.nan
    out['close_j'] = close_j
    out['n_bars_j'] = len(b)
    out['arm_m'] = j_m

    # arm_index: bars i in [0, j-1] whose bar i+1's high reached (running HOD through i) + 0.01.
    # i+1 <= j always -- both bars used are known at the close of bar j.
    arm_idx = 0
    running_hod = float(h[0])
    for i in range(0, j):
        if i > 0:
            running_hod = max(running_hod, float(h[i]))
        if float(h[i + 1]) >= running_hod + 0.01:
            arm_idx += 1
    out['arm_index'] = arm_idx

    pm = bars[(bars.m >= PM_OPEN_M) & (bars.m < RTH_OPEN_M)]
    out['pm_dollar_vol'] = float((pm.v * pm.c).sum()) if len(pm) else 0.0

    # ---- item 2's bar-derived sub-features (K = HodBreakParams().consol_bars = 5) ----
    K = min(CONSOL_K, j + 1)
    last_k_v = v[j - K + 1: j + 1]
    if K >= 2:
        out['consol_vol_slope'] = float(np.polyfit(np.arange(K, dtype=float), last_k_v, 1)[0])
    else:
        out['consol_vol_slope'] = np.nan
    mean_all = float(v[:j + 1].mean())
    out['consol_vol_ratio'] = (float(last_k_v.mean()) / mean_all) if mean_all > 0 else np.nan

    last_k_l = l[j - K + 1: j + 1]
    out['higher_lows_count'] = int(np.sum(np.diff(last_k_l) > 0)) if len(last_k_l) >= 2 else np.nan

    if level is not None and not (isinstance(level, float) and np.isnan(level)):
        touch = (l[:j + 1] <= level * (1 + LEVEL_TOUCH_PCT)) & (h[:j + 1] >= level * (1 - LEVEL_TOUCH_PCT))
        out['level_touches'] = int(touch.sum())
    else:
        out['level_touches'] = np.nan

    if j + 1 > K:
        pre_high = float(h[:j - K + 1].max())
        consol_low = float(l[j - K + 1: j + 1].min())
        out['pullback_depth_pct'] = (pre_high - consol_low) / pre_high * 100 if pre_high > 0 else np.nan
    else:
        out['pullback_depth_pct'] = np.nan

    typical = (h[:j + 1] + l[:j + 1] + c[:j + 1]) / 3.0
    cum_pv = np.cumsum(typical * v[:j + 1])
    cum_vv = np.cumsum(v[:j + 1])
    with np.errstate(invalid='ignore', divide='ignore'):
        vwap_series = np.where(cum_vv > 0, cum_pv / np.where(cum_vv == 0, np.nan, cum_vv), np.nan)
    vwap_j = float(vwap_series[j])
    out['vwap_dist_pct'] = (close_j - vwap_j) / vwap_j * 100 if not np.isnan(vwap_j) and vwap_j != 0 else np.nan
    n15 = min(15, j + 1)
    if n15 >= 2:
        vwap_tail = vwap_series[j - n15 + 1: j + 1]
        if not np.any(np.isnan(vwap_tail)):
            out['vwap_slope'] = float(np.polyfit(np.arange(n15, dtype=float), vwap_tail, 1)[0])
        else:
            out['vwap_slope'] = np.nan
    else:
        out['vwap_slope'] = np.nan

    gaps = np.diff(m)
    out['halt_proxy'] = int(np.any(gaps >= 5)) if len(gaps) else 0
    return out


# ================================================================================================
# PIT daily parquet: prior-day volume/rvol and the symbol-persistence proxy (item 5)
# ================================================================================================

def build_daily_volume_panel(instrument_ids):
    """Causal prior-day volume / ADV20-of-the-prior-day, and the item-5 persistence indicator,
    from the same PIT daily parquet as cell_1445.build_daily_panel -- every join is a shift(1) or
    later, so nothing from the signal day (or later) ever enters a row."""
    df = pd.read_parquet(c1445.DAILY_PARQUET,
                          columns=['bar_date', 'symbol', 'instrument_id', 'open', 'high', 'low',
                                   'close', 'volume'])
    df = df[df.instrument_id.isin(instrument_ids)].copy()
    df['bar_date'] = pd.to_datetime(df['bar_date'])
    df = df.sort_values(['instrument_id', 'bar_date']).reset_index(drop=True)
    g = df.groupby('instrument_id', sort=False)
    df['pd_volume'] = g['volume'].shift(1)                                            # prior session's volume
    df['pd_adv20'] = g['volume'].transform(lambda s: s.shift(2).rolling(20, min_periods=20).mean())
    df['pd_rvol'] = df['pd_volume'] / df['pd_adv20']

    # item 5 proxy: share of the PRIOR 60 sessions (strictly before the signal day) where
    # (high-open)/open >= 5% AND close > open*1.05 -- a daily-bar proxy for the PREREG's intraday
    # "≥5% above open at 11:00 and closed above that price" persistence rule (11:00 intraday state
    # is not in the daily parquet; disclosed deviation, FEATURES_A.md).
    big_day = ((df['high'] - df['open']) / df['open'] >= 0.05) & (df['close'] > df['open'] * 1.05)
    df['big_day_ind'] = big_day.astype(float)
    df['symbol_persistence'] = df.groupby('instrument_id', sort=False)['big_day_ind'].transform(
        lambda s: s.shift(1).rolling(60, min_periods=10).mean())
    return df


def attach_volume_panel(df, instr_by_sd):
    daily = build_daily_volume_panel(set(instr_by_sd.values()))
    idx = daily.set_index(['instrument_id', 'bar_date'])
    cols = ['pd_volume', 'pd_rvol', 'symbol_persistence']
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
    return pd.DataFrame(recs, index=df.index)


# ================================================================================================
# Main
# ================================================================================================

def main(arm_bar_closed=None, out_csv=None):
    if arm_bar_closed is None:
        arm_bar_closed = ARM_BAR_CLOSED
    if out_csv is None:
        out_csv = OUT_CSV
    log(f'main: arm_bar_closed={arm_bar_closed}, out_csv={out_csv}')
    t_start = time.time()
    fills = load_fills()
    n = len(fills)

    # ---------- bar-derived features, fresh from bars_fills_1478.db ONLY ----------
    bar_groups = load_bars_from_store()
    recs = []
    n_missing = 0
    for i, r in enumerate(fills.itertuples()):
        bars = bar_groups.get((r.symbol, r.day))
        feat = bar_features_for_fill(bars, r.fill_min, r.level, arm_bar_closed=arm_bar_closed) \
            if bars is not None else {}
        if not feat:
            n_missing += 1
        recs.append(feat)
        if (i + 1) % 2000 == 0:
            log(f'bar features: {i + 1}/{n} fills done ({time.time() - t_start:.0f}s elapsed)')
    bar_df = pd.DataFrame(recs)
    log(f'bar features: {n_missing}/{n} fills had NO RTH bar before fill_min in bars_fills_1478.db '
        f'(all bar-derived columns NaN for those -- {"WARNING" if n_missing else "none"})')
    out = pd.concat([fills.reset_index(drop=True), bar_df], axis=1)

    # rv_j needs ADV20 -- research/bf_zero/universe.csv, keyed by (symbol, bar_date)
    uni = pd.read_csv(UNIVERSE_CSV, usecols=['symbol', 'bar_date', 'adv20'])
    uni = uni.drop_duplicates(subset=['symbol', 'bar_date'])
    uni_lookup = uni.set_index(['symbol', 'bar_date'])['adv20'].to_dict()
    out['adv20'] = [uni_lookup.get((s, d), np.nan) for s, d in zip(out.symbol, out.day)]
    out['rv_j'] = [rv_profile(cv, a, mm) if pd.notna(a) else np.nan
                   for cv, a, mm in zip(out.cum_volume_j, out.adv20, out.arm_m)]
    log(f'rv_j: adv20 join coverage {out.adv20.notna().mean():.1%}')

    # ---------- item 1 non-bar features: float, level vs prior-day high / 20-session high, gap ----------
    map_df = c1445.load_symbol_map()
    instr_by_sd = c1445.resolve_instrument_ids(list(zip(out.symbol, out.day)), map_df)
    n_resolved = sum(1 for sd in zip(out.symbol, out.day) if sd in instr_by_sd)
    log(f'instrument id resolved for {n_resolved}/{len(out)} fills')

    daily = c1445.build_daily_panel(set(instr_by_sd.values()))
    didx = daily.set_index(['instrument_id', 'bar_date'])
    prev_close, prev_high, high20 = [], [], []
    for r in out.itertuples():
        iid = instr_by_sd.get((r.symbol, r.day))
        day_ts = pd.Timestamp(r.day)
        if iid is not None and (iid, day_ts) in didx.index:
            row = didx.loc[(iid, day_ts)]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            prev_close.append(row['prev_close']); prev_high.append(row['prev_high']); high20.append(row['high20'])
        else:
            prev_close.append(np.nan); prev_high.append(np.nan); high20.append(np.nan)
    out['prev_close'] = prev_close
    out['prev_high'] = prev_high
    out['high20'] = high20
    out['gap_vs_prior_close_pct'] = (out.level - out.prev_close) / out.prev_close * 100
    out['level_vs_prior_high_pct'] = (out.level - out.prev_high) / out.prev_high * 100
    out['level_vs_high20_pct'] = (out.level - out.high20) / out.high20 * 100

    ext = c1457.attach_daily_ext(out[['day', 'symbol']].copy())
    out['atr14_pct'] = ext['atr14_pct'].to_numpy()
    out['prior_range_pct'] = ext['prior_range_pct'].to_numpy()

    float_lookup = c1445.load_float_lookup()
    out['float_shares'] = [float_lookup.get(s, np.nan) for s in out.symbol]

    nbbo_lookup = c1445.load_nbbo_lookup()
    half_entry, net_R_costfix, nbbo_fb = c1445.corrected_cost(out, nbbo_lookup)
    out['half_entry'] = half_entry
    out['spread_frac_at_fill'] = 2.0 * out.half_entry / out.fill
    # ask_distance: the WEEKEND_QUEUE definition (pre-trigger NBBO ask/level - 1) needs a quote feed
    # this task has no access to (Alpaca minute BARS only, per the task's own data-source list) --
    # proxy'd by the realized fill's distance from the level, disclosed as a deviation below.
    out['ask_distance_proxy_pct'] = (out.fill - out.level) / out.level * 100

    out['time_of_day_min'] = out.fill_min
    out['R_pct'] = out.R / out.fill * 100
    out['dow'] = pd.to_datetime(out.day).dt.dayofweek

    # ---------- item 2 non-bar feature: prior-day volume/rvol; item 5: symbol persistence ----------
    vol_panel = attach_volume_panel(out[['day', 'symbol']].copy(), instr_by_sd)
    out['prev_day_volume'] = vol_panel['pd_volume'].to_numpy()
    out['prev_day_rvol'] = vol_panel['pd_rvol'].to_numpy()
    out['symbol_persistence'] = vol_panel['symbol_persistence'].to_numpy()

    # ---------- decoy columns for the metadata-only leak model ----------
    sd_pairs = list(zip(out.symbol, out.day))
    sip_counts = batched_bar_counts(BARS_SIP_DB, 'bars', 'symbol', 'day', sd_pairs, rth_only=True)
    cache_counts = batched_bar_counts(CACHE_DB, 'intraday_bars_1min', 'symbol', 'bar_date', sd_pairs, rth_only=True)
    store_served_1438 = []
    rth_bar_count_1438 = []
    for sd in sd_pairs:
        sip_n = sip_counts.get(sd, 0)
        served = 1 if sip_n == 0 else 0
        store_served_1438.append(served)
        rth_bar_count_1438.append(cache_counts.get(sd, 0) if served else sip_n)
    out['store_served_1438'] = store_served_1438
    out['rth_bar_count_1438'] = rth_bar_count_1438
    log(f'decoy: store_served_1438 mean {np.mean(store_served_1438):.3f} '
        f'(share of fills whose (symbol,day) has ZERO rows in bars_sip.db)')

    c_feat = pd.read_csv(CELL_C_CSV, low_memory=False)
    c_feat = c_feat[['day', 'symbol', 'fill_min', 'has_prebreak']].rename(
        columns={'has_prebreak': 'tick_window_has_bar_j'})
    before = len(out)
    out = out.merge(c_feat, on=['day', 'symbol', 'fill_min'], how='left')
    assert len(out) == before, 'features_1478_C join changed row count -- duplicate keys in C'

    # ---------- write ----------
    keep_meta = ['day', 'symbol', 'fill_min', 'split']
    feature_cols = [c for c in out.columns if c not in keep_meta and c not in
                    ['status', 'fill', 'stop', 'level', 'exit_m', 'exit_price', 'why', 'R', 'raw_R',
                     'cost_R', 'net_R', 'exit_half_src', 'wk', 'half', 'variant', 'n_cross', 'b0_net_R']]
    final_cols = keep_meta + feature_cols
    out[final_cols].to_csv(out_csv, index=False)
    log(f'WROTE {out_csv}: {len(out)} rows, {len(feature_cols)} feature columns')

    cov = {c: float(out[c].notna().mean()) for c in feature_cols}
    log('coverage per feature:')
    for k, v in sorted(cov.items()):
        log(f'  {k}: {v:.1%}')
    log(f'TOTAL elapsed {time.time() - t_start:.0f}s')
    return cov


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm-bar-closed', action='store_true', default=None,
                    help='Amendment 3 fix: arm bar j = last RTH bar fully closed before fill_min '
                         '(m <= floor(fill_min) - 1). Default: old leaky rule (m < fill_min).')
    p.add_argument('--out-csv', default=None, help=f'Output CSV path (default {OUT_CSV}).')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    main(arm_bar_closed=args.arm_bar_closed, out_csv=args.out_csv)
