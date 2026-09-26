#!/usr/bin/env python3
"""INDEPENDENT REBUILD of cells 1,457-1,462 (ceiling + causal big-day predictors + spread joint)
and the 1,463 stop-limit re-execution (20 bps variant only), written from
research/hod_entry/PREREG_1457.md prose ALONE by an agent that has not read cell_1457.py,
test_cell_1457.py or RESULT_1457.md.

Base book: research/hod_entry/causal_arming_causal.csv, status == 'fill', TRAIN-H2 + VAL
(9,911 rows per the PREREG; a WARNING is logged, not a hard failure, if the count differs).

Cost standard: the 1,445 standard (research/hod_entry/cell_1445.py: half_entry recovered from
cost_R/R/exit_price/exit_half_src and added back once; measured per-trade stop slip from cell
1,443's cache, holdout-pooled-mean fallback when uncached) with ONE fix the PREREG names: an
unmeasured EOD exit falls back to the EOD-specific pooled mean slip (computed HERE from the same
cache, not copied from any builder document), not the stop-derived 35 bps mean. The flat-30bps
variant is unaffected (it is already uniform) and is reused unchanged from cell_1445.corrected_cost
/ apply_stop_slip.

Cells (verbatim from PREREG_1457.md):
  1,457 CEILING (report-only, look-ahead by design): full-day range (high-low)/low >= 10% on the
        PIT Databento daily bar for the signal day itself.
  1,458: pre-market dollar volume (04:00-09:29 ET, sum of v*c) >= $500K. Source is bars_sip.db for
        any (symbol, day) that has at least one row there; cache.db intraday_bars_1min otherwise
        (the "1,928 cache-only days" cohort -- both tables carry UTC timestamps, converted to ET
        here independently of causal_arming._rth, which is an RTH-only filter).
  1,459: ATR14 as % of the prior close >= 4% -- true range averaged over the 14 sessions strictly
        before the signal day (Databento daily parquet, causal shift).
  1,460: prior session's (high-low)/low >= 5% (Databento daily parquet, causal shift).
  1,461: news catalyst -- data/research/orb_news_catalyst_nightly.csv, n_articles >= 1 on
        (symbol, day); NaN for a (symbol, day) absent from that CSV (its generator's own universe
        was not identified within this rebuild's budget -- see the coverage line in the log and
        the caveat in the printed summary; VOID if computable coverage < 80%, per PREREG).
  1,462: quoted spread at the fill <= 10 bps (spread_frac = 2*half_entry/fill, independently of
        cell_1445's own `spread_frac` column name -- computed the same way because the PREREG
        points at "1,454's" method, which IS that formula) AND flag_1458.
  1,463 COST (20 bps variant only; 50bps and cells 1,464-1,465 are out of this rebuild's scope):
        stop-LIMIT re-execution on cell 1,443's measured stop/stop_bar exits. limit = stop *
        (1 - 20bps). If the cached bid_250 (the NBBO bid at t0+250ms, already on file in
        sip_cache_stopslip/ and requiring NO new tape fetch) is >= limit, the fill is bid_250
        unchanged (this is exactly cell 1,443's own fill rule, so slip is unchanged for the ~
        already-inside-20bps rows). Otherwise a real trade-tape re-fetch (causal_arming.fetch_window,
        same window cell 1,443 used) is required to find the first print >= limit after t0, or the
        window's last print if none breaches it (the no-fill tail). Fetching that tape for the
        FULL population (thousands of rows) is out of this rebuild's step/time budget, so 1,463 is
        computed ONLY for the rows that overlap (by day, symbol, fill_min) with the builder's own
        research/hod_entry/cell_1457_features_DRYRUN.csv (the only "final comparison" file that
        exists; a full cell_1457_features.csv does not) -- this is exactly the set the independent
        check needs and bounds the fetch to at most a few hundred rows. Every row outside that
        intersection is left NaN with a WARNING logged (a disclosed scope limit, not a silent drop).

Usage:
    nice -n 19 python3 research/hod_entry/cell_1457_rebuild.py

Outputs: research/hod_entry/cell_1457_rebuild.csv (day, symbol, fill_min, split, flag_1457..1462,
net_R_corr_v2, net_R_corr_flat30, slip_bps_1463_20bps) and a printed comparison against
cell_1457_features_DRYRUN.csv (flag agreement, VAL kept-mean net R mine vs builder's, 1,463 mean
slip mine vs builder's).
"""
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

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from research.hod_entry import cell_1445 as c45           # noqa: E402 -- reuse the cost standard
import causal_arming as ca                                 # noqa: E402 -- fetch_window (1,463 tape)
import sip_rebuild as sr                                    # noqa: E402 -- prevailing_quote

CACHE_DB = os.path.join(REPO, 'data/cache.db')
BARS_SIP_DB = os.path.join(REPO, 'research/bf_zero/bars_sip.db')
DAILY_PARQUET = os.path.join(REPO, 'data/research/databento/equs_daily_2025_2026.parquet')
NEWS_CSV = os.path.join(REPO, 'data/research/orb_news_catalyst_nightly.csv')
BUILDER_DRYRUN_CSV = os.path.join(HERE, 'cell_1457_features_DRYRUN.csv')
OUT_CSV = os.path.join(HERE, 'cell_1457_rebuild.csv')

PREMARKET_LO_M = 240   # 04:00 ET
PREMARKET_HI_M = 570   # 09:30 ET (exclusive -- matches OPEN_M)
COVERAGE_VOID_BAR = 0.80


def log(msg):
    """Verbose progress line, flushed immediately (print() is buffered under nohup otherwise)."""
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


# --------------------------------------------------------------------------------------------
# Step 0: base book + the 1,445 cost standard, with the EOD-fallback fix
# --------------------------------------------------------------------------------------------

def eod_fallback_bps(fills):
    """The EOD-specific pooled mean of MEASURED slip_bps (why == 'eod'), computed independently
    from cell 1,443's cache -- the PREREG's fix in place of the stop-derived 35 bps fallback."""
    eod = fills[fills.why == 'eod']
    vals = []
    for row in eod.itertuples():
        m = c45.measured_slip_bps_one(row.day, row.symbol, row.exit_m, row.why, row.fill_min)
        if m is not None:
            vals.append(m)
    if not vals:
        log('WARNING eod_fallback_bps: zero measured eod rows found -- falling back to the stop '
            'mean (35 bps); the EOD-fallback fix could not be computed')
        return c45.FLAT_SLIP_FALLBACK_BPS
    bps = float(np.mean(vals))
    log(f'eod_fallback_bps: {len(vals)} measured eod rows, pooled mean {bps:.2f} bps '
        f'(replaces the 35 bps stop mean for unmeasured eod rows)')
    return bps


def apply_stop_slip_v2(fills, eod_bps):
    """Copy of cell_1445.apply_stop_slip with ONE change: an unmeasured 'eod' row falls back to
    `eod_bps` (the EOD-specific pooled mean), not FLAT_SLIP_FALLBACK_BPS (the stop mean)."""
    slip_R = np.zeros(len(fills))
    n_measured = n_fb_eod = n_fb_stop = 0
    for i, row in enumerate(fills.itertuples()):
        if row.why not in c45.SLIP_APPLICABLE_WHY:
            continue
        m = c45.measured_slip_bps_one(row.day, row.symbol, row.exit_m, row.why, row.fill_min)
        if m is not None:
            bps = m
            n_measured += 1
        elif row.why == 'eod':
            bps = eod_bps
            n_fb_eod += 1
        else:
            bps = c45.FLAT_SLIP_FALLBACK_BPS
            n_fb_stop += 1
        base_price = row.exit_price if row.why == 'eod' else row.stop
        slip_R[i] = base_price * bps / 1e4 / row.R
    log(f'apply_stop_slip_v2: {n_measured} measured, {n_fb_eod} eod-fallback ({eod_bps:.1f} bps), '
        f'{n_fb_stop} stop-fallback (35 bps)')
    return slip_R


# --------------------------------------------------------------------------------------------
# Step 1: causal daily-bar features (1,457 ceiling, 1,459 ATR14, 1,460 prior range)
# --------------------------------------------------------------------------------------------

def build_daily_panel_ext(instrument_ids):
    """Databento daily panel restricted to instrument_ids, with prev_close/prev_high/prev_low
    (causal shift) and ATR14 (true range averaged over the 14 PRIOR sessions, shifted so the
    signal day's own bar never enters its own ATR14)."""
    df = pd.read_parquet(DAILY_PARQUET,
                          columns=['bar_date', 'symbol', 'instrument_id', 'open', 'high', 'low',
                                   'close', 'volume'])
    df = df[df.instrument_id.isin(instrument_ids)].copy()
    df['bar_date'] = pd.to_datetime(df['bar_date'])
    df = df.sort_values(['instrument_id', 'bar_date']).reset_index(drop=True)
    g = df.groupby('instrument_id', sort=False)
    df['prev_close'] = g['close'].shift(1)
    df['prev_high'] = g['high'].shift(1)
    df['prev_low'] = g['low'].shift(1)
    tr = pd.concat([df['high'] - df['low'], (df['high'] - df['prev_close']).abs(),
                    (df['low'] - df['prev_close']).abs()], axis=1).max(axis=1)
    df['tr'] = tr
    df['atr14'] = df.groupby('instrument_id', sort=False)['tr'] \
        .transform(lambda s: s.rolling(14, min_periods=14).mean().shift(1))
    return df


# --------------------------------------------------------------------------------------------
# Step 2: pre-market dollar volume (1,458) -- independent of causal_arming._rth (RTH-only)
# --------------------------------------------------------------------------------------------

def _premarket(g, tcol):
    """Pre-market minute frame (m,o,h,l,c,v), 04:00-09:29 ET, from a raw UTC-timestamped bar
    frame. Independently written (not causal_arming._rth, which is RTH-only)."""
    ts = pd.to_datetime(g[tcol], utc=True).dt.tz_convert('America/New_York')
    gg = g.assign(m=(ts.dt.hour * 60 + ts.dt.minute).values).sort_values('m').drop_duplicates('m')
    return gg[(gg.m >= PREMARKET_LO_M) & (gg.m < PREMARKET_HI_M)]


def premarket_dollar_vol_by_day(symbol_days):
    """{(symbol, day): premarket_dollar_vol or NaN} -- bars_sip.db if that (symbol,day) has ANY
    row there, else data/cache.db intraday_bars_1min (the 1,928 cache-only cohort); NaN if neither
    source has a row for that (symbol, day)."""
    sip_con = sqlite3.connect(f'file:{BARS_SIP_DB}?mode=ro', uri=True)
    cache_con = sqlite3.connect(f'file:{CACHE_DB}?mode=ro', uri=True)
    by_day = {}
    for sym, day in symbol_days:
        by_day.setdefault(day, set()).add(sym)
    out = {}
    n_sip_days = n_cache_days = n_neither = 0
    days_sorted = sorted(by_day)
    for di, day in enumerate(days_sorted):
        syms = sorted(by_day[day])
        qmarks = ','.join('?' * len(syms))
        sip = pd.read_sql(f'select symbol, t, o, h, l, c, v from bars where day=? and symbol in '
                           f'({qmarks})', sip_con, params=[day] + syms)
        has_sip = set(sip.symbol.unique()) if len(sip) and 'symbol' in sip.columns else set()
        need_cache = [s for s in syms if s not in has_sip]
        cache_df = pd.DataFrame()
        if need_cache:
            qmarks2 = ','.join('?' * len(need_cache))
            cache_df = pd.read_sql(
                f'select symbol, timestamp as t, open as o, high as h, low as l, close as c, '
                f'volume as v from intraday_bars_1min where bar_date=? and symbol in ({qmarks2})',
                cache_con, params=[day] + need_cache)
        if len(sip) and 'symbol' in sip.columns:
            for s, gg in sip.groupby('symbol'):
                pm = _premarket(gg, 't')
                out[(s, day)] = float((pm.v * pm.c).sum())
                n_sip_days += 1
        if len(cache_df) and 'symbol' in cache_df.columns:
            for s, gg in cache_df.groupby('symbol'):
                pm = _premarket(gg, 't')
                out[(s, day)] = float((pm.v * pm.c).sum())
                n_cache_days += 1
        for s in syms:
            if (s, day) not in out:
                out[(s, day)] = np.nan
                n_neither += 1
        if di % 50 == 0 or di == len(days_sorted) - 1:
            log(f'premarket_dollar_vol_by_day: day {di + 1}/{len(days_sorted)} ({day})')
    sip_con.close()
    cache_con.close()
    log(f'premarket_dollar_vol_by_day: {n_sip_days} symbol-days from bars_sip.db, '
        f'{n_cache_days} from cache.db (cache-only cohort), {n_neither} with neither source')
    return out


# --------------------------------------------------------------------------------------------
# Step 3: news catalyst (1,461)
# --------------------------------------------------------------------------------------------

def load_news_lookup():
    n = pd.read_csv(NEWS_CSV)
    dup = n.duplicated(subset=['symbol', 'day']).sum()
    if dup:
        log(f'WARNING orb_news_catalyst_nightly.csv has {dup} duplicate (symbol,day) rows -- '
            f'keeping the first')
        n = n.drop_duplicates(subset=['symbol', 'day'], keep='first')
    return n.set_index(['symbol', 'day'])['n_articles'].to_dict()


# --------------------------------------------------------------------------------------------
# Step 4: 1,463 stop-limit re-execution (20 bps variant), bounded to the builder-overlap rows
# --------------------------------------------------------------------------------------------

def stop_limit_slip_20bps(row, cache_rec):
    """One measured stop/stop_bar row -> slip_bps under a stop-LIMIT at stop*(1-20bps), or NaN if
    a real tape re-fetch was needed and failed/was out of scope. `cache_rec` is the sip_cache_stopslip
    record already on file (has t0, bid_250) -- NO new fetch is needed when bid_250 already clears
    the limit, which is most of the population (limit is *below* the plain-stop bid comparison)."""
    stop = row.stop
    limit = stop * (1 - 20 / 1e4)
    bid_250 = cache_rec['bid_250']
    if bid_250 >= limit:
        return (stop - bid_250) / stop * 1e4          # unchanged from the plain-stop fill
    # bid_250 < limit: need the real tape to find the first print >= limit after t0, else the
    # window's last print (the no-fill tail). This is the only branch needing a network re-fetch.
    m = int(np.floor(row.fill_min)) if row.why == 'stop_bar' else int(row.exit_m)
    try:
        t, q = ca.fetch_window(row.symbol, row.day, m, m + 1)
    except Exception as e:                                          # noqa: BLE001
        log(f'  WARNING 1463 tape fetch failed {row.symbol} {row.day} m={m}: '
            f'{type(e).__name__}: {e}')
        return np.nan
    if not len(t):
        return np.nan
    after = t[t.ts > cache_rec['t0']].sort_values('ts', kind='stable')
    hit = after[after.price >= limit]
    if len(hit):
        fill_price = float(hit.price.iloc[0])
    else:
        fill_price = float(t.sort_values('ts', kind='stable').price.iloc[-1])   # no-fill tail
    return (stop - fill_price) / stop * 1e4


def compute_1463_for_overlap(fills, builder_keys):
    """1,463 20bps-variant slip for the subset of fills overlapping the builder's DRYRUN sample
    (see module docstring for why the full book is out of scope). Returns {index: slip_bps}."""
    out = {}
    cand = fills[fills.why.isin(['stop', 'stop_bar']) &
                 fills.set_index(['day', 'symbol', 'fill_min']).index.isin(builder_keys)]
    log(f'compute_1463_for_overlap: {len(cand)} stop/stop_bar rows overlap the builder sample')
    n_no_cache = n_done = 0
    for row in cand.itertuples():
        rec = c45.load_stopslip_cache(row.day).get(
            c45.row_key(row.symbol, row.exit_m, row.why, row.fill_min))
        if rec is None or not rec.get('measured'):
            n_no_cache += 1
            continue
        out[row.Index] = stop_limit_slip_20bps(row, rec)
        n_done += 1
    log(f'compute_1463_for_overlap: {n_done} computed, {n_no_cache} had no measured 1,443 record')
    return out


# --------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------

def main():
    log('loading base book (causal_arming_causal.csv, status==fill, TRAIN-H2+VAL)')
    fills = c45.load_base_book()

    log('Step 1: corrected cost (half_entry) -- reused unchanged from cell_1445')
    nbbo_lookup = c45.load_nbbo_lookup()
    half_entry, net_R_costfix, nbbo_fallback = c45.corrected_cost(fills, nbbo_lookup)
    fills = fills.copy()
    fills['half_entry'] = half_entry
    fills['net_R_costfix'] = net_R_costfix

    log('Step 1: measured stop slip, EOD-fallback fix applied')
    eod_bps = eod_fallback_bps(fills)
    slip_R_v2 = apply_stop_slip_v2(fills, eod_bps)
    slip_R_flat30 = c45.apply_stop_slip(fills, flat_bps=c45.FLAT_SLIP_VARIANT_BPS)
    fills['net_R_corr_v2'] = fills['net_R_costfix'] - slip_R_v2
    fills['net_R_corr_flat30'] = fills['net_R_costfix'] - slip_R_flat30

    log('Step 2: Databento instrument ids + daily panel (prev_close/high/low, ATR14, day range)')
    map_df = c45.load_symbol_map()
    instr_by_sd = c45.resolve_instrument_ids(list(zip(fills.symbol, fills.day)), map_df)
    n_resolved = sum(1 for sd in zip(fills.symbol, fills.day) if sd in instr_by_sd)
    log(f'  instrument id resolved for {n_resolved}/{len(fills)} fills')
    daily = build_daily_panel_ext(set(instr_by_sd.values()))
    daily_idx = daily.set_index(['instrument_id', 'bar_date'])

    log('Step 3: pre-market dollar volume (bars_sip.db / cache.db fallback)')
    pm_vol = premarket_dollar_vol_by_day(list(zip(fills.symbol, fills.day)))

    log('Step 4: news catalyst lookup')
    news_lookup = load_news_lookup()

    rows = []
    n_no_daily = 0
    for r in fills.itertuples():
        day_ts = pd.Timestamp(r.day)
        iid = instr_by_sd.get((r.symbol, r.day))
        prev_close = prev_high = prev_low = atr14 = day_high = day_low = np.nan
        if iid is not None and (iid, day_ts) in daily_idx.index:
            drow = daily_idx.loc[(iid, day_ts)]
            if isinstance(drow, pd.DataFrame):
                drow = drow.iloc[0]
            prev_close, prev_high, prev_low = drow['prev_close'], drow['prev_high'], drow['prev_low']
            atr14 = drow['atr14']
            day_high, day_low = drow['high'], drow['low']
        else:
            n_no_daily += 1
        rows.append(dict(prev_close=prev_close, prev_high=prev_high, prev_low=prev_low,
                          atr14=atr14, day_high=day_high, day_low=day_low))
    feat = pd.DataFrame(rows)
    if n_no_daily:
        log(f'WARNING {n_no_daily}/{len(fills)} fills had no Databento daily-panel match')
    fills = pd.concat([fills.reset_index(drop=True), feat], axis=1)

    fills['placebo_range_pct'] = (fills.day_high - fills.day_low) / fills.day_low * 100
    fills['atr14_pct'] = fills.atr14 / fills.prev_close * 100
    fills['prior_range_pct'] = (fills.prev_high - fills.prev_low) / fills.prev_low * 100
    fills['premkt_dvol'] = fills.apply(lambda r: pm_vol.get((r.symbol, r.day), np.nan), axis=1)
    fills['n_articles'] = fills.apply(lambda r: news_lookup.get((r.symbol, r.day), np.nan), axis=1)
    fills['spread_bps'] = 2 * fills.half_entry / fills.fill * 1e4

    news_coverage = float(fills.n_articles.notna().mean())
    log(f'news coverage: {news_coverage:.1%} of fills have a (symbol,day) row in the nightly CSV'
        f'{" -- VOID per PREREG (<80%)" if news_coverage < COVERAGE_VOID_BAR else ""}')

    flag_1457 = (fills.placebo_range_pct >= 10).fillna(False).astype(float)
    flag_1458 = (fills.premkt_dvol >= 500_000)
    flag_1458 = flag_1458.where(fills.premkt_dvol.notna())      # NaN stays NaN, not False
    flag_1459 = (fills.atr14_pct >= 4).where(fills.atr14_pct.notna())
    flag_1460 = (fills.prior_range_pct >= 5).where(fills.prior_range_pct.notna())
    flag_1461 = fills.n_articles.apply(lambda x: np.nan if pd.isna(x) else float(x >= 1))
    flag_1462 = ((fills.spread_bps <= 10) & (flag_1458 == 1)).where(
        fills.spread_bps.notna() & flag_1458.notna())

    out = pd.DataFrame({
        'day': fills.day, 'symbol': fills.symbol, 'fill_min': fills.fill_min,
        'split': fills.holdout,
        'flag_1457': flag_1457.astype(float),
        'flag_1458': flag_1458.astype(float),
        'flag_1459': flag_1459.astype(float),
        'flag_1460': flag_1460.astype(float),
        'flag_1461': flag_1461.astype(float),
        'flag_1462': flag_1462.astype(float),
        'net_R_corr_v2': fills.net_R_corr_v2,
        'net_R_corr_flat30': fills.net_R_corr_flat30,
    })

    log('Step 5: 1,463 stop-limit re-execution (20bps), bounded to the builder-overlap rows')
    if os.path.exists(BUILDER_DRYRUN_CSV):
        builder = pd.read_csv(BUILDER_DRYRUN_CSV)
        builder_keys = pd.MultiIndex.from_frame(builder[['day', 'symbol', 'fill_min']])
        slip_map = compute_1463_for_overlap(fills, builder_keys)
        out['slip_bps_1463_20bps'] = pd.Series(slip_map, dtype=float).reindex(out.index)
    else:
        log('WARNING no builder DRYRUN csv found -- 1,463 left entirely NaN (no bounding set)')
        out['slip_bps_1463_20bps'] = np.nan

    out.to_csv(OUT_CSV, index=False)
    log(f'wrote {OUT_CSV} ({len(out)} rows)')

    if os.path.exists(BUILDER_DRYRUN_CSV):
        compare(out, pd.read_csv(BUILDER_DRYRUN_CSV))
    return out


# --------------------------------------------------------------------------------------------
# Comparison against the builder's DRYRUN features (the only "final comparison" file that exists)
# --------------------------------------------------------------------------------------------

def compare(mine, builder):
    """Prints, per cell: flag agreement % (NaN==NaN counts as agreement) on the overlapping
    (day, symbol, fill_min) rows; VAL kept-set mean net R under my flag vs the builder's flag
    (each side scored on ITS OWN net_R_corr_v2, matching how PREREG's own scoring works); and for
    1,463 the mean slip mine vs the builder's on rows both sides computed."""
    m = mine.merge(builder, on=['day', 'symbol', 'fill_min'], suffixes=('_mine', '_bld'))
    print(f'\n=== overlap: {len(m)} / {len(mine)} mine, {len(builder)} builder rows ===')
    for cell in ['1457', '1458', '1459', '1460', '1461', '1462']:
        a, b = m[f'flag_{cell}_mine'], m[f'flag_{cell}_bld']
        both_nan = a.isna() & b.isna()
        agree = (both_nan | (a == b)).mean()
        val = m[m.split_bld == 'VAL'] if 'split_bld' in m.columns else m[m.split == 'VAL']
        kept_mine = val.net_R_corr_v2_mine[val[f'flag_{cell}_mine'] == 1].mean()
        kept_bld = val.net_R_corr_v2_bld[val[f'flag_{cell}_bld'] == 1].mean()
        n_cmp = int((a == b).sum() + both_nan.sum())
        print(f'cell {cell}: agreement={agree:.1%} (n={n_cmp}/{len(m)}) '
              f'VAL kept mean net R: mine={kept_mine:.4f} builder={kept_bld:.4f} '
              f'(n_kept mine={int((val[f"flag_{cell}_mine"]==1).sum())} '
              f'bld={int((val[f"flag_{cell}_bld"]==1).sum())})')
        mism = m[~(both_nan | (a == b))]
        if len(mism):
            r = mism.iloc[0]
            print(f'    example mismatch: {r.day} {r.symbol} fill_min={r.fill_min:.2f} '
                  f'mine={r[f"flag_{cell}_mine"]} builder={r[f"flag_{cell}_bld"]}')
    both = m.slip_bps_1463_20bps_mine.notna() & m.slip_bps_1463_20bps_bld.notna()
    if both.sum():
        mm = m[both]
        diff = (mm.slip_bps_1463_20bps_mine - mm.slip_bps_1463_20bps_bld)
        print(f'\ncell 1463 (20bps): n_compared={both.sum()} '
              f'mean slip mine={mm.slip_bps_1463_20bps_mine.mean():.2f}bps '
              f'builder={mm.slip_bps_1463_20bps_bld.mean():.2f}bps '
              f'mean|diff|={diff.abs().mean():.2f}bps max|diff|={diff.abs().max():.2f}bps')
    else:
        print('\ncell 1463: zero overlapping rows with both sides computed')


if __name__ == '__main__':
    main()
