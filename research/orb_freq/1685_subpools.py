"""Cell 1,685 -- ORB admission sub-pools (PREREG_1684.md amendments 1 & 1b, owner 10/1).

20 sub-pools: gap band {A=[2,3)%, B=[3,4)%, C=[4,5)%} x feature {F1 relative volume at 09:35,
F2 pre-market $ volume, F3 above-VWAP+top-half-close, F4 within 5% of 52w high, F5 prior-day
range>=1.5xATR14, F6 day-2 of a >=10% gapper} = 18, plus recovery sub-pools 19 (F1) / 20 (F2) on
gap>=5%,$3-30,prior-day volume [100K,500K) -- the names production's 500K floor drops.

REUSE (zero/low new minute-bar cost, per this cell's budget -- same discipline as cell 1,684's
idea1/idea2 fastpath): band B/C's minute-bar range features (entry_price, range_total_volume,
range_vwap_distance_pct, range_close_position, avg_daily_volume_20d, ...) come from the ALREADY-
BUILT wide-seed CSVs (research/orb_seed_wide/out/*, gap>=3%, in-regime) and cell 1,684's
fresh_out_regime/idea1_features.csv (gap[3,5)%, out-regime) -- no new minute bars. Band A's F6 and
the F6 feature generally reuse cell 1,684's fresh_{in,out}_regime/idea11_features.csv (prev-day
gap>=10%, ANY today-gap, both windows already built).

SCOPED OUT (stated plainly, not hidden -- same discipline as 1684's idea1 <3% / 2023H1 cuts):
  * Band A (gap[2,3)%) x {F1,F2,F3,F4,F5}: the general gap-[2,3)% population has NO existing
    minute-bar feature build (wide-seed's own floor is >=3%); building one fresh (study_orb_features
    + backfill, as cell 1,684 did for idea10/idea11) is a 6th new population and does not fit this
    cell's budget on top of bands B/C + recovery + the owner's two priority reads. Daily-bar
    ADMISSION FREQUENCY is still reported for band A (cheap, no minute bars needed).
  * F2 (pre-market $ volume, 04:00-09:30) for EVERY band/pool: no existing feature build carries
    pre-market bars (study_orb_features computes the 09:30-09:35 opening range only); a fresh
    pre-market intraday aggregation is a separate engineering task, out of scope here.
  * Recovery sub-pool 19 (F1) out-of-regime leg, and recovery sub-pool 20 (F2) both legs: the
    100K-500K volume slice is NOT in any existing out-of-regime feature build (idea1/2/10/11 all
    used the production 500K floor); only the in-regime F1 leg is attempted (wide-seed carries NO
    volume floor at the seed level -- verified empirically below, not assumed).

EXIT: every scored sub-pool uses the PRODUCTION PIPELINE's live-rule exit only (study_orb_pipeline_
static_lock.py's baked-in exit). The PREREG's 3-way TRAIN exit menu (live rule; no-exit-at-2R->half-
at-3R->trail MFE-1R; 50%-at-+1R) needs the 1,679 walker (687 lines, raw-minute-bar re-walk, BarStore/
f1668/f1670/f1678 cross-module state) -- re-running that machinery per sub-pool is outside this
cell's budget. THIS IS A STATED DEVIATION from the PREREG, not a silent one: every sub-pool below is
scored on the live rule only, not a TRAIN-selected exit.

Selection chain: feeding each sub-pool's candidates through study_orb_pipeline_static_lock.py gives
it the SAME per-pool ranking/8-slot cap (_composite desc, cell 1,328) that cell 1,684 relied on --
"own selection chain" by construction, no extra code needed.

Usage: python3 research/orb_freq/1685_subpools.py --stage {daily,pools,pipeline,score,all}
"""
import argparse
import json
import logging
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/ec2-user/onemil')
OUT = ROOT / 'research/orb_freq'
POOLDIR = OUT / 'subpools_1685'
POOLDIR.mkdir(exist_ok=True)

sys.path.insert(0, str(OUT))
sys.path.insert(0, str(ROOT))
from pools_1684_lib import CACHE_DB, BARS_SIP, WIDE_CSVS, IN_REGIME, OUT_REGIME  # noqa: E402
from study_orb_pipeline_static_lock import build_atr14_lookup  # noqa: E402

EQUS_2024H2 = ROOT / 'data/research/databento/equs_daily_2024H2.parquet'
XNAS_2023_2024H1 = ROOT / 'data/research/databento/xnas_daily_2023_2024H1.parquet'
DAILY_SRC_IN = OUT / 'daily_source_in_regime.parquet'
DAILY_SRC_OUT = OUT / 'daily_source_out_regime.parquet'
FRESH_IN = OUT / 'fresh_in_regime'
FRESH_OUT = OUT / 'fresh_out_regime'

PRICE_MIN, PRICE_MAX = 3.0, 30.0
VOL_FLOOR = 300_000            # amendment 1's lower floor for sub-pools 1-18
PROD_VOL_FLOOR = 500_000       # production's own floor
REC_VOL_LO, REC_VOL_HI = 100_000, 500_000   # recovery band (names the floor drops)
PROD_GAP_MIN = 5.0
BANDS = {'A': (2.0, 3.0), 'B': (3.0, 4.0), 'C': (4.0, 5.0)}
TRAIN_YEAR = 2025

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                     handlers=[logging.StreamHandler(),
                               logging.FileHandler(OUT / '1685_subpools.log', mode='a')],
                     force=True)  # pools_1684_lib's OWN import-time basicConfig() (no handlers=)
# installs a StreamHandler-only root config first; without force=True, basicConfig() here is a
# documented no-op once the root logger has ANY handler -- this cell's FileHandler silently never
# attached on the first run (log console output was real and complete; the .log FILE was empty).
log = logging.getLogger('1685')


def _ro(p):
    return sqlite3.connect(f'file:{p}?mode=ro', uri=True)


def _clean_symbol(s):
    return s.astype(str).str.replace(r'\+$', '.WS', regex=True)


def load_long_daily():
    """Full daily-bar panel, symbol-agnostic, stitched xnas_daily_2023_2024H1.parquet (2023-01..
    2024-06-02) + cache.db daily_bars (2024-06-03.. ; read-only, cache.db is the source of record
    on any overlap). Gives >=252-session causal lookback for F4/F5/F6 everywhere this cell needs it
    (in-regime starts 2025-01, out-regime 2024-07 -- both have a full prior year on this panel)."""
    xnas = pd.read_parquet(XNAS_2023_2024H1, columns=['bar_date', 'symbol', 'open', 'high', 'low', 'close', 'volume'])
    xnas['symbol'] = _clean_symbol(xnas['symbol'])
    con = _ro(CACHE_DB)
    cache = pd.read_sql_query("SELECT symbol,bar_date,open,high,low,close,volume FROM daily_bars", con)
    con.close()
    df = pd.concat([xnas[xnas.bar_date < '2024-06-03'], cache], ignore_index=True)
    df = df.drop_duplicates(['symbol', 'bar_date']).sort_values(['symbol', 'bar_date']).reset_index(drop=True)
    log.info('load_long_daily: %d rows, %d symbols, %s..%s', len(df), df.symbol.nunique(),
              df.bar_date.min(), df.bar_date.max())
    return _add_daily_derived(df)


def load_out_regime_panel():
    """EQUS 2024H2 (the out-of-regime window's admission source of record, per cell 1,684) with the
    xnas 2023-2024H1 prefix ONLY as lookback history for F4/F5 (never as admission rows)."""
    equs = pd.read_parquet(EQUS_2024H2, columns=['bar_date', 'symbol', 'open', 'high', 'low', 'close', 'volume'])
    equs['symbol'] = _clean_symbol(equs['symbol'])
    xnas = pd.read_parquet(XNAS_2023_2024H1, columns=['bar_date', 'symbol', 'open', 'high', 'low', 'close', 'volume'])
    xnas['symbol'] = _clean_symbol(xnas['symbol'])
    df = pd.concat([xnas, equs], ignore_index=True).drop_duplicates(['symbol', 'bar_date'])
    df = df.sort_values(['symbol', 'bar_date']).reset_index(drop=True)
    log.info('load_out_regime_panel: %d rows (incl. lookback), %d symbols', len(df), df.symbol.nunique())
    return _add_daily_derived(df)


def _add_daily_derived(df):
    g = df.groupby('symbol')
    df['prev_close'] = g['close'].shift(1)
    df['prev_high'] = g['high'].shift(1)
    df['prev_low'] = g['low'].shift(1)
    df['prev_volume'] = g['volume'].shift(1)
    df['gap_pct'] = (df['open'] - df['prev_close']) / df['prev_close'] * 100
    g = df.groupby('symbol')  # re-group: gap_pct is a new column
    df['prev_day_gap_pct'] = g['gap_pct'].shift(1)
    df['roll252_high'] = g['high'].transform(lambda s: s.rolling(252, min_periods=200).max().shift(1))
    return df


def band_label(gap_pct):
    out = pd.Series(np.where(gap_pct.isna(), None, None), index=gap_pct.index, dtype=object)
    for name, (lo, hi) in BANDS.items():
        out = out.where(~((gap_pct >= lo) & (gap_pct < hi)), name)
    return out


# ================================================================== stage: daily ==================
def stage_daily():
    """Daily-bar-only pass: band-A frequency (context, not scored), the floor-dropped count (how
    many >=5% gappers/day the 500K volume floor removes), and the production book's fills split by
    prior-day volume tercile. Writes 1685_reads.csv (long: window,symbol,date,gap_pct,band,prev_
    volume,f4,f5_avail inputs,f6) and prints the floor/tercile material for RESULT_1685.md."""
    long_panel = load_long_daily()
    in_win = long_panel[(long_panel.bar_date >= IN_REGIME[0]) & (long_panel.bar_date <= IN_REGIME[1])].copy()
    in_win['window'] = 'in_regime'
    out_panel = load_out_regime_panel()
    out_win = out_panel[(out_panel.bar_date >= OUT_REGIME[0]) & (out_panel.bar_date <= OUT_REGIME[1])].copy()
    out_win['window'] = 'out_regime'

    rows = []
    for win, df in (('in_regime', in_win), ('out_regime', out_win)):
        d = df.dropna(subset=['prev_close', 'prev_volume']).copy()
        d['band'] = band_label(d['gap_pct'])
        d['price_ok'] = (d['open'] >= PRICE_MIN) & (d['open'] <= PRICE_MAX)
        d['is_prod'] = d['price_ok'] & (d['gap_pct'] >= PROD_GAP_MIN) & (d['prev_volume'] >= PROD_VOL_FLOOR)
        d['floor_dropped'] = d['price_ok'] & (d['gap_pct'] >= PROD_GAP_MIN) & (d['prev_volume'] < PROD_VOL_FLOOR)
        d['is_recovery'] = d['price_ok'] & (d['gap_pct'] >= PROD_GAP_MIN) & \
            (d['prev_volume'] >= REC_VOL_LO) & (d['prev_volume'] < REC_VOL_HI)
        d['f6'] = d['prev_day_gap_pct'] >= 10.0
        d['f4'] = d['roll252_high'].notna() & (d['open'] >= 0.95 * d['roll252_high'])
        n_days = d['bar_date'].nunique()
        n_prod = d['is_prod'].sum()
        n_dropped = d['floor_dropped'].sum()
        log.info('%s: %d calendar days, band-A(2-3%%) rows=%d (%.2f/day), production(>=5%%,>=500K)=%d '
                  '(%.2f/day), floor-dropped(>=5%%,<500K)=%d (%.2f/day), recovery-band(100-500K)=%d',
                  win, n_days, (d.band == 'A').sum(), (d.band == 'A').sum() / n_days,
                  n_prod, n_prod / n_days, n_dropped, n_dropped / n_days, d['is_recovery'].sum())
        keep = d[d['band'].notna() | d['is_prod'] | d['floor_dropped']]
        rows.append(keep[['window', 'symbol', 'bar_date', 'open', 'prev_close', 'prev_volume', 'gap_pct',
                           'band', 'is_prod', 'floor_dropped', 'is_recovery', 'f4', 'f6']])
    reads = pd.concat(rows, ignore_index=True)
    reads.to_csv(OUT / '1685_reads.csv', index=False)
    log.info('wrote %s rows=%d', OUT / '1685_reads.csv', len(reads))

    # Tercile split of the production book's own fills by prior-day volume.
    prod_books = {'in_regime': OUT / 'fastpath/prod_true.csv',
                  'out_regime': ROOT / 'research/orb_2024/book_1415_liveexit.csv'}
    daily_lookup = {'in_regime': in_win, 'out_regime': out_win}
    for win, path in prod_books.items():
        if not path.exists():
            log.warning('tercile split %s: production book missing at %s -- skipped', win, path)
            continue
        book = pd.read_csv(path, keep_default_na=False, na_values=[''])
        book = book[book['entered'].astype(str).isin(['1', 'True', 'true'])].copy()
        d = daily_lookup[win][['symbol', 'bar_date', 'prev_volume']].rename(columns={'bar_date': 'date'})
        m = book.merge(d, on=['symbol', 'date'], how='left')
        n_nomatch = m['prev_volume'].isna().sum()
        m = m.dropna(subset=['prev_volume'])
        m['R'] = m['_sized_pnl'].astype(float) / 375.0
        try:
            m['tercile'] = pd.qcut(m['prev_volume'], 3, labels=['low', 'mid', 'high'], duplicates='drop')
            tg = m.groupby('tercile', observed=True)['R'].agg(['count', 'mean', 'std'])
            log.info('%s tercile split (n_nomatch=%d/%d): \n%s', win, n_nomatch, len(book), tg.to_string())
        except ValueError as e:
            log.warning('%s tercile split failed (%s) -- prev_volume likely too coarse/degenerate', win, e)


# ================================================================== stage: pools ==================
def _train_profile_f1(wide):
    """F1's TRAIN cross-sectional 09:35 profile: median(range_total_volume/avg_daily_volume_20d)
    over TRAIN-year (2025), non-production, price/volume-OK rows of the wide-seed population. Frozen
    once here; applied unchanged to VAL/TEST/out-regime reads below (no re-fit on held-out data)."""
    yr = pd.to_datetime(wide['date']).dt.year
    pool = wide[(yr == TRAIN_YEAR) & (wide['gap_pct'] < PROD_GAP_MIN) &
                (wide['prev_volume'] >= VOL_FLOOR) & (wide['open'] >= PRICE_MIN) & (wide['open'] <= PRICE_MAX)]
    raw = pool['range_total_volume'] / pool['avg_daily_volume_20d']
    prof = raw.replace([np.inf, -np.inf], np.nan).median()
    log.info('F1 TRAIN(2025) cross-sectional profile median=%.6f (n=%d rows)', prof, len(pool))
    return prof


def _load_wide_seed_joined():
    """Wide-seed CSVs (gap>=3%, in-regime) joined to daily_bars for open/prev_close/prev_high/
    prev_volume (the wide CSV itself carries only derived %s) -- same join cell 1,684's fastpath
    used. NOTE: verified empirically that the wide-seed CSV carries NO volume floor at the seed
    level (rows with prev_volume<500K are present), which is what makes bands B/C AND recovery
    sub-pool 19's in-regime leg buildable from this one source."""
    frames = [pd.read_csv(p, keep_default_na=False, na_values=['']) for p in WIDE_CSVS]
    wide = pd.concat(frames, ignore_index=True).drop_duplicates(['symbol', 'date'])
    syms = sorted(wide.symbol.unique())
    con = _ro(CACHE_DB)
    ph = ','.join('?' * len(syms))
    daily = pd.read_sql_query(
        f"SELECT symbol, bar_date, open, high, low, close, volume FROM daily_bars "
        f"WHERE symbol IN ({ph}) AND bar_date >= '2024-11-01'", con, params=syms)
    con.close()
    daily = daily.sort_values(['symbol', 'bar_date']).reset_index(drop=True)
    g = daily.groupby('symbol')
    daily['prev_close'] = g['close'].shift(1)
    daily['prev_volume'] = g['volume'].shift(1)
    m = wide.merge(daily[['symbol', 'bar_date', 'open', 'prev_close', 'prev_volume']],
                    left_on=['symbol', 'date'], right_on=['symbol', 'bar_date'], how='left')
    m = m.dropna(subset=['open', 'prev_close'])
    n_sub500k = (m['prev_volume'] < PROD_VOL_FLOOR).sum()
    if n_sub500k == 0:
        log.warning('wide-seed joined: 0/%d rows have prev_volume<500K -- the wide-seed CSV\'s own '
                     'historical scan already enforced a ~500K floor at admission time (contrary to '
                     'this cell\'s working assumption); recovery sub-pool 19 (F1) is NOT buildable '
                     'from this source for EITHER window -- scoped out, same as sub-pool 20', len(m))
    else:
        log.info('wide-seed joined: %d rows, %d with prev_volume<500K', len(m), n_sub500k)
    return m, wide.columns.tolist()


def stage_pools():
    """Build per-sub-pool candidate feature CSVs for every REUSE-able sub-pool (bands B/C x F1,F3,
    F4,F5,F6 both windows; band A x F6 both windows; recovery-19/F1 in-regime only). Merges daily-
    bar F4/F6 flags and F5 (via build_atr14_lookup, the shared helper) onto the existing minute-bar-
    featured rows; computes F1/F3 directly from existing wide-seed/fresh columns."""
    long_panel = load_long_daily()
    daily_cols = ['symbol', 'bar_date', 'prev_day_gap_pct', 'roll252_high', 'prev_high', 'prev_low']
    daily_in = long_panel[daily_cols].rename(columns={'bar_date': 'date'})
    out_panel = load_out_regime_panel()
    daily_out = out_panel[daily_cols].rename(columns={'bar_date': 'date'})
    # f1o (fresh_out_regime/idea1_features.csv) carries NO raw open/prev_volume (same 31-col
    # study_orb_features schema as wide-seed -- derived %s only); unlike `wide`, it was never
    # joined to a daily-bar source, so it needs open/prev_volume too, not just the F4/F5/F6 inputs.
    daily_out_full = out_panel[daily_cols + ['open', 'prev_volume']].rename(columns={'bar_date': 'date'})

    manifest = []

    # ---- bands B/C, in-regime: wide-seed reuse -----------------------------------------------
    wide, wide_cols = _load_wide_seed_joined()
    f1_profile = _train_profile_f1(wide)
    wide = wide.merge(daily_in, on=['symbol', 'date'], how='left')
    wide['f6'] = wide['prev_day_gap_pct'] >= 10.0
    wide['f4'] = wide['roll252_high'].notna() & (wide['open'] >= 0.95 * wide['roll252_high'])
    wide['f1_raw'] = wide['range_total_volume'] / wide['avg_daily_volume_20d']
    wide['f1'] = (wide['f1_raw'] / f1_profile) >= 3.0
    wide['f3'] = (wide['range_vwap_distance_pct'] > 0) & (wide['range_close_position'] >= 0.5)
    base = (wide['open'] >= PRICE_MIN) & (wide['open'] <= PRICE_MAX) & (wide['prev_volume'] >= VOL_FLOOR) \
        & (wide['gap_pct'] < PROD_GAP_MIN)
    wide['band'] = band_label(wide['gap_pct'])
    atr_pairs_bc = list(zip(wide.loc[base & wide['band'].isin(['B', 'C']), 'symbol'],
                             wide.loc[base & wide['band'].isin(['B', 'C']), 'date']))
    atr_lut = build_atr14_lookup(atr_pairs_bc, db_path=str(CACHE_DB))
    wide['atr14'] = [atr_lut.get((s, dt)) for s, dt in zip(wide.symbol, wide.date)]
    wide['f5'] = wide['atr14'].notna() & ((wide['prev_high'] - wide['prev_low']) >= 1.5 * wide['atr14'])
    log.info('in-regime bands B/C base pool (pre-feature): %d rows; ATR14 avail %d/%d',
              base.sum(), wide['atr14'].notna().sum(), len(atr_pairs_bc))

    for band in ('B', 'C'):
        for feat in ('f1', 'f3', 'f4', 'f5'):
            mask = base & (wide['band'] == band) & wide[feat].fillna(False)
            _write_pool(wide, mask, wide_cols, f'{band}{feat.upper()}', 'in_regime')
            manifest.append((f'{band}{feat.upper()}', 'in_regime', int(mask.sum())))

    # recovery sub-pool 19 (F1), in-regime: same wide-seed source, gap>=5%, vol[100K,500K)
    rec_base = (wide['open'] >= PRICE_MIN) & (wide['open'] <= PRICE_MAX) & (wide['gap_pct'] >= PROD_GAP_MIN) \
        & (wide['prev_volume'] >= REC_VOL_LO) & (wide['prev_volume'] < REC_VOL_HI)
    rec_mask = rec_base & wide['f1'].fillna(False)
    _write_pool(wide, rec_mask, wide_cols, 'REC19F1', 'in_regime')
    manifest.append(('REC19F1', 'in_regime', int(rec_mask.sum())))
    log.info('recovery-19/F1 in-regime: base=%d admitted(F1)=%d', rec_base.sum(), rec_mask.sum())

    # ---- bands B/C, out-of-regime: fresh_out_regime/idea1_features.csv reuse (gap[3,5)%) -------
    idea1_fp = FRESH_OUT / 'idea1_features.csv'
    if idea1_fp.exists():
        f1o = pd.read_csv(idea1_fp, keep_default_na=False, na_values=[''])
        f1o = f1o.merge(daily_out_full, on=['symbol', 'date'], how='left')
        f1o['f6'] = f1o['prev_day_gap_pct'] >= 10.0
        f1o['f4'] = f1o['roll252_high'].notna() & (f1o['open'] >= 0.95 * f1o['roll252_high'])
        f1o['f1_raw'] = f1o['range_total_volume'] / f1o['avg_daily_volume_20d']
        f1o['f1'] = (f1o['f1_raw'] / f1_profile) >= 3.0   # SAME frozen TRAIN profile, no re-fit
        f1o['f3'] = (f1o['range_vwap_distance_pct'] > 0) & (f1o['range_close_position'] >= 0.5)
        f1o['band'] = band_label(f1o['gap_pct'])
        atr_pairs_o = list(zip(f1o.symbol, f1o.date))
        atr_lut_o = build_atr14_lookup(atr_pairs_o, daily_source=str(DAILY_SRC_OUT))
        f1o['atr14'] = [atr_lut_o.get((s, dt)) for s, dt in zip(f1o.symbol, f1o.date)]
        f1o['f5'] = f1o['atr14'].notna() & ((f1o['prev_high'] - f1o['prev_low']) >= 1.5 * f1o['atr14'])
        for band in ('B', 'C'):
            for feat in ('f1', 'f3', 'f4', 'f5'):
                mask = (f1o['band'] == band) & f1o[feat].fillna(False)
                _write_pool(f1o, mask, f1o.columns.tolist(), f'{band}{feat.upper()}', 'out_regime')
                manifest.append((f'{band}{feat.upper()}', 'out_regime', int(mask.sum())))
    else:
        log.warning('out-regime bands B/C: %s missing -- skipped (cell 1,684 should have built it)', idea1_fp)

    # ---- F6, all bands, both windows: idea11 reuse (prev_day_gap>=10%, any today-gap) ----------
    for win, fp in (('in_regime', FRESH_IN / 'idea11_features.csv'), ('out_regime', FRESH_OUT / 'idea11_features.csv')):
        if not fp.exists():
            log.warning('F6 %s: %s missing -- skipped', win, fp)
            continue
        f6df = pd.read_csv(fp, keep_default_na=False, na_values=[''])
        f6df['band'] = band_label(f6df['gap_pct'])
        for band in ('A', 'B', 'C'):
            mask = f6df['band'] == band
            _write_pool(f6df, mask, f6df.columns.tolist(), f'{band}F6', win)
            manifest.append((f'{band}F6', win, int(mask.sum())))

    # ---- band A context frequency (NOT scored) --------------------------------------------------
    for win, df in (('in_regime', long_panel[(long_panel.bar_date >= IN_REGIME[0]) & (long_panel.bar_date <= IN_REGIME[1])]),
                     ('out_regime', out_panel[(out_panel.bar_date >= OUT_REGIME[0]) & (out_panel.bar_date <= OUT_REGIME[1])])):
        band_a = (df['open'] >= PRICE_MIN) & (df['open'] <= PRICE_MAX) & (df['prev_volume'] >= VOL_FLOOR) \
            & (df['gap_pct'] >= 2.0) & (df['gap_pct'] < 3.0)
        log.info('band A (2-3%%) %s: %d candidate rows (CONTEXT ONLY, not scored -- no minute-bar build)',
                  win, band_a.sum())

    pd.DataFrame(manifest, columns=['pool', 'window', 'n_candidates']).to_csv(POOLDIR / 'manifest.csv', index=False)
    log.info('stage_pools DONE -- manifest: %s', POOLDIR / 'manifest.csv')


def _write_pool(df, mask, cols, pool_id, window):
    sub = df.loc[mask, [c for c in cols if c in df.columns]]
    fp = POOLDIR / f'{pool_id}_{window}_features.csv'
    sub.to_csv(fp, index=False)
    log.info('%s/%s: %d candidate rows -> %s', pool_id, window, len(sub), fp)


# ================================================================== stage: pipeline ================
# Env is keyed by WHERE the candidate's minute bars actually live, not just by window:
#   - bands B/C (F1/F3/F4/F5), in-regime: wide-seed reuse -> cache.db default (cell 1,684 fastpath's
#     own idea1/idea2 in-regime env -- these candidates' bars are in the normal production store).
#   - everything else (F6 any band any window; bands B/C out-regime): the candidates' minute bars
#     were fetched into bars_sip.db by cell 1,684's fresh-path builds (idea10/idea11 both windows,
#     idea1/idea2 out-regime) -- MUST override ORB_BT_BARS_DB/ORB_BT_DAILY_SOURCE or the pipeline
#     looks in cache.db, finds nothing, and its own >2%-missing-bars gate aborts the run (rc=1,
#     verified: AF6/BF6/CF6 in-regime all failed this way on the first attempt with ~30% missing).
BARS_SIP_ENV = {'in_regime': {'ORB_BT_BARS_DB': str(BARS_SIP), 'ORB_BT_DAILY_SOURCE': str(DAILY_SRC_IN)},
                'out_regime': {'ORB_BT_BARS_DB': str(BARS_SIP), 'ORB_BT_DAILY_SOURCE': str(DAILY_SRC_OUT)}}
CACHE_DB_ENV = {'in_regime': {}, 'out_regime': {}}  # {} = defaults (cache.db)


def _pipeline_env_for(pool, window):
    if pool.endswith('F6') or window == 'out_regime':
        return BARS_SIP_ENV[window]
    return CACHE_DB_ENV[window]


def stage_pipeline():
    """Run study_orb_pipeline_static_lock.py once per (pool,window) features CSV written by
    stage_pools, sequentially (one process, nice -n 10), ORB_CATALYST_VETO=0 (the LIVE config)."""
    import os
    manifest = pd.read_csv(POOLDIR / 'manifest.csv')
    for _, row in manifest.iterrows():
        pool, window, n = row['pool'], row['window'], row['n_candidates']
        fp = POOLDIR / f'{pool}_{window}_features.csv'
        outp = POOLDIR / f'{pool}_{window}_true.csv'
        if n == 0:
            log.warning('%s/%s: 0 candidates -- skipping pipeline run', pool, window)
            continue
        env = dict(os.environ, ORB_BT_FEATURES_CSV=str(fp), ORB_BT_BOOK_OUT=str(outp),
                   ORB_CATALYST_VETO='0', PYTHONPATH=str(ROOT), **_pipeline_env_for(pool, window))
        log_path = POOLDIR / f'{pool}_{window}_true.log'
        log.info('running pipeline for %s/%s (n=%d) -> %s', pool, window, n, outp)
        with open(log_path, 'w') as lf:
            rc = subprocess.run(['nice', '-n', '10', 'python3', 'study_orb_pipeline_static_lock.py'],
                                 cwd=str(ROOT), env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
        log.info('%s/%s pipeline rc=%d', pool, window, rc)


# ================================================================== stage: score ===================
SCOPED_OUT = {
    'AF1': 'band A (2-3%) has no existing minute-bar feature build (wide-seed floor is >=3%)',
    'AF2': 'no pre-market (04:00-09:30) bars in any existing feature build, any band',
    'AF3': 'band A (2-3%) has no existing minute-bar feature build',
    'AF4': 'band A (2-3%) has no existing minute-bar feature build',
    'AF5': 'band A (2-3%) has no existing minute-bar feature build',
    'BF2': 'no pre-market bars in any existing feature build',
    'CF2': 'no pre-market bars in any existing feature build',
    'REC19F1': 'wide-seed already floors prev_volume>=500K at its own admission (0/22,606 rows <500K) -- the 100-500K slice is not in ANY existing feature build',
    'REC20F2': 'no pre-market bars AND the 100-500K slice is not in any existing feature build',
}
SCORED_POOLS = ['AF6', 'BF1', 'BF3', 'BF4', 'BF5', 'BF6', 'CF1', 'CF3', 'CF4', 'CF5', 'CF6']
IN_HALVES = (('2025', 2025), ('2026', 2026))


def _score1684():
    import importlib.util
    spec = importlib.util.spec_from_file_location('score1684', OUT / '1684_score.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def stage_score():
    """Load every scored sub-pool's pipeline output + production, compute stats/union/cadence
    (reusing 1684_score.py's functions verbatim), apply the PREREG pass bar, write
    1685_pool_books.csv, and print the full material for RESULT_1685.md."""
    from datetime import date
    s = _score1684()
    IN_LO, IN_HI = date(2025, 1, 1), date(2026, 9, 18)
    OUT_LO, OUT_HI = date(2024, 7, 1), date(2024, 12, 31)

    prod_in = s.load_book(OUT / 'fastpath/prod_true.csv')
    prod_out = s.load_book(ROOT / 'research/orb_2024/book_1415_liveexit.csv')
    print(f"production in_regime  n={0 if prod_in is None else len(prod_in)}")
    print(f"production out_regime n={0 if prod_out is None else len(prod_out)}")

    all_rows = []
    verdicts = {}
    for pool in SCORED_POOLS:
        for win, prod in (('in_regime', prod_in), ('out_regime', prod_out)):
            path = POOLDIR / f'{pool}_{win}_true.csv'
            df = s.load_book(path)
            if df is not None and len(df):
                tag = df.copy()
                tag['window'] = win
                tag['pool'] = pool
                all_rows.append(tag[['window', 'pool', 'date', 'symbol', 'entry_price', '_sized_pnl', 'R', '_composite']])
            n = 0 if df is None else len(df)
            print(f"\n--- {pool} / {win} (n={n}) ---")
            if win == 'in_regime' and df is not None and len(df):
                for label, yr in IN_HALVES:
                    print(s.fmt(s.stats(df[df.date.dt.year == yr], f'{pool}/{win}/{label}')))
            st = s.stats(df, f'{pool}/{win}/FULL')
            print(s.fmt(st))
            verdicts.setdefault(pool, {})[win] = st
            if df is not None and len(df):
                union, addon, raw_overlap = s.union_book(prod, df)
                print(f"  raw_overlap_with_prod={raw_overlap:.1%} added_after_excl={len(addon)}")
                print(s.fmt(s.stats(union, f'{pool}/{win}/UNION')))
                lo, hi = (IN_LO, IN_HI) if win == 'in_regime' else (OUT_LO, OUT_HI)
                print(s.cadence_report(union, f'{pool}/{win}/UNION', lo, hi))
                print(s.cadence_report(prod, f'{win}/prod-alone', lo, hi))

    if all_rows:
        pd.concat(all_rows, ignore_index=True).to_csv(OUT / '1685_pool_books.csv', index=False)
        print(f"\nwrote {OUT / '1685_pool_books.csv'} rows={sum(len(r) for r in all_rows)}")

    print("\n=== PASS BAR (own mean R>=+0.05 & dc_t>=2.0 in-regime, >=0 out-of-regime, ex-top5%>0) ===")
    n_pass = 0
    for pool, by_win in verdicts.items():
        sin = by_win.get('in_regime', {})
        sout = by_win.get('out_regime', {})
        ok_in = sin.get('n', 0) > 0 and sin.get('mean_r', -9) >= 0.05 and (sin.get('dc_t') or -9) >= 2.0 \
            and (sin.get('ex_top5') or -9) > 0
        ok_out = sout.get('n', 0) == 0 or sout.get('mean_r', -9) >= 0.0
        verdict = 'PASS' if (ok_in and ok_out) else 'FAIL'
        if verdict == 'PASS':
            n_pass += 1
        print(f"{pool}: in-regime n={sin.get('n', 0)} meanR={sin.get('mean_r', float('nan')):+.3f} "
              f"dc_t={sin.get('dc_t', float('nan')):.2f} exTop5={sin.get('ex_top5', float('nan')):+.3f} | "
              f"out-regime n={sout.get('n', 0)} meanR={sout.get('mean_r', float('nan')):+.3f} -> {verdict}")

    print(f"\n=== Scoped-out sub-pools (not pipeline-scored, reason stated) ===")
    for pool, reason in SCOPED_OUT.items():
        print(f"{pool}: {reason}")

    print(f"\n=== STAGE 2 (pairs, pre-declared: ONLY for stage-1 passers) ===")
    print(f"{n_pass}/{len(SCORED_POOLS)} scored sub-pools passed stage 1.")
    if n_pass == 0:
        print("0 passers -> stage 2 NOT RUN (pre-declared in PREREG_1684.md's amendment 1).")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['daily', 'pools', 'pipeline', 'score', 'all'])
    a = ap.parse_args()
    if a.stage in ('daily', 'all'):
        stage_daily()
    if a.stage in ('pools', 'all'):
        stage_pools()
    if a.stage in ('pipeline', 'all'):
        stage_pipeline()
    if a.stage in ('score', 'all'):
        stage_score()
