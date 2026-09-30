#!/usr/bin/env python3
"""Cell 1,676: candle shapes, in depth -- the seven holes of 1,668-1,671.

PREREG: research/hod_entry/PREREG_1676.md (FROZEN 2026-09-30 06:50 UTC;
amendment 1 at 07:05 UTC, BEFORE any number was scored, added G7 rolling
post-entry shapes -- verified against the file on disk, not just the
relayed message, before this script was written).

Population: f1668.load_population() -- the SAME 1663_features.csv (5,506-row
floored primary book) + fill_id/split + level, joined 1:1 on (date,symbol).
Halves: TRAIN-H2 / VAL (matches 1669/1670/1673's convention). Tercile /
quintile edges fit on TRAIN-H2 only, applied to both halves.

Frames: 1-min bars as stored in bars_sip.db; 5/10/15-min bars built by
aggregating 1-min bars aligned to 09:30 ET (minute 570). A frame bar is
"closed" (usable for G2-G5, G7) only if its end-minute is <= the 1-min bar
START at the decision instant (strict: never a bar that closes at/after the
instant). G1's break-bar geometry is the one deliberate exception written
into its own spec line ("bar containing/ending at the level bar"): it uses
the PARTIAL in-progress frame candle built only from 1-min bars up to and
including the instant bar -- still causal (nothing after the instant), just
not "closed".

Reuses (imported via importlib, module name starts with a digit):
  f1668.BarStore, .find_fill_index, .find_break_bar, .shape_features,
  .talib_features, .walk_k, .dR_cut, .load_population, .minute_of_day,
  .et_offset_minutes, .check_disk, .CDL_NAMES, .TALIB_AVAILABLE,
  .BARS_DB, .EOD_M, .LEVEL_TOL, .ENTRY_BPS, .CUT_BPS, .Z_MDE
Reuses (copied verbatim, attributed -- same convention 1668 used for the ET
timing functions "per PREREG's instruction to reuse the working, DST-aware
conversion"; these are pure stat helpers from 1667_sweep.py, not worth a
fragile whole-module import):
  iid_t, day_clustered_t, ex_top5_mean, fills_per_week, mde,
  stats_for_subset, paired_day_clustered_t, make_edges, assign_bucket
Reuses (read, not re-fit): research/hod_entry/1670_per_fill_k.csv's
p_stop_ALL column (the 1,670 ALL-family per-fill probability) joined on
(fill_id, k) for R3's "does shape add to path+volume" test.

Scope disclosure (stated up front, not hidden): G7's continuous shape
features (CLV/body/wick/range/vol-ratio/close-in-R) and climax/effort-
residual run on the FULL (k in {1..15,20,30,45,60}) x (w in {5,10,15}) grid
(57 cells/fill). The 61 TA-Lib flags on the trailing series -- the expensive
part on a shared 2-CPU node -- run at full k-grid for w=5 (finest
resolution) and at all three widths for the modeled k-subset {1,5,15,30}
(the placebo k's plus k=1). This bounds node load; it is disclosed here and
in RESULT_1676.md, not silently dropped.

Usage:
    python3 1676_shapes.py --resume [--limit N]

Outputs (research/hod_entry/):
    1676_features.csv, 1676_reads.csv, 1676_patterns.csv, 1676_shapes.log,
    RESULT_1676.md
"""
import argparse
import importlib.util
import logging
import math
import os
import sys
import time
import warnings

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '1')

import numpy as np
import pandas as pd

try:
    import talib
except ImportError:
    talib = None

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance

warnings.filterwarnings('ignore', category=RuntimeWarning)

HERE = os.path.dirname(os.path.abspath(__file__))

_spec = importlib.util.spec_from_file_location('f1668', os.path.join(HERE, '1668_failure.py'))
f1668 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(f1668)

FEATURES_CSV = os.path.join(HERE, '1676_features.csv')
READS_CSV = os.path.join(HERE, '1676_reads.csv')
PATTERNS_CSV = os.path.join(HERE, '1676_patterns.csv')
LOG_FILE = os.path.join(HERE, '1676_shapes.log')
RESULT_MD = os.path.join(HERE, 'RESULT_1676.md')
PANEL_PARQUET = os.path.join(os.path.dirname(HERE), 'overnight_high', 'panel_2024_2026.parquet')
K1670_CSV = os.path.join(HERE, '1670_per_fill_k.csv')

EOD_M = f1668.EOD_M
LEVEL_TOL = f1668.LEVEL_TOL
ENTRY_BPS = f1668.ENTRY_BPS
CUT_BPS = f1668.CUT_BPS
Z_MDE = f1668.Z_MDE
CDL_NAMES = f1668.CDL_NAMES
TALIB_AVAILABLE = f1668.TALIB_AVAILABLE
minute_of_day = f1668.minute_of_day

FRAMES = {'1m': 1, '5m': 5, '10m': 10, '15m': 15}
G6_KS = [5, 15]
G7_KS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 30, 45, 60]
G7_WS = [5, 10, 15]
G7_TALIB_FULLGRID_W = 5
G7_TALIB_KEY_KS = [1, 5, 15, 30]
G7_PLACEBO_KS = [1, 5, 15, 30]
G7_IMPORTANCE_KS = [1, 5, 15]
TAUS_ARM = [0.5, 0.6, 0.7]
TAUS_G7 = [0.6, 0.7]
LOG_EVERY = 500

logger = logging.getLogger('1676')


def setup_logging():
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(LOG_FILE)
    fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter('%(levelname)s %(message)s'))
    logger.addHandler(sh)


# ---------------------------------------------------------------------------
# Stats/bucketing toolkit -- copied verbatim from research/hod_entry/
# 1667_sweep.py per the PREREG's reuse instruction (same convention 1668
# used for the ET timing functions: small, pure, foundational -> copy with
# attribution rather than a fragile whole-module import).
# ---------------------------------------------------------------------------

def iid_t(vals):
    vals = pd.Series(vals).dropna()
    n = len(vals)
    if n < 2:
        return np.nan
    s = vals.std(ddof=1)
    if s == 0 or np.isnan(s):
        return np.nan
    return vals.mean() / (s / math.sqrt(n))


def day_clustered_t(df, datecol='date', valcol='net_R'):
    day_means = df.groupby(datecol)[valcol].mean()
    n_days = len(day_means)
    if n_days < 2:
        return np.nan
    s = day_means.std(ddof=1)
    if s == 0 or np.isnan(s):
        return np.nan
    return day_means.mean() / (s / math.sqrt(n_days))


def ex_top5_mean(vals):
    vals = pd.Series(vals).dropna().sort_values()
    n = len(vals)
    if n == 0:
        return np.nan
    k = int(math.ceil(n * 0.05))
    if k >= n:
        return vals.mean()
    return vals.iloc[:n - k].mean()


def fills_per_week(df, datecol='date'):
    if len(df) == 0:
        return 0.0
    dates = pd.to_datetime(df[datecol])
    span_days = (dates.max() - dates.min()).days
    weeks = max(span_days / 7.0, 1.0)
    return len(df) / weeks


def mde(sd, n):
    if n <= 0 or pd.isna(sd):
        return np.nan
    return Z_MDE * sd / math.sqrt(n)


def stats_for_subset(df, datecol='date', valcol='net_R'):
    n = len(df)
    if n == 0:
        return dict(n=0, mean=np.nan, iid_t=np.nan, day_t=np.nan,
                     ex_top5=np.nan, fpw=0.0, mde=np.nan)
    sd = df[valcol].std(ddof=1)
    return dict(
        n=n, mean=df[valcol].mean(), iid_t=iid_t(df[valcol]),
        day_t=day_clustered_t(df, datecol, valcol),
        ex_top5=ex_top5_mean(df[valcol]), fpw=fills_per_week(df, datecol),
        mde=mde(sd, n))


def paired_day_clustered_t(df, in_a, datecol='date', valcol='net_R'):
    tmp = df[[datecol, valcol]].copy()
    tmp['grp'] = in_a
    day_stats = tmp.groupby([datecol, 'grp'])[valcol].mean().unstack('grp')
    if True not in day_stats.columns or False not in day_stats.columns:
        return np.nan
    delta = (day_stats[True] - day_stats[False]).dropna()
    n_days = len(delta)
    if n_days < 2:
        return np.nan
    s = delta.std(ddof=1)
    if s == 0 or np.isnan(s):
        return np.nan
    return delta.mean() / (s / math.sqrt(n_days))


def make_edges(pooled_vals, k):
    vals = pd.Series(pooled_vals).dropna()
    if len(vals) < k * 5:
        logger.warning('too few pooled values (%d) for %d-way bucketing', len(vals), k)
    try:
        _, edges = pd.qcut(vals, k, retbins=True, duplicates='drop')
    except ValueError as e:
        logger.warning('qcut(%d) failed: %s', k, e)
        edges = np.array([vals.min(), vals.max()])
    return edges


def assign_bucket(vals, edges):
    n_bins = len(edges) - 1
    labels = [f'B{i + 1}' for i in range(n_bins)]
    return pd.cut(vals, bins=edges, labels=labels, include_lowest=True)


# ---------------------------------------------------------------------------
# Frame construction
# ---------------------------------------------------------------------------

def build_frame(bars1m, width):
    """Aggregate 1-min bars into `width`-min bars aligned to 09:30 ET
    (minute 570). Returns a DataFrame of CLOSED bars only (bucket, o,h,l,c,
    v, n, start_min, end_min); the partial/in-progress bucket is handled
    separately by partial_candle() since it must never look past the
    decision instant."""
    minarr = bars1m['minarr']
    mask = minarr >= 570
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return pd.DataFrame(columns=['bucket', 'o', 'h', 'l', 'c', 'v', 'n', 'start_min', 'end_min'])
    bucket = np.floor((minarr[idx] - 570) / width).astype(int)
    df = pd.DataFrame({'bucket': bucket, 'o': bars1m['o'][idx], 'h': bars1m['h'][idx],
                        'l': bars1m['l'][idx], 'c': bars1m['c'][idx], 'v': bars1m['v'][idx]})
    agg = df.groupby('bucket').agg(o=('o', 'first'), h=('h', 'max'), l=('l', 'min'),
                                    c=('c', 'last'), v=('v', 'sum'), n=('o', 'size')).reset_index()
    agg['start_min'] = 570 + width * agg['bucket']
    agg['end_min'] = 570 + width * (agg['bucket'] + 1)
    # only truly complete buckets (n == width, i.e. every constituent minute present)
    agg = agg[agg['n'] >= width].reset_index(drop=True)
    return agg


def closed_before(frame_df, cutoff_min):
    """Bars whose end_min <= cutoff_min (strictly closed before the instant
    when cutoff_min is the 1-min bar's own start minute)."""
    return frame_df[frame_df['end_min'] <= cutoff_min].sort_values('bucket')


def partial_candle(bars1m, i_instant, width):
    """The in-progress frame candle containing bar i_instant, built ONLY
    from 1-min bars in [bucket_start, i_instant] -- never past the instant.
    G1's deliberate exception (break-bar 'containing/ending at the level
    bar')."""
    t = bars1m['minarr'][i_instant]
    if t < 570:
        return None
    bucket_start = 570 + width * math.floor((t - 570) / width)
    sel = np.where((bars1m['minarr'] >= bucket_start) & (np.arange(len(bars1m['minarr'])) <= i_instant))[0]
    if len(sel) == 0:
        return None
    return dict(o=bars1m['o'][sel[0]], h=bars1m['h'][sel].max(), l=bars1m['l'][sel].min(),
                c=bars1m['c'][i_instant], v=bars1m['v'][sel].sum(), n=len(sel))


def candle_shape(o, h, l, c):
    rng = h - l
    if not (rng > 0):
        return dict(clv=np.nan, body=np.nan, uwick=np.nan, lwick=np.nan)
    return dict(clv=(c - l) / rng, body=abs(c - o) / rng,
                uwick=(h - max(o, c)) / rng, lwick=(min(o, c) - l) / rng)


def ols_slope(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.all(x == x[0]):
        return np.nan, np.nan
    xm, ym = x.mean(), y.mean()
    sxx = ((x - xm) ** 2).sum()
    if sxx == 0:
        return np.nan, np.nan
    b = ((x - xm) * (y - ym)).sum() / sxx
    a = ym - b * xm
    resid = y[-1] - (a + b * x[-1])
    return b, resid


# ---------------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------------

def g1_features(bars1m, i_instant, atr14, mean_vol, level):
    out = {}
    for fname, w in FRAMES.items():
        cndl = partial_candle(bars1m, i_instant, w) if w > 1 else dict(
            o=bars1m['o'][i_instant], h=bars1m['h'][i_instant], l=bars1m['l'][i_instant],
            c=bars1m['c'][i_instant], v=bars1m['v'][i_instant], n=1)
        if cndl is None:
            continue
        sh = candle_shape(cndl['o'], cndl['h'], cndl['l'], cndl['c'])
        rng = cndl['h'] - cndl['l']
        out[f'g1_{fname}_clv'] = sh['clv']
        out[f'g1_{fname}_body'] = sh['body']
        out[f'g1_{fname}_uwick'] = sh['uwick']
        out[f'g1_{fname}_lwick'] = sh['lwick']
        out[f'g1_{fname}_range_atr'] = rng / atr14 if atr14 and atr14 > 0 else np.nan
        out[f'g1_{fname}_vol_ratio'] = cndl['v'] / mean_vol if mean_vol and mean_vol > 0 else np.nan
        out[f'g1_{fname}_close_vs_level'] = (cndl['c'] - level) / level if pd.notna(level) and level else np.nan
    return out


def g2_features(frame_df, cutoff_min, level):
    out = {}
    closed = closed_before(frame_df, cutoff_min)
    for n_win in (10, 30):
        w = closed.tail(n_win)
        pfx = f'n{n_win}'
        if len(w) < 3:
            for suf in ('rejections', 'higher_lows', 'range_contract', 'tightness',
                        'red_share', 'net_progress', 'vol_slope'):
                out[f'g2_{pfx}_{suf}'] = np.nan
            continue
        h, l, c, o, v = w['h'].values, w['l'].values, w['c'].values, w['o'].values, w['v'].values
        out[f'g2_{pfx}_rejections'] = int((pd.notna(level)) and np.sum(np.abs(h - level) <= LEVEL_TOL * 5)) if pd.notna(level) else np.nan
        out[f'g2_{pfx}_higher_lows'] = int(np.sum(np.diff(l) > 0))
        last5_range = (w.tail(5)['h'].max() - w.tail(5)['l'].min()) if len(w) >= 5 else np.nan
        prior10_range = (w.tail(min(len(w), 10))['h'].max() - w.tail(min(len(w), 10))['l'].min())
        out[f'g2_{pfx}_range_contract'] = (last5_range / prior10_range) if prior10_range and prior10_range > 0 else np.nan
        atr_local = c.std(ddof=1) if len(c) > 1 else np.nan
        out[f'g2_{pfx}_tightness'] = (c.std(ddof=1) / atr_local) if atr_local and atr_local > 0 else np.nan
        out[f'g2_{pfx}_red_share'] = float(np.mean(c < o))
        out[f'g2_{pfx}_net_progress'] = (c[-1] - c[0]) / len(c)
        slope, _ = ols_slope(np.arange(len(v)), v)
        out[f'g2_{pfx}_vol_slope'] = slope
    return out


def g3_features(frame_df, cutoff_min, atr14):
    out = {}
    closed = closed_before(frame_df, cutoff_min).tail(10)
    if len(closed) < 3:
        return dict(g3_climax_flag=np.nan, g3_climax_mins_since=np.nan, g3_effort_result_resid=np.nan)
    h, l, c, o, v = closed['h'].values, closed['l'].values, closed['c'].values, closed['o'].values, closed['v'].values
    rng = h - l
    mean_v, mean_r = v.mean(), rng.mean()
    clv = np.where(rng > 0, (c - l) / np.where(rng > 0, rng, np.nan), np.nan)
    climax = (v >= 3 * mean_v) & (rng >= 2 * mean_r) & (clv <= 0.5)
    out['g3_climax_flag'] = int(climax.any())
    out['g3_climax_mins_since'] = int((len(closed) - 1 - np.where(climax)[0][-1])) if climax.any() else np.nan
    closed30 = closed_before(frame_df, cutoff_min).tail(30)
    if len(closed30) >= 5:
        _, resid = ols_slope(closed30['v'].values, (closed30['h'] - closed30['l']).values)
        out['g3_effort_result_resid'] = resid
    else:
        out['g3_effort_result_resid'] = np.nan
    return out


def g4_features(frame_df, cutoff_min):
    """61 TA-Lib flags at the instant + interaction with prior-10-bar return
    sign. Returns (features dict, fired_names set for the fire-rate/R2
    tally)."""
    out = {}
    fired = set()
    closed = closed_before(frame_df, cutoff_min)
    if not TALIB_AVAILABLE or len(closed) < 15:
        return out, fired, np.nan
    o, h, l, c = closed['o'].values, closed['h'].values, closed['l'].values, closed['c'].values
    ctx = np.nan
    if len(c) >= 11:
        prior_ret = (c[-1] - c[-11])
        ctx = 1 if prior_ret > 0 else (-1 if prior_ret < 0 else 0)
    for name in CDL_NAMES:
        try:
            vals = getattr(talib, name)(o, h, l, c)
        except Exception:
            continue
        v = int(vals[-1])
        out[f'g4_cdl_{name}'] = v
        if v != 0:
            fired.add(name)
    return out, fired, ctx


def g5_features(panel_sym, date_str, bars1m, i_arm, level, atr14, open_px):
    out = {}
    prior = panel_sym[panel_sym['bar_date'] < date_str].sort_values('bar_date')
    day_hi = bars1m['h'][:i_arm + 1].max()
    day_lo = bars1m['l'][:i_arm + 1].min()
    day_rng = day_hi - day_lo
    out['g5_day_clv_so_far'] = (bars1m['c'][i_arm] - day_lo) / day_rng if day_rng > 0 else np.nan
    out['g5_level_in_day_range'] = (level - day_lo) / day_rng if day_rng > 0 and pd.notna(level) else np.nan
    out['g5_day_range_atr'] = day_rng / atr14 if atr14 and atr14 > 0 else np.nan
    if len(prior) >= 1:
        p = prior.iloc[-1]
        sh = candle_shape(p['open'], p['high'], p['low'], p['close'])
        out['g5_prior_clv'] = sh['clv']
        out['g5_prior_body'] = sh['body']
        out['g5_prior_uwick'] = sh['uwick']
        out['g5_prior_gap_pct'] = (open_px / p['close'] - 1) * 100 if p['close'] else np.nan
    if TALIB_AVAILABLE and len(prior) >= 20:
        tail = prior.tail(29)
        do = np.append(tail['open'].values, open_px)
        dh = np.append(tail['high'].values, day_hi)
        dl = np.append(tail['low'].values, day_lo)
        dc = np.append(tail['close'].values, bars1m['c'][i_arm])
        for name in CDL_NAMES:
            try:
                vals = getattr(talib, name)(do, dh, dl, dc)
                out[f'g5_daily_cdl_{name}'] = int(vals[-1])
            except Exception:
                continue
    return out


def g6_features(bars1m, i0, k, width, entry, R_unit):
    """1,668's shape_features/talib_features generalized to any frame,
    applied over the wall-clock window (k minutes) expressed in that
    frame's own bars (k=5min -> 1 five-min bar, k=15min -> 3 five-min
    bars)."""
    n_bars = max(1, k // width)
    last = i0 + n_bars * width
    if last >= len(bars1m['o']):
        return {}
    if width == 1:
        f = f1668.shape_features(bars1m, i0, k)
    else:
        frame = build_frame(bars1m, width)
        seg = frame[(frame['start_min'] >= bars1m['minarr'][i0 + 1]) & (frame['end_min'] <= bars1m['minarr'][last] + 1)]
        if len(seg) == 0:
            return {}
        last_row = seg.iloc[-1]
        sh = candle_shape(last_row['o'], last_row['h'], last_row['l'], last_row['c'])
        f = {'clv_last': sh['clv'], 'wick_last': sh['uwick'], 'body_last': sh['body'],
             'clv_mean': float(seg.apply(lambda r: candle_shape(r['o'], r['h'], r['l'], r['c'])['clv'], axis=1).mean()),
             'red_share': float((seg['c'] < seg['o']).mean())}
    return {f'g6_k{k}_{fname}_{key}': val for key, val in f.items() for fname in [f'{width}m']}


def g7_trailing_candle(bars1m, i0, k, w):
    end = i0 + k
    start = max(i0 + 1, end - w + 1)
    if end >= len(bars1m['o']) or start > end:
        return None
    o = bars1m['o'][start]
    h = bars1m['h'][start:end + 1].max()
    l = bars1m['l'][start:end + 1].min()
    c = bars1m['c'][end]
    v = bars1m['v'][start:end + 1].sum()
    return dict(o=o, h=h, l=l, c=c, v=v, start=start, end=end)


def g7_features(bars1m, i0, entry, R_unit, atr14, mean_vol):
    """G7 rolling post-entry shapes: full (k,w) grid for continuous
    features + climax/effort; TA-Lib bounded per the scope disclosure in the
    module docstring (full k-grid at w=5, all widths at the key k-subset)."""
    out = {}
    for k in G7_KS:
        for w in G7_WS:
            cndl = g7_trailing_candle(bars1m, i0, k, w)
            pfx = f'g7_k{k}_w{w}'
            if cndl is None:
                continue
            sh = candle_shape(cndl['o'], cndl['h'], cndl['l'], cndl['c'])
            rng = cndl['h'] - cndl['l']
            out[f'{pfx}_clv'] = sh['clv']
            out[f'{pfx}_body'] = sh['body']
            out[f'{pfx}_uwick'] = sh['uwick']
            out[f'{pfx}_lwick'] = sh['lwick']
            out[f'{pfx}_range_atr'] = rng / atr14 if atr14 and atr14 > 0 else np.nan
            out[f'{pfx}_vol_ratio'] = cndl['v'] / mean_vol if mean_vol and mean_vol > 0 else np.nan
            out[f'{pfx}_close_R'] = (cndl['c'] - entry) / R_unit if R_unit else np.nan
            # last-5-non-overlapping trailing candles of width w, ending at k
            series = []
            for j in range(5):
                cj = g7_trailing_candle(bars1m, i0, k - j * w, w)
                if cj is not None:
                    series.append(cj)
            series = series[::-1]
            if len(series) >= 2:
                sv = np.array([s['v'] for s in series])
                sr = np.array([s['h'] - s['l'] for s in series])
                sc = np.array([candle_shape(s['o'], s['h'], s['l'], s['c'])['clv'] for s in series])
                climax = (sv >= 3 * sv.mean()) & (sr >= 2 * sr.mean()) & (np.nan_to_num(sc, nan=1.0) <= 0.5)
                out[f'{pfx}_climax'] = int(climax.any())
                if len(series) >= 3:
                    _, resid = ols_slope(sv, sr)
                    out[f'{pfx}_effort_resid'] = resid
            do_talib = TALIB_AVAILABLE and (w == G7_TALIB_FULLGRID_W or k in G7_TALIB_KEY_KS) and len(series) >= 3
            if do_talib:
                so = np.array([s['o'] for s in series])
                sh_ = np.array([s['h'] for s in series])
                sl_ = np.array([s['l'] for s in series])
                sc_ = np.array([s['c'] for s in series])
                nfired = 0
                for name in CDL_NAMES:
                    try:
                        vals = getattr(talib, name)(so, sh_, sl_, sc_)
                        if vals[-1] != 0:
                            nfired += 1
                    except Exception:
                        continue
                out[f'{pfx}_talib_nfired'] = nfired
    return out


# ---------------------------------------------------------------------------
# Labels -- ONE precedence-resolved forward walk per fill (EOD, then stop,
# then target -- same order f1668.walk_k uses) gives exit_idx/exit_type_self,
# reused for fail_any/fail10 (ARM/entry-relative) and every G6/G7
# fail_after_k (forward-from-k, conditioned on "still open at k" so a trade
# already resolved by k is excluded, not mislabeled). success15 /
# success_next15_from_k are pure forward-MFE price checks, unconditional.
# ---------------------------------------------------------------------------

def resolve_walk(bars1m, i0, stop, target):
    n = len(bars1m['o'])
    for j in range(i0 + 1, n):
        if bars1m['minarr'][j] >= EOD_M:
            return j, 'eod'
        if bars1m['l'][j] <= stop:
            return j, 'stop'
        if bars1m['h'][j] >= target:
            return j, 'target'
    return None, None


def mfe_ge_1R(bars1m, i0, k, entry, R_unit, horizon=15):
    n = len(bars1m['o'])
    start, end = i0 + k + 1, min(i0 + k + horizon, n - 1)
    if start > end or R_unit <= 0:
        return np.nan
    return int((bars1m['h'][start:end + 1].max() - entry) >= R_unit)


def compute_labels(bars1m, i0, entry, stop, target, R_unit):
    out = {}
    exit_idx, exit_type_self = resolve_walk(bars1m, i0, stop, target)
    out['fail_any'] = int(exit_type_self == 'stop') if exit_type_self else np.nan
    out['fail10'] = int(exit_type_self == 'stop' and exit_idx <= i0 + 10) if exit_type_self else np.nan
    out['success15'] = mfe_ge_1R(bars1m, i0, 0, entry, R_unit, 15)
    out['_exit_idx'] = exit_idx if exit_idx is not None else np.nan
    out['_exit_type_self'] = exit_type_self
    for k in G6_KS:
        eligible = exit_idx is not None and exit_idx > i0 + k
        out[f'fail_after_{k}'] = (int(exit_type_self == 'stop') if eligible else np.nan)
        out[f'success_next15_from_{k}'] = mfe_ge_1R(bars1m, i0, k, entry, R_unit, 15)
    for k in G7_KS:
        eligible = exit_idx is not None and exit_idx > i0 + k
        out[f'g7fail_after_{k}'] = (int(exit_type_self == 'stop') if eligible else np.nan)
        out[f'g7success_next15_from_{k}'] = mfe_ge_1R(bars1m, i0, k, entry, R_unit, 15)
    return out


# ---------------------------------------------------------------------------
# Money-read mechanics at a gated k: cut (f1668.dR_cut, reused exactly),
# short (1,673 geometry: enter short at open k+1, stop = short_entry + R,
# target = the long's original stop; EOD/stop/target precedence walk mirrors
# walk_k's polarity), add (increase size at open k+1, ride to the SAME
# realized exit already in the population -- the incremental R earned by a
# later, possibly better-priced, entry).
# ---------------------------------------------------------------------------

def money_reads_at_k(bars1m, i0, k, entry, stop, R_unit, exit_price, exit_type):
    n = len(bars1m['o'])
    nx = i0 + k + 1
    if nx >= n:
        return dict(cut=np.nan, short=np.nan, add=np.nan)
    next_open = bars1m['o'][nx]
    base_net_R = (exit_price - entry) / R_unit - (ENTRY_BPS * entry + CUT_BPS * exit_price) / R_unit
    cut = f1668.dR_cut(entry, stop, base_net_R, next_open)
    short_entry = next_open
    short_stop = short_entry + R_unit
    short_target = stop
    s_exit_idx, s_exit_type = None, None
    for j in range(nx + 1, n):
        if bars1m['minarr'][j] >= EOD_M:
            s_exit_idx, s_exit_type = j, 'eod'
            break
        if bars1m['h'][j] >= short_stop:
            s_exit_idx, s_exit_type = j, 'stop'
            break
        if bars1m['l'][j] <= short_target:
            s_exit_idx, s_exit_type = j, 'target'
            break
    if s_exit_idx is None:
        short = np.nan
    else:
        s_exit_price = short_stop if s_exit_type == 'stop' else (short_target if s_exit_type == 'target' else bars1m['c'][s_exit_idx])
        short = (short_entry - s_exit_price) / R_unit - (ENTRY_BPS * short_entry + CUT_BPS * s_exit_price) / R_unit
    add = (exit_price - next_open) / R_unit - (ENTRY_BPS * next_open + CUT_BPS * exit_price) / R_unit
    return dict(cut=cut, short=short, add=add)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all():
    pop = f1668.load_population()
    k1670 = pd.read_csv(K1670_CSV, usecols=['fill_id', 'k', 'p_stop_ALL'])
    k1670_wide = k1670.pivot_table(index='fill_id', columns='k', values='p_stop_ALL', aggfunc='first')
    k1670_wide.columns = [f'k1670_ALL_k{c}' for c in k1670_wide.columns]
    avail_ks_1670 = sorted(int(c.split('_k')[-1]) for c in k1670_wide.columns)
    panel = pd.read_parquet(PANEL_PARQUET, columns=['symbol', 'bar_date', 'open', 'high', 'low', 'close'])
    pop_syms = set(pop['symbol'].unique())
    panel = panel[panel['symbol'].isin(pop_syms)]
    panel_by_sym = {s: g.sort_values('bar_date').reset_index(drop=True) for s, g in panel.groupby('symbol')}
    logger.info('population %d rows, panel filtered to %d symbols, 1670 k-grid=%s',
                len(pop), len(panel_by_sym), avail_ks_1670)
    return pop, k1670_wide, avail_ks_1670, panel_by_sym


def nearest_k(target, avail_ks):
    le = [k for k in avail_ks if k <= target]
    return max(le) if le else (min(avail_ks) if avail_ks else None)


# ---------------------------------------------------------------------------
# Per-fill processing
# ---------------------------------------------------------------------------

def process_fill(r, store, panel_by_sym):
    sym, date = r['symbol'], r['date']
    bars1m = store.day_bars(sym, date)
    if bars1m is None or len(bars1m['o']) < 5:
        return None
    i0 = f1668.find_fill_index(bars1m, r['fill_min'])
    if i0 is None or i0 + 1 >= len(bars1m['o']):
        return None
    entry, stop, target = r['entry_price'], r['stop'], r['target_price']
    R_unit = entry - stop
    if not (R_unit > 0):
        return None
    level = r.get('level', np.nan)
    atr14 = r.get('atr14', np.nan)
    mean_vol = bars1m['v'][:i0 + 1].mean() if i0 >= 1 else np.nan
    row = {'fill_id': r['fill_id'], 'date': date, 'symbol': sym, 'half': r['half'], 'net_R': r['net_R']}

    i_arm = f1668.find_break_bar(bars1m, r['fill_min'], level)
    if i_arm is not None:
        row.update(g1_features(bars1m, i_arm, atr14, mean_vol, level))
        frames = {w: build_frame(bars1m, w) for w in FRAMES.values()}
        cutoff = bars1m['minarr'][i_arm]
        for fname, w in FRAMES.items():
            g2 = g2_features(frames[w], cutoff, level)
            row.update({f'{k[:3]}{fname}_{k[3:]}': v for k, v in g2.items()})
            g3 = g3_features(frames[w], cutoff, atr14)
            row.update({f'{k[:3]}{fname}_{k[3:]}': v for k, v in g3.items()})
            g4, fired, ctx = g4_features(frames[w], cutoff)
            row.update({(f'{k[:3]}{fname}_{k[7:]}' if k.startswith('g4_cdl_') else k): v for k, v in g4.items()})
            row[f'_g4fired_{fname}'] = ','.join(sorted(fired))
            row[f'_g4ctx_{fname}'] = ctx
        day_open = bars1m['o'][0]
        row.update(g5_features(panel_by_sym.get(sym, pd.DataFrame(columns=['bar_date', 'open', 'high', 'low', 'close'])),
                                date, bars1m, i_arm, level, atr14, day_open))
    for k in G6_KS:
        for fname, w in (('1m', 1), ('5m', 5)):
            row.update(g6_features(bars1m, i0, k, w, entry, R_unit))
    row.update(g7_features(bars1m, i0, entry, R_unit, atr14, mean_vol))
    row.update(compute_labels(bars1m, i0, entry, stop, target, R_unit))
    exit_price, exit_type = r['exit_price'], r['exit_type']
    for k in G7_KS:
        mr = money_reads_at_k(bars1m, i0, k, entry, stop, R_unit, exit_price, exit_type)
        row[f'g7_k{k}_act_cut'] = mr['cut']
        row[f'g7_k{k}_act_short'] = mr['short']
        row[f'g7_k{k}_act_add'] = mr['add']
    return row


def sweep(pop, store, panel_by_sym, resume=False, limit=None):
    done_ids = set()
    existing = None
    if resume and os.path.exists(FEATURES_CSV):
        existing = pd.read_csv(FEATURES_CSV, dtype={'fill_id': str})
        done_ids = set(existing['fill_id'].astype(str))
        logger.info('resume: %d fills already in %s', len(done_ids), FEATURES_CSV)
    pop = pop.copy()
    pop['fill_id'] = pop['fill_id'].astype(str)
    todo = pop[~pop['fill_id'].isin(done_ids)]
    if limit:
        todo = todo.head(limit)
    logger.info('sweeping %d fills (todo)', len(todo))
    rows = [] if existing is None else existing.to_dict('records')
    t0 = time.time()
    n_ok = n_bad = 0
    for i, (_, r) in enumerate(todo.iterrows()):
        try:
            row = process_fill(r, store, panel_by_sym)
        except Exception as e:
            logger.warning('fill %s/%s raised %s: %s', r['symbol'], r['date'], type(e).__name__, e)
            row = None
        if row is None:
            n_bad += 1
        else:
            rows.append(row)
            n_ok += 1
        if (i + 1) % LOG_EVERY == 0 or (i + 1) == len(todo):
            elapsed = time.time() - t0
            logger.info('progress %d/%d fills (ok=%d bad=%d) %.1fs elapsed, %.2f fills/s',
                        i + 1, len(todo), n_ok, n_bad, elapsed, (i + 1) / max(elapsed, 1e-9))
            df_out = pd.DataFrame(rows)
            tmp = FEATURES_CSV + '.tmp'
            df_out.to_csv(tmp, index=False)
            os.replace(tmp, FEATURES_CSV)
    logger.info('sweep done: %d ok, %d bad, %d total rows written', n_ok, n_bad, len(rows))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Modeling / reads stage
# ---------------------------------------------------------------------------

def build_group_cols(df, avail_ks_1670):
    cols = df.columns
    g = {}
    g['G1'] = [c for c in cols if c.startswith('g1_')]
    g['G2'] = [c for c in cols if c.startswith('g2_')]
    g['G3'] = [c for c in cols if c.startswith('g3_')]
    g['G4'] = [c for c in cols if c.startswith('g4_')]
    g['G5'] = [c for c in cols if c.startswith('g5_')]
    g['G6'] = [c for c in cols if c.startswith('g6_')]
    g['ALL-shape'] = g['G1'] + g['G2'] + g['G3'] + g['G4'] + g['G5'] + g['G6']
    k0 = nearest_k(0, avail_ks_1670)
    all1670 = f'k1670_ALL_k{k0}' if k0 is not None and f'k1670_ALL_k{k0}' in cols else None
    g['ALL-shape+1670'] = g['ALL-shape'] + ([all1670] if all1670 else [])
    for k in G7_KS:
        g[f'G7_k{k}'] = [c for c in cols if c.startswith(f'g7_k{k}_') and '_act_' not in c]
        kk = nearest_k(k, avail_ks_1670)
        c1670 = f'k1670_ALL_k{kk}' if kk is not None and f'k1670_ALL_k{kk}' in cols else None
        g[f'G7_k{k}+ALL'] = g[f'G7_k{k}'] + ([c1670] if c1670 else [])
    return g


def fit_both_scorings(df, feat_cols, label_col, do_importance=False, do_placebo=True, seed=1676):
    feat_cols = [c for c in feat_cols if c in df.columns]
    if not feat_cols:
        return []
    sub = df[['date', 'half', label_col] + feat_cols].dropna(subset=[label_col]).copy()
    sub[label_col] = sub[label_col].astype(int)
    for c in feat_cols:
        sub[c] = pd.to_numeric(sub[c], errors='coerce')
    halves = {'TRAIN-H2': sub[sub.half == 'TRAIN-H2'], 'VAL': sub[sub.half == 'VAL']}
    results = []
    for train_h, test_h in [('TRAIN-H2', 'VAL'), ('VAL', 'TRAIN-H2')]:
        tr, te = halves[train_h], halves[test_h]
        rec = dict(scoring=f'{train_h}->{test_h}', n_train=len(tr), n_test=len(te),
                   auc=np.nan, placebo_auc=np.nan, importances='', _pred_index=None, _pred=None)
        if len(tr) < 30 or len(te) < 30 or tr[label_col].nunique() < 2 or te[label_col].nunique() < 2:
            results.append(rec)
            continue
        try:
            Xtr, ytr = tr[feat_cols].values, tr[label_col].values
            Xte, yte = te[feat_cols].values, te[label_col].values
            model = HistGradientBoostingClassifier(max_iter=200, random_state=seed)
            model.fit(Xtr, ytr)
            p = model.predict_proba(Xte)[:, 1]
            rec['auc'] = roc_auc_score(yte, p)
            rec['_pred_index'] = te.index.values
            rec['_pred'] = p
            if do_placebo:
                rng = np.random.RandomState(seed)
                ytr_shuf = tr.assign(_y=ytr).groupby('date')['_y'].transform(
                    lambda s: rng.permutation(s.values)).values
                mp = HistGradientBoostingClassifier(max_iter=200, random_state=seed)
                mp.fit(Xtr, ytr_shuf)
                rec['placebo_auc'] = roc_auc_score(yte, mp.predict_proba(Xte)[:, 1])
            if do_importance:
                pi = permutation_importance(model, Xte, yte, n_repeats=5, random_state=seed, n_jobs=1)
                order = np.argsort(pi.importances_mean)[::-1][:10]
                rec['importances'] = ';'.join(f'{feat_cols[j]}:{pi.importances_mean[j]:.4f}' for j in order)
        except Exception as e:
            logger.warning('fit_both_scorings failed for %s/%s/%s: %s', label_col, train_h, test_h, e)
        results.append(rec)
    return results


def run_R3(df, group_cols):
    rows, pred_store = [], {}
    labels_arm = ['fail_any', 'fail10', 'success15']
    arm_groups = ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'ALL-shape', 'ALL-shape+1670']
    for grp in arm_groups:
        for label in labels_arm:
            res = fit_both_scorings(df, group_cols[grp], label, do_importance=True, do_placebo=True)
            for r in res:
                rows.append(dict(read='R3', group=grp, frame='ALL' if grp not in ('G1', 'G2', 'G3', 'G4', 'G5', 'G6') else 'multi',
                                  label=label, k=np.nan, n_feat=len(group_cols[grp]), **{k: v for k, v in r.items() if not k.startswith('_')}))
                pred_store[(grp, label, r['scoring'])] = (r['_pred_index'], r['_pred'])
        logger.info('R3 done: group=%s', grp)
    for k in G7_KS:
        for grp in (f'G7_k{k}', f'G7_k{k}+ALL'):
            for label in (f'g7fail_after_{k}', f'g7success_next15_from_{k}'):
                res = fit_both_scorings(df, group_cols[grp], label,
                                        do_importance=(k in G7_IMPORTANCE_KS),
                                        do_placebo=(k in G7_PLACEBO_KS))
                for r in res:
                    rows.append(dict(read='R3_G7', group=grp, frame='rolling', label=label, k=k,
                                      n_feat=len(group_cols[grp]), **{kk: v for kk, v in r.items() if not kk.startswith('_')}))
                    pred_store[(grp, label, r['scoring'])] = (r['_pred_index'], r['_pred'])
        if k % 5 == 0 or k in (1, 2, 3):
            logger.info('R3_G7 done through k=%d', k)
    return pd.DataFrame(rows), pred_store


def run_R1(df):
    rows = []
    cand = [c for c in df.columns if (c.startswith(('g1_', 'g2_', 'g3_', 'g5_')) and 'cdl' not in c)
            and pd.api.types.is_numeric_dtype(df[c])]
    train = df[df.half == 'TRAIN-H2']
    for feat in cand:
        for kbucket in (3, 5):
            edges = make_edges(train[feat], kbucket)
            if len(edges) - 1 < 2:
                continue
            for half in ('TRAIN-H2', 'VAL'):
                sub = df[df.half == half].copy()
                sub['_bucket'] = assign_bucket(sub[feat], edges)
                cats = list(sub['_bucket'].cat.categories)
                for b in cats:
                    bd = sub[sub['_bucket'] == b]
                    st = stats_for_subset(bd)
                    rows.append(dict(read='R1', feature=feat, kbucket=kbucket, half=half, bucket=str(b), **st))
                if len(cats) >= 2:
                    top = (sub['_bucket'] == cats[-1]).values
                    t = paired_day_clustered_t(sub, top)
                    dR = sub.loc[top, 'net_R'].mean() - sub.loc[~top, 'net_R'].mean()
                    rows.append(dict(read='R1_topvrest', feature=feat, kbucket=kbucket, half=half, bucket='top_vs_rest',
                                      n=int(top.sum()), mean=dR, day_t=t, iid_t=np.nan, ex_top5=np.nan, fpw=np.nan, mde=np.nan))
    return pd.DataFrame(rows)


def run_R2(df):
    import re
    rows = []
    pat_cols = [c for c in df.columns if re.match(r'^g4_\d+m_CDL', c)]
    for c in pat_cols:
        m = re.match(r'^g4_(\d+m)_(CDL\w+)$', c)
        frame, pat = m.group(1), m.group(2)
        ctxcol = f'_g4ctx_{frame}'
        if ctxcol not in df.columns:
            continue
        for ctx_sign in (-1, 1):
            fired_all = (df[c] != 0) & (df[ctxcol] == ctx_sign)
            if fired_all.sum() < 50:
                continue
            for half in ('TRAIN-H2', 'VAL'):
                sub = df[df.half == half]
                m2 = (sub[c] != 0) & (sub[ctxcol] == ctx_sign)
                bd = sub[m2]
                if len(bd) == 0:
                    continue
                st = stats_for_subset(bd)
                book = stats_for_subset(sub)
                rows.append(dict(pattern=pat, frame=frame, context=ctx_sign, half=half,
                                  lift_vs_book=st['mean'] - book['mean'], **st))
    return pd.DataFrame(rows)


def run_R4(df, r1_df, pred_store):
    rows = []
    tv = r1_df[r1_df.read == 'R1_topvrest'] if len(r1_df) else pd.DataFrame()
    if len(tv):
        train_rows = tv[tv.half == 'TRAIN-H2'].copy()
        train_rows['abs_t'] = train_rows['day_t'].abs()
        train_rows = train_rows.dropna(subset=['abs_t']).sort_values('abs_t', ascending=False)
        if len(train_rows):
            best = train_rows.iloc[0]
            for half in ('TRAIN-H2', 'VAL'):
                hit = tv[(tv.feature == best['feature']) & (tv.kbucket == best['kbucket']) & (tv.half == half)]
                if len(hit):
                    hr = hit.iloc[0]
                    rows.append(dict(read='R4_best_cut', feature=best['feature'], kbucket=best['kbucket'],
                                      half=half, n=hr['n'], mean_dR=hr['mean'], day_t=hr['day_t'],
                                      selected_on='TRAIN-H2 max|day_t| (pre-committed mechanical rule)'))

    for scoring in ['TRAIN-H2->VAL', 'VAL->TRAIN-H2']:
        key = ('ALL-shape', 'success15', scoring)
        if key not in pred_store or pred_store[key][0] is None:
            continue
        idx, pred = pred_store[key]
        test_half = scoring.split('->')[1]
        sub = df.loc[idx].copy()
        sub['_p'] = pred
        book = stats_for_subset(sub)
        for tau in TAUS_ARM:
            gated_mask = (sub['_p'] >= tau).values
            gated = sub[gated_mask]
            if gated_mask.sum() < 5:
                rows.append(dict(read='R4_gated_entry', tau=tau, scoring=scoring, half=test_half,
                                  n=int(gated_mask.sum()), mean_dR=np.nan, day_t=np.nan, ex_top5=np.nan, fpw=np.nan, mde=np.nan))
                continue
            t = paired_day_clustered_t(sub, gated_mask)
            st = stats_for_subset(gated)
            rows.append(dict(read='R4_gated_entry', tau=tau, scoring=scoring, half=test_half,
                              n=st['n'], mean_dR=st['mean'] - book['mean'], day_t=t,
                              ex_top5=st['ex_top5'], fpw=st['fpw'], mde=st['mde']))

    for k in G7_KS:
        for scoring in ['TRAIN-H2->VAL', 'VAL->TRAIN-H2']:
            test_half = scoring.split('->')[1]
            fail_key = (f'G7_k{k}', f'g7fail_after_{k}', scoring)
            succ_key = (f'G7_k{k}', f'g7success_next15_from_{k}', scoring)
            for action, pkey in [('cut', fail_key), ('short', fail_key), ('add', succ_key)]:
                if pkey not in pred_store or pred_store[pkey][0] is None:
                    continue
                idx, pred = pred_store[pkey]
                col = f'g7_k{k}_act_{action}'
                if col not in df.columns:
                    continue
                sub = df.loc[idx].copy()
                sub['_p'] = pred
                sub['_act'] = sub[col]
                elig = sub.dropna(subset=['_act'])
                if len(elig) < 10:
                    continue
                book_mean = elig['_act'].mean()
                for tau in TAUS_G7:
                    gated_mask = (elig['_p'] >= tau).values
                    gated = elig[gated_mask]
                    if gated_mask.sum() < 5:
                        continue
                    tmp = elig[['date']].copy()
                    tmp['net_R'] = elig['_act']
                    t = paired_day_clustered_t(tmp, gated_mask)
                    rows.append(dict(read='R4_G7_money', k=k, action=action, tau=tau, scoring=scoring, half=test_half,
                                      n=int(gated_mask.sum()), mean_action=gated['_act'].mean(),
                                      book_mean=book_mean, mean_dR=gated['_act'].mean() - book_mean, day_t=t,
                                      ex_top5=ex_top5_mean(gated['_act']), fpw=fills_per_week(gated)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# RESULT.md
# ---------------------------------------------------------------------------

def write_result_md(feats, r1_df, r2_df, r3_df, r4_df):
    lines = []
    lines.append('# RESULT -- cell 1,676: candle shapes, in depth (G1-G7, 1/5/10/15-min frames)')
    lines.append('')
    lines.append(f'PREREG: research/hod_entry/PREREG_1676.md, FROZEN 2026-09-30 06:50 UTC, amendment 1 (G7) '
                  '07:05 UTC -- verified against the file on disk before G7 was built, not taken on a relayed '
                  'message alone. Population n={} ({}); halves: {}.'.format(
                      len(feats), 'primary r_pct>=1.5%', dict(feats['half'].value_counts())))
    lines.append('')
    lines.append('## Scope disclosure')
    lines.append('- G6 R3 uses forward-from-k labels (fail_after_k / success_next15_from_k) at k=5, not the base '
                  'ARM-relative fail_any/fail10/success15, to avoid the k=15 window leaking bars past an '
                  'already-resolved outcome into the feature.')
    lines.append('- G7 TA-Lib (61 flags) runs the full k-grid at w=5 and all widths at k in {1,5,15,30}; '
                  'continuous shape features and climax/effort run the full (k,w) grid. Bounds the shared '
                  '2-CPU node; disclosed, not dropped silently.')
    lines.append('- "Short" money geometry follows 1,673 (stop=short_entry+1R, target=long stop); "add" is the '
                  'incremental R of a later same-direction entry riding to the SAME realized exit.')
    lines.append('')

    lines.append('## R3 -- AUC table (best rows by group, both scorings; full table in 1676_reads.csv)')
    lines.append('')
    if len(r3_df):
        arm = r3_df[r3_df.read == 'R3'].copy()
        arm['auc_gap'] = arm['auc'] - arm['placebo_auc']
        best = arm.sort_values('auc', ascending=False).groupby(['group', 'label']).head(1)
        best = best.sort_values('auc', ascending=False).head(20)
        lines.append('| group | label | scoring | n_test | AUC | placebo AUC | gap |')
        lines.append('|---|---|---|---|---|---|---|')
        for _, r in best.iterrows():
            lines.append(f"| {r['group']} | {r['label']} | {r['scoring']} | {int(r['n_test'])} | "
                          f"{r['auc']:.3f} | {r['placebo_auc']:.3f} | {r['auc_gap']:.3f} |")
        lines.append('')
        g7 = r3_df[r3_df.read == 'R3_G7'].dropna(subset=['auc'])
        if len(g7):
            g7best = g7.sort_values('auc', ascending=False).head(15)
            lines.append('G7 rolling -- top 15 (group, label, k) by AUC:')
            lines.append('')
            lines.append('| group | label | k | scoring | n_test | AUC | placebo AUC |')
            lines.append('|---|---|---|---|---|---|---|')
            for _, r in g7best.iterrows():
                lines.append(f"| {r['group']} | {r['label']} | {int(r['k'])} | {r['scoring']} | {int(r['n_test'])} | "
                              f"{r['auc']:.3f} | {r['placebo_auc'] if pd.notna(r['placebo_auc']) else float('nan'):.3f} |")
        lines.append('')
        does_shape_add = arm[arm.group.isin(['ALL-shape', 'ALL-shape+1670'])]
        lines.append('Does shape add to path+volume (ALL-shape vs ALL-shape+1670, mean AUC both scorings):')
        for lbl in does_shape_add['label'].unique():
            a = does_shape_add[(does_shape_add.group == 'ALL-shape') & (does_shape_add.label == lbl)]['auc'].mean()
            b = does_shape_add[(does_shape_add.group == 'ALL-shape+1670') & (does_shape_add.label == lbl)]['auc'].mean()
            lines.append(f'- {lbl}: ALL-shape {a:.3f} -> +1670-ALL {b:.3f} (delta {b - a:+.3f})')
    else:
        lines.append('(no R3 rows -- model fitting produced nothing; see log)')
    lines.append('')

    lines.append('## R1 -- top-vs-rest cells with |day-clustered t| >= 2 in either half')
    lines.append('')
    if len(r1_df):
        tv = r1_df[r1_df.read == 'R1_topvrest'].copy()
        piv = tv.pivot_table(index=['feature', 'kbucket'], columns='half', values=['mean', 'day_t', 'n'], aggfunc='first')
        for stat in ('mean', 'day_t', 'n'):
            for half in ('TRAIN-H2', 'VAL'):
                if (stat, half) not in piv.columns:
                    piv[(stat, half)] = np.nan
        flag = (piv[('day_t', 'TRAIN-H2')].abs() >= 2) | (piv[('day_t', 'VAL')].abs() >= 2)
        sel = piv.loc[flag.fillna(False).values].copy()
        sort_key = sel[('day_t', 'VAL')].abs().fillna(-1)
        sel = sel.iloc[np.argsort(-sort_key.values)].head(30)
        lines.append('| feature | k | TRAIN-H2 n/mean/t | VAL n/mean/t | same sign |')
        lines.append('|---|---|---|---|---|')
        for pos in range(len(sel)):
            r = sel.iloc[pos]
            feat, kb = sel.index[pos]
            ntr, nva = r[('n', 'TRAIN-H2')], r[('n', 'VAL')]
            mtr, mva = float(r[('mean', 'TRAIN-H2')]), float(r[('mean', 'VAL')])
            same_sign = (mtr > 0) == (mva > 0) if pd.notna(mtr) and pd.notna(mva) else False
            lines.append(f"| {feat} | {kb} | {int(ntr) if pd.notna(ntr) else 'NA'}/{mtr:.4f}/{float(r[('day_t','TRAIN-H2')]):.2f} | "
                          f"{int(nva) if pd.notna(nva) else 'NA'}/{mva:.4f}/{float(r[('day_t','VAL')]):.2f} | {same_sign} |")
    else:
        lines.append('(no R1 rows)')
    lines.append('')

    lines.append('## R2 -- pattern x context cells (>=50 fires), top 20 by |t| either half')
    lines.append('')
    if len(r2_df):
        r2p = r2_df.copy()
        r2p['abs_t'] = r2p['day_t'].abs()
        top = r2p.sort_values('abs_t', ascending=False).head(20)
        lines.append('| pattern | frame | context | half | n | mean net_R | day_t | lift vs book |')
        lines.append('|---|---|---|---|---|---|---|---|')
        for _, r in top.iterrows():
            lines.append(f"| {r['pattern']} | {r['frame']} | {r['context']} | {r['half']} | {int(r['n'])} | "
                          f"{r['mean']:.4f} | {r['day_t']:.2f} | {r['lift_vs_book']:.4f} |")
    else:
        lines.append('(no R2 cells reached the >=50-fire threshold)')
    lines.append('')

    lines.append('## R4 -- money reads')
    lines.append('')
    if len(r4_df):
        bc = r4_df[r4_df.read == 'R4_best_cut']
        if len(bc):
            lines.append('Best pre-entry cut (selected on TRAIN-H2 max|day_t|, mechanical, reported both halves):')
            for _, r in bc.iterrows():
                lines.append(f"- {r['feature']} k={r['kbucket']} half={r['half']}: n={int(r['n'])} dR={r['mean_dR']:.4f} t={r['day_t']:.2f}")
        ge = r4_df[r4_df.read == 'R4_gated_entry']
        if len(ge):
            lines.append('')
            lines.append('Shape-gated entry (ALL-shape model, P(success15)>=tau):')
            lines.append('| tau | scoring | half | n | dR vs book | day_t | ex_top5 | fpw |')
            lines.append('|---|---|---|---|---|---|---|---|')
            for _, r in ge.iterrows():
                lines.append(f"| {r['tau']} | {r['scoring']} | {r['half']} | {int(r['n'])} | {r['mean_dR']:.4f} | "
                              f"{r['day_t']:.2f} | {r['ex_top5']:.4f} | {r['fpw']:.1f} |")
        g7m = r4_df[r4_df.read == 'R4_G7_money'].dropna(subset=['day_t'])
        if len(g7m):
            g7m = g7m.copy()
            g7m['abs_t'] = g7m['day_t'].abs()
            top = g7m.sort_values('abs_t', ascending=False).head(15)
            lines.append('')
            lines.append('G7 money reads (cut/short/add gated by the k-model), top 15 by |t|:')
            lines.append('| k | action | tau | scoring | half | n | dR vs book | day_t | ex_top5 |')
            lines.append('|---|---|---|---|---|---|---|---|---|')
            for _, r in top.iterrows():
                lines.append(f"| {int(r['k'])} | {r['action']} | {r['tau']} | {r['scoring']} | {r['half']} | "
                              f"{int(r['n'])} | {r['mean_dR']:.4f} | {r['day_t']:.2f} | {r['ex_top5']:.4f} |")
    else:
        lines.append('(no R4 rows)')
    lines.append('')

    lines.append('## Verdicts vs the pass bar (dR>=+0.05R, t>=2.5 day-clustered BOTH halves/scorings, '
                  'ex_top5>0 both, >=3 fills/week)')
    lines.append('')
    passed = []
    if len(r4_df):
        ge = r4_df[r4_df.read == 'R4_gated_entry']
        for tau in ge['tau'].unique() if len(ge) else []:
            rows_t = ge[ge.tau == tau]
            ok = (rows_t['day_t'].abs() >= 2.5).all() and (rows_t['mean_dR'] >= 0.05).all() and \
                 (rows_t['ex_top5'] > 0).all() and (rows_t['fpw'] >= 3).all() and len(rows_t) >= 2
            if ok:
                passed.append(f'shape-gated entry tau={tau}')
        g7m = r4_df[r4_df.read == 'R4_G7_money']
        for (k, act, tau), grp in g7m.groupby(['k', 'action', 'tau']):
            if len(grp) >= 2 and (grp['day_t'].abs() >= 2.5).all() and (grp['mean_dR'] >= 0.05).all() and \
               (grp['ex_top5'] > 0).all():
                passed.append(f'G7 k={k} {act} tau={tau}')
    lines.append(f'Passing cuts/gates: {passed if passed else "NONE at this pass bar"}.')
    lines.append('')
    lines.append('## Adequacy')
    lines.append('This is the first-pass build closing the seven holes + the G7 amendment; nothing here is an '
                  'owner-facing claim yet -- per PREREG, any pass requires an independent reimplementation from '
                  'this prose before it is reported. Multiplicity is large (R1 ~1,300 + R2 ~100 + R3 ~64 + R4 ~12 '
                  'base, plus ~1,100 more from G7); the both-halves/both-scorings rule and the sign line are the '
                  'protection, not any single t. MDE at this n (~2,700/half) reported per-cell in 1676_reads.csv.')
    lines.append('')
    lines.append('Files: 1676_features.csv, 1676_reads.csv, 1676_patterns.csv, 1676_shapes.py, 1676_shapes.log.')

    text = '\n'.join(lines)
    if len(lines) > 200:
        text = '\n'.join(lines[:198] + ['', '(truncated to the 200-line budget)'])
    tmp = RESULT_MD + '.tmp'
    with open(tmp, 'w') as fh:
        fh.write(text + '\n')
    os.replace(tmp, RESULT_MD)
    logger.info('wrote %s (%d lines)', RESULT_MD, len(text.splitlines()))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--skip-sweep', action='store_true')
    args = ap.parse_args()
    setup_logging()
    logger.info('=== cell 1,676 start (limit=%s resume=%s skip_sweep=%s) ===', args.limit, args.resume, args.skip_sweep)
    f1668.check_disk(5.0)
    pop, k1670_wide, avail_ks_1670, panel_by_sym = load_all()

    if args.skip_sweep and os.path.exists(FEATURES_CSV):
        feats = pd.read_csv(FEATURES_CSV, dtype={'fill_id': str, 'date': str, 'symbol': str})
        logger.info('skip-sweep: loaded %d rows from %s', len(feats), FEATURES_CSV)
    else:
        store = f1668.BarStore(f1668.BARS_DB)
        feats = sweep(pop, store, panel_by_sym, resume=args.resume, limit=args.limit)
        store.close()

    feats['fill_id'] = feats['fill_id'].astype(str)
    k1670_wide2 = k1670_wide.reset_index()
    k1670_wide2['fill_id'] = k1670_wide2['fill_id'].astype(str)
    feats = feats.merge(k1670_wide2, on='fill_id', how='left')
    f1668.check_disk(5.0)

    group_cols = build_group_cols(feats, avail_ks_1670)
    logger.info('starting R1 (%d candidate ARM-instant features)', len(feats.columns))
    r1_df = run_R1(feats)
    logger.info('R1 done: %d rows', len(r1_df))
    logger.info('starting R2')
    r2_df = run_R2(feats)
    logger.info('R2 done: %d pattern-context cells', len(r2_df))
    logger.info('starting R3 (long pole: HGB fits across groups/frames/labels/scorings + G7 k-grid)')
    r3_df, pred_store = run_R3(feats, group_cols)
    logger.info('R3 done: %d rows', len(r3_df))
    logger.info('starting R4')
    r4_df = run_R4(feats, r1_df, pred_store)
    logger.info('R4 done: %d rows', len(r4_df))

    reads_all = pd.concat([r1_df, r3_df, r4_df], ignore_index=True, sort=False)
    tmp = READS_CSV + '.tmp'
    reads_all.to_csv(tmp, index=False)
    os.replace(tmp, READS_CSV)
    tmp2 = PATTERNS_CSV + '.tmp'
    r2_df.to_csv(tmp2, index=False)
    os.replace(tmp2, PATTERNS_CSV)
    write_result_md(feats, r1_df, r2_df, r3_df, r4_df)
    logger.info('=== cell 1,676 done ===')


if __name__ == '__main__':
    main()
