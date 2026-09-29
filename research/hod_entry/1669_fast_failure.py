#!/usr/bin/env python3
"""Cell 1,669: fast failures -- anatomy, prediction at minute 0-1, and the
value decomposition of cutting.

PREREG: research/hod_entry/PREREG_1669.md (FROZEN 2026-09-29 19:45 UTC).

Owner ask (2026-09-29) on 1,668's "failure is predictable but not tradable":
"check yourself, maybe move some of the params/duration... focus on a
specific sub-set of failure, e.g. those that stop out in sub 5 min or even
sub 2 min... find an interesting bucket/population of failures and use them
for both the model and the candle story."

Reuses (imported, not copied) from research/hod_entry/1668_failure.py:
BarStore, find_fill_index, find_break_bar, walk_k, continuous_features,
shape_features, talib_features, load_population, dR_cut, iid_t,
day_clustered_t, ex_top5_mean, mde, check_disk, minute_of_day,
et_offset_minutes, CDL_NAMES, TALIB_AVAILABLE, EOD_M, LEVEL_TOL, ENTRY_BPS,
CUT_BPS. This file adds only what 1,668 did not build: an UNBOUNDED
bar-by-bar walk to the actual exit (1668's walk_k stops at a fixed k), the
fill-bar's (k=0) own-bar features, and the value decomposition.

Part 1: anatomy of every fill's actual exit (minutes-to-exit buckets, MFE
before the exit, level re-cross, fill-bar and fill+1-bar shape) plus a
TA-Lib fire-rate table on bars fill-2..fill+1 split by exit class and, for
stops, by failure speed (<=5 min vs >5 min).

Part 2: HistGradientBoostingClassifier on FF2/FF5/FF10 (stop-out within
2/5/10 min of the fill bar) at decision instants k=0 (close of the fill bar)
and k=1 (close of bar fill+1), TRAIN-H2<->VAL, out-of-sample AUC and
precision/recall at fixed P thresholds, within-day label-shuffle placebo,
permutation importance.

Part 3: cut rule C(label,k,tau) [+ variant C' requiring current loss <=
0.25R] evaluated on the SAME out-of-sample P(fail) from Part 2: paired dR,
day-clustered t, ex-top-5% dR, MDE, and the saved/forgone/cost decomposition
with an explicit numerical identity check.

Usage:
    python3 1669_fast_failure.py [--resume] [--limit N]

Outputs (research/hod_entry/):
    1669_per_fill.csv  -- fill-level anatomy + Part2 features/labels/P(fail) + Part3 dR_cut
    1669_anatomy.csv   -- Part 1 descriptive tables (anatomy stats + candle fire-rate)
    1669_reads.csv      -- every Part 2 (family=B) and Part 3 (family=C) statistical read
    1669_fast_failure.log
    RESULT_1669.md
"""
import argparse
import importlib.util
import logging
import math
import os
import sys
import time

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

HERE = os.path.dirname(os.path.abspath(__file__))

# --- reuse 1668_failure.py's tested machinery (module name starts with a
# digit, so importlib.util rather than a normal `import` statement). ---
_spec = importlib.util.spec_from_file_location('f1668', os.path.join(HERE, '1668_failure.py'))
f1668 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(f1668)

F1667_CSV = os.path.join(HERE, '1667_features.csv')
BARS_DB = f1668.BARS_DB
PER_FILL_CSV = os.path.join(HERE, '1669_per_fill.csv')
ANATOMY_CSV = os.path.join(HERE, '1669_anatomy.csv')
READS_CSV = os.path.join(HERE, '1669_reads.csv')
LOG_FILE = os.path.join(HERE, '1669_fast_failure.log')
RESULT_MD = os.path.join(HERE, 'RESULT_1669.md')

HALVES = ['TRAIN-H2', 'VAL']
LABELS = ['FF2', 'FF5', 'FF10']
LABEL_MIN = {'FF2': 2, 'FF5': 5, 'FF10': 10}
KS = [0, 1]
TAUS = [0.5, 0.6, 0.7, 0.8]
PREC_THRESH = [0.5, 0.6, 0.7, 0.8]
OPEN_M = 570  # 09:30 ET in minutes-of-day, matches f1668.EOD_M=955 (15:55 ET)
RNG_SEED = 1669
Z_MDE = f1668.Z_MDE
CDL_NAMES = f1668.CDL_NAMES
TALIB_AVAILABLE = f1668.TALIB_AVAILABLE

logger = logging.getLogger('1669')


def setup_logging():
    """Log to both 1669_fast_failure.log and stdout, verbose (INFO); also
    routes 1668_failure's own logger (load_population etc.) through the same
    handlers so its progress/warnings land in this cell's log too."""
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(LOG_FILE, mode='w')
    fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter('%(levelname)s %(message)s'))
    logger.addHandler(fh)
    logger.addHandler(sh)
    f1668.logger.handlers = [fh, sh]
    f1668.logger.setLevel(logging.INFO)
    f1668.logger.propagate = False


# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------

def load_population():
    """1668_failure.load_population() (1663_features join incl. fill_id,
    split, level) + F11-F15 arm-time features from 1667_features.csv, joined
    1:1 on (date,symbol) -- both keys verified unique (0 dupes) beforehand."""
    pop = f1668.load_population()
    f17 = pd.read_csv(F1667_CSV, dtype={'date': str, 'symbol': str},
                       usecols=['date', 'symbol', 'F11', 'F12', 'F13', 'F14', 'F15'])
    before = len(pop)
    pop = pop.merge(f17, on=['date', 'symbol'], how='left', validate='one_to_one')
    n_miss = pop[['F11', 'F12', 'F13', 'F14', 'F15']].isna().any(axis=1).sum()
    logger.info('1667 F11-F15 join: %d/%d rows (%d with >=1 missing F-col)', before, len(pop), n_miss)
    pop['minutes_since_open'] = pop['fill_min'] - OPEN_M
    return pop


# ---------------------------------------------------------------------------
# Unbounded walk to the actual exit (Part 1) -- same precedence as
# f1668.walk_k (EOD time cutoff, then stop, then target) but not capped at a
# fixed k; needed because Part 1 asks for the true minutes-to-exit, and
# f1668 only ever evaluated fixed horizons k in {3,5,10,15}.
# ---------------------------------------------------------------------------

def walk_to_exit(bars, i0, stop, target):
    """First bar at/after i0+1 that resolves the trade. Returns (exit_idx,
    'stop'|'target'|'eod') or (None, None) if the store's day ends first."""
    n = len(bars['o'])
    for j in range(i0 + 1, n):
        if bars['minarr'][j] >= f1668.EOD_M:
            return j, 'eod'
        if bars['l'][j] <= stop:
            return j, 'stop'
        if bars['h'][j] >= target:
            return j, 'target'
    return None, None


def bar0_features(bars, i0, entry, R, atr14, break_bar_v):
    """k=0 decision-instant features: the fill bar's OWN CLV/body/range-ATR/
    volume-ratio/close-vs-fill, computed only from bars[0..i0] -- no bar
    after the fill bar is touched."""
    o, h, l, c, v = bars['o'][i0], bars['h'][i0], bars['l'][i0], bars['c'][i0], bars['v'][i0]
    rng = h - l
    return dict(
        close0=c,
        clv0=(c - l) / rng if rng > 0 else np.nan,
        body0=abs(c - o) / rng if rng > 0 else np.nan,
        rangeATR0=(rng / atr14) if (pd.notna(atr14) and atr14 > 0) else np.nan,
        volratio0=(v / break_bar_v) if (pd.notna(break_bar_v) and break_bar_v > 0) else np.nan,
        closeR0=(c - entry) / R,
    )


def talib_bar0(bars, i0):
    """talib_features at k=0: window [i0+1:i0+1] is empty (nbull/nbear
    trivially 0), but last_flags is the fill bar's OWN pattern flags -- the
    useful part of this read, computed causally from bars[0..i0] only."""
    if not TALIB_AVAILABLE:
        return {}
    tf, _ = f1668.talib_features(bars, i0, 0, want_fired_any=False)
    return {f'{k}_0': v for k, v in tf.items()}


def talib_anatomy_window(bars, i0):
    """Part 1 'candle story' read: fire-any per pattern over bars fill-2..
    fill+1 inclusive. Each pattern is computed over the FULL day from bar 0
    (TA-Lib's own lookback requirement) and only the OUTPUT is windowed --
    never the input -- matching f1668.talib_features' convention."""
    if not TALIB_AVAILABLE:
        return {}
    n = len(bars['o'])
    lo, hi = max(0, i0 - 2), min(n - 1, i0 + 1)
    o, h, l, c = bars['o'][:hi + 1], bars['h'][:hi + 1], bars['l'][:hi + 1], bars['c'][:hi + 1]
    fired = {}
    for name in CDL_NAMES:
        vals = getattr(talib, name)(o, h, l, c)
        fired[name] = bool((vals[lo:hi + 1] != 0).any())
    return fired


# ---------------------------------------------------------------------------
# Main sweep: one pass per fill builds Part 1 anatomy + Part 2 k=0/k=1
# feature blocks + the dR_cut inputs Part 3 needs.
# ---------------------------------------------------------------------------

def sweep(pop, store, resume=False, limit=None):
    if limit:
        pop = pop.iloc[:limit].copy()
    if resume and os.path.exists(PER_FILL_CSV):
        cached = pd.read_csv(PER_FILL_CSV, dtype={'date': str, 'symbol': str})
        if len(cached) == len(pop):
            logger.info('--resume: reusing complete cache %s (%d rows)', PER_FILL_CSV, len(cached))
            return cached, pd.DataFrame(), 0
        logger.info('--resume requested but cache incomplete (%d vs %d) -- recomputing', len(cached), len(pop))

    rows, fire_rows = [], []
    n_no_bars = n_no_fill_idx = n_no_exit = n_mismatch = 0
    t0 = time.time()
    pop = pop.reset_index(drop=True)
    for i, r in pop.iterrows():
        rec = dict(fill_id=r['fill_id'], date=r['date'], symbol=r['symbol'], split=r['half'],
                    entry_price=r['entry_price'], stop=r['stop'], target_price=r['target_price'],
                    level=r['level'], r_pct=r['r_pct'], atr14=r['atr14'], atr14_pct=r['atr14_pct'],
                    fill_min=r['fill_min'], minutes_since_open=r['minutes_since_open'],
                    F11=r['F11'], F12=r['F12'], F13=r['F13'], F14=r['F14'], F15=r['F15'],
                    base_exit_type=r['exit_type'], base_net_R=r['net_R'])
        bars = store.day_bars(r['symbol'], r['date'])
        if bars is None:
            n_no_bars += 1
            rows.append(rec)
            continue
        i0 = f1668.find_fill_index(bars, r['fill_min'])
        if i0 is None:
            n_no_fill_idx += 1
            rows.append(rec)
            continue

        entry, stop, target = r['entry_price'], r['stop'], r['target_price']
        R = entry - stop
        break_idx = f1668.find_break_bar(bars, r['fill_min'], r['level'])
        break_bar_v = bars['v'][break_idx] if break_idx is not None else np.nan

        # --- Part 1: unbounded walk to the true exit ---
        exit_idx, exit_kind = walk_to_exit(bars, i0, stop, target)
        if exit_idx is None:
            n_no_exit += 1
            rows.append(rec)
            continue
        if exit_kind != r['exit_type']:
            n_mismatch += 1
        minutes_to_exit = float(bars['minarr'][exit_idx] - bars['minarr'][i0])
        seg = slice(i0 + 1, exit_idx + 1)
        mfe_R = (bars['h'][seg].max() - entry) / R
        recrossed_level = (bool((bars['h'][seg] >= r['level'] - f1668.LEVEL_TOL).any())
                            if pd.notna(r['level']) else np.nan)
        exit_class_speed = (('stop_fast' if minutes_to_exit <= 5 else 'stop_slow')
                             if r['exit_type'] == 'stop' else r['exit_type'])
        rec.update(minutes_to_exit=minutes_to_exit, mfe_R=mfe_R, recrossed_level=recrossed_level,
                    exit_class_speed=exit_class_speed)
        for lab in LABELS:
            rec[lab] = bool(r['exit_type'] == 'stop' and minutes_to_exit <= LABEL_MIN[lab])

        n = len(bars['o'])
        if i0 + 1 < n:
            j = i0 + 1
            rng1 = bars['h'][j] - bars['l'][j]
            rec['clv_p1'] = (bars['c'][j] - bars['l'][j]) / rng1 if rng1 > 0 else np.nan
            rec['ret_p1'] = (bars['c'][j] - entry) / R
            rec['volratio_p1'] = (bars['v'][j] / break_bar_v) if (pd.notna(break_bar_v) and break_bar_v > 0) else np.nan

        fr = dict(fill_id=r['fill_id'], date=r['date'], split=r['half'], exit_class_speed=exit_class_speed)
        fr.update(talib_anatomy_window(bars, i0))
        fire_rows.append(fr)

        # --- Part 2: k=0 (close of fill bar) ---
        rec.update(bar0_features(bars, i0, entry, R, r['atr14'], break_bar_v))
        rec.update(talib_bar0(bars, i0))
        next_open0 = bars['o'][i0 + 1] if (i0 + 1) < n else None
        rec['k0_computable'] = True
        rec['next_open_0'] = next_open0
        rec['dR_cut_0'] = f1668.dR_cut(entry, stop, r['net_R'], next_open0)

        # --- Part 2: k=1 (close of bar fill+1), reusing f1668's own
        # walk_k/continuous_features/shape_features/talib_features exactly
        # as 1668 used them at k=5/10 -- 'still open through fill+1' gate is
        # w1['preempt']=='' (a stop/target/eod hit AT fill+1 makes cutting
        # at that instant moot, same convention as 1668). ---
        w1 = f1668.walk_k(bars, i0, 1, stop, target)
        k1_computable = (w1 is not None) and (w1['preempt'] == '')
        rec['k1_computable'] = k1_computable
        if k1_computable:
            spy_bars = store.spy_bars(r['date'])
            spy_at_fill = f1668.spy_close_at_or_before(spy_bars, r['fill_min'])
            spy_ret = None
            if spy_at_fill is not None and spy_at_fill != 0:
                spy_at_k = f1668.spy_close_at_or_before(spy_bars, bars['minarr'][w1['last_idx']])
                if spy_at_k is not None:
                    spy_ret = spy_at_k / spy_at_fill - 1.0
            tp = (bars['h'] + bars['l'] + bars['c']) / 3.0
            v_cum = np.cumsum(bars['v'])
            tpv_cum = np.cumsum(tp * bars['v'])
            vwap_k = (tpv_cum[w1['last_idx']] / v_cum[w1['last_idx']]) if v_cum[w1['last_idx']] > 0 else None
            cf = f1668.continuous_features(w1, r['level'], entry, stop, break_bar_v, spy_ret, vwap_k)
            sf = f1668.shape_features(bars, i0, 1)
            rec.update({f'{k}_1': v for k, v in cf.items()})
            rec.update({f'{k}_1': v for k, v in sf.items()})
            if TALIB_AVAILABLE:
                tf, _ = f1668.talib_features(bars, i0, 1, want_fired_any=False)
                rec.update({f'{k}_1': v for k, v in tf.items()})
            rec['close_1'] = w1['close_k']
            rec['next_open_1'] = w1['next_open']
            rec['dR_cut_1'] = f1668.dR_cut(entry, stop, r['net_R'], w1['next_open'])

        rows.append(rec)
        if (i + 1) % 500 == 0 or (i + 1) == len(pop):
            elapsed = time.time() - t0
            logger.info('sweep %d/%d fills (%.1fs, %d no-bars, %d no-fill-idx, %d no-exit-found, '
                        '%d exit-kind mismatches vs base_exit_type)',
                        i + 1, len(pop), elapsed, n_no_bars, n_no_fill_idx, n_no_exit, n_mismatch)
            out = pd.DataFrame(rows)
            tmp = PER_FILL_CSV + '.tmp'
            out.to_csv(tmp, index=False)
            os.replace(tmp, PER_FILL_CSV)
            if fire_rows:
                ftmp = ANATOMY_CSV + '.fired.tmp'
                pd.DataFrame(fire_rows).to_csv(ftmp, index=False)
                os.replace(ftmp, ANATOMY_CSV + '.fired.csv')

    logger.info('sweep done: %d fills, %d no-bars, %d no-fill-idx, %d no-exit-found, %d exit-kind mismatches',
                len(pop), n_no_bars, n_no_fill_idx, n_no_exit, n_mismatch)
    return pd.DataFrame(rows), pd.DataFrame(fire_rows), n_mismatch


# ---------------------------------------------------------------------------
# Part 1 -- anatomy + candle fire-rate tables
# ---------------------------------------------------------------------------

def anatomy_tables(per_fill, fire_df):
    """Tidy-long anatomy CSV: one block of per-(half, exit_type) descriptive
    stats, one block of per-(half, exit_class_speed, pattern) TA-Lib fire
    rates. `table` distinguishes the two blocks."""
    rows = []
    for half in HALVES:
        for et in ['stop', 'target', 'eod']:
            sub = per_fill[(per_fill['split'] == half) & (per_fill['base_exit_type'] == et)
                            & per_fill['minutes_to_exit'].notna()]
            n = len(sub)
            if n == 0:
                continue
            m = sub['minutes_to_exit']
            rows.append(dict(
                table='anatomy', half=half, exit_type=et, n=n,
                share_le1=(m <= 1).mean(), share_le2=(m <= 2).mean(), share_le5=(m <= 5).mean(),
                share_le10=(m <= 10).mean(), share_le30=(m <= 30).mean(), share_gt30=(m > 30).mean(),
                median_minutes=m.median(), mean_mfe_R=sub['mfe_R'].mean(),
                share_recrossed_level=sub['recrossed_level'].mean(),
                mean_clv0=sub['clv0'].mean(), mean_rangeATR0=sub['rangeATR0'].mean(),
                mean_clv_p1=sub['clv_p1'].mean(), mean_ret_p1_R=sub['ret_p1'].mean(),
                mean_volratio_p1=sub['volratio_p1'].mean(),
            ))
    anatomy_df = pd.DataFrame(rows)

    fire_rows = []
    if not fire_df.empty and TALIB_AVAILABLE:
        for name in CDL_NAMES:
            if name not in fire_df.columns:
                continue
            for half in HALVES:
                for cls in ['stop_fast', 'stop_slow', 'target', 'eod']:
                    sub = fire_df[(fire_df['split'] == half) & (fire_df['exit_class_speed'] == cls)]
                    if len(sub) == 0:
                        continue
                    fire_rows.append(dict(table='fire_rate', half=half, exit_class_speed=cls,
                                            pattern=name, n=len(sub), fire_rate=sub[name].mean()))
    fire_rate_df = pd.DataFrame(fire_rows)
    return anatomy_df, fire_rate_df


# ---------------------------------------------------------------------------
# Part 2 -- fast-failure classifier
# ---------------------------------------------------------------------------

ARM_FEATS = ['r_pct', 'atr14_pct', 'F11', 'F12', 'F13', 'F14', 'F15', 'minutes_since_open']
BAR0_FEATS = ['clv0', 'body0', 'rangeATR0', 'volratio0', 'closeR0']
BAR1_CONT = ['cS1_dist_level_1', 'cS2_mfe_1', 'cS3_ret_1', 'cS4_volratio_1', 'cS5_spyret_1',
             'cS6_dist_vwap_1', 'cS7_mae_1', 'cA1_progvol_1']
BAR1_SHAPE = ['clv_last_1', 'wick_last_1', 'body_last_1', 'clv_mean_1', 'red_share_1']


def feat_cols_for_k(per_fill, k):
    cols = list(ARM_FEATS) + list(BAR0_FEATS)
    if TALIB_AVAILABLE:
        cols += [c for c in per_fill.columns if c.startswith('talib_') and c.endswith('_0')]
        cols += [c for c in per_fill.columns if c.startswith('cdl_') and c.endswith('_0')]
    if k == 1:
        cols += list(BAR1_CONT) + list(BAR1_SHAPE)
        if TALIB_AVAILABLE:
            cols += [c for c in per_fill.columns if c.startswith('talib_') and c.endswith('_1')]
            cols += [c for c in per_fill.columns if c.startswith('cdl_') and c.endswith('_1')]
    return [c for c in cols if c in per_fill.columns]


def build_xy(per_fill, half, k, label):
    comp = per_fill[f'k{k}_computable'] == True  # noqa: E712
    sub = per_fill[(per_fill['split'] == half) & comp & per_fill[label].notna()].copy()
    feat_cols = feat_cols_for_k(per_fill, k)
    X = sub[feat_cols].astype(float)
    y = sub[label].astype(int)
    dr = sub[f'dR_cut_{k}'].astype(float)
    extra = sub[['date', 'entry_price', 'stop', f'next_open_{k}', f'close_{k}' if k == 1 else 'close0']].copy()
    extra.columns = ['date', 'entry', 'stop', 'next_open', 'close_k']
    return X, y, dr, extra, feat_cols, sub.index


def run_part2(per_fill):
    reads = []
    importance_tables = {}
    scored = {}  # (label,k,direction) -> DataFrame[p,y,dR,date,entry,stop,next_open,close_k]
    rng = np.random.RandomState(RNG_SEED)
    for label in LABELS:
        for k in KS:
            Xtr, ytr, drtr, extr, feat_cols, _ = build_xy(per_fill, 'TRAIN-H2', k, label)
            Xva, yva, drva, exva, _, _ = build_xy(per_fill, 'VAL', k, label)
            logger.info('Part2 %s k=%d: TRAIN-H2 n=%d (%.2f%% pos), VAL n=%d (%.2f%% pos), %d features',
                        label, k, len(ytr), 100 * ytr.mean(), len(yva), 100 * yva.mean(), len(feat_cols))
            directions = {
                'TRAIN->VAL': (Xtr, ytr, drtr, extr, Xva, yva, drva, exva),
                'VAL->TRAIN-H2 (swap)': (Xva, yva, drva, exva, Xtr, ytr, drtr, extr),
            }
            for direction, (Xa, ya, dra, exa, Xb, yb, drb, exb) in directions.items():
                if ya.nunique() < 2:
                    logger.warning('Part2 %s k=%d %s: training half has a single class -- skipped', label, k, direction)
                    continue
                model = HistGradientBoostingClassifier(max_iter=200, random_state=RNG_SEED)
                model.fit(Xa, ya)
                p_score = model.predict_proba(Xb)[:, 1]
                auc = roc_auc_score(yb, p_score) if yb.nunique() > 1 else np.nan

                ya_shuf = pd.Series(ya.values, index=ya.index).groupby(exa['date'].values).transform(
                    lambda s: rng.permutation(s.values))
                auc_pl = np.nan
                if ya_shuf.nunique() > 1:
                    model_pl = HistGradientBoostingClassifier(max_iter=200, random_state=RNG_SEED)
                    model_pl.fit(Xa, ya_shuf)
                    p_score_pl = model_pl.predict_proba(Xb)[:, 1]
                    auc_pl = roc_auc_score(yb, p_score_pl) if yb.nunique() > 1 else np.nan

                read = dict(family='B', label=label, k=k, direction=direction,
                            n_train=len(ya), n_score=len(yb), auc=auc, placebo_auc=auc_pl)
                for thr in PREC_THRESH:
                    pred = p_score >= thr
                    tp = int((pred & (yb.values == 1)).sum())
                    fp = int((pred & (yb.values == 0)).sum())
                    fn = int((~pred & (yb.values == 1)).sum())
                    read[f'precision_{thr}'] = tp / (tp + fp) if (tp + fp) > 0 else np.nan
                    read[f'recall_{thr}'] = tp / (tp + fn) if (tp + fn) > 0 else np.nan
                    read[f'n_fired_{thr}'] = tp + fp
                reads.append(read)

                sdf = exb.copy()
                sdf['p'] = p_score
                sdf['y'] = yb.values
                sdf['dR'] = drb.values
                scored[(label, k, direction)] = sdf

                if direction == 'TRAIN->VAL':
                    logger.info('Part2 %s k=%d permutation importance on VAL (n_repeats=5, n_jobs=1)', label, k)
                    pi = permutation_importance(model, Xb, yb, n_repeats=5, random_state=RNG_SEED, n_jobs=1)
                    importance_tables[(label, k)] = pd.DataFrame({
                        'feature': feat_cols, 'importance_mean': pi.importances_mean,
                        'importance_std': pi.importances_std}).sort_values('importance_mean', ascending=False)
            logger.info('Part2 %s k=%d done', label, k)
    return pd.DataFrame(reads), importance_tables, scored


# ---------------------------------------------------------------------------
# Part 3 -- value decomposition of a cut
# ---------------------------------------------------------------------------

def decompose(sdf, tau, variant, sd_scored):
    """One (label,k,tau,variant,direction) read: paired dR + the
    saved/forgone/cost decomposition, with the identity
    mean(dR|fired) == saved_contrib - forgone_contrib - cost_mean verified
    numerically (residual reported, not assumed)."""
    R_unit = sdf['entry'] - sdf['stop']
    cost = f1668.CUT_BPS * sdf['next_open'] / R_unit
    fire = sdf['p'] >= tau
    if variant == "C'":
        loss_now = np.maximum(0.0, (sdf['entry'] - sdf['close_k']) / R_unit)
        fire = fire & (loss_now <= 0.25)
    fired = sdf[fire]
    cost_f = cost[fire]
    n_scored, n_fired = len(sdf), len(fired)
    out = dict(n_scored=n_scored, n_fired=n_fired, share_cut=(n_fired / n_scored if n_scored else np.nan))
    if n_fired == 0:
        return out
    dR = fired['dR']
    tp_mask = fired['y'] == 1
    fp_mask = fired['y'] == 0
    saved_vals = dR[tp_mask] + cost_f[tp_mask]
    forgone_vals = -(dR[fp_mask] + cost_f[fp_mask])
    saved_mean = saved_vals.mean() if tp_mask.any() else np.nan
    forgone_mean = forgone_vals.mean() if fp_mask.any() else np.nan
    cost_mean = cost_f.mean()
    share_tp = float(tp_mask.mean())
    saved_contrib = share_tp * saved_mean if pd.notna(saved_mean) else 0.0
    forgone_contrib = (1 - share_tp) * forgone_mean if pd.notna(forgone_mean) else 0.0
    identity_rhs = saved_contrib - forgone_contrib - cost_mean
    if pd.notna(saved_mean) and pd.notna(forgone_mean):
        denom = saved_mean + forgone_mean + cost_mean
        breakeven_prec = ((forgone_mean + cost_mean) / denom) if denom else np.nan
    else:
        breakeven_prec = np.nan  # undefined without both a TP and an FP in the fired set
    out.update(
        mean_dR=dR.mean(), iid_t=f1668.iid_t(dR), day_t=f1668.day_clustered_t(fired['date'], dR),
        ex_top5_dR=f1668.ex_top5_mean(dR), mde=f1668.mde(sd_scored, n_fired),
        achieved_precision=share_tp, saved_mean_TP=saved_mean, forgone_mean_FP=forgone_mean,
        cost_mean=cost_mean, saved_contrib=saved_contrib, forgone_contrib=forgone_contrib,
        identity_lhs_mean_dR=dR.mean(), identity_rhs=identity_rhs,
        identity_residual=dR.mean() - identity_rhs, breakeven_precision=breakeven_prec,
    )
    return out


def run_part3(scored, sd_by_half):
    reads = []
    dir_to_scored_half = {'TRAIN->VAL': 'VAL', 'VAL->TRAIN-H2 (swap)': 'TRAIN-H2'}
    for (label, k, direction), sdf in scored.items():
        sd_scored = sd_by_half[dir_to_scored_half[direction]]
        for tau in TAUS:
            for variant in ['C', "C'"]:
                res = decompose(sdf, tau, variant, sd_scored)
                res.update(family='C', label=label, k=k, direction=direction, tau=tau, variant=variant)
                reads.append(res)
    return pd.DataFrame(reads)


# ---------------------------------------------------------------------------
# RESULT.md
# ---------------------------------------------------------------------------

def write_result_md(per_fill, anatomy_df, fire_rate_df, reads_b, reads_c, importance_tables, n_pop, n_mismatch):
    L = []
    L.append('# RESULT 1,669 -- fast failures: anatomy, prediction at minute 0-1, value decomposition\n')
    L.append(f'PREREG: `research/hod_entry/PREREG_1669.md` (FROZEN 2026-09-29 19:45 UTC). '
             f'Population n={n_pop} (same 5,506-row 1.5%-floored primary book as cell 1,668). '
             f'Exit-kind (unbounded walk) vs base_exit_type mismatches: {n_mismatch}.\n')

    L.append('## Part 1 -- anatomy of the actual exit (both halves)')
    L.append('| half | exit | n | <=1m | <=2m | <=5m | <=10m | <=30m | >30m | med min | MFE(R) | recross | clv0 | rng/ATR0 | clv+1 | ret+1(R) | volratio+1 |')
    L.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for _, r in anatomy_df.iterrows():
        L.append(f"| {r['half']} | {r['exit_type']} | {r['n']} | {r['share_le1']:.2f} | {r['share_le2']:.2f} | "
                  f"{r['share_le5']:.2f} | {r['share_le10']:.2f} | {r['share_le30']:.2f} | {r['share_gt30']:.2f} | "
                  f"{r['median_minutes']:.1f} | {r['mean_mfe_R']:.3f} | {r['share_recrossed_level']:.2f} | "
                  f"{r['mean_clv0']:.2f} | {r['mean_rangeATR0']:.2f} | {r['mean_clv_p1']:.2f} | "
                  f"{r['mean_ret_p1_R']:.3f} | {r['mean_volratio_p1']:.2f} |")
    L.append('')

    L.append('## Candle story -- TA-Lib fire rate on bars fill-2..fill+1, top 12 by |stop_fast - stop_slow| gap (pooled)')
    if fire_rate_df.empty:
        L.append('NOT RUN (TA-Lib unavailable).' if not TALIB_AVAILABLE else 'no rows.')
    else:
        piv = fire_rate_df.pivot_table(index='pattern', columns='exit_class_speed', values='fire_rate', aggfunc='mean')
        cnt = fire_rate_df.pivot_table(index='pattern', columns='exit_class_speed', values='n', aggfunc='sum')
        for c in ['stop_fast', 'stop_slow', 'target', 'eod']:
            if c not in piv.columns:
                piv[c] = np.nan
        piv['gap'] = (piv['stop_fast'] - piv['stop_slow']).abs()
        top = piv.sort_values('gap', ascending=False).head(12)
        L.append('| pattern | stop_fast rate (n) | stop_slow rate (n) | target rate (n) | eod rate (n) |')
        L.append('|---|---|---|---|---|')
        for pat, r in top.iterrows():
            def cell(c):
                v = r.get(c, np.nan)
                n = cnt.loc[pat, c] if (pat in cnt.index and c in cnt.columns) else 0
                return f'{v:.3f} ({int(n) if pd.notna(n) else 0})' if pd.notna(v) else 'n/a'
            L.append(f"| {pat} | {cell('stop_fast')} | {cell('stop_slow')} | {cell('target')} | {cell('eod')} |")
    L.append('')

    L.append('## Part 2 -- fast-failure classifier (out of sample)')
    L.append('| label | k | direction | n tr | n sc | AUC | placebo AUC | P/R@.5 | P/R@.6 | P/R@.7 | P/R@.8 |')
    L.append('|---|---|---|---|---|---|---|---|---|---|---|')
    for _, r in reads_b.iterrows():
        def pr(t):
            p, rc = r.get(f'precision_{t}'), r.get(f'recall_{t}')
            return f'{p:.2f}/{rc:.2f}' if pd.notna(p) else 'n/a'
        L.append(f"| {r['label']} | {r['k']} | {r['direction']} | {r['n_train']} | {r['n_score']} | "
                  f"{r['auc']:.3f} | {r['placebo_auc']:.3f} | {pr(0.5)} | {pr(0.6)} | {pr(0.7)} | {pr(0.8)} |")
    L.append('')

    L.append('## Permutation importance (VAL-scored, TRAIN-H2->VAL, n_repeats=5, top 6)')
    L.append('| label | k | feature | importance mean | std |')
    L.append('|---|---|---|---|---|')
    for (label, k), imp in importance_tables.items():
        for _, r in imp.head(6).iterrows():
            L.append(f"| {label} | {k} | {r['feature']} | {r['importance_mean']:.4f} | {r['importance_std']:.4f} |")
    L.append('')

    L.append('## Part 3 -- value decomposition at tau=0.7 (full tau grid in 1669_reads.csv)')
    L.append('Formulas: saved (TP) = cut_gross - base_R; forgone (FP) = base_R - cut_gross; '
             'cost = CUT_BPS*next_open/R_unit (all fired). Identity: mean(dR|fired) = '
             'share_TP*saved - share_FP*forgone - cost. Break-even precision = (forgone+cost)/(saved+forgone+cost), '
             'PREREG formula, reported beside achieved precision.')
    L.append('| label | k | variant | direction | n fired | share cut | dR | day t | ex-top5% | achieved P | '
             'break-even P | saved | forgone | cost | identity residual |')
    L.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    sub = reads_c[reads_c['tau'] == 0.7]
    passers = []
    for (label, k, variant), g in reads_c.groupby(['label', 'k', 'variant']):
        gg = g.set_index('direction')
        if not set(gg.index) >= {'TRAIN->VAL', 'VAL->TRAIN-H2 (swap)'}:
            continue
        for tau in TAUS:
            gt = g[g['tau'] == tau].set_index('direction')
            if not set(gt.index) >= {'TRAIN->VAL', 'VAL->TRAIN-H2 (swap)'}:
                continue
            ok = all(pd.notna(gt.loc[d, 'mean_dR']) and gt.loc[d, 'mean_dR'] >= 0.05 and
                     gt.loc[d, 'day_t'] >= 2.5 and pd.notna(gt.loc[d, 'ex_top5_dR']) and gt.loc[d, 'ex_top5_dR'] > 0
                     for d in ['TRAIN->VAL', 'VAL->TRAIN-H2 (swap)'])
            if ok:
                passers.append((label, k, variant, tau))
    for _, r in sub.iterrows():
        if r['n_fired'] == 0 or pd.isna(r.get('mean_dR')):
            continue
        L.append(f"| {r['label']} | {r['k']} | {r['variant']} | {r['direction']} | {r['n_fired']} | "
                  f"{r['share_cut']:.3f} | {r['mean_dR']:.3f} | {r['day_t']:.2f} | {r['ex_top5_dR']:.3f} | "
                  f"{r['achieved_precision']:.2f} | {r['breakeven_precision']:.2f} | {r['saved_mean_TP']:.3f} | "
                  f"{r['forgone_mean_FP']:.3f} | {r['cost_mean']:.4f} | {r['identity_residual']:.2e} |")
    L.append('')
    L.append(f'**Pass bar (dR>=+0.05R, day t>=2.5, ex-top5%>0, BOTH out-of-sample scorings) across all '
             f'{len(TAUS)} tau x 2 variants x 3 labels x 2 k = 96 cells: {len(passers)} pass -- '
             f'{passers if passers else "none"}.**')
    L.append('')

    L.append('## Verdicts')
    L.append(f'* Part 3 cells clearing the pass bar on both out-of-sample scorings: {len(passers)}/96.')
    L.append('* cS5/S5 (SPY-relative) is void wherever bars_sip carries SPY for a single day only (same as cell 1,668); '
             'HistGradientBoostingClassifier handles the resulting NaNs natively, no imputation.')
    L.append('')

    L.append('## Adequacy review')
    L.append('* minutes_to_exit is measured from the FILL BAR (bars["minarr"][i0]), matching Part 2\'s own label '
             'wording ("stop-out within k min of the fill bar"), not from the fractional fill instant -- a fill '
             'landing late in its own bar reads as up to ~1 minute faster here than a fill-instant clock would show.')
    L.append('* MFE and level re-cross include the exit bar\'s own high (intrabar order of high vs low is unknown, '
             'same convention as 1668\'s cS2_mfe); this is an optimistic (upper-bound) MFE, not a certified touch.')
    L.append('* Exit-kind from the unbounded walk matched base_exit_type on '
             f'{100 * (1 - n_mismatch / max(n_pop, 1)):.1f}% of fills; mismatches (if any) are logged, not silently dropped.')
    L.append('* No Part 2/3 feature or label used a bar after its own decision instant (k=0 uses bars[0..i0] only; '
             'k=1 gates on walk_k\'s preempt=="" so a fill already resolved by bar fill+1 never enters the k=1 model).')
    L.append('* A null in Part 3 is a claim about these τ/k/label/variant cells specifically -- MDE at achieved n is '
             'reported per cell in 1669_reads.csv; break-even precision is compared to achieved precision, not asserted.')
    with open(RESULT_MD, 'w') as f:
        f.write('\n'.join(L) + '\n')
    logger.info('wrote %s (%d lines)', RESULT_MD, len(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()

    setup_logging()
    logger.info('cell 1,669 starting (resume=%s limit=%s talib=%s)', args.resume, args.limit, TALIB_AVAILABLE)
    f1668.check_disk()

    pop = load_population()
    if args.limit:
        pop = pop.iloc[:args.limit].copy()
        logger.info('--limit %d: population truncated to %d rows', args.limit, len(pop))

    store = f1668.BarStore(BARS_DB)
    try:
        per_fill, fire_df, n_mismatch = sweep(pop, store, resume=args.resume, limit=None)
    finally:
        store.close()

    sd_by_half = {h: pop.loc[pop['half'] == h, 'net_R'].std(ddof=1) for h in HALVES}
    logger.info('book SD (net_R) by half: %s', sd_by_half)

    anatomy_df, fire_rate_df = anatomy_tables(per_fill, fire_df)
    tmp = ANATOMY_CSV + '.tmp'
    pd.concat([anatomy_df, fire_rate_df], ignore_index=True, sort=False).to_csv(tmp, index=False)
    os.replace(tmp, ANATOMY_CSV)
    logger.info('wrote %s (%d anatomy rows, %d fire-rate rows)', ANATOMY_CSV, len(anatomy_df), len(fire_rate_df))

    reads_b, importance_tables, scored = run_part2(per_fill)
    for (label, k, direction), sdf in scored.items():
        col = f'pstop_{label}_{k}'
        if col not in per_fill.columns:
            per_fill[col] = np.nan
        per_fill.loc[sdf.index, col] = sdf['p'].values

    reads_c = run_part3(scored, sd_by_half)

    tmp = PER_FILL_CSV + '.tmp'
    per_fill.to_csv(tmp, index=False)
    os.replace(tmp, PER_FILL_CSV)
    logger.info('final %s: %d rows, %d cols', PER_FILL_CSV, *per_fill.shape)

    reads_all = pd.concat([reads_b, reads_c], ignore_index=True, sort=False)
    reads_all.to_csv(READS_CSV, index=False)
    logger.info('wrote %s (%d rows)', READS_CSV, len(reads_all))

    write_result_md(per_fill, anatomy_df, fire_rate_df, reads_b, reads_c, importance_tables, len(pop), n_mismatch)
    logger.info('cell 1,669 done')


if __name__ == '__main__':
    main()
