#!/usr/bin/env python3
"""Cell 1,668: post-entry failure detection -- cut the trade early when the
first minutes say it will stop out.

PREREG: research/hod_entry/PREREG_1668.md (FROZEN 2026-09-29 18:31 UTC),
including amendments 1 (S9 effort-without-result), 2 (bar-shape features),
3 (TA-Lib candlestick patterns).

Owner ask (2026-09-29): "we entered, but then we find out that we better cut
our losses now due to indicators that increase the likelihood that this trade
will stop out."

Population: 1663_features.csv (the 5,506-row floored primary book, r_pct >=
1.5%), joined 1:1 on (date,symbol) to fills_1658.csv for (fill_id, split) and
to causal_arming_causal.csv (status=='fill') for `level`. Post-entry path from
research/bf_zero/bars_sip.db, opened read-only, one query per (symbol,day) on
the (symbol,day,t) primary key -- never a full scan.

Part A: 9 pre-declared early-cut rules (S1-S9) x 4 horizons (k=3,5,10,15) x
2 halves, evaluated bar-by-bar; a stop/target/EOD hit at or before minute k
takes precedence over any cut (stop before target in a bar).

Part B: a HistGradientBoosting failure classifier on continuous features at
k=5 and k=10 (continuous S1-S7 + progress-per-volume + bar-shape + TA-Lib
candlestick patterns), trained TRAIN-H2->VAL and swapped, tau fitted on the
training half only, scored out of sample, with a within-day label-shuffle
placebo.

Usage:
    python3 1668_failure.py [--resume] [--limit N]

Outputs (research/hod_entry/):
    1668_per_fill.csv -- fill-level: fire flag + dR per rule x k, Part B
                          continuous/shape/TA-Lib features at k=5/10, and the
                          out-of-sample P(stop) at k=5/10
    1668_reads.csv     -- every Part A read (family=A) and Part B read (family=B)
    1668_failure.log   -- verbose progress log
    RESULT_1668.md      -- coverage, tables, verdicts, adequacy review
"""
import argparse
import logging
import math
import os
import sqlite3
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

# Constrain thread usage before numpy/sklearn import -- shared 2-CPU node,
# another research job runs beside this one; no multiprocessing anywhere.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '1')

import numpy as np
import pandas as pd

try:
    import talib
    TALIB_AVAILABLE = True
    CDL_NAMES = talib.get_function_groups()['Pattern Recognition']
except ImportError:
    TALIB_AVAILABLE = False
    CDL_NAMES = []

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance

_ET = ZoneInfo('America/New_York')
_et_offset_cache = {}

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_CSV = os.path.join(HERE, '1663_features.csv')
FILLS_CSV = os.path.join(HERE, 'fills_1658.csv')
CAUSAL_CSV = os.path.join(HERE, 'causal_arming_causal.csv')
BARS_DB = os.path.join(os.path.dirname(HERE), 'bf_zero', 'bars_sip.db')

PER_FILL_CSV = os.path.join(HERE, '1668_per_fill.csv')
READS_CSV = os.path.join(HERE, '1668_reads.csv')
LOG_FILE = os.path.join(HERE, '1668_failure.log')
RESULT_MD = os.path.join(HERE, 'RESULT_1668.md')

HALVES = ['TRAIN-H2', 'VAL']
ALL_K = [3, 5, 10, 15]
PARTB_K = [5, 10]
RULES = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9']

EOD_M = 955        # 15:55 ET, matches research/hod_entry/sip_rebuild.py's EOD_M
LEVEL_TOL = 0.01   # $ tolerance matching a bar's high to the level (matches 1667_sweep.py)

# Costs (PREREG population section): entry 7bps and the base exit's own cost are
# already inside the CSV's net_R. The early-cut alternative gets the SAME entry
# cost convention (it cancels exactly in every paired dR, since it is a per-fill
# constant added on both sides) plus its own 6bps marketable-sell cost.
ENTRY_BPS = 0.0007
CUT_BPS = 0.0006

Z_MDE = 1.959964 + 0.841621  # two-sided alpha .05 (1.96) + 80% power (0.84)
RNG_SEED = 1668

logger = logging.getLogger('1668')


def setup_logging():
    """Log to both 1668_failure.log and stdout, verbose (INFO) per project rules."""
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(LOG_FILE, mode='w')
    fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter('%(levelname)s %(message)s'))
    logger.addHandler(fh)
    logger.addHandler(sh)


def check_disk(min_gb=5.0):
    """Abort if free space on / is below min_gb (project rule)."""
    st = os.statvfs('/')
    free_gb = st.f_bavail * st.f_frsize / (1024 ** 3)
    logger.info('disk free on /: %.1f GB', free_gb)
    if free_gb < min_gb:
        logger.error('disk free %.1f GB < %.1f GB floor -- aborting', free_gb, min_gb)
        sys.exit(1)


# ---------------------------------------------------------------------------
# ET timing -- copied from research/hod_entry/1667_sweep.py per PREREG's
# instruction to reuse the working, DST-aware conversion.
# ---------------------------------------------------------------------------

def et_offset_minutes(day_str):
    """UTC->ET offset in minutes for a trading day (handles DST); cached per day."""
    if day_str not in _et_offset_cache:
        dt_utc = datetime.fromisoformat(day_str + 'T12:00:00+00:00')
        dt_et = dt_utc.astimezone(_ET)
        _et_offset_cache[day_str] = dt_et.utcoffset().total_seconds() / 60.0
    return _et_offset_cache[day_str]


def minute_of_day(t_iso, day_str):
    """Parse bars_sip's UTC 'YYYY-MM-DDTHH:MM:SS+00:00' into ET minutes-since-
    midnight of `day_str` (the trading day)."""
    hh = int(t_iso[11:13])
    mm = int(t_iso[14:16])
    utc_minute = hh * 60 + mm
    if t_iso[:10] != day_str:
        utc_minute += 1440
    return utc_minute + et_offset_minutes(day_str)


# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------

def load_population():
    """1663_features.csv (5,506-row floored primary book) + fill_id/split from
    fills_1658.csv + level from causal_arming_causal.csv, all joined 1:1 on
    (date,symbol) -- the key RESULT_1663 verified unique."""
    base = pd.read_csv(BASE_CSV, dtype={'date': str, 'symbol': str})
    logger.info('loaded %s: %d rows, halves=%s', BASE_CSV, len(base),
                dict(base['half'].value_counts()))

    fills = pd.read_csv(FILLS_CSV, dtype={'date': str, 'symbol': str},
                         usecols=['fill_id', 'date', 'symbol', 'split'])
    merged = base.merge(fills, on=['date', 'symbol'], how='left', validate='one_to_one')
    n_miss = merged['fill_id'].isna().sum()
    if n_miss:
        logger.error('%d/%d rows failed to match fill_id in fills_1658 join', n_miss, len(merged))
    bad_split = (merged['split'] != merged['half']).sum()
    if bad_split:
        logger.warning('%d rows where fills_1658.split != 1663.half', bad_split)

    causal = pd.read_csv(CAUSAL_CSV, dtype={'day': str, 'symbol': str})
    causal = causal[causal['status'] == 'fill'][['day', 'symbol', 'level', 'fill', 'stop']]
    causal = causal.rename(columns={'fill': 'c_fill', 'stop': 'c_stop'})
    dup = causal.duplicated(['day', 'symbol']).sum()
    if dup:
        logger.warning('causal_arming_causal has %d duplicate (day,symbol) keys among fills', dup)
    merged = merged.merge(causal, left_on=['date', 'symbol'], right_on=['day', 'symbol'], how='left')
    n_miss_level = merged['level'].isna().sum()
    if n_miss_level:
        logger.warning('%d/%d rows failed to match level in causal join', n_miss_level, len(merged))
    bad_entry = (merged['c_fill'].notna() & ((merged['c_fill'] - merged['entry_price']).abs() > 0.01)).sum()
    bad_stop = (merged['c_stop'].notna() & ((merged['c_stop'] - merged['stop']).abs() > 0.01)).sum()
    if bad_entry or bad_stop:
        logger.warning('cross-check: %d entry_price and %d stop mismatches vs causal fill/stop', bad_entry, bad_stop)
    merged = merged.drop(columns=['day', 'c_fill', 'c_stop'])
    logger.info('population ready: %d rows, %d with a matched level (%.1f%%)',
                len(merged), merged['level'].notna().sum(), 100.0 * merged['level'].notna().mean())
    return merged


# ---------------------------------------------------------------------------
# bars_sip.db access -- one connection, one query per (symbol,day) on the
# (symbol,day,t) primary key.
# ---------------------------------------------------------------------------

class BarStore:
    """Read-only bars_sip.db access. Each (symbol,day) is queried once per
    fill (the population is 1:1 on (date,symbol)); SPY bars are cached per
    day since many fills share a trading day."""

    def __init__(self, path):
        self.con = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
        self.cur = self.con.cursor()
        self._spy_cache = {}

    def day_bars(self, symbol, day):
        rows = self.cur.execute(
            'SELECT t,o,h,l,c,v FROM bars WHERE symbol=? AND day=? ORDER BY t',
            (symbol, day)).fetchall()
        if not rows:
            return None
        minarr = np.array([minute_of_day(r[0], day) for r in rows], dtype=float)
        return dict(
            minarr=minarr,
            o=np.array([r[1] for r in rows], dtype=float),
            h=np.array([r[2] for r in rows], dtype=float),
            l=np.array([r[3] for r in rows], dtype=float),
            c=np.array([r[4] for r in rows], dtype=float),
            v=np.array([r[5] for r in rows], dtype=float),
        )

    def spy_bars(self, day):
        if day not in self._spy_cache:
            self._spy_cache[day] = self.day_bars('SPY', day)
        return self._spy_cache[day]

    def close(self):
        self.con.close()


def find_fill_index(bars, fill_min):
    """Index of the bar whose 1-minute window contains the fill instant: the
    LAST bar at or before fill_min. fill_min is a fractional ET instant
    (e.g. 644.8) while bar minutes are whole integers, so this must be a
    floor/last-at-or-before match, never nearest-neighbor -- nearest-neighbor
    can jump to the bar AFTER the fill (across a gap, or simply because the
    fill is >0.5 min into its own bar), which then mislabels a bar that
    starts after the fill as 'the fill bar' (the fractional-minute look-ahead
    pitfall from cell 1,667: 9/26 memory note). This matches the project's
    established causal 'last bar at or before' convention (1667_sweep.py)."""
    idx = np.where(bars['minarr'] <= fill_min)[0]
    if len(idx) == 0:
        return None
    return int(idx[-1])


def find_break_bar(bars, fill_min, level):
    """First bar at or before fill_min whose high is within LEVEL_TOL of
    `level` (the breakout bar) -- same search 1667_sweep.py uses for F11-F16."""
    if pd.isna(level):
        return None
    cutoff_idx = np.where(bars['minarr'] <= fill_min)[0]
    for i in cutoff_idx:
        if abs(bars['h'][i] - level) <= LEVEL_TOL:
            return int(i)
    return None


def spy_close_at_or_before(spy_bars, target_min):
    if spy_bars is None:
        return None
    idx = np.where(spy_bars['minarr'] <= target_min)[0]
    if len(idx) == 0:
        return None
    return spy_bars['c'][idx[-1]]


# ---------------------------------------------------------------------------
# Bar-by-bar precedence walk and the shared early-cut dR primitive
# ---------------------------------------------------------------------------

def walk_k(bars, i0, k, stop, target):
    """Bar-by-bar precedence walk over bars i0+1..i0+k. None if bar k doesn't
    exist (not computable). Else a dict with `preempt` ('' / 'stop' /
    'target' / 'eod') and the k-bar aggregates S1-S9 need. A stop or target
    touch takes precedence over reaching minute k clean (stop wins a tie)."""
    n = len(bars['o'])
    last_idx = i0 + k
    if last_idx >= n:
        return None
    preempt = ''
    for j in range(i0 + 1, i0 + k + 1):
        if bars['minarr'][j] >= EOD_M:
            preempt = 'eod'
            break
        if bars['l'][j] <= stop:
            preempt = 'stop'
            break
        if bars['h'][j] >= target:
            preempt = 'target'
            break
    seg = slice(i0 + 1, i0 + k + 1)
    return dict(
        preempt=preempt, last_idx=last_idx,
        close_k=bars['c'][last_idx], open_k=bars['o'][last_idx],
        high_max=bars['h'][seg].max(), low_min=bars['l'][seg].min(),
        mean_v=bars['v'][seg].mean(), sum_v=bars['v'][seg].sum(),
        next_open=(bars['o'][last_idx + 1] if (last_idx + 1) < n else None),
    )


def dR_cut(entry, stop, base_net_R, next_open):
    """dR of cutting at the open of bar k+1 vs the recorded base exit.
    Cost: entry ENTRY_BPS (same convention as the base's own net_R -- a
    per-fill constant that cancels exactly in this difference) + CUT_BPS of
    the cut price (a marketable sell). Falls back to 0 (no cut executable)
    when there is no bar k+1 to exit into -- the position rides to whatever
    the base exit already was."""
    if next_open is None:
        return 0.0
    R_unit = entry - stop
    gross = (next_open - entry) / R_unit
    cost = (ENTRY_BPS * entry + CUT_BPS * next_open) / R_unit
    return (gross - cost) - base_net_R


def eval_rules(walk, level, entry, stop, break_bar_v, spy_ret, vwap_k):
    """S1-S9 fire flags at this horizon (PREREG Part A + amendment 1).
    NaN when a needed input is missing; False (not fired, not unknown) when
    the trade already closed before this horizon -- the cut mechanism is
    moot, not undefined."""
    if walk['preempt'] != '':
        return {r: False for r in RULES}
    close_k, high_max, low_min, mean_v = walk['close_k'], walk['high_max'], walk['low_min'], walk['mean_v']
    have_bb = pd.notna(break_bar_v) and break_bar_v > 0
    out = {}
    out['S1'] = (close_k < level) if pd.notna(level) else np.nan
    out['S2'] = bool(high_max <= entry)
    out['S3'] = bool(close_k < entry)
    out['S4'] = (mean_v < 0.5 * break_bar_v) if have_bb else np.nan
    out['S5'] = (spy_ret < -0.002) if spy_ret is not None else np.nan
    out['S6'] = (close_k < vwap_k) if vwap_k is not None else np.nan
    out['S7'] = bool(low_min <= entry - 0.5 * (entry - stop))
    out['S8'] = (out['S2'] and out['S4']) if have_bb else np.nan
    out['S9'] = ((high_max <= entry + 0.25 * (entry - stop)) and (mean_v >= 1.5 * break_bar_v)) if have_bb else np.nan
    return out


def continuous_features(walk, level, entry, stop, break_bar_v, spy_ret, vwap_k):
    """Part B continuous versions of S1-S7 + amendment-1 progress-per-volume."""
    R = entry - stop
    close_k, high_max, low_min, mean_v, sum_v = (
        walk['close_k'], walk['high_max'], walk['low_min'], walk['mean_v'], walk['sum_v'])
    have_bb = pd.notna(break_bar_v) and break_bar_v > 0
    f = {}
    f['cS1_dist_level'] = (close_k - level) / R if pd.notna(level) else np.nan
    f['cS2_mfe'] = (high_max - entry) / R
    f['cS3_ret'] = (close_k - entry) / R
    f['cS4_volratio'] = (mean_v / break_bar_v) if have_bb else np.nan
    f['cS5_spyret'] = spy_ret if spy_ret is not None else np.nan
    f['cS6_dist_vwap'] = (close_k - vwap_k) / R if vwap_k is not None else np.nan
    f['cS7_mae'] = (entry - low_min) / R
    volratio_sum = (sum_v / break_bar_v) if have_bb else np.nan
    f['cA1_progvol'] = (f['cS3_ret'] / volratio_sum) if pd.notna(volratio_sum) and volratio_sum != 0 else np.nan
    return f


def shape_features(bars, i0, k):
    """Amendment 2: CLV / wick / body / red-share on bars fill+1..fill+k
    (never the fill bar). CLV, wick and body use the LAST bar only; mean CLV
    and red share use all k bars."""
    last = i0 + k
    hi_l, lo_l, cl_l, op_l = bars['h'][last], bars['l'][last], bars['c'][last], bars['o'][last]
    rng_l = hi_l - lo_l
    f = {
        'clv_last': (cl_l - lo_l) / rng_l if rng_l > 0 else np.nan,
        'wick_last': (min(op_l, cl_l) - lo_l) / rng_l if rng_l > 0 else np.nan,
        'body_last': abs(cl_l - op_l) / rng_l if rng_l > 0 else np.nan,
    }
    clvs, reds = [], []
    for j in range(i0 + 1, last + 1):
        rng = bars['h'][j] - bars['l'][j]
        if rng > 0:
            clvs.append((bars['c'][j] - bars['l'][j]) / rng)
        reds.append(bars['c'][j] < bars['o'][j])
    f['clv_mean'] = float(np.mean(clvs)) if clvs else np.nan
    f['red_share'] = float(np.mean(reds)) if reds else np.nan
    return f


def talib_features(bars, i0, k, want_fired_any=False):
    """Amendment 3: 61 CDL* patterns on bars 0..fill+k (lookback may reach
    bars before the fill; nothing after bar fill+k). Returns (features for
    the per-fill table, optional per-pattern 'fired anywhere in
    fill+1..fill+k' dict for the informational fire-rate table)."""
    last = i0 + k
    o, h, l, c = bars['o'][:last + 1], bars['h'][:last + 1], bars['l'][:last + 1], bars['c'][:last + 1]
    n_bull, n_bear = 0, 0
    last_flags = {}
    fired_any = {} if want_fired_any else None
    for name in CDL_NAMES:
        vals = getattr(talib, name)(o, h, l, c)
        window = vals[i0 + 1:last + 1]
        n_bull += int((window > 0).sum())
        n_bear += int((window < 0).sum())
        last_flags[f'cdl_{name}'] = int(vals[last])
        if want_fired_any:
            fired_any[name] = bool((window != 0).any())
    out = {'talib_nbull': n_bull, 'talib_nbear': n_bear}
    out.update(last_flags)
    return out, fired_any


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def sweep(pop, store, resume=False, limit=None):
    """One pass per fill: locate the fill bar and break bar, then for every
    k in ALL_K walk the precedence and evaluate S1-S9; for k in PARTB_K also
    build the Part B feature block. Writes 1668_per_fill.csv incrementally
    (atomic tmp+replace) every 500 fills."""
    if limit:
        pop = pop.iloc[:limit].copy()
    if resume and os.path.exists(PER_FILL_CSV):
        cached = pd.read_csv(PER_FILL_CSV, dtype={'date': str, 'symbol': str})
        if len(cached) == len(pop):
            logger.info('--resume: reusing complete cache %s (%d rows)', PER_FILL_CSV, len(cached))
            return cached, pd.DataFrame()
        logger.info('--resume requested but cache incomplete (%d vs %d rows) -- recomputing',
                     len(cached), len(pop))

    rows, fired_any_rows = [], []
    n_no_bars = n_no_fill_idx = n_no_break = n_high_check = n_high_bad = 0
    t0 = time.time()
    pop = pop.reset_index(drop=True)
    for i, r in pop.iterrows():
        rec = dict(fill_id=r['fill_id'], date=r['date'], symbol=r['symbol'], split=r['half'],
                    entry_price=r['entry_price'], stop=r['stop'], target_price=r['target_price'],
                    level=r['level'], r_pct=r['r_pct'], atr14_pct=r['atr14_pct'],
                    base_exit_type=r['exit_type'], base_net_R=r['net_R'])
        bars = store.day_bars(r['symbol'], r['date'])
        if bars is None:
            n_no_bars += 1
            rows.append(rec)
            continue
        i0 = find_fill_index(bars, r['fill_min'])
        if i0 is None:
            n_no_fill_idx += 1
            rows.append(rec)
            continue
        if n_high_check < 25:
            n_high_check += 1
            if bars['h'][i0] < r['entry_price'] - 0.005:
                n_high_bad += 1
                logger.warning('%s %s: fill bar high %.4f < fill price %.4f (fill_min=%.1f) -- spot check',
                                r['date'], r['symbol'], bars['h'][i0], r['entry_price'], r['fill_min'])

        break_idx = find_break_bar(bars, r['fill_min'], r['level'])
        break_bar_v = bars['v'][break_idx] if break_idx is not None else np.nan
        if break_idx is None:
            n_no_break += 1

        spy_bars = store.spy_bars(r['date'])
        spy_at_fill = spy_close_at_or_before(spy_bars, r['fill_min'])
        v_cum = np.cumsum(bars['v'])
        tp = (bars['h'] + bars['l'] + bars['c']) / 3.0
        tpv_cum = np.cumsum(tp * bars['v'])

        for k in ALL_K:
            w = walk_k(bars, i0, k, r['stop'], r['target_price'])
            rec[f'k{k}_computable'] = w is not None
            if w is None:
                continue
            rec[f'k{k}_preempt'] = w['preempt']
            spy_ret = None
            if spy_at_fill is not None and spy_at_fill != 0:
                spy_at_k = spy_close_at_or_before(spy_bars, bars['minarr'][w['last_idx']])
                if spy_at_k is not None:
                    spy_ret = spy_at_k / spy_at_fill - 1.0
            vwap_k = (tpv_cum[w['last_idx']] / v_cum[w['last_idx']]) if v_cum[w['last_idx']] > 0 else None
            fired = eval_rules(w, r['level'], r['entry_price'], r['stop'], break_bar_v, spy_ret, vwap_k)
            dcut = dR_cut(r['entry_price'], r['stop'], r['net_R'], w['next_open']) if w['preempt'] == '' else 0.0
            for s in RULES:
                rec[f'{s}_{k}'] = fired[s]
                rec[f'dR_{s}_{k}'] = np.nan if pd.isna(fired[s]) else (dcut if fired[s] else 0.0)

            if k in PARTB_K and w['preempt'] == '':
                cf = continuous_features(w, r['level'], r['entry_price'], r['stop'], break_bar_v, spy_ret, vwap_k)
                sf = shape_features(bars, i0, k)
                rec.update({f'{key}_{k}': val for key, val in cf.items()})
                rec.update({f'{key}_{k}': val for key, val in sf.items()})
                rec[f'label_stop_{k}'] = (r['exit_type'] == 'stop')
                rec[f'dR_cut_{k}'] = dcut
                if TALIB_AVAILABLE:
                    tf, fired_any = talib_features(bars, i0, k, want_fired_any=(k == 5))
                    rec.update({f'{key}_{k}': val for key, val in tf.items()})
                    if k == 5:
                        fa_rec = dict(fill_id=r['fill_id'], date=r['date'], split=r['half'],
                                       base_exit_type=r['exit_type'])
                        fa_rec.update(fired_any)
                        fired_any_rows.append(fa_rec)

        rows.append(rec)
        if (i + 1) % 500 == 0 or (i + 1) == len(pop):
            elapsed = time.time() - t0
            logger.info('sweep %d/%d fills (%.1fs elapsed, %d no-bars, %d no-fill-idx, %d no-break-bar)',
                         i + 1, len(pop), elapsed, n_no_bars, n_no_fill_idx, n_no_break)
            out = pd.DataFrame(rows)
            tmp = PER_FILL_CSV + '.tmp'
            out.to_csv(tmp, index=False)
            os.replace(tmp, PER_FILL_CSV)

    logger.info('sweep done: %d fills, %d no-bars-at-all, %d no-fill-bar-match, %d no-break-bar-match '
                '(fill-bar high>=price spot check: %d/%d ok)',
                len(pop), n_no_bars, n_no_fill_idx, n_no_break, n_high_check - n_high_bad, n_high_check)
    return pd.DataFrame(rows), pd.DataFrame(fired_any_rows)


# ---------------------------------------------------------------------------
# Statistics (formulas match research/hod_entry/1667_sweep.py's conventions)
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


def day_clustered_t(dates, vals):
    df = pd.DataFrame({'date': dates, 'v': vals}).dropna()
    day_means = df.groupby('date')['v'].mean()
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
    kk = int(math.ceil(n * 0.05))
    if kk >= n:
        return vals.mean()
    return vals.iloc[:n - kk].mean()


def mde(sd, n):
    if n <= 0 or pd.isna(sd):
        return np.nan
    return Z_MDE * sd / math.sqrt(n)


# ---------------------------------------------------------------------------
# Part A reads
# ---------------------------------------------------------------------------

def part_a_reads(per_fill, sd_by_half):
    """Every rule x horizon x half: n, share fired, paired dR on the whole
    (computable) book with iid/day-clustered t, ex-top-5% dR, MDE, plus the
    dR on the fired subset alone."""
    reads = []
    for half in HALVES:
        sub_half = per_fill[per_fill['split'] == half]
        for k in ALL_K:
            comp_col = f'k{k}_computable'
            comp = sub_half[sub_half[comp_col] == True]  # noqa: E712
            for s in RULES:
                fcol, dcol = f'{s}_{k}', f'dR_{s}_{k}'
                d = comp[['date', fcol, dcol]].dropna(subset=[dcol])
                n = len(d)
                n_fired = int(d[fcol].sum()) if n else 0
                fired_rows = d[d[fcol] == True]  # noqa: E712
                reads.append(dict(
                    family='A', rule=s, k=k, half=half, n=n, n_fired=n_fired,
                    share_fired=(n_fired / n if n else np.nan),
                    mean_dR=(d[dcol].mean() if n else np.nan),
                    iid_t=iid_t(d[dcol]), day_t=day_clustered_t(d['date'], d[dcol]),
                    ex_top5_dR=ex_top5_mean(d[dcol]), mde=mde(sd_by_half[half], n),
                    fired_mean_dR=(fired_rows[dcol].mean() if n_fired else np.nan),
                ))
    return pd.DataFrame(reads)


# ---------------------------------------------------------------------------
# Part B: trained failure classifier
# ---------------------------------------------------------------------------

def build_xy(per_fill, half, k):
    label_col = f'label_stop_{k}'
    sub = per_fill[(per_fill['split'] == half) & per_fill[label_col].notna()].copy()
    prefixes = ('cS1_', 'cS2_', 'cS3_', 'cS4_', 'cS5_', 'cS6_', 'cS7_', 'cA1_',
                'clv_', 'wick_', 'body_', 'red_share_', 'talib_', 'cdl_')
    feat_cols = [c for c in sub.columns if c.endswith(f'_{k}') and c.startswith(prefixes)]
    feat_cols += ['r_pct', 'atr14_pct']
    X = sub[feat_cols].astype(float)
    y = sub[label_col].astype(int)
    dr = sub[f'dR_cut_{k}']
    return X, y, dr, sub['date'], feat_cols, sub.index


def fit_tau(y_true, p_stop, dr_cut):
    """Grid search tau in [0.05,0.95] maximising mean paired dR on THIS
    (training) half -- tau is fitted, never hand-picked."""
    grid = np.arange(0.05, 0.951, 0.01)
    best_tau, best_dr = 0.5, -np.inf
    for tau in grid:
        dr = np.where(p_stop > tau, dr_cut, 0.0).mean()
        if dr > best_dr:
            best_dr, best_tau = dr, tau
    return best_tau, best_dr


def run_part_b(per_fill):
    reads = []
    importance_tables = {}
    pstop_series = {}
    rng = np.random.RandomState(RNG_SEED)
    for k in PARTB_K:
        Xtr, ytr, drtr, datetr, feat_cols, idxtr = build_xy(per_fill, 'TRAIN-H2', k)
        Xval, yval, drval, dateval, _, idxval = build_xy(per_fill, 'VAL', k)
        logger.info('Part B k=%d: TRAIN-H2 n=%d (%.1f%% stop), VAL n=%d (%.1f%% stop), %d features',
                     k, len(ytr), 100 * ytr.mean(), len(yval), 100 * yval.mean(), len(feat_cols))

        directions = {
            'TRAIN->VAL': (Xtr, ytr, drtr, datetr, Xval, yval, drval, dateval, idxval),
            'VAL->TRAIN-H2 (swap)': (Xval, yval, drval, dateval, Xtr, ytr, drtr, datetr, idxtr),
        }
        k_pstop = pd.Series(index=per_fill.index, dtype=float)
        for direction, (Xa, ya, dra, datea, Xb, yb, drb, dateb, idxb) in directions.items():
            model = HistGradientBoostingClassifier(max_iter=200, random_state=RNG_SEED)
            model.fit(Xa, ya)
            p_train = model.predict_proba(Xa)[:, 1]
            tau, train_dr_insample = fit_tau(ya.values, p_train, dra.values)
            p_score = model.predict_proba(Xb)[:, 1]
            k_pstop.loc[idxb] = p_score
            auc = roc_auc_score(yb, p_score) if yb.nunique() > 1 else np.nan
            fire = p_score > tau
            dr_vals = np.where(fire, drb.values, 0.0)

            ya_shuf = pd.Series(ya.values, index=ya.index).groupby(datea.values).transform(
                lambda s: rng.permutation(s.values))
            model_pl = HistGradientBoostingClassifier(max_iter=200, random_state=RNG_SEED)
            model_pl.fit(Xa, ya_shuf)
            p_train_pl = model_pl.predict_proba(Xa)[:, 1]
            tau_pl, _ = fit_tau(ya_shuf.values, p_train_pl, dra.values)
            p_score_pl = model_pl.predict_proba(Xb)[:, 1]
            auc_pl = roc_auc_score(yb, p_score_pl) if yb.nunique() > 1 else np.nan
            dr_pl = np.where(p_score_pl > tau_pl, drb.values, 0.0).mean()

            reads.append(dict(
                family='B', k=k, direction=direction, n_train=len(ya), n_score=len(yb),
                tau=tau, auc=auc, mean_dR=dr_vals.mean(),
                iid_t=iid_t(dr_vals), day_t=day_clustered_t(dateb.values, dr_vals),
                ex_top5_dR=ex_top5_mean(dr_vals), share_cut=float(fire.mean()),
                train_insample_dR=train_dr_insample,
                placebo_auc=auc_pl, placebo_dR=dr_pl,
            ))

            if direction == 'TRAIN->VAL':
                logger.info('Part B k=%d permutation importance on VAL (n_repeats=5, n_jobs=1)', k)
                pi = permutation_importance(model, Xb, yb, n_repeats=5, random_state=RNG_SEED, n_jobs=1)
                importance_tables[k] = pd.DataFrame({
                    'feature': feat_cols, 'importance_mean': pi.importances_mean,
                    'importance_std': pi.importances_std}).sort_values('importance_mean', ascending=False)
        pstop_series[k] = k_pstop
        logger.info('Part B k=%d done', k)
    return pd.DataFrame(reads), importance_tables, pstop_series


def fire_rate_table(fired_any_df):
    """Amendment 3 informational table: fire rate of each of the 61 patterns
    on bars fill+1..fill+5, later stop-outs vs later target/EOD, both halves."""
    if fired_any_df.empty:
        return pd.DataFrame()
    rows = []
    for name in CDL_NAMES:
        if name not in fired_any_df.columns:
            continue
        for half in HALVES:
            sub = fired_any_df[fired_any_df['split'] == half]
            stop_sub = sub[sub['base_exit_type'] == 'stop']
            other_sub = sub[sub['base_exit_type'] != 'stop']
            rows.append(dict(
                pattern=name, half=half,
                stop_fire_rate=(stop_sub[name].mean() if len(stop_sub) else np.nan), stop_n=len(stop_sub),
                other_fire_rate=(other_sub[name].mean() if len(other_sub) else np.nan), other_n=len(other_sub),
            ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# RESULT.md
# ---------------------------------------------------------------------------

def write_result_md(per_fill, reads_a, reads_b, importance_tables, fire_rates, sd_by_half, n_pop):
    lines = []
    lines.append('# RESULT 1,668 -- post-entry failure detection\n')
    lines.append(f'PREREG: `research/hod_entry/PREREG_1668.md` (FROZEN 2026-09-29 18:31 UTC) '
                 f'+ amendments 1-3. Population n={n_pop} (5,506 x1.5%-floored primary book).\n')

    lines.append('## Coverage')
    for k in ALL_K:
        comp = per_fill[f'k{k}_computable']
        lines.append(f'* k={k}: computable {100 * comp.mean():.1f}% ({int(comp.sum())}/{len(per_fill)})')
    lines.append(f'* level matched (causal join): {100 * per_fill["level"].notna().mean():.1f}%')
    spy_cov = per_fill['S5_5'].notna().mean() if 'S5_5' in per_fill.columns else float('nan')
    lines.append(f'* S5 (SPY) coverage at k=5: {100 * spy_cov:.1f}% -- bars_sip has SPY for a single day only; '
                 'S5/cS5 are VOID by the rule\'s own escape clause, not reported further.')
    lines.append('')

    lines.append('## Part A -- every rule x horizon, both halves (paired dR vs the base, whole computable book)')
    lines.append('| Rule | k | n TR/VAL | fired% TR/VAL | dR TR/VAL | day-t TR/VAL | ex-top5% TR/VAL | fired-only dR TR/VAL |')
    lines.append('|---|---|---|---|---|---|---|---|')
    passers = []
    for (rule, k), g in reads_a.groupby(['rule', 'k'], sort=False):
        gg = g.set_index('half')
        if not (set(HALVES) <= set(gg.index)):
            continue
        tr, va = gg.loc['TRAIN-H2'], gg.loc['VAL']
        lines.append(f"| {rule} | {k} | {tr['n']}/{va['n']} | {tr['share_fired']:.2f}/{va['share_fired']:.2f} | "
                     f"{tr['mean_dR']:.3f}/{va['mean_dR']:.3f} | {tr['day_t']:.1f}/{va['day_t']:.1f} | "
                     f"{tr['ex_top5_dR']:.3f}/{va['ex_top5_dR']:.3f} | "
                     f"{tr['fired_mean_dR']:.3f}/{va['fired_mean_dR']:.3f} |")
        ok = all(gg.loc[h, 'mean_dR'] >= 0.05 and gg.loc[h, 'day_t'] >= 2.5 and gg.loc[h, 'ex_top5_dR'] > 0
                 for h in HALVES)
        if ok:
            passers.append((rule, k))
    lines.append('')
    med_n = reads_a['n'].median()
    lines.append(f'MDE at the median n ({med_n:.0f}), fixed book SD: TRAIN-H2 {mde(sd_by_half["TRAIN-H2"], med_n):.3f} R, '
                 f'VAL {mde(sd_by_half["VAL"], med_n):.3f} R -- full per-cell MDE in `1668_reads.csv`.')
    lines.append(f'**Part A pass bar (dR>=+0.05R, day t>=2.5, ex-top5%>0, BOTH halves): '
                 f'{len(passers)}/36 rule x k cells pass -- {passers if passers else "none"}.**')
    lines.append('')

    lines.append('## Part B -- trained classifier (out of sample)')
    lines.append('| k | direction | n train | n score | tau | AUC | mean dR | iid t | day t | ex-top5% dR | '
                 'share cut | train-half dR (in-sample, caveat) | placebo AUC | placebo dR |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for _, r in reads_b.iterrows():
        lines.append(f"| {r['k']} | {r['direction']} | {r['n_train']} | {r['n_score']} | {r['tau']:.2f} | "
                     f"{r['auc']:.3f} | {r['mean_dR']:.4f} | {r['iid_t']:.2f} | {r['day_t']:.2f} | "
                     f"{r['ex_top5_dR']:.4f} | {r['share_cut']:.3f} | {r['train_insample_dR']:.4f} | "
                     f"{r['placebo_auc']:.3f} | {r['placebo_dR']:.4f} |")
    lines.append('')
    b_pass = all((reads_b['mean_dR'] >= 0.05) & (reads_b['day_t'] >= 2.5) & (reads_b['ex_top5_dR'] > 0))
    lines.append(f'**Part B pass bar (both out-of-sample scorings, dR>=+0.05R, day t>=2.5, ex-top5%>0): '
                 f'{"PASS" if b_pass else "fail"}.**')
    lines.append('')

    lines.append('## Permutation importance (VAL-scored model, TRAIN-H2->VAL direction, n_repeats=5, top 8)')
    lines.append('| k | feature | importance mean | importance std |')
    lines.append('|---|---|---|---|')
    for k, imp in importance_tables.items():
        for _, r in imp.head(8).iterrows():
            lines.append(f"| {k} | {r['feature']} | {r['importance_mean']:.4f} | {r['importance_std']:.4f} |")
    lines.append('')

    lines.append('## Amendment 3 -- pattern fire rate on bars fill+1..fill+5, both halves (informational, not pass/fail)')
    if fire_rates.empty:
        lines.append('NOT RUN (TA-Lib unavailable).' if not TALIB_AVAILABLE else 'no rows.')
    else:
        top = fire_rates.copy()
        top['gap'] = (top['stop_fire_rate'] - top['other_fire_rate']).abs()
        top = top.sort_values('gap', ascending=False).head(12)
        lines.append('Top 12 by |stop fire rate - target/EOD fire rate|, pooled rank (both halves shown):')
        lines.append('| pattern | half | stop rate (n) | target/EOD rate (n) |')
        lines.append('|---|---|---|---|')
        for _, r in top.iterrows():
            lines.append(f"| {r['pattern']} | {r['half']} | {r['stop_fire_rate']:.3f} ({r['stop_n']}) | "
                         f"{r['other_fire_rate']:.3f} ({r['other_n']}) |")
    lines.append('')

    lines.append('## Verdicts')
    lines.append(f'* Part A: {len(passers)}/72 (9 rules x 4 k x 2 halves paired reads, '
                 f'36 rule x k cells) clear the pass bar in both halves.')
    lines.append(f'* Part B: {"PASS" if b_pass else "no cell clears the out-of-sample pass bar on both scorings"}.')
    lines.append('* S5 (market) is VOID throughout: bars_sip.db carries SPY for one distinct day only.')
    lines.append('')

    lines.append('## Adequacy review')
    lines.append(f'* Book SD (net_R) used for MDE: TRAIN-H2={sd_by_half["TRAIN-H2"]:.3f}, VAL={sd_by_half["VAL"]:.3f}.')
    lines.append('* This is a null-heavy design by construction: S1-S9 fire on a minority of still-open fills at '
                 'each k, so most reads carry wide MDEs relative to the +0.05R bar; a non-pass here is a claim '
                 'about a specific mechanically-defined cut rule, not about post-entry information in general.')
    lines.append('* No read used a bar after its own cut minute; a stop/target/EOD hit at or before k always '
                 'preempted the rule (mean dR forced to 0), matching the PREREG precedence rule.')
    lines.append('* Base-exit-derived costs (entry/stop/target/EOD) were taken as-is from 1663_features.csv\'s '
                 'net_R; only the early-cut leg (entry+cut bps) was computed here, on the same convention.')
    with open(RESULT_MD, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    logger.info('wrote %s (%d lines)', RESULT_MD, len(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()

    setup_logging()
    logger.info('cell 1,668 starting (resume=%s limit=%s talib=%s)', args.resume, args.limit, TALIB_AVAILABLE)
    check_disk()

    pop = load_population()
    if args.limit:
        pop = pop.iloc[:args.limit].copy()
        logger.info('--limit %d: population truncated to %d rows', args.limit, len(pop))

    store = BarStore(BARS_DB)
    try:
        per_fill, fired_any_df = sweep(pop, store, resume=args.resume, limit=None)
    finally:
        store.close()

    sd_by_half = {h: pop.loc[pop['half'] == h, 'net_R'].std(ddof=1) for h in HALVES}
    logger.info('book SD (net_R) by half: %s', sd_by_half)

    reads_a = part_a_reads(per_fill, sd_by_half)
    reads_b, importance_tables, pstop_series = run_part_b(per_fill)

    for k, s in pstop_series.items():
        per_fill[f'pstop_{k}'] = s.reindex(per_fill.index).values
    tmp = PER_FILL_CSV + '.tmp'
    per_fill.to_csv(tmp, index=False)
    os.replace(tmp, PER_FILL_CSV)
    logger.info('final %s: %d rows, %d cols', PER_FILL_CSV, *per_fill.shape)

    reads_all = pd.concat([reads_a, reads_b], ignore_index=True, sort=False)
    reads_all.to_csv(READS_CSV, index=False)
    logger.info('wrote %s (%d rows)', READS_CSV, len(reads_all))

    fire_rates = fire_rate_table(fired_any_df)
    if not fire_rates.empty:
        fire_rates.to_csv(os.path.join(HERE, '1668_pattern_fire_rates.csv'), index=False)

    write_result_md(per_fill, reads_a, reads_b, importance_tables, fire_rates, sd_by_half, len(pop))
    logger.info('cell 1,668 done')


if __name__ == '__main__':
    main()
