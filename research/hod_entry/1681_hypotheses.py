#!/usr/bin/env python3
"""research/hod_entry/1681_hypotheses.py

Registry of the 35 mid-flight HOD-break exit hypotheses (H1-H35) from
PREREG_1681.md (FROZEN 2026-09-30 17:50 UTC). Owner: "can't be that HOD
cannot use the candle-shape story line or the trained model or a
combination to make better exit decisions mid-flight."

Every hypothesis is a FIXED rule (no fitted threshold, nothing re-fit)
evaluated over a per-fill continuous 1-minute price path, entry bar -> EOD,
loaded ONCE from the read-only bar store and cached to 1681_paths.parquet.

Precedence walk (shared by every hypothesis, same convention as
1668_failure.py's walk_k and hod_exit_lab's x5_lock): starting at the bar
after entry, at each forward bar j:
  1. a decision queued at bar j-1 (a hypothesis condition true on data
     through j-1) executes at bar j's OPEN - 6 bps (PREREG's exit
     convention; never uses bar j's own high/low/close to decide);
  2. else the (possibly rule-reshaped) stop/target/EOD is checked against
     bar j's low/high/minute, same-bar fill (gap-or-touch for a stop,
     the level for a target, the EOD bar's open for a session close) --
     this is the base rule's own mechanics, reused verbatim unless an F5
     hypothesis has reshaped stop/target as of this bar;
  3. else the hypothesis's condition is evaluated on data THROUGH bar j
     (causal) and, if true, queues the exit for bar j+1.
A rule that never fires (or whose condition never precedes the base's own
exit) rides to the SAME exit as the base -- dR = 0 for that fill, and the
STORED base_R is reused exactly (no reconstruction noise) for that case.

Cost convention (reverse-engineered from 1663_features.csv and matching
1668_failure.py's ENTRY_BPS/CUT_BPS): every exit price P nets to
    R = [(P - entry) - ENTRY_BPS*entry - EXIT_BPS*P] / (entry - stop0)
with ENTRY_BPS=7bps (a per-fill constant that cancels in every dR) and
EXIT_BPS=6bps -- this IS "the next bar's open minus 6bps" for a
rule-driven cut, and the same uniform convention the base population's own
net_R already carries on stop/target/eod exits (verified: sample fill cost
= 13bps = 7+6 exactly).

Model-based hypotheses NEVER re-fit or re-infer: they reuse the PERSISTED,
already-scored probabilities sitting on disk --
  P(+1R next 15)  <- 1677_per_fill.csv:p_success, keyed (fill_id,k,scoring),
                      k in {5,10,15,30,60}, scoring in
                      {TRAIN-H2->VAL, VAL->TRAIN-H2} (each scoring's rows
                      are the model trained on the source half, applied to
                      the target half's fills -- the target half is what
                      "read" selects for a model-based hypothesis).
  P(stop after k) <- 1670_per_fill_k.csv:p_stop_ALL, keyed (fill_id,k),
                      k in {0,1,2,5,10,30,60}. This column is already each
                      fill's own out-of-sample value (no train/val split in
                      the source file), so H14 (and anything using it) gets
                      ONE effective read, reported under both scoring labels
                      for CSV-shape consistency with the other model rules
                      (the two rows will be identical/near-identical by
                      construction -- this is documented, not a bug).

Population: the 1663/1676 join on (date,symbol,half), r_pct floored >=1.5%,
n=5,506 (verified: net_R agrees between the two files for every row).

CLI:
    python3 research/hod_entry/1681_hypotheses.py --run H1,H2,...  [--lens]
    python3 research/hod_entry/1681_hypotheses.py --run ALL --lens
appends one row per (hypothesis,read) to 1681_reads.csv and per-fill rows
to 1681_per_fill.csv; both writes are atomic (tmp + os.replace) and
idempotent per hypothesis id (existing rows for that id are dropped before
the new ones are appended). Without --lens the consistency-lens columns
are written as NaN (cheap default run); --lens fills them in (per-week
grouping, heavier).
"""
import argparse
import importlib.util
import logging
import math
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BF_ZERO = os.path.join(os.path.dirname(HERE), 'bf_zero')

POP_1663 = os.path.join(HERE, '1663_features.csv')
JOIN_1676 = os.path.join(HERE, '1676_features.csv')
PSUCCESS_1677 = os.path.join(HERE, '1677_per_fill.csv')
PSTOP_1670 = os.path.join(HERE, '1670_per_fill_k.csv')

PATHS_PARQUET = os.path.join(HERE, '1681_paths.parquet')
READS_CSV = os.path.join(HERE, '1681_reads.csv')
PER_FILL_CSV = os.path.join(HERE, '1681_per_fill.csv')
LOG_FILE = os.path.join(HERE, '1681_hypotheses.log')

# ---------------------------------------------------------------------------
# Reuse the project's own bar-store / ET-minute / day-clustered-stats module
# instead of re-implementing it (CLAUDE.md: "use the main code with flags,
# not bespoke scripts"). Same _load_module pattern 1669/1677/1678 use.
# ---------------------------------------------------------------------------


def _load_module(name, fname, root=HERE):
    path = os.path.join(root, fname)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


f1668 = _load_module('f1668_for_1681', '1668_failure.py')

BarStore = f1668.BarStore
minute_of_day = f1668.minute_of_day
find_fill_index = f1668.find_fill_index
EOD_M = f1668.EOD_M
BARS_DB = f1668.BARS_DB
iid_t = f1668.iid_t
day_clustered_t = f1668.day_clustered_t
ex_top5_mean = f1668.ex_top5_mean
mde = f1668.mde

ENTRY_BPS = 0.0007
EXIT_BPS = 0.0006  # PREREG: "exits execute at the NEXT bar's open - 6 bps"

SUCCESS_KS = [5, 10, 15, 30, 60]
STOP_KS = [0, 1, 2, 5, 10, 30, 60]
SCORINGS = ['TRAIN-H2->VAL', 'VAL->TRAIN-H2']
HALVES = ['TRAIN-H2', 'VAL']

logger = logging.getLogger('1681')


def setup_logging():
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(LOG_FILE, mode='a')
    fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter('%(levelname)s %(message)s'))
    logger.addHandler(fh)
    logger.addHandler(sh)


# ---------------------------------------------------------------------------
# Population + per-fill continuous path cache
# ---------------------------------------------------------------------------

def load_population():
    """1663 join 1676 on (date,symbol,half) -> fill_id + entry/stop/target/
    fill_min/base_R. Asserts net_R agrees (the join key is correct)."""
    p63 = pd.read_csv(POP_1663)
    p76 = pd.read_csv(JOIN_1676, usecols=['fill_id', 'date', 'symbol', 'half', 'net_R'])
    m = p76.merge(p63, on=['date', 'symbol', 'half'], suffixes=('_1676', '_1663'), how='inner')
    mismatch = (m['net_R_1676'] - m['net_R_1663']).abs().max()
    logger.info('population: 1663 n=%d, 1676 n=%d, joined n=%d, net_R max mismatch=%.6g',
                len(p63), len(p76), len(m), mismatch)
    if len(m) != len(p63) or mismatch > 1e-6:
        logger.error('join did not reproduce the 1663 population 1:1 (n=%d vs %d) or net_R mismatched -- abort', len(m), len(p63))
        sys.exit(1)
    m = m.rename(columns={'net_R_1663': 'base_R', 'stop': 'stop0', 'target_price': 'target0', 'entry_price': 'entry'})
    m['R_unit'] = m['entry'] - m['stop0']
    return m[['fill_id', 'date', 'symbol', 'half', 'entry', 'stop0', 'target0', 'R_unit',
              'fill_min', 'base_R', 'exit_type', 'exit_price', 'r_pct']].reset_index(drop=True)


def build_paths(pop, rebuild=False):
    """Per fill: 1-min OHLCV from the fill (entry) bar to the EOD bar, PLUS
    the full trading-day bars (needed for session VWAP) and the entry-bar
    index within the day. Cached long-form to 1681_paths.parquet, loaded
    ONCE per run thereafter. Returns {fill_id: dict(...)}."""
    if os.path.exists(PATHS_PARQUET) and not rebuild:
        logger.info('loading cached paths from %s', PATHS_PARQUET)
        long_df = pd.read_parquet(PATHS_PARQUET)
        out = {}
        for fid, g in long_df.groupby('fill_id'):
            g = g.sort_values('j')
            out[int(fid)] = dict(
                m=g['m'].to_numpy(), o=g['o'].to_numpy(), h=g['h'].to_numpy(),
                l=g['l'].to_numpy(), c=g['c'].to_numpy(), v=g['v'].to_numpy(),
                vwap=g['vwap'].to_numpy(),
            )
        missing = set(pop['fill_id']) - set(out)
        if missing:
            logger.warning('%d fills missing from the path cache -- rebuilding those', len(missing))
            pop_missing = pop[pop['fill_id'].isin(missing)]
            out.update(_fetch_paths(pop_missing))
            _persist_paths(out)
        return out
    store = BarStore(BARS_DB)
    try:
        out = _fetch_paths(pop, store)
    finally:
        store.close()
    _persist_paths(out)
    return out


def _fetch_paths(pop, store=None):
    own_store = store is None
    if own_store:
        store = BarStore(BARS_DB)
    out = {}
    t0 = time.time()
    n_ok, n_bad = 0, 0
    for i, row in enumerate(pop.itertuples()):
        bars = store.day_bars(row.symbol, row.date)
        if bars is None or len(bars['o']) == 0:
            logger.warning('fill_id=%s %s %s: no bars in the store -- skipped', row.fill_id, row.symbol, row.date)
            n_bad += 1
            continue
        i0 = find_fill_index(bars, row.fill_min)
        if i0 is None:
            logger.warning('fill_id=%s %s %s: fill_min=%.2f before first bar -- skipped', row.fill_id, row.symbol, row.date, row.fill_min)
            n_bad += 1
            continue
        pv = np.cumsum(bars['c'] * bars['v']) / np.maximum(np.cumsum(bars['v']), 1e-9)
        out[int(row.fill_id)] = dict(
            m=bars['minarr'][i0:], o=bars['o'][i0:], h=bars['h'][i0:],
            l=bars['l'][i0:], c=bars['c'][i0:], v=bars['v'][i0:],
            vwap=pv[i0:],
        )
        n_ok += 1
        if (i + 1) % 500 == 0:
            logger.info('paths: %d/%d fetched (%d ok, %d bad) in %.1fs', i + 1, len(pop), n_ok, n_bad, time.time() - t0)
    if own_store:
        store.close()
    logger.info('paths done: %d ok, %d bad, %.1fs total', n_ok, n_bad, time.time() - t0)
    return out


def _persist_paths(paths):
    rows = []
    for fid, d in paths.items():
        n = len(d['m'])
        rows.append(pd.DataFrame(dict(
            fill_id=fid, j=np.arange(n), m=d['m'], o=d['o'], h=d['h'],
            l=d['l'], c=d['c'], v=d['v'], vwap=d['vwap'],
        )))
    long_df = pd.concat(rows, ignore_index=True)
    tmp = PATHS_PARQUET + '.tmp'
    long_df.to_parquet(tmp, index=False)
    os.replace(tmp, PATHS_PARQUET)
    logger.info('paths cached: %s (%d fills, %d rows)', PATHS_PARQUET, len(paths), len(long_df))


def load_model_probs():
    """p_success[(fill_id,k,scoring)] -> prob ; p_stop[(fill_id,k)] -> prob.
    Reused verbatim from disk -- never re-fit, never re-inferred (PREREG)."""
    ps = pd.read_csv(PSUCCESS_1677, usecols=['fill_id', 'k', 'scoring', 'p_success'])
    p_success = {(int(r.fill_id), int(r.k), r.scoring): r.p_success for r in ps.itertuples()}
    pp = pd.read_csv(PSTOP_1670, usecols=['fill_id', 'k', 'p_stop_ALL'])
    p_stop = {(int(r.fill_id), int(r.k)): r.p_stop_ALL for r in pp.itertuples()}
    logger.info('model probs: p_success %d rows, p_stop %d rows', len(ps), len(pp))
    return p_success, p_stop


# ---------------------------------------------------------------------------
# Per-fill context: rolling-candle + shape + model-lookup helpers, all
# causal (bar j only ever sees bars 0..j).
# ---------------------------------------------------------------------------

def _clv(h, l, c):
    return 0.5 if h <= l else (c - l) / (h - l)


class FillCtx:
    def __init__(self, fid, row, path, p_success, p_stop, scoring):
        self.fid = fid
        self.entry = row.entry
        self.stop0 = row.stop0
        self.target0 = row.target0
        self.R = row.R_unit
        self.base_R = row.base_R
        self.date = row.date
        self.o, self.h, self.l, self.c, self.v = path['o'], path['h'], path['l'], path['c'], path['v']
        self.m = path['m']  # ET minute-of-day per bar
        self.vwap = path['vwap']
        self.n = len(self.o)
        self.p_success = p_success
        self.p_stop = p_stop
        self.scoring = scoring
        # running favorable-excursion (high-based) and its "last new high" bar
        hi_R = (self.h - self.entry) / self.R
        run_max = np.maximum.accumulate(hi_R)
        self.run_hi_R = run_max
        new_high_bar = np.zeros(self.n, dtype=int)
        best = -1e18
        best_j = 0
        for j in range(self.n):
            if hi_R[j] > best:
                best = hi_R[j]
                best_j = j
            new_high_bar[j] = best_j
        self._new_high_bar = new_high_bar
        # checkpoint bar index (last bar at or before entry_minute+k)
        self._kbar = {}
        # k is a simple BAR-INDEX offset from entry (i0+k), matching
        # 1668_failure.py's walk_k `last_idx = i0 + k` -- the project's
        # established convention the 1670/1677 caches were built under,
        # NOT a minute-of-day lookup (verified: switching this to a
        # minute-based search broke H16's reproduction of 1677's own
        # k=60 number).
        for k in sorted(set(SUCCESS_KS) | set(STOP_KS)):
            self._kbar[k] = k if k < self.n else None

    def close_R(self, j):
        return (self.c[j] - self.entry) / self.R

    def mins(self, j):
        return self.m[j] - self.m[0]

    def mins_since_new_high(self, j):
        return self.m[j] - self.m[self._new_high_bar[j]]

    def roll(self, j, w):
        """Trailing w-bar rolling candle ENDING AT bar j (inclusive)."""
        lo = max(0, j - w + 1)
        seg = slice(lo, j + 1)
        o, h, l, c, v = self.o[lo], self.h[seg].max(), self.l[seg].min(), self.c[j], self.v[seg].sum()
        return dict(o=o, h=h, l=l, c=c, v=v)

    def mean_vol(self, j, upto_excl=True):
        hi = j if upto_excl else j + 1
        seg = self.v[0:max(hi, 1)]
        return seg.mean() if len(seg) else np.nan

    def kbar(self, k):
        return self._kbar.get(k)

    def psucc(self, k):
        return self.p_success.get((self.fid, k, self.scoring))

    def pstop(self, k):
        return self.p_stop.get((self.fid, k))


def leg_R(entry, R_unit, price):
    return ((price - entry) - ENTRY_BPS * entry - EXIT_BPS * abs(price)) / R_unit


def _gap_or_touch(open_, low_, stop):
    return open_ if open_ <= stop else stop


# ---------------------------------------------------------------------------
# Generic engines: trigger (full exit), reshape (stop/target change only),
# partial (scale-outs). All share the same stop/target/EOD precedence.
# ---------------------------------------------------------------------------

def run_trigger(ctx, cond_fn):
    """cond_fn(ctx,j)->bool, evaluated on data through bar j (causal). Fires
    -> exit at bar j+1's open, -6bps. Base stop/target/eod (UNCHANGED)
    always takes precedence in time; if it comes first the STORED base_R
    is returned exactly (no reconstruction noise)."""
    n = ctx.n
    for j in range(1, n):
        if ctx.m[j] >= EOD_M:
            return ctx.base_R, False, None
        if ctx.l[j] <= ctx.stop0:
            return ctx.base_R, False, None
        if ctx.h[j] >= ctx.target0:
            return ctx.base_R, False, None
        if cond_fn(ctx, j):
            if j + 1 >= n:
                return ctx.base_R, False, None
            price = ctx.o[j + 1]
            return leg_R(ctx.entry, ctx.R, price), True, ctx.mins(j)
    return ctx.base_R, False, None


def run_reshape(ctx, reshape_fn):
    """reshape_fn(ctx,j)->Optional[(new_stop,new_target)], evaluated causally
    on bar j; if given, takes effect from bar j+1 on. If stop/target never
    actually changes before the walk's own exit, the STORED base_R is
    returned exactly."""
    n = ctx.n
    cur_stop, cur_target = ctx.stop0, ctx.target0
    changed = False
    for j in range(1, n):
        if ctx.m[j] >= EOD_M:
            price = ctx.o[j]
            return (ctx.base_R if not changed else leg_R(ctx.entry, ctx.R, price)), changed, ctx.mins(j)
        if ctx.l[j] <= cur_stop:
            price = _gap_or_touch(ctx.o[j], ctx.l[j], cur_stop)
            return (ctx.base_R if not changed else leg_R(ctx.entry, ctx.R, price)), changed, ctx.mins(j)
        if ctx.h[j] >= cur_target:
            price = cur_target
            return (ctx.base_R if not changed else leg_R(ctx.entry, ctx.R, price)), changed, ctx.mins(j)
        upd = reshape_fn(ctx, j)
        if upd is not None:
            new_stop, new_target = upd
            if new_stop != cur_stop or new_target != cur_target:
                changed = True
            cur_stop, cur_target = max(cur_stop, new_stop), min(cur_target, new_target)
    return ctx.base_R, changed, None


def run_partial(ctx, legs, remainder_reshape=None):
    """legs: ordered [(cond_fn, fraction), ...], each fires at most once
    (bar j's data -> exit at j+1's open -6bps for that fraction). After the
    LAST leg fires, remainder_reshape(ctx,j)->Optional[(stop,target)] may
    keep reshaping the remaining fraction's stop/target (else it rides
    unchanged). Whatever fraction is unsold when base stop/target/eod (at
    the CURRENT, possibly-reshaped levels) triggers exits together at that
    price. Returns (ruleR, any_leg_fired, first_fire_minute)."""
    n = ctx.n
    cur_stop, cur_target = ctx.stop0, ctx.target0
    sold = 0.0
    realized = 0.0
    leg_idx = 0
    any_fired = False
    first_fire = None
    pending = None  # (fraction, decided_at_j) queued for j+1 open
    for j in range(1, n):
        if pending is not None:
            frac, _ = pending
            realized += frac * leg_R(ctx.entry, ctx.R, ctx.o[j])
            sold += frac
            pending = None
        if sold >= 1.0 - 1e-12:
            return realized, any_fired, first_fire
        if ctx.m[j] >= EOD_M:
            realized += (1.0 - sold) * leg_R(ctx.entry, ctx.R, ctx.o[j])
            return realized, any_fired, first_fire
        if ctx.l[j] <= cur_stop:
            price = _gap_or_touch(ctx.o[j], ctx.l[j], cur_stop)
            realized += (1.0 - sold) * leg_R(ctx.entry, ctx.R, price)
            return realized, any_fired, first_fire
        if ctx.h[j] >= cur_target:
            realized += (1.0 - sold) * leg_R(ctx.entry, ctx.R, cur_target)
            return realized, any_fired, first_fire
        if leg_idx < len(legs):
            cond_fn, frac = legs[leg_idx]
            if cond_fn(ctx, j):
                pending = (frac, j)
                leg_idx += 1
                any_fired = True
                if first_fire is None:
                    first_fire = ctx.mins(j)
                continue
        elif remainder_reshape is not None:
            upd = remainder_reshape(ctx, j)
            if upd is not None:
                new_stop, new_target = upd
                cur_stop, cur_target = max(cur_stop, new_stop), min(cur_target, new_target)
    if sold < 1.0:
        realized += (1.0 - sold) * ctx.base_R if sold == 0.0 else (1.0 - sold) * leg_R(ctx.entry, ctx.R, ctx.c[n - 1])
    return realized, any_fired, first_fire


# ---------------------------------------------------------------------------
# H1-H35 condition functions (each takes ctx,j -> bool, or is itself the
# top-level callable registered below). Comments flag any reading PREREG
# left ambiguous (timeframe of a single "candle" with no stated window).
# ---------------------------------------------------------------------------

def _h1(ctx, j):
    return ctx.mins(j) >= 30 and ctx.close_R(j) < 0.25


def _h2(ctx, j):
    return ctx.mins(j) >= 60 and ctx.close_R(j) < 0.5


def _h3(ctx, j):
    return ctx.mins(j) >= 90 and ctx.close_R(j) < 0.75


def _h4(ctx, j):
    return ctx.mins_since_new_high(j) > 20 and ctx.close_R(j) < 0.5


def _h5(ctx, j):
    return ctx.run_hi_R[j] >= 1.0 and ctx.mins_since_new_high(j) >= 15


def _h6(ctx, j):
    if j - 10 < 0 or ctx.close_R(j) < 0.5:
        return False
    cur, prev = ctx.roll(j, 5), ctx.roll(j - 5, 5)
    return (cur['c'] < cur['o'] and prev['c'] > prev['o']
            and cur['o'] >= prev['c'] and cur['c'] <= prev['o'])


def _h7(ctx, j):
    if j - 1 < 0 or ctx.close_R(j) < 0.5:
        return False
    prev = ctx.roll(j - 1, 10)
    return ctx.c[j] < prev['l']


def _h8(ctx, j):
    return ctx.close_R(j) >= 0.5 and ctx.c[j] < ctx.vwap[j]


def _h9(ctx, j):
    # timeframe unspecified in PREREG for H9/H10 (unlike H6/H7/H8/H11/H12,
    # which name 5/10/15-min) -- read as the 1-min bar, the finest "candle".
    if ctx.close_R(j) < 1.0:
        return False
    o, h, l, c = ctx.o[j], ctx.h[j], ctx.l[j], ctx.c[j]
    body = abs(c - o)
    uwick = h - max(o, c)
    return uwick >= 2 * body and uwick > 0


def _h10(ctx, j):
    if ctx.close_R(j) <= 0:
        return False
    mv = ctx.mean_vol(j, upto_excl=True)
    if not np.isfinite(mv) or mv <= 0:
        return False
    return ctx.v[j] >= 3 * mv and _clv(ctx.h[j], ctx.l[j], ctx.c[j]) <= 0.5


def _h11(ctx, j):
    if j - 14 < 0 or ctx.run_hi_R[j] < 1.0:
        return False
    cur = ctx.roll(j, 15)
    return cur['c'] < cur['o']


def _h12(ctx, j):
    if j - 10 < 0 or ctx.close_R(j) < 0.5:
        return False
    w0, w1, w2 = ctx.roll(j, 5), ctx.roll(j - 5, 5), ctx.roll(j - 10, 5)
    return w0['h'] < w1['h'] < w2['h']


def _h13_at(ctx, k):
    j = ctx.kbar(k)
    if j is None:
        return False
    p = ctx.psucc(k)
    return p is not None and ctx.close_R(j) >= 0.5 and p < 0.3


def _h13(ctx, j):
    for k in SUCCESS_KS:
        if ctx.kbar(k) == j:
            return _h13_at(ctx, k)
    return False


def _h14_at(ctx, k):
    j = ctx.kbar(k)
    if j is None:
        return False
    p = ctx.pstop(k)
    return p is not None and ctx.close_R(j) >= 0.25 and p > 0.7


def _h14(ctx, j):
    for k in STOP_KS:
        if ctx.kbar(k) == j:
            return _h14_at(ctx, k)
    return False


def _h15(ctx, j):
    for k in (set(SUCCESS_KS) & set(STOP_KS)):
        if ctx.kbar(k) == j:
            return _h13_at(ctx, k) and _h14_at(ctx, k)
    return False


def _h16(ctx, j):
    k = 60
    jj = ctx.kbar(k)
    if jj is None or jj != j:
        return False
    p = ctx.psucc(k)
    return p is not None and ctx.close_R(j) >= 1.0 and p < 0.3


def _h23(ctx, j):
    if j - 29 < 0:
        return None
    cur = ctx.roll(j, 30)
    if _clv(cur['h'], cur['l'], cur['c']) < 0.5:
        return (ctx.stop0, ctx.entry + 1.5 * ctx.R)
    return None


def _h24_leg(ctx, j):
    for k in SUCCESS_KS:
        if ctx.kbar(k) == j and ctx.close_R(j) >= 1.0:
            p = ctx.psucc(k)
            if p is not None and p > 0.7:
                return (ctx.stop0, ctx.entry + 3.0 * ctx.R)
    return None


def _h25(ctx, j):
    if ctx.h[j] >= ctx.entry + 1.0 * ctx.R:
        return (ctx.entry, ctx.target0)
    return None


def _h26(ctx, j):
    if ctx.h[j] >= ctx.entry + 1.5 * ctx.R:
        return (ctx.entry + 0.5 * ctx.R, ctx.target0)
    return None


def _h27(ctx, j):
    if ctx.run_hi_R[j] >= 1.0:
        return (ctx.roll(j, 10)['l'], ctx.target0)
    return None


def _h28(ctx, j):
    if ctx.run_hi_R[j] >= 1.0:
        return (ctx.vwap[j], ctx.target0)
    return None


def _h29(ctx, j):
    if ctx.mins(j) < 20 or ctx.close_R(j) >= 0.5:
        return False
    break_v = ctx.v[0]
    if break_v <= 0:
        return False
    last5 = ctx.roll(j, 5)
    return (last5['v'] / 5.0) < 0.3 * break_v


def _h30(ctx, j):
    if ctx.close_R(j) < 0.5:
        return False
    mv = ctx.mean_vol(j, upto_excl=True)
    return ctx.c[j] < ctx.o[j] and np.isfinite(mv) and mv > 0 and ctx.v[j] >= 3 * mv


def _h31(ctx, j):
    if ctx.close_R(j) < 0.5:
        return False
    lo = max(0, j - 9)
    seg_o, seg_c, seg_v = ctx.o[lo:j + 1], ctx.c[lo:j + 1], ctx.v[lo:j + 1]
    up = seg_v[seg_c > seg_o].sum()
    down = seg_v[seg_c < seg_o].sum()
    return up <= down


def _h32(ctx, j):
    for k in SUCCESS_KS:
        if ctx.kbar(k) == j:
            return _h13_at(ctx, k) and _h7(ctx, j)
    return False


def _h34(ctx, j):
    for k in STOP_KS:
        if ctx.kbar(k) == j:
            return _h4(ctx, j) and _h14_at(ctx, k)
    return False


def _h35_leg1(ctx, j):
    return ctx.h[j] >= ctx.entry + 1.0 * ctx.R and ctx.c[j] < ctx.vwap[j]


def _h21_remainder(ctx, j):
    return (ctx.entry, ctx.target0)


# ---------------------------------------------------------------------------
# Top-level hypothesis runners: each takes ctx -> (ruleR, fired, fired_min)
# ---------------------------------------------------------------------------

def _trig(cond):
    return lambda ctx: run_trigger(ctx, cond)


def _resh(fn):
    return lambda ctx: run_reshape(ctx, fn)


def _h17(ctx):
    n = ctx.n
    for j in range(1, n):
        if ctx.m[j] >= EOD_M or ctx.l[j] <= ctx.stop0:
            return ctx.base_R, False, None
        if ctx.h[j] >= ctx.target0:
            return ctx.base_R, False, None
        for k in SUCCESS_KS:
            if ctx.kbar(k) == j and ctx.close_R(j) >= 1.0:
                p = ctx.psucc(k)
                if p is None:
                    return ctx.base_R, False, None
                frac = round((1.0 - p) * 4) / 4.0
                if frac <= 0:
                    return ctx.base_R, False, None
                if frac >= 1.0:
                    if j + 1 >= n:
                        return ctx.base_R, False, None
                    return leg_R(ctx.entry, ctx.R, ctx.o[j + 1]), True, ctx.mins(j)
                legs = [(lambda c, jj: True, frac)]
                r2, fired, fmin = run_partial(ctx, legs)
                return r2, True, ctx.mins(j)
    return ctx.base_R, False, None


def _h18(ctx):
    legs = [(lambda c, jj: c.close_R(jj) >= 1.0, 0.5)]
    return run_partial(ctx, legs)


def _h19(ctx):
    legs = [(lambda c, jj: c.close_R(jj) >= 1.0, 1.0 / 3),
            (lambda c, jj: c.close_R(jj) >= 2.0, 1.0 / 3)]

    def trail(c, jj):
        return (c.run_hi_R[jj] * c.R + c.entry - c.R, c.target0)
    return run_partial(ctx, legs, remainder_reshape=trail)


def _h20(ctx):
    def cond(c, jj):
        if c.close_R(jj) < 1.0:
            return False
        for k in SUCCESS_KS:
            if c.kbar(k) == jj:
                p = c.psucc(k)
                return p is not None and p < 0.5
        return False
    legs = [(cond, 0.5)]
    return run_partial(ctx, legs)


def _h21(ctx):
    legs = [(lambda c, jj: c.close_R(jj) >= 1.0, 0.5)]
    return run_partial(ctx, legs, remainder_reshape=_h21_remainder)


def _h22(ctx):
    legs = [(lambda c, jj: c.close_R(jj) >= 0.5, 0.25)]
    return run_partial(ctx, legs)


def _h33(ctx):
    legs = [(lambda c, jj: c.close_R(jj) >= 1.0, 0.5)]
    return run_partial(ctx, legs, remainder_reshape=_h24_leg)


def _h35(ctx):
    legs = [(_h35_leg1, 0.5)]
    return run_partial(ctx, legs, remainder_reshape=_h21_remainder)


def _h36(ctx):
    """Joint rule (cell 1,681 synthesis, FROZEN on TRAIN): H35 (H21 AND H8,
    TRAIN score S=+0.0063, the top scorer) has precedence at every bar over
    H16 (the 1,677 reference, S=+0.0024, the only other positive-score
    hypothesis that both passes the ex5/P10 constraints and acts on a
    different trigger). Mechanically this is an OR of two mutually
    exclusive actions on the SAME untouched path, checked in precedence
    order each bar, sharing the run_partial engine's stop/target/EOD
    precedence and next-bar-open pricing: if H35's leg (mtm>=+1R AND 5m
    close<VWAP) fires first, its own mechanism runs unchanged (50% out,
    breakeven remainder via _h21_remainder); only on a fill where that leg
    has NOT yet fired does H16 (bar k=60, mtm>=+1R, P(+1R,15m)<0.3) get a
    chance to fire a full exit instead. Once H35's leg has fired, H16 is no
    longer checked (the position is already reshaped, not a candidate for
    a second, full-exit action) -- the two never compound on one fill.
    """
    n = ctx.n
    cur_stop, cur_target = ctx.stop0, ctx.target0
    sold = 0.0
    realized = 0.0
    any_fired = False
    first_fire = None
    pending = None       # (fraction, decided_at_j) queued for j+1's open
    h35_fired = False    # has H35's leg already fired (then reshape only)?
    for j in range(1, n):
        if pending is not None:
            frac, _ = pending
            realized += frac * leg_R(ctx.entry, ctx.R, ctx.o[j])
            sold += frac
            pending = None
        if sold >= 1.0 - 1e-12:
            return realized, any_fired, first_fire
        if ctx.m[j] >= EOD_M:
            realized += (1.0 - sold) * leg_R(ctx.entry, ctx.R, ctx.o[j])
            return realized, any_fired, first_fire
        if ctx.l[j] <= cur_stop:
            price = _gap_or_touch(ctx.o[j], ctx.l[j], cur_stop)
            realized += (1.0 - sold) * leg_R(ctx.entry, ctx.R, price)
            return realized, any_fired, first_fire
        if ctx.h[j] >= cur_target:
            realized += (1.0 - sold) * leg_R(ctx.entry, ctx.R, cur_target)
            return realized, any_fired, first_fire
        if not h35_fired:
            if _h35_leg1(ctx, j):
                pending = (0.5, j)
                h35_fired = True
                any_fired = True
                if first_fire is None:
                    first_fire = ctx.mins(j)
                continue
            if _h16(ctx, j):
                pending = (1.0, j)
                any_fired = True
                if first_fire is None:
                    first_fire = ctx.mins(j)
                continue
        else:
            upd = _h21_remainder(ctx, j)
            if upd is not None:
                new_stop, new_target = upd
                cur_stop, cur_target = max(cur_stop, new_stop), min(cur_target, new_target)
    if sold < 1.0:
        realized += (1.0 - sold) * ctx.base_R if sold == 0.0 else (1.0 - sold) * leg_R(ctx.entry, ctx.R, ctx.c[n - 1])
    return realized, any_fired, first_fire


HYPOTHESES = {
    'H1': dict(family='F1', model=False, desc='exit at 30 min if mtm<+0.25R', fn=_trig(_h1)),
    'H2': dict(family='F1', model=False, desc='exit at 60 min if mtm<+0.5R', fn=_trig(_h2)),
    'H3': dict(family='F1', model=False, desc='exit at 90 min if mtm<+0.75R', fn=_trig(_h3)),
    'H4': dict(family='F1', model=False, desc='>20min since last new high AND mtm<+0.5R', fn=_trig(_h4)),
    'H5': dict(family='F1', model=False, desc='after +1R, no new high in 15min', fn=_trig(_h5)),
    'H6': dict(family='F2', model=False, desc='bearish engulfing 5m candle while >=+0.5R', fn=_trig(_h6)),
    'H7': dict(family='F2', model=False, desc='close below trailing 10m low while >=+0.5R', fn=_trig(_h7)),
    'H8': dict(family='F2', model=False, desc='5m close below day VWAP while >=+0.5R', fn=_trig(_h8)),
    'H9': dict(family='F2', model=False, desc='upper-wick rejection (wick>=2x body) while >=+1R', fn=_trig(_h9)),
    'H10': dict(family='F2', model=False, desc='climax bar (vol>=3x mean, CLV<=0.5) at any profit', fn=_trig(_h10)),
    'H11': dict(family='F2', model=False, desc='trailing 15m candle turns red after +1R', fn=_trig(_h11)),
    'H12': dict(family='F2', model=False, desc='two consecutive lower 5m highs while >=+0.5R', fn=_trig(_h12)),
    'H13': dict(family='F3', model=True, desc='P(+1R next15)<0.3 while >=+0.5R (rolling from k=5)', fn=_trig(_h13)),
    'H14': dict(family='F3', model=True, desc='P(stop after k)>0.7 while >=+0.25R', fn=_trig(_h14)),
    'H15': dict(family='F3', model=True, desc="H13's and H14's conditions both hold", fn=_trig(_h15)),
    'H16': dict(family='F3', model=True, desc='the 1,677 reference: k>=60, mtm>=+1R, P<0.3', fn=_trig(_h16)),
    'H17': dict(family='F3', model=True, desc='P-weighted partial at +1R: sell 1-P(+1Rnext15)', fn=_h17),
    'H18': dict(family='F4', model=False, desc='50% out at +1R, rest as the live rule', fn=_h18),
    'H19': dict(family='F4', model=False, desc='33% at +1R, 33% at +2R, rest trails MFE-1R', fn=_h19),
    'H20': dict(family='F4', model=True, desc='50% out at +1R only if P(+1Rnext15)<0.5', fn=_h20),
    'H21': dict(family='F4', model=False, desc='50% out at +1R and stop to breakeven for the rest', fn=_h21),
    'H22': dict(family='F4', model=False, desc='25% out at +0.5R, rest as the live rule', fn=_h22),
    'H23': dict(family='F5', model=False, desc='target to +1.5R when trailing 30m candle red (CLV<0.5)', fn=_resh(_h23)),
    'H24': dict(family='F5', model=True, desc='target to +3R when P(+1Rnext15)>0.7 at +1R', fn=_resh(_h24_leg)),
    'H25': dict(family='F5', model=False, desc='stop to breakeven at +1R (exit-lab X5 replicate)', fn=_resh(_h25)),
    'H26': dict(family='F5', model=False, desc='stop to +0.5R at +1.5R', fn=_resh(_h26)),
    'H27': dict(family='F5', model=False, desc="trail = trailing 10m candle's low once >=+1R", fn=_resh(_h27)),
    'H28': dict(family='F5', model=False, desc='trail = day VWAP once >=+1R', fn=_resh(_h28)),
    'H29': dict(family='F6', model=False, desc='last5 mean vol<0.3x break-bar AND mtm<+0.5R after 20min', fn=_trig(_h29)),
    'H30': dict(family='F6', model=False, desc='red bar with vol>=3x mean while >=+0.5R', fn=_trig(_h30)),
    'H31': dict(family='F6', model=False, desc='hold to +1R only while up-vol>down-vol over last10, else exit >=+0.5R', fn=_trig(_h31)),
    'H32': dict(family='F7', model=True, desc='H13 AND H7', fn=_trig(_h32)),
    'H33': dict(family='F7', model=True, desc='H18 then H24 on the remainder', fn=_h33),
    'H34': dict(family='F7', model=True, desc='H4 AND H14', fn=_trig(_h34)),
    'H35': dict(family='F7', model=False, desc='H21 AND H8', fn=_h35),
    'H36': dict(family='JOINT', model=True, desc='synthesis: H35 precedence over H16 (TRAIN-scored)', fn=_h36),
}

assert len(HYPOTHESES) == 36, f'expected 36 hypotheses, got {len(HYPOTHESES)}'


# ---------------------------------------------------------------------------
# Stats, give-back decomposition, consistency lens
# ---------------------------------------------------------------------------

def stats_block(dR, dates):
    dR = pd.Series(dR).reset_index(drop=True)
    dates = pd.Series(dates).reset_index(drop=True)
    return dict(
        n=len(dR), dR=dR.mean() if len(dR) else np.nan,
        iid_t=iid_t(dR), day_t=day_clustered_t(dates, dR),
        ex5=ex_top5_mean(dR), mde=mde(dR.std(ddof=1) if len(dR) > 1 else np.nan, len(dR)),
    )


def decompose(fired_mask, rule_R, base_R):
    """saved = mean dR on fills where the rule's exit beat the base
    (give-back it avoided); forgone = mean dR (negative) on fills where the
    rule exited worse than the base would have (continuation it gave up);
    cost = mean dR on all fired fills minus (saved_contrib+forgone_contrib)
    reconciliation is exact by construction: saved_sum+forgone_sum ==
    sum(dR on fired) -- no separate 'cost' term is needed beyond what the
    unified leg_R formula already charges (which is already inside dR), so
    cost is reported as the mean EXIT_BPS/ENTRY_BPS drag realized on fired
    legs for transparency."""
    dR = rule_R - base_R
    fired = dR[fired_mask]
    saved = fired[fired > 0]
    forgone = fired[fired <= 0]
    cost = pd.Series(np.full(len(fired), EXIT_BPS))  # bps drag charged on every fired leg's exit price, informational
    return dict(
        saved=saved.mean() if len(saved) else 0.0, saved_n=len(saved),
        forgone=forgone.mean() if len(forgone) else 0.0, forgone_n=len(forgone),
        cost=cost.mean() if len(cost) else 0.0,
    )


def _weekly(dates, R):
    df = pd.DataFrame({'date': pd.to_datetime(dates), 'R': R})
    df['week'] = df['date'].dt.isocalendar().year.astype(str) + '-W' + df['date'].dt.isocalendar().week.astype(str)
    wk = df.groupby('week')['R'].sum()
    return wk


def _maxdd(dates, R):
    df = pd.DataFrame({'date': pd.to_datetime(dates), 'R': R}).sort_values('date')
    cum = df['R'].cumsum()
    peak = cum.cummax()
    return (cum - peak).min()


def consistency_lens(dates, base_R, rule_R):
    day_base = pd.DataFrame({'date': dates, 'R': base_R}).groupby('date')['R'].sum()
    day_rule = pd.DataFrame({'date': dates, 'R': rule_R}).groupby('date')['R'].sum()
    wk_base, wk_rule = _weekly(dates, base_R), _weekly(dates, rule_R)
    return dict(
        green_day_base=(day_base > 0).mean(), green_day_rule=(day_rule > 0).mean(),
        green_week_base=(wk_base > 0).mean(), green_week_rule=(wk_rule > 0).mean(),
        weekly_p10_base=wk_base.quantile(0.10), weekly_p10_rule=wk_rule.quantile(0.10),
        weekly_sharpe_base=(wk_base.mean() / wk_base.std(ddof=1)) if wk_base.std(ddof=1) else np.nan,
        weekly_sharpe_rule=(wk_rule.mean() / wk_rule.std(ddof=1)) if wk_rule.std(ddof=1) else np.nan,
        maxdd_base=_maxdd(dates, base_R), maxdd_rule=_maxdd(dates, rule_R),
    )


LENS_NA = dict(green_day_base=np.nan, green_day_rule=np.nan, green_week_base=np.nan, green_week_rule=np.nan,
               weekly_p10_base=np.nan, weekly_p10_rule=np.nan, weekly_sharpe_base=np.nan, weekly_sharpe_rule=np.nan,
               maxdd_base=np.nan, maxdd_rule=np.nan)

READS_COLUMNS = ['id', 'read', 'n', 'dR', 'iid_t', 'day_t', 'ex5', 'mde', 'fired_share', 'saved', 'forgone', 'cost',
                  'green_day_base', 'green_day_rule', 'green_week_base', 'green_week_rule',
                  'weekly_p10_base', 'weekly_p10_rule', 'weekly_sharpe_base', 'weekly_sharpe_rule',
                  'maxdd_base', 'maxdd_rule']


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _atomic_replace_by_id(path, new_df, id_col='id'):
    if os.path.exists(path):
        old = pd.read_csv(path)
        ids = set(new_df[id_col].unique())
        old = old[~old[id_col].isin(ids)]
        out = pd.concat([old, new_df], ignore_index=True)
    else:
        out = new_df
    tmp = path + '.tmp'
    out.to_csv(tmp, index=False)
    os.replace(tmp, path)


def run_hypothesis(hid, pop, paths, p_success, p_stop, lens=False):
    spec = HYPOTHESES[hid]
    t0 = time.time()
    reads_rows = []
    per_fill_rows = []
    read_list = SCORINGS if spec['model'] else HALVES
    for read in read_list:
        if spec['model']:
            target_half = 'VAL' if read == 'TRAIN-H2->VAL' else 'TRAIN-H2'
            sub = pop[pop['half'] == target_half]
            scoring = read
        else:
            sub = pop[pop['half'] == read]
            scoring = None
        rule_Rs, base_Rs, fired_flags, dates, fids = [], [], [], [], []
        for row in sub.itertuples():
            path = paths.get(row.fill_id)
            if path is None or len(path['o']) < 2:
                continue
            ctx = FillCtx(row.fill_id, row, path, p_success, p_stop, scoring)
            r, fired, fmin = spec['fn'](ctx)
            rule_Rs.append(r)
            base_Rs.append(row.base_R)
            fired_flags.append(fired)
            dates.append(row.date)
            fids.append(row.fill_id)
            per_fill_rows.append(dict(id=hid, read=read, fill_id=row.fill_id, date=row.date,
                                       symbol=row.symbol, base_R=row.base_R, rule_R=r,
                                       dR=r - row.base_R, fired=fired, fired_min=fmin))
        rule_Rs, base_Rs = np.array(rule_Rs), np.array(base_Rs)
        fired_flags = np.array(fired_flags)
        dates_arr = np.array(dates)
        dR = rule_Rs - base_Rs
        # PREREG ground rule: "paired dR vs the base ... on the whole book"
        # -- every fill counts, unfired ones contributing dR=0 by
        # construction (matches hod_exit_lab's X-cell convention, which is
        # what H25's validation target was built under). "n" = the whole
        # read's fill count; "fired_share" (below) reports how often the
        # rule actually acted. See the module docstring / VALIDATION NOTE
        # for cell 1,677's OWN table, which instead reports its headline
        # mean_dR conditional on firing (n=n_fired) -- reproduced
        # separately, not stored here, when validating H16 against it.
        s = stats_block(dR, dates_arr)
        dc = decompose(fired_flags, rule_Rs, base_Rs)
        lens_vals = consistency_lens(dates, base_Rs, rule_Rs) if lens else dict(LENS_NA)
        reads_rows.append(dict(
            id=hid, read=read, n=s['n'], dR=s['dR'], iid_t=s['iid_t'], day_t=s['day_t'],
            ex5=s['ex5'], mde=s['mde'], fired_share=fired_flags.mean() if len(fired_flags) else np.nan,
            saved=dc['saved'], forgone=dc['forgone'], cost=dc['cost'], **lens_vals,
        ))
        logger.info('%s [%s]: n=%d dR=%.4f t_day=%.2f fired=%.1f%%', hid, read, s['n'], s['dR'],
                    s['day_t'] if pd.notna(s['day_t']) else float('nan'),
                    100 * (fired_flags.mean() if len(fired_flags) else 0))
    reads_df = pd.DataFrame(reads_rows, columns=READS_COLUMNS)
    per_fill_df = pd.DataFrame(per_fill_rows)
    _atomic_replace_by_id(READS_CSV, reads_df, 'id')
    _atomic_replace_by_id(PER_FILL_CSV, per_fill_df, 'id')
    logger.info('%s done in %.1fs, %d reads written', hid, time.time() - t0, len(reads_rows))
    return reads_df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True, help='comma list of hypothesis ids, e.g. H1,H2, or ALL')
    ap.add_argument('--lens', action='store_true', help='also compute the consistency lens (green-day/week, P10, Sharpe, maxdd)')
    ap.add_argument('--rebuild-paths', action='store_true', help='force-rebuild 1681_paths.parquet')
    args = ap.parse_args()
    setup_logging()
    logger.info('=== 1681_hypotheses start: run=%s lens=%s ===', args.run, args.lens)

    ids = list(HYPOTHESES.keys()) if args.run.strip().upper() == 'ALL' else [s.strip() for s in args.run.split(',') if s.strip()]
    bad = [i for i in ids if i not in HYPOTHESES]
    if bad:
        logger.error('unknown hypothesis id(s): %s', bad)
        sys.exit(1)

    pop = load_population()
    paths = build_paths(pop, rebuild=args.rebuild_paths)
    p_success, p_stop = load_model_probs()

    for hid in ids:
        run_hypothesis(hid, pop, paths, p_success, p_stop, lens=args.lens)

    logger.info('=== 1681_hypotheses done ===')


if __name__ == '__main__':
    main()
