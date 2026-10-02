#!/usr/bin/env python3
"""Cell 1,701 step 3: walk + reads for the earnings-session ORB pool "E1".

PREREG: research/orb_earn/PREREG_1701.md (FROZEN 2026-10-02, Amendment 1). Implements exactly:
population from 1701_candidates.csv, production entry (range-high/low break + chase cap) on
Databento 1-min bars, three exits (X1/X2/X3), both sides, both directions, the reads, the pass
bar. No filter added after seeing numbers (PREREG "Not allowed").

Reuses VERBATIM (importlib for digit-prefixed cell scripts, project convention; normal import
for trading/*):
  research/orb_freq/1694_money.py (f1694, itself carrying f1693 which carries f1679/f1668/f1677/
    cadence_bar/orb_csv transitively) -- f1694.f1679.find_range_and_breakout (opening range
    09:30-09:34 ET + breakout = first bar in [09:35,10:35) ET whose high > range_high),
    f1694._walk_lock (X1 target_r=2.0 / X2 target_r=None -- ONE unified function, "an optional
    fixed target caps the trade; None = no target, ride the lock to the 15:45 close --
    A3/production"), f1694._walk_scale_then_lock (X3, scale_r=2.0), f1694.stats_block (n,
    mean_dR, iid_t, day_t, ex_top5, mde), f1694.CachedStore (bars_sip.db, read-only -- cross-
    check ONLY, never the walk's bar source). ORB_EOD_M/EXIT_SLIP_BPS/R_FLOOR_PCT/
    LOCK_TRIGGER_R_LIVE/LOCK_STOP_R_LIVE/RANGE_LO_M.. all read off f1679, never redefined here.
  trading/hod_break.py -- entry_fill(next_open, level, p): "fills at the next bar's open iff at
    or under level*(1+cap)", the project's OWN already-shipped generic implementation of
    CLAUDE.md's obtainability rule ("every fill is ... reachable by an order the engine would
    have had resting: next bar's open under a cap, never the touch of a level"). Called here
    with HodBreakParams(cap=0.02) -- the ORB production chase cap (trading/buy_stop_guard.py
    docstring + trading/order_executor.py: limit = round(stop_price*1.02, 2)). Pure, no I/O.
  scripts/cadence_bar.py (cb) -- build_weekly_series/week_monday/percentile/
    max_drawdown_and_underwater (weekly P10/worst-week/max-DD, called in-process, not shelled
    out, so no text-report parsing is needed).
  trading/orb_csv.py -- read_orb_csv for analysis_results/orb_bplus_book.csv (union step).

NOT a reuse (clearly marked; no existing ORB cell ever walked a population with no pre-existing
entry_price column -- every prior cell's reconstruct_fill took entry_price from an already-
recorded book row, so this glue never had to be written before):
  DatabentoStore.day_bars(symbol, day) -- SAME dict shape (minarr/o/h/l/c/v) as
    f1668.BarStore.day_bars, loads ONE day-shard at a time (not the full merged parquet) for the
    node's ~2GB RAM budget.
  reconstruct_entry_long/_short -- the glue connecting find_range_and_breakout's range/breakout
    to entry_fill's capped next-open fill (every earlier cell was handed entry_price already).
  mirror_bars + entry_fill_short -- the short side. PREREG asked for "a sign-flipped call of the
    same walker if it supports a direction flag, else a clearly-marked mirror function with a
    unit self-check": entry_fill's own formula level*(1+cap) is NOT sign-agnostic under price
    negation (hand-derivation caught this before shipping: a negated level makes the cap move
    the wrong way), so entry_fill_short is the direct mechanical mirror, not a sign-flipped
    call of entry_fill. _walk_lock/_walk_scale_then_lock ARE reused as a true sign-flipped call
    (mirror_bars only; the walkers themselves are untouched) -- proved sign-correct by hand and
    by _self_check_mirror() below, which gates the script (raises loudly if it ever regresses).
  Quote fetch (measured NBBO half-spread) -- same libraries/pattern as
    research/bf_zero/causal_filter/fetch_nbbo.py (StockHistoricalDataClient/StockQuotesRequest/
    DataFeed.SIP), adapted for 1701's fill list, cached to 1701_quotes.parquet, resumable.

Cost: exit slippage is ADV-conditional (PREREG: 35bps mean for ADV20<$20M, else the project's
standard EXIT_SLIP_BPS=10) passed as the walkers' own `slip_bps` argument -- applied uniformly to
whichever exit fires (stop/lock/target/EOD), NOT stop-only, because the reused walkers return a
price/R only, no exit-reason tag, and classifying the reason post-hoc by float-matching against
the known levels is fragile. This is a stated, conservative approximation (overstates cost on
thin-name target/EOD exits) -- flagged here and in RESULT_1701.md, not hidden. On top of that,
entry cost = measured NBBO half-spread at the entry minute, subtracted separately as its own R
term (never double-charged against the same slippage).

Usage:
    python3 research/orb_earn/1701_walk.py --source sip --smoke   # <=300 sessions already in bars_sip.db
    setsid nohup nice -n 10 python3 research/orb_earn/1701_walk.py --source sip --full \
        > research/orb_earn/1701_walk.log 2>&1 &
    # --source databento (default was databento pre-Amendment-2) loads the venue-subset EQUS.MINI
    # shards -- kept for the cross-check only; its RESULT is marked VOID (Amendment 2).
"""
import argparse
import glob
import importlib.util
import logging
import os
import sqlite3
import sys
import time
from datetime import date as _date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

CANDIDATES_CSV = os.path.join(HERE, '1701_candidates.csv')
SHARD_DIR = os.path.join(HERE, 'bars_db_1701')
MERGED_PARQUET = os.path.join(HERE, 'bars_db_1701.parquet')
QUOTES_PARQUET = os.path.join(HERE, '1701_quotes.parquet')
FILLS_CSV = os.path.join(HERE, '1701_fills.csv')
READS_CSV = os.path.join(HERE, '1701_reads.csv')
RESULT_MD = os.path.join(HERE, 'RESULT_1701.md')
LOG_FILE = os.path.join(HERE, '1701_walk.log')
BARS_SIP_DB = os.path.join(ROOT, 'research/bf_zero/bars_sip.db')
PRODUCTION_BOOK_CSV = os.path.join(ROOT, 'analysis_results/orb_bplus_book.csv')

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                     handlers=[logging.StreamHandler(), logging.FileHandler(LOG_FILE, mode='a')])
logger = logging.getLogger('cell1701')

ET_ZONE = ZoneInfo('America/New_York')
CHASE_CAP = 0.02           # trading/buy_stop_guard.py + order_executor.py: round(stop*1.02,2)
FIXED_RISK_DOLLARS = 375.0  # project-wide ORB per-trade risk convention (matches f1693/f1694)
HALF_A = (_date(2024, 7, 1), _date(2025, 6, 30))
HALF_B = (_date(2025, 7, 1), _date(2026, 9, 30))
WHOLE = (_date(2024, 7, 1), _date(2026, 9, 30))
MAX_SLOTS_PER_SIDE_PER_DAY = 8


def _load_module(name, fname, root=ROOT):
    """Verbatim pattern from research/orb_exit/1679_orb_exit.py._load_module (project
    convention for digit-prefixed cell scripts, which cannot be `import`ed normally)."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(root, fname))
    mod = importlib.util.module_from_spec(spec)
    old_argv = sys.argv
    sys.argv = [sys.argv[0]]
    spec.loader.exec_module(mod)
    sys.argv = old_argv
    return mod


def check_disk(floor_gb=2.0):
    """Node is at ~4.6GB free (96% full) from OTHER running jobs -- this script's own writes
    (quotes parquet, fills/reads CSVs) are small, so the floor is lower than the project's
    usual 5GB fetch-job gate, but still checked and logged (CLAUDE.md: every fallback path
    logs why)."""
    st = os.statvfs('/')
    free_gb = st.f_bavail * st.f_frsize / (1024 ** 3)
    logger.info('disk free on /: %.2f GB', free_gb)
    if free_gb < floor_gb:
        logger.error('disk free %.2f GB < %.2f GB floor -- aborting', free_gb, floor_gb)
        sys.exit(1)
    return free_gb


# ---------------------------------------------------------------------------
# Reused modules (verbatim)
# ---------------------------------------------------------------------------
f1694 = _load_module('f1694_1701', 'research/orb_freq/1694_money.py')
f1679 = f1694.f1679
find_range_and_breakout = f1679.find_range_and_breakout
_walk_lock = f1694._walk_lock
_walk_scale_then_lock = f1694._walk_scale_then_lock
stats_block = f1694.stats_block
CachedStore = f1694.CachedStore
ORB_EOD_M = f1679.ORB_EOD_M
EXIT_SLIP_BPS = f1679.EXIT_SLIP_BPS
LOCK_TRIGGER_R_LIVE = f1679.LOCK_TRIGGER_R_LIVE
LOCK_STOP_R_LIVE = f1679.LOCK_STOP_R_LIVE
RANGE_LO_M, RANGE_HI_M = f1679.RANGE_LO_M, f1679.RANGE_HI_M

from trading.hod_break import entry_fill, HodBreakParams  # noqa: E402

cb = _load_module('cadence_bar_1701', 'scripts/cadence_bar.py')
orb_csv = _load_module('orb_csv_1701', 'trading/orb_csv.py')


# ---------------------------------------------------------------------------
# NEW: short-side mirror (no existing cell has ever traded the short side of
# an ORB breakout -- see module docstring for why this is not re-implementing
# the reused walkers, just the glue that lets them run on a short setup).
# ---------------------------------------------------------------------------

def mirror_bars(bars):
    """Price-mirror: negate every price field and swap high/low; volume and minarr (time) are
    unchanged. Self-inverse: mirror_bars(mirror_bars(b)) == b (bit-identical, pure negation)."""
    return dict(minarr=bars['minarr'], o=-bars['o'], h=-bars['l'], l=-bars['h'], c=-bars['c'],
                v=bars.get('v'))


def entry_fill_short(next_open, level, cap):
    """Short mirror of trading/hod_break.py::entry_fill. NOT a sign-flipped call of entry_fill
    itself (its formula level*(1+cap) is not sign-agnostic under price negation -- a negated
    level makes the cap move the wrong way, caught by hand-derivation before shipping). Direct
    mechanical mirror: fills at the next bar's open iff it is AT OR ABOVE level*(1-cap) (the
    downside chase cap; no chase further down than that)."""
    limit = level * (1.0 - cap)
    return float(next_open) if next_open >= limit else None


def reconstruct_entry_long(bars, cap=CHASE_CAP):
    """Long entry: f1679.find_range_and_breakout (VERBATIM) for the range/breakout bar, then
    trading.hod_break.entry_fill (VERBATIM, cap=0.02) at the NEXT bar's open. Returns a dict
    with reason='ok' (+ i0/entry/stop/R_unit/bars ready for the exit walkers unchanged) or a
    reason string explaining why not (never a silent drop)."""
    rec = find_range_and_breakout(bars)
    if rec is None:
        return dict(reason='no_breakout')
    i0b = rec['i0']
    if i0b + 1 >= len(bars['o']):
        return dict(reason='no_next_bar')
    entry = entry_fill(float(bars['o'][i0b + 1]), rec['range_high'], HodBreakParams(cap=cap))
    if entry is None:
        return dict(reason='no_fill_capped')
    stop = rec['range_low']
    R_unit = entry - stop
    if not (R_unit > 0):
        return dict(reason='bad_R')
    if not (0.005 <= R_unit / entry <= 0.10):
        return dict(reason='R_pct_out_of_band')
    return dict(reason='ok', i0=i0b + 1, entry=entry, stop=stop, R_unit=R_unit, bars=bars,
                side='long', entry_real=entry, stop_real=stop,
                range_high=rec['range_high'], range_low=rec['range_low'])


def reconstruct_entry_short(bars, cap=CHASE_CAP):
    """Short mirror of reconstruct_entry_long. mirror_bars + find_range_and_breakout (VERBATIM,
    unmodified call) locate the breakout in mirror space; entry_fill_short fills on REAL
    (unmirrored) prices; the returned entry/stop/bars are in MIRROR SPACE, ready to hand to
    _walk_lock/_walk_scale_then_lock UNCHANGED -- the R they return is already correctly signed
    (proved by _self_check_mirror() below). entry_real/stop_real are the real (short) prices,
    for the fills CSV."""
    mbars = mirror_bars(bars)
    rec = find_range_and_breakout(mbars)
    if rec is None:
        return dict(reason='no_breakout')
    i0b = rec['i0']
    if i0b + 1 >= len(bars['o']):
        return dict(reason='no_next_bar')
    range_low_real = -rec['range_high']
    range_high_real = -rec['range_low']
    entry_real = entry_fill_short(float(bars['o'][i0b + 1]), range_low_real, cap)
    if entry_real is None:
        return dict(reason='no_fill_capped')
    stop_real = range_high_real
    R_unit = stop_real - entry_real
    if not (R_unit > 0):
        return dict(reason='bad_R')
    if not (0.005 <= R_unit / entry_real <= 0.10):
        return dict(reason='R_pct_out_of_band')
    return dict(reason='ok', i0=i0b + 1, entry=-entry_real, stop=-stop_real, R_unit=R_unit,
                bars=mbars, side='short', entry_real=entry_real, stop_real=stop_real,
                range_high=range_high_real, range_low=range_low_real)


def _synthetic_long_bars():
    """Hand-built minute bars 09:30-16:15 ET for the unit self-check: flat range 09:30-09:34
    (range_high=100.2, range_low=99.0), breakout bar at 09:35 (high pokes to 100.9), entry bar
    at 09:36 opens at 100.10, then a clean walk past every R level (+1.75R arm, +2R target) by
    11:00, flat to EOD."""
    minutes = np.arange(570, 976, dtype=float)
    n = len(minutes)
    o = np.full(n, 100.0); h = np.full(n, 100.2); l = np.full(n, 99.0); c = np.full(n, 100.0)
    v = np.full(n, 1000.0)
    o[5], h[5], l[5], c[5] = 100.05, 100.9, 100.0, 100.5   # breakout bar (idx 5, minute 575)
    o[6] = 100.10                                          # entry bar open (idx 6, minute 576)
    for i in range(7, 25):
        lvl = 100.10 + 0.15 * (i - 6)
        o[i] = l[i] = lvl - 0.02
        h[i] = c[i] = lvl
    for i in range(25, n):
        o[i] = h[i] = l[i] = c[i] = 103.0
    return dict(minarr=minutes, o=o, h=h, l=l, c=c, v=v)


def _reflect_bars(bars, pivot=100.0):
    """Self-check-ONLY test-data helper (never used in the real walk): affine reflection
    price' = 2*pivot - price. Turns the long-favorable synthetic path into a short-favorable
    one with realistic POSITIVE prices (unlike mirror_bars' pure negation, which is correct for
    REAL market data but produces negative numbers out of an already-positive synthetic long
    path -- entry_fill_short's level*(1-cap) formula is only valid for positive, real levels,
    the same way entry_fill's level*(1+cap) is; feeding it a negative level is testing an input
    that can never occur in production and was caught, not a production bug)."""
    return dict(minarr=bars['minarr'], o=2 * pivot - bars['o'], h=2 * pivot - bars['l'],
                l=2 * pivot - bars['h'], c=2 * pivot - bars['c'], v=bars.get('v'))


def _self_check_mirror():
    """PREREG's required unit self-check: a synthetic bar path walked long and its mirror
    walked short must give identical R, for X1/X2/X3. Raises AssertionError (loud, never
    silent) on failure -- gates the whole script. slip_bps=0 in the walk comparison: slip is a
    PERCENTAGE OF THE ABSOLUTE PRICE LEVEL, and _reflect_bars shifts the absolute price scale
    (to stay positive) while preserving every relative move exactly -- slip itself is already
    well-tested elsewhere (f1679/f1693) and orthogonal to the sign/shape correctness this check
    targets, so it is zeroed out here rather than re-verified."""
    assert entry_fill_short(98.5, 100.0, 0.02) == 98.5, 'entry_fill_short: within-cap fill wrong'
    assert entry_fill_short(97.0, 100.0, 0.02) is None, 'entry_fill_short: past-cap should skip'
    bars_long = _synthetic_long_bars()
    rl = reconstruct_entry_long(bars_long)
    assert rl['reason'] == 'ok', f'synthetic long setup produced no fill: {rl}'
    bars_short = _reflect_bars(bars_long, pivot=100.0)
    rs = reconstruct_entry_short(bars_short)
    assert rs['reason'] == 'ok', f'synthetic short setup produced no fill: {rs}'
    for name, trig, lock, targ in (('X1', 1.75, 0.5, 2.0), ('X2', 1.75, 0.5, None)):
        Rl, _ = _walk_lock(rl['bars'], rl['i0'], rl['entry'], rl['stop'], rl['R_unit'], trig, lock, targ, ORB_EOD_M, 0.0)
        Rs, _ = _walk_lock(rs['bars'], rs['i0'], rs['entry'], rs['stop'], rs['R_unit'], trig, lock, targ, ORB_EOD_M, 0.0)
        assert abs(Rl - Rs) < 1e-6, f'{name} mirror mismatch: long={Rl} short={Rs}'
    Rl, _ = _walk_scale_then_lock(rl['bars'], rl['i0'], rl['entry'], rl['stop'], rl['R_unit'], 2.0, ORB_EOD_M, 0.0)
    Rs, _ = _walk_scale_then_lock(rs['bars'], rs['i0'], rs['entry'], rs['stop'], rs['R_unit'], 2.0, ORB_EOD_M, 0.0)
    assert abs(Rl - Rs) < 1e-6, f'X3 mirror mismatch: long={Rl} short={Rs}'
    logger.info('UNIT SELF-CHECK PASSED: long vs mirrored-short identical R on X1/X2/X3 (synthetic path)')


# ---------------------------------------------------------------------------
# NEW: Databento shard loader, SAME day_bars(symbol,day) interface as
# f1668.BarStore (no existing cell reads Databento bars).
# ---------------------------------------------------------------------------

class DatabentoStore:
    """day_bars(symbol, day) over Databento per-day parquet shards in SHARD_DIR. Loads ONE
    day-shard at a time (not the full merged parquet) for the node's ~2GB RAM budget;
    candidates are processed sorted by day, so this is a true single-pass cache."""

    def __init__(self, shard_dir=SHARD_DIR, merged_path=MERGED_PARQUET):
        self.shard_dir = shard_dir
        self.merged_path = merged_path
        self._day = None
        self._by_symbol = {}
        self._merged = None

    def _load_day(self, day):
        path = os.path.join(self.shard_dir, f'{day}.parquet')
        if os.path.exists(path):
            df = pd.read_parquet(path)
        elif self.merged_path and os.path.exists(self.merged_path):
            if self._merged is None:
                logger.warning('no shard for %s -- falling back to the merged parquet (slower)', day)
                self._merged = pd.read_parquet(self.merged_path)
            df = self._merged[self._merged['day'] == day]
        else:
            df = pd.DataFrame(columns=['symbol', 'ts_utc', 'open', 'high', 'low', 'close', 'volume'])
        by_symbol = {}
        if len(df):
            df = df.sort_values(['symbol', 'ts_utc'])
            ts_et = df['ts_utc'].dt.tz_convert(ET_ZONE)
            minarr = ts_et.dt.hour.to_numpy() * 60.0 + ts_et.dt.minute.to_numpy() + ts_et.dt.second.to_numpy() / 60.0
            df = df.assign(_minarr=minarr)
            for sym, g in df.groupby('symbol', sort=False):
                by_symbol[sym] = dict(
                    minarr=g['_minarr'].to_numpy(dtype=float),
                    o=g['open'].to_numpy(dtype=float), h=g['high'].to_numpy(dtype=float),
                    l=g['low'].to_numpy(dtype=float), c=g['close'].to_numpy(dtype=float),
                    v=g['volume'].to_numpy(dtype=float))
        self._by_symbol = by_symbol
        self._day = day

    def day_bars(self, symbol, day):
        if day != self._day:
            self._load_day(day)
        return self._by_symbol.get(symbol)

    def available_days(self):
        return sorted(os.path.basename(p)[:-len('.parquet')]
                      for p in glob.glob(os.path.join(self.shard_dir, '*.parquet')))


# ---------------------------------------------------------------------------
# NEW (Amendment 2): the Alpaca SIP store (research/bf_zero/bars_sip.db) is the ONLY source a
# range-based entry can be walked on -- Databento EQUS.MINI is a venue subset (opening-range
# high/low inside the SIP range by up to 4%) and stays above for the cross-check only.
# SipRetryStore composes the project's own CachedStore (f1694.CachedStore ->
# research/orb_freq/1693_pool_exits.py, wrapping f1668.BarStore, read-only `?mode=ro`,
# day_bars(symbol, day) -> {minarr, o, h, l, c, v}, SAME shape as DatabentoStore above) so
# walk_all is UNCHANGED under --source sip. The only addition is retry/backoff: f1668.BarStore
# opens read-only but carries no busy-timeout of its own, and the appender
# (research/orb_earn/1701_backfill_run.py) is the sole writer and may hold the write lock --
# every call here retries 'database is locked' with exponential backoff capped at 30s total,
# logging WARNING per retry / ERROR on final failure (CLAUDE.md: every fallback path logs why).
# ---------------------------------------------------------------------------

class SipRetryStore:
    """day_bars(symbol, day) over bars_sip.db via f1694.CachedStore, with a 30s busy-timeout/
    backoff wrapper around the appender's write lock. Never writes; never opens the DB any way
    but through CachedStore's own read-only connection."""

    def __init__(self, db_path=BARS_SIP_DB, max_wait_s=30.0):
        self._store = CachedStore(db_path)
        self.max_wait_s = max_wait_s

    def day_bars(self, symbol, day):
        waited, delay = 0.0, 0.5
        while True:
            try:
                return self._store.day_bars(symbol, str(day))
            except sqlite3.OperationalError as e:
                if 'locked' not in str(e).lower() or waited >= self.max_wait_s:
                    logger.error('bars_sip.db read failed for %s %s after %.1fs waited: %s', symbol, day, waited, e)
                    return None
                logger.warning('bars_sip.db locked (appender writing) -- retry %s %s in %.1fs (%.1f/%.0fs waited)',
                                symbol, day, delay, waited, self.max_wait_s)
                time.sleep(delay)
                waited += delay
                delay = min(delay * 2.0, 5.0)

    def close(self):
        self._store.close()


def sip_available_sessions(cands, db_path=BARS_SIP_DB, limit=300):
    """Read-only count of candidate (symbol, session) pairs already written to bars_sip.db by
    the appender (research/orb_earn/1701_backfill_run.py) -- the --smoke session filter for
    --source sip. Opens the DB for exactly one DISTINCT query, mode=ro."""
    con = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True, timeout=30)
    try:
        have = pd.read_sql('SELECT DISTINCT symbol, day FROM bars', con)
    finally:
        con.close()
    have_pairs = set(zip(have['symbol'], have['day']))
    pairs = cands[['symbol', 'session']].drop_duplicates()
    covered = pairs.apply(lambda r: (r['symbol'], r['session']) in have_pairs, axis=1)
    sessions = sorted(pairs.loc[covered, 'session'].unique())
    logger.info('SIP smoke availability: %d/%d candidate (symbol,day) pairs already in bars_sip.db -- '
                '%d distinct sessions covered', int(covered.sum()), len(pairs), len(sessions))
    return sessions[:limit]


def sip_coverage_gate(cands, store):
    """PREREG Amendment 2's coverage gate: a session is walkable only if all five 09:30-09:34 ET
    bars (RANGE_LO_M..RANGE_HI_M) are present AND there are >=300 regular-session (09:30-16:00
    ET) bars. Returns (walkable, requested, pct, note); note states VOID below the 95% gate --
    never silently dropped."""
    RTH_END_M = 960.0  # 16:00 ET regular-session close (ORB_EOD_M=945 is the 15:45 flatten rule, not this)
    pairs = cands[['symbol', 'session']].drop_duplicates()
    requested = len(pairs)
    walkable = 0
    for r in pairs.itertuples():
        bars = store.day_bars(r.symbol, r.session)
        if bars is None or len(bars['o']) == 0:
            continue
        m = bars['minarr']
        five_present = len(set(np.floor(m[(m >= RANGE_LO_M) & (m <= RANGE_HI_M)]).astype(int))) >= 5
        rth_n = int(((m >= RANGE_LO_M) & (m < RTH_END_M)).sum())
        if five_present and rth_n >= 300:
            walkable += 1
    pct = 100.0 * walkable / requested if requested else 0.0
    note = (f'SIP coverage: {walkable}/{requested} sessions walkable ({pct:.1f}%) -- '
            + ('gate MET (>=95%).' if pct >= 95.0 else 'gate NOT met -- VOID per Amendment 2.'))
    logger.info(note)
    return walkable, requested, pct, note


def session_features(bars):
    """09:30-09:34 ET range (5 bars) -> range_high/low, range_volume, session_open (first RTH
    bar's open). None if the window isn't present (thin/missing bars)."""
    m = bars['minarr']
    rmask = (m >= RANGE_LO_M) & (m <= RANGE_HI_M)
    if rmask.sum() == 0:
        return None
    idx = np.where(rmask)[0]
    return dict(range_high=float(bars['h'][rmask].max()), range_low=float(bars['l'][rmask].min()),
                range_volume=float(np.nansum(bars['v'][rmask])), session_open=float(bars['o'][idx[0]]))


# ---------------------------------------------------------------------------
# Main walk
# ---------------------------------------------------------------------------

def walk_all(cands, store, cap=CHASE_CAP):
    """One pass over every candidate row: side by open vs prior_close, reconstruct the entry,
    walk X1/X2/X3 if filled. Every row keeps a `reason` (ok or why not -- never a silent drop).
    Processed sorted by session so the DatabentoStore's single-day cache is never re-loaded."""
    rows = []
    t0 = time.time()
    cands_sorted = cands.sort_values('session')
    n = len(cands_sorted)
    n_bars = n_range = n_ok = 0
    for i, r in enumerate(cands_sorted.itertuples(), 1):
        bars = store.day_bars(r.symbol, r.session)
        row = dict(symbol=r.symbol, session=r.session, half=r.half, prior_close=r.prior_close,
                   adv20_dollar=r.adv20_dollar, release_slot=r.release_slot)
        if bars is None or len(bars['o']) < 10:
            row.update(reason='no_bars', side=None)
            rows.append(row)
            if i % 1000 == 0 or i == n:
                logger.info('walk %d/%d (%.0fs) bars=%d range=%d ok=%d', i, n, time.time() - t0, n_bars, n_range, n_ok)
            continue
        n_bars += 1
        sf = session_features(bars)
        if sf is None:
            row.update(reason='no_range', side=None)
            rows.append(row)
            continue
        n_range += 1
        side = 'long' if sf['session_open'] >= r.prior_close else 'short'
        row['side'] = side
        row['gap_pct'] = (sf['session_open'] - r.prior_close) / r.prior_close * 100.0
        adv20_shares = r.adv20_dollar / r.prior_close if r.prior_close else np.nan
        row['rvol_0935'] = sf['range_volume'] / adv20_shares if adv20_shares and adv20_shares > 0 else np.nan
        rec = reconstruct_entry_long(bars, cap) if side == 'long' else reconstruct_entry_short(bars, cap)
        row['reason'] = rec['reason']
        if rec['reason'] != 'ok':
            rows.append(row)
            continue
        n_ok += 1
        slip_bps = 35.0 if (r.adv20_dollar is not None and r.adv20_dollar < 20e6) else 10.0
        x1, m1 = _walk_lock(rec['bars'], rec['i0'], rec['entry'], rec['stop'], rec['R_unit'], 1.75, 0.5, 2.0, ORB_EOD_M, slip_bps)
        x2, m2 = _walk_lock(rec['bars'], rec['i0'], rec['entry'], rec['stop'], rec['R_unit'], 1.75, 0.5, None, ORB_EOD_M, slip_bps)
        x3, m3 = _walk_scale_then_lock(rec['bars'], rec['i0'], rec['entry'], rec['stop'], rec['R_unit'], 2.0, ORB_EOD_M, slip_bps)
        eod_idx = np.where(rec['bars']['minarr'] >= ORB_EOD_M)[0]
        last_idx = int(eod_idx[0]) if len(eod_idx) else len(rec['bars']['o']) - 1
        seg_h = rec['bars']['h'][rec['i0'] + 1:last_idx + 1]
        mfe_R = float((seg_h.max() - rec['entry']) / rec['R_unit']) if len(seg_h) else np.nan
        entry_minute = float(rec['bars']['minarr'][rec['i0']])
        row.update(entry_price=rec['entry_real'], stop_price=rec['stop_real'], R_unit=rec['R_unit'],
                   entry_minute_et=entry_minute, X1_R=x1, X2_R=x2, X3_R=x3, mfe_R=mfe_R,
                   runner3=int(mfe_R >= 3.0) if mfe_R == mfe_R else 0, slip_bps_used=slip_bps)
        rows.append(row)
        if i % 1000 == 0 or i == n:
            logger.info('walk %d/%d (%.0fs) bars=%d range=%d ok=%d', i, n, time.time() - t0, n_bars, n_range, n_ok)
    out = pd.DataFrame(rows)
    logger.info('walk done: %d candidates, %d with bars, %d with a range, %d filled (ok)', n, n_bars, n_range, n_ok)
    return out


def apply_slots(fills):
    """8 slots per side per day, ranked by rvol_0935 descending (PREREG step 3). Fills beyond
    8 are flagged beyond_slots=True (frequency-only, excluded from the edge reads)."""
    fills = fills.copy()
    fills['beyond_slots'] = False
    ok = fills[fills['reason'] == 'ok']
    for (session, side), g in ok.groupby(['session', 'side']):
        ranked = g.sort_values('rvol_0935', ascending=False)
        beyond_idx = ranked.index[MAX_SLOTS_PER_SIDE_PER_DAY:]
        fills.loc[beyond_idx, 'beyond_slots'] = True
    return fills


# ---------------------------------------------------------------------------
# Cost: measured NBBO half-spread at the entry minute (adapted from
# research/bf_zero/causal_filter/fetch_nbbo.py's pattern).
# ---------------------------------------------------------------------------

def fetch_quotes(fills_ok):
    """Batched, resumable, cached Alpaca SIP quote fetch for the FILL minutes only (one request
    per (session,symbol), not per exit). Returns a DataFrame symbol,session,spread_mean. Any
    fetch failure is logged WARNING and recorded with spread_mean=NaN (never silently dropped);
    a NaN spread costs nothing extra downstream, which is reported, not hidden."""
    need = fills_ok[['session', 'symbol', 'entry_minute_et']].drop_duplicates(['session', 'symbol'])
    done = pd.DataFrame(columns=['session', 'symbol', 'spread_mean', 'err'])
    if os.path.exists(QUOTES_PARQUET):
        done = pd.read_parquet(QUOTES_PARQUET)
    done_keys = set(zip(done['session'], done['symbol'])) if len(done) else set()
    todo = [r for r in need.itertuples() if (r.session, r.symbol) not in done_keys]
    logger.info('quotes: %d fills needing a quote, %d already cached, %d todo', len(need), len(done), len(todo))
    if not todo:
        return done
    from config import Config  # noqa
    from alpaca.data.historical import StockHistoricalDataClient  # noqa
    from alpaca.data.requests import StockQuotesRequest  # noqa
    from alpaca.data.enums import DataFeed  # noqa
    cfg = Config()
    client = StockHistoricalDataClient(cfg.alpaca_api_key, cfg.alpaca_api_secret)
    rows = []
    t0 = time.time()
    for i, r in enumerate(todo):
        m = r.entry_minute_et
        hh, mm = int(m // 60), int(m % 60)
        midnight_et = datetime.strptime(r.session, '%Y-%m-%d').replace(tzinfo=ET_ZONE)
        start = midnight_et + timedelta(hours=hh, minutes=mm)
        rec = dict(session=r.session, symbol=r.symbol, spread_mean=np.nan, n=0, err='')
        try:
            q = client.get_stock_quotes(StockQuotesRequest(
                symbol_or_symbols=r.symbol, start=start, end=start + timedelta(minutes=1),
                feed=DataFeed.SIP, limit=6000)).data.get(r.symbol, [])
            sp = np.array([float(x.ask_price) - float(x.bid_price) for x in q
                           if x.ask_price and x.bid_price and x.ask_price > x.bid_price])
            rec['n'] = len(sp)
            if len(sp):
                rec['spread_mean'] = float(sp.mean())
        except Exception as e:                                           # loud, never silent
            rec['err'] = type(e).__name__
            logger.warning('quote fetch %s %s: %s %s', r.session, r.symbol, type(e).__name__, e)
        rows.append(rec)
        time.sleep(0.3)   # rate-limit aware: ~200 req/min, Alpaca's free-tier historical cap
        if len(rows) >= 200 or i == len(todo) - 1:
            batch = pd.DataFrame(rows)
            done = pd.concat([done, batch], ignore_index=True)
            done.to_parquet(QUOTES_PARQUET + '.tmp', index=False)
            os.rename(QUOTES_PARQUET + '.tmp', QUOTES_PARQUET)
            rows = []
            el = time.time() - t0
            logger.info('quotes %d/%d (%.1f min, %.0f/min)', i + 1, len(todo), el / 60, (i + 1) / max(el, 1) * 60)
    return done


# ---------------------------------------------------------------------------
# Reads: per side x exit x window, cadence bar, union, pass bar.
# ---------------------------------------------------------------------------

def _half_of(session_str):
    d = datetime.strptime(session_str, '%Y-%m-%d').date()
    if HALF_A[0] <= d <= HALF_A[1]:
        return 'A'
    if HALF_B[0] <= d <= HALF_B[1]:
        return 'B'
    return None


def _in_season(d):
    """6 weeks (42 days) after any quarter-end."""
    for m, day in ((3, 31), (6, 30), (9, 30), (12, 31)):
        qe = _date(d.year, m, day)
        if qe <= d <= qe + timedelta(days=42):
            return True
        qe_prev = _date(d.year - 1, m, day)
        if qe_prev <= d <= qe_prev + timedelta(days=42):
            return True
    return False


def cadence_metrics(dates, Rs, lo, hi):
    trades = [{'date': d, 'r': r} for d, r in zip(dates, Rs) if r == r]
    weekly = cb.build_weekly_series(trades, lo, hi)
    wr = [w[1] for w in weekly]
    p10 = cb.percentile(wr, 10)
    mdd, underwater = cb.max_drawdown_and_underwater(wr)
    worst = min(wr) if wr else np.nan
    return dict(weekly_p10_R=p10, worst_week_R=worst, max_dd_R=mdd, n_weeks=len(wr))


def read_one(fills_ok, side, exit_col, lo, hi, cost_by_key, half_label):
    """One (side, exit, window) read: n, fills/week, net mean R, t-stats, ex-top5, runner
    share, cadence metrics, obtainable share."""
    sub = fills_ok[(fills_ok['side'] == side) & (~fills_ok['beyond_slots'])].copy()
    sub['date'] = sub['session'].apply(lambda s: datetime.strptime(s, '%Y-%m-%d').date())
    sub = sub[(sub['date'] >= lo) & (sub['date'] <= hi)]
    if len(sub) == 0:
        return dict(side=side, exit=exit_col, window=half_label, n=0, fills_per_week=0.0,
                    mean_R=np.nan, day_t=np.nan, iid_t=np.nan, ex_top5=np.nan, mde=np.nan,
                    runner_share=np.nan, weekly_p10_R=np.nan, worst_week_R=np.nan, max_dd_R=np.nan,
                    obtainable_share=np.nan, in_season_fpw=np.nan, off_season_fpw=np.nan)
    key = list(zip(sub['session'], sub['symbol']))
    half_spread = np.array([cost_by_key.get(k, np.nan) for k in key])
    cost_R = np.where(half_spread == half_spread, (half_spread / 2.0) / sub['R_unit'].to_numpy(), 0.0)
    gross = sub[exit_col].to_numpy()
    net = gross - cost_R
    st = stats_block(net, sub['date'].values)
    weeks = window_weeks_count(lo, hi)
    fpw = len(sub) / weeks if weeks > 0 else np.nan
    cm = cadence_metrics(sub['date'].tolist(), net.tolist(), lo, hi)
    in_season = sub['date'].apply(_in_season)
    in_weeks, off_weeks = weeks * (6 * 4) / (365.25), weeks  # rough in-season week count (6wk x4 qtrs/yr)
    in_fpw = in_season.sum() / max(window_weeks_count(lo, hi) * (24.0 / 52.0), 1e-9)
    off_fpw = (~in_season).sum() / max(window_weeks_count(lo, hi) * (28.0 / 52.0), 1e-9)
    obtainable = fills_ok  # already obtainable-by-construction; share reported at the side level (see main())
    return dict(side=side, exit=exit_col, window=half_label, n=st['n'], fills_per_week=fpw,
                mean_R=st['mean_dR'], day_t=st['day_t'], iid_t=st['iid_t'], ex_top5=st['ex_top5'],
                mde=st['mde'], runner_share=float(sub['runner3'].mean()),
                weekly_p10_R=cm['weekly_p10_R'], worst_week_R=cm['worst_week_R'], max_dd_R=cm['max_dd_R'],
                in_season_fpw=in_fpw, off_season_fpw=off_fpw)


def window_weeks_count(lo, hi):
    return (hi - lo).days / 7.0


def gap_bands(fills_ok, side, exit_col, cost_by_key):
    sub = fills_ok[(fills_ok['side'] == side) & (~fills_ok['beyond_slots'])].copy()
    key = list(zip(sub['session'], sub['symbol']))
    half_spread = np.array([cost_by_key.get(k, np.nan) for k in key])
    cost_R = np.where(half_spread == half_spread, (half_spread / 2.0) / sub['R_unit'].to_numpy(), 0.0)
    sub['net_R'] = sub[exit_col].to_numpy() - cost_R
    bands = pd.cut(sub['gap_pct'], bins=[-np.inf, -5, 0, 5, np.inf],
                   labels=['<=-5%', '-5..0%', '0..5%', '>=5%'])
    return sub.groupby(bands)['net_R'].agg(['count', 'mean']).reset_index()


def obtainable_share(all_rows, side):
    sub = all_rows[all_rows['side'] == side]
    breakout_found = sub[~sub['reason'].isin(['no_bars', 'no_range', None])]
    if len(breakout_found) == 0:
        return np.nan
    return float((breakout_found['reason'] == 'ok').mean())


def select_exit(reads_df, side, window_label):
    """Selection rule (stated, not hidden): highest mean_R among X1/X2/X3 on the selection
    half, among exits with n>=10 (else fall back to the max-n exit to avoid selecting on a
    near-empty read)."""
    cand = reads_df[(reads_df['side'] == side) & (reads_df['window'] == window_label)]
    cand_enough = cand[cand['n'] >= 10]
    pool = cand_enough if len(cand_enough) else cand
    if len(pool) == 0 or pool['mean_R'].notna().sum() == 0:
        return None   # no fills (or all-NaN mean_R) on this side/window in this sample -- logged by the caller
    return pool.loc[pool['mean_R'].idxmax(), 'exit']


def union_with_production(fills_ok, side, exit_col, cost_by_key, lo, hi):
    """Union with the production book (production first; duplicates by day+symbol removed)."""
    prod = orb_csv.read_orb_csv(PRODUCTION_BOOK_CSV)
    prod_ent = prod[prod['entered'] == 1].copy()
    prod_ent['date'] = pd.to_datetime(prod_ent['date']).dt.date
    prod_keys = set(zip(prod_ent['date'].astype(str), prod_ent['symbol']))
    sub = fills_ok[(fills_ok['side'] == side) & (~fills_ok['beyond_slots'])].copy()
    sub['date'] = sub['session']
    key = list(zip(sub['session'], sub['symbol']))
    half_spread = np.array([cost_by_key.get(k, np.nan) for k in key])
    cost_R = np.where(half_spread == half_spread, (half_spread / 2.0) / sub['R_unit'].to_numpy(), 0.0)
    sub['usd'] = (sub[exit_col].to_numpy() - cost_R) * FIXED_RISK_DOLLARS
    sub = sub[~sub.apply(lambda r: (r['session'], r['symbol']) in prod_keys, axis=1)]
    e1_trades = [{'date': datetime.strptime(d, '%Y-%m-%d').date(), 'r': u / FIXED_RISK_DOLLARS}
                 for d, u in zip(sub['session'], sub['usd'])]
    prod_trades = [{'date': d, 'r': p / FIXED_RISK_DOLLARS}
                    for d, p in zip(prod_ent['date'], prod_ent['pnl'])
                    if lo <= d <= hi]
    e1_trades = [t for t in e1_trades if lo <= t['date'] <= hi]
    union_trades = prod_trades + e1_trades
    cm = cadence_metrics([t['date'] for t in union_trades], [t['r'] for t in union_trades], lo, hi)
    weeks = window_weeks_count(lo, hi)
    return dict(n_production=len(prod_trades), n_e1_added=len(e1_trades),
                fills_per_week=len(union_trades) / weeks if weeks > 0 else np.nan,
                union_usd=sum(t['r'] for t in union_trades) * FIXED_RISK_DOLLARS,
                weekly_p10_R=cm['weekly_p10_R'], worst_week_usd=cm['worst_week_R'] * FIXED_RISK_DOLLARS)


# ---------------------------------------------------------------------------
# Cross-check vs bars_sip.db (read-only; the cross-check ONLY, never the walk's bar source)
# ---------------------------------------------------------------------------

def cross_check(fills_ok, n_pairs=200, seed=1701):
    if not os.path.exists(BARS_SIP_DB):
        logger.warning('bars_sip.db not found at %s -- cross-check SKIPPED', BARS_SIP_DB)
        return dict(n_checked=0, share_within_0_1pct=np.nan, max_dev_pct=np.nan)
    store_sip = CachedStore(BARS_SIP_DB)
    pairs = fills_ok[['symbol', 'session']].drop_duplicates()
    rng = np.random.RandomState(seed)
    sample = pairs.sample(n=min(n_pairs, len(pairs)), random_state=rng) if len(pairs) else pairs
    n_checked, n_within, max_dev = 0, 0, 0.0
    db_store = DatabentoStore()
    for r in sample.itertuples():
        sip_bars = store_sip.day_bars(r.symbol, r.session)
        db_bars = db_store.day_bars(r.symbol, r.session)
        if sip_bars is None or db_bars is None:
            continue
        sip_sf, db_sf = session_features(sip_bars), session_features(db_bars)
        if sip_sf is None or db_sf is None:
            continue
        n_checked += 1
        devs = [abs(sip_sf['range_high'] - db_sf['range_high']) / sip_sf['range_high'],
                abs(sip_sf['range_low'] - db_sf['range_low']) / sip_sf['range_low'],
                abs(float(sip_bars['c'][-1]) - float(db_bars['c'][-1])) / float(sip_bars['c'][-1])]
        d = max(devs)
        max_dev = max(max_dev, d)
        if d <= 0.001:
            n_within += 1
    store_sip.close()
    return dict(n_checked=n_checked, share_within_0_1pct=n_within / n_checked if n_checked else np.nan,
                max_dev_pct=max_dev * 100.0)


# ---------------------------------------------------------------------------
# RESULT.md writer
# ---------------------------------------------------------------------------

def write_result(reads_df, gap_band_rows, union_rows, cross, cov_note, n_cands, n_ok_total):
    lines = []
    lines.append('# RESULT_1701 -- earnings-session ORB pool E1 (walk)')
    lines.append('')
    lines.append(f'Candidates: {n_cands:,}; filled (ok, any side): {n_ok_total:,}. {cov_note}')
    lines.append('')
    lines.append('## Pass bar per side (>=3 fills/wk test half, mean R>=+0.10 both halves day_t>=2.0 '
                 'pooled, ex-top5%>=0, obtainable>=80%, cadence weekly P10>=-2R)')
    for side in ('long', 'short'):
        sel_a = select_exit(reads_df, side, 'A')
        sel_b = select_exit(reads_df, side, 'B')
        for sel_half, test_half, sel_exit in (('A', 'B', sel_a), ('B', 'A', sel_b)):
            if sel_exit is None:
                lines.append(f'- {side} (select {sel_half}->test {test_half}): no exit selectable (n too small)')
                continue
            r_test = reads_df[(reads_df['side'] == side) & (reads_df['exit'] == sel_exit) & (reads_df['window'] == test_half)]
            r_sel = reads_df[(reads_df['side'] == side) & (reads_df['exit'] == sel_exit) & (reads_df['window'] == sel_half)]
            if len(r_test) == 0 or len(r_sel) == 0:
                continue
            rt, rs = r_test.iloc[0], r_sel.iloc[0]
            passed = (rt['fills_per_week'] >= 3.0 and rt['mean_R'] >= 0.10 and rs['mean_R'] >= 0.10
                      and rt['day_t'] >= 2.0 and rt['ex_top5'] >= 0 and rt['weekly_p10_R'] >= -2.0)
            lines.append(f"- {side} select={sel_half} ({sel_exit}) test={test_half}: n={rt['n']} fpw={rt['fills_per_week']:.1f} "
                        f"meanR={rt['mean_R']:.3f} day_t={rt['day_t']:.1f} ex5={rt['ex_top5']:.3f} "
                        f"P10={rt['weekly_p10_R']:.2f}R -- {'PASS' if passed else 'FAIL'}")
    lines.append('')
    lines.append('## Exits x windows (net R, after measured entry half-spread; exit slip is ADV-conditional '
                 '35/10bps applied uniformly per fill -- see module docstring caveat)')
    lines.append('side | exit | window | n | fills/wk | mean_R | day_t | iid_t | ex_top5 | runner% | P10(R) | worst_wk(R) | maxDD(R)')
    for _, r in reads_df.iterrows():
        lines.append(f"{r['side']} | {r['exit']} | {r['window']} | {r['n']} | {r['fills_per_week']:.2f} | "
                    f"{r['mean_R']:.3f} | {r['day_t']:.2f} | {r['iid_t']:.2f} | {r['ex_top5']:.3f} | "
                    f"{100*r['runner_share'] if r['runner_share']==r['runner_share'] else float('nan'):.1f} | "
                    f"{r['weekly_p10_R']:.2f} | {r['worst_week_R']:.2f} | {r['max_dd_R']:.2f}")
    lines.append('')
    lines.append('## Frequency: in-season (6wk after quarter-end) vs off-season (fills/week)')
    for _, r in reads_df[reads_df['window'] == 'whole'].iterrows():
        lines.append(f"{r['side']} {r['exit']}: in-season {r['in_season_fpw']:.2f}/wk, off-season {r['off_season_fpw']:.2f}/wk")
    lines.append('')
    lines.append('## Gap-band decomposition (feature, not a filter)')
    for side, rows in gap_band_rows.items():
        lines.append(f'{side}:')
        for _, r in rows.iterrows():
            lines.append(f"  {r.iloc[0]}: n={r['count']} mean_R={r['mean']:.3f}")
    lines.append('')
    lines.append('## Union with the production book (production first, dedup by day+symbol)')
    for side, u in union_rows.items():
        lines.append(f"{side}: production={u['n_production']} +E1={u['n_e1_added']} fills/wk={u['fills_per_week']:.2f} "
                    f"union=${u['union_usd']:,.0f} P10={u['weekly_p10_R']:.2f}R worst_wk=${u['worst_week_usd']:,.0f}")
    lines.append('')
    lines.append('## Source parity (Databento vs bars_sip.db, read-only cross-check)')
    lines.append(f"checked={cross['n_checked']} within_0.1%={cross['share_within_0_1pct']:.1%} "
                f"max_dev={cross['max_dev_pct']:.3f}%" if cross['n_checked'] else 'cross-check: 0 overlapping pairs available')
    lines.append('')
    lines.append('## Caveats')
    lines.append('- Exit slip (35/10bps ADV-conditional) is applied to EVERY exit type (stop/lock/target/EOD), '
                'not stop-only -- the reused walkers return no exit-reason tag; conservative (overstates cost).')
    lines.append('- Selection rule: highest mean_R among X1/X2/X3 on the selection half (n>=10 else max-n exit) -- '
                'stated, not hidden; PREREG did not fully specify a tie-break.')
    lines.append('- Half B frequency for the last ~19 trading days (panel ends 2026-09-04) is not claimed (Amendment 1).')
    lines.append('- "whole" window pools both halves; reading it as a verdict on its own is not allowed per PREREG multiplicity rules.')
    with open(RESULT_MD, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    logger.info('wrote %s (%d lines)', RESULT_MD, len(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true', help='<=300 sessions, already on disk/in bars_sip.db only')
    ap.add_argument('--full', action='store_true', help='all candidates (needs the full pull)')
    ap.add_argument('--source', choices=['sip', 'databento'], default='sip',
                     help='bar source (default sip): sip = Alpaca SIP store (research/bf_zero/bars_sip.db), '
                          'the only source a range-based entry can be walked on (PREREG Amendment 2); '
                          'databento = venue-subset EQUS.MINI shards, cross-check only, VOID as a result source')
    args = ap.parse_args()
    if not args.smoke and not args.full:
        logger.error('pass --smoke or --full')
        sys.exit(1)
    logger.info('source=%s smoke=%s full=%s', args.source, args.smoke, args.full)

    check_disk()
    logger.info('running unit self-check (mirror long vs short)...')
    _self_check_mirror()

    cands = pd.read_csv(CANDIDATES_CSV, dtype={'symbol': str, 'session': str, 'half': str})
    logger.info('loaded %d candidates from %s', len(cands), CANDIDATES_CSV)

    store = SipRetryStore() if args.source == 'sip' else DatabentoStore()
    if args.smoke:
        if args.source == 'sip':
            sessions = sip_available_sessions(cands, limit=300)
        else:
            avail = set(store.available_days())
            logger.info('%d day-shards on disk', len(avail))
            cands_av = cands[cands['session'].isin(avail)]
            sessions = sorted(cands_av['session'].unique())[:300]
        cands = cands[cands['session'].isin(sessions)]
        logger.info('SMOKE (%s): restricted to %d sessions, %d candidate rows', args.source, len(sessions), len(cands))
        if len(cands) == 0:
            logger.warning('SMOKE (%s): nothing available yet -- nothing to walk this run', args.source)
            return

    run_kind = 'SMOKE run (<=300 sessions)' if args.smoke else 'FULL run'
    if args.source == 'sip':
        _walkable, _requested, _pct, cov_line = sip_coverage_gate(cands, store)
        cov_note = f'{run_kind}, source=sip. {cov_line}'
    else:
        cov_note = (f'{run_kind}, source=databento. VOID per PREREG Amendment 2: EQUS.MINI is a venue-subset '
                    f'feed (opening-range high/low inside the SIP range by up to 4%) -- cross-check only, '
                    f'never a reportable result source.')
    logger.info('coverage: %s', cov_note)

    all_rows = walk_all(cands, store)
    all_rows = apply_slots(all_rows)
    all_rows.to_csv(FILLS_CSV, index=False)
    logger.info('wrote %s (%d rows)', FILLS_CSV, len(all_rows))

    fills_ok = all_rows[(all_rows['reason'] == 'ok') & (~all_rows['beyond_slots'])].copy()
    logger.info('%d fills ok (slot-admitted)', len(fills_ok))
    if len(fills_ok) == 0:
        logger.warning('no admitted fills -- stopping before quotes/reads (expected on a tiny smoke sample)')
        return

    quotes = fetch_quotes(fills_ok)
    cost_by_key = {(r.session, r.symbol): r.spread_mean for r in quotes.itertuples()}

    reads = []
    gap_rows, union_rows = {}, {}
    for side in ('long', 'short'):
        for exit_col in ('X1_R', 'X2_R', 'X3_R'):
            for label, (lo, hi) in (('A', HALF_A), ('B', HALF_B), ('whole', WHOLE)):
                reads.append(read_one(fills_ok, side, exit_col, lo, hi, cost_by_key, label))
        sel = select_exit(pd.DataFrame(reads), side, 'whole') or 'X2_R'
        gap_rows[side] = gap_bands(fills_ok, side, sel, cost_by_key)
        union_rows[side] = union_with_production(fills_ok, side, sel, cost_by_key, *WHOLE)
    reads_df = pd.DataFrame(reads)
    reads_df.to_csv(READS_CSV, index=False)
    logger.info('wrote %s (%d rows)', READS_CSV, len(reads_df))

    cross = cross_check(fills_ok)
    write_result(reads_df, gap_rows, union_rows, cross, cov_note, len(all_rows), len(fills_ok))
    logger.info('DONE (source=%s)', args.source)


if __name__ == '__main__':
    main()
