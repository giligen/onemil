#!/usr/bin/env python3
"""STAGE B pass 1 — candidates4.csv: one row per SIGNAL (not per fill), both fill models, the level's pre-signal
history, MAE/MFE, the H1 stop variants and the H9 second-chance families.

=====================================================================================================================
THE SCORER CONTRACT FOR STAGE C (score5.py).  Anything that scores this file MUST use exactly this:

  POPULATION (PLAN.md §1 gates, applied at SCORING, never here):
    entry >= $5 (the FILL price of the model being scored, not `price`), entry_m <= 841 (14:01), r_pct >= 1.0 of the
    variant being scored, and for F5-F14 `range_so_far_pct >= 5` (bars STRICTLY BEFORE the signal bar; this file
    emits the column and applies NO range floor).  Splits: TRAIN 2025-01-02..2025-12-31, VAL 2026-01-01..2026-05-31,
    TEST 2026-06-01..2026-09-11 — TEST IS READ ONCE, after the stage's selection is frozen in writing.

  COST, corrected per H7/A0 (probe_costs.md; this REPLACES score4.py's contract):
    half_cc = 0.5 * (spread_cc_bps / 100) / max(r_pct, 0.05)            # half the cost-curve spread, in R units
    entry leg :  0.25 * half_cc  for the NEXT-OPEN fill  (the fill is already a printed, ask-side price:
                                                          bf_zero/REPORT.md §8 puts the signal minute's last ask
                                                          +5.9 bps ABOVE the next open, and probe_costs.md measures
                                                          our live entries at a median 0.0 bps vs the contemporaneous
                                                          quote — 0.25x is the conservative quartile, not the median)
                 1.00 * half_cc  for the RESTING fill    (an arrival execution: probe_costs.md, macd_wave ratio 1.00x)
    exit leg  :  stop 0.875 * half_cc · eod 0.412 * half_cc · target 0.875 * half_cc, or 0.0 for a target ONLY when
                 the engine rests a take-profit leg (the HOD bracket does; ORB's lock-stops cross like stops).
                 Report both target conventions; never silently take the free one.
    `spread_pct` (the 1.90/1.20/0.80/0.60/0.50 price-band table) is kept ONLY for backward comparison with score4.
    It is 3.8x too wide for the names we trade — do not score with it except to reproduce the old number.
    Separately report the live liquidity gate as a row: quoted spread <= 15% of R.

  BOOK: trading.hod_break.run_book(rows, 12, 4) — 12 a day, 4 concurrent, first-come, alphabetical tie-break,
        causal slot freeing.  rows = (day, entry_m, exit_m, symbol, net, wk).

  GATE (PLAN.md §1 H10): G1 TRAIN mean net R > 0, t >= 2.0, >= 5 trades/week.  G2 VAL mean net R > 0, t >= 1.0,
        >= 55% weeks green, bar raised by 1 SE of weekly R per 10 cells that passed G1.  G3 TEST read once.
        Every G2 survivor also gets: permutation search-adjusted p over ALL cells of this stage, tail removal
        (top 1% / top 5%), winner cap at +3R, and a per-month table.  Report the cell count.
=====================================================================================================================

WHAT IS IN A ROW.  One row per (day, symbol, fam, cfg) SIGNAL.  A row exists whenever the family signalled; the
fill columns of a model that does not fill are EMPTY.  This is the fix for the defect probe_stops.md §3 found:
candidates3.csv only ever contained signals whose next-bar open came back under the cap, so it is conditioned to
hold cheap fills and cannot answer the resting-fill question at all.

  `price` IS THE LEVEL, not a fill.  candidates3.csv's `price` was the next-open fill; use `entry_next` for that.
  Every context feature (dist_open_pct, vwap_dist_pct, spread bands, mae_pct denominator) is therefore
  fill-independent and known at the signal bar.

TWO FILLS
  entry_next = o[i+1]                 iff  <= level*1.006   (the shipped HOD engine: it reacts after the bar closes)
  entry_rest = max(o[i], level)       iff  <= level*1.006   (an ORB-style resting stop-limit at the level:
                                                             trading/orb_engine.py submit_stop_limit_bracket)
  entry_rest is obtainable inside the signal bar by construction (the bar's high reached the level); `rest_obtain`
  records the check anyway, and `rest_queue_ok` = sig_v >= 5 * shares at $100 risk.
  F11/F12/F13 are CLOSE-triggered families (the decision needs the bar's close), so a resting order cannot express
  them: entry_rest is empty for those three by construction, not by failure.

EXITS, per fill model.  Exits start the bar AFTER the fill bar — for entry_rest the fill bar IS the signal bar.
  rr_2r / rr_hold          build_candidates3.walk verbatim: eod (m>=955, fill at that bar's open) beats stop
                           (l<=stop, fill min(stop,open)*0.999) beats target (a bar CLOSE >= entry+2R, fill AT it).
  rr_2r_closestop          H1 S1: the stop fires on a bar CLOSE at/below the stop level, filled at the NEXT bar's
                           open * 0.999 (a market order after the close).  Target/eod unchanged.
  rr_2r_stopm1             H1 S4: stop = structural stop - 1% of price, R REDEFINED (r_pct_m1); rr is in the NEW R,
                           so at a constant $100 of risk pnl100 = 100*rr — that is the constant-dollar-risk
                           comparison probe_stops.md §6 says is the only lever with the right order of magnitude.
  rr_lock                  ORB static lock, hold to close, no target: a CLOSED bar's high reaching +1.75R arms the
                           lock from the FOLLOWING bar and moves the stop to +0.5R.  (Live ORB arms intrabar; this
                           is conservative.)
  mae_pct / mfe_r          over the 2R walk's own holding window, [fill bar+1 .. exit bar].

FAMILIES (14 configs, the cell list Stage C will score — pre-registered in B/REPORT.md):
  F1 {"P":0.12}  F5 {"K":5,"X":0.04} (reference only — gross-negative at zero cost, probe_stops.md §6)
  F6 {}          F8 {"N":5|15|30}    F9 {"G":0.05}   F10 {}
  F11 {"base":"F8","N":15} / {"base":"F6"}   close confirmation: the break bar must CLOSE above the level
  F12 {"base":"F8","N":15} / {"base":"F6"}   retest: after the break, a bar's low comes within 0.3% of the level
                                             within 30 min and the NEXT bar's low holds at/above it; that hold bar
                                             is the signal; stop = the retest bar's low
  F13 {"K":5,"X":0.04}                       sweep-and-reclaim: the F5 consolidation low is pierced by <= 1%, then a
                                             bar closes back above it within 5 bars; stop = the sweep low
  F14 {"N":15}                               second break: the first F8-15 break filled and STOPPED OUT under the 2R
                                             exit; the next break of the same level that day; stop = the lowest low
                                             from the stop bar through the signal bar

UNIVERSE / CAUSALITY unchanged from build_candidates3.py: research/bf_zero/universe.csv (the point-in-time >=5%-range
day list), bars from data/cache.db then research/bf_zero/bars_sip.db (BFZ_SIP_STORE), RTH only, >= 10 bars, every
field computed on bars at or before the signal bar.  The universe is NOT causal (it is an end-of-day range list) —
that is why the scorer must apply range_so_far_pct >= 5.  H6 (a causal universe) is Stage E.

RUN:  setsid nohup bash -c "cd /home/ec2-user/onemil; ulimit -v 3500000; nice -n 10 \
        python3 research/fuckup_audit/B/build_candidates4.py > research/fuckup_audit/B/build4.log 2>&1; \
        echo EXIT=\$? >> research/fuckup_audit/B/build4.log" >/dev/null 2>&1 </dev/null &
Resumable per day via build4_state.json; appends with a PINNED column list (probe_stops.md: appending dicts with
conditional keys silently mis-aligns the CSV).  BFZ_DAYS=N limits to the first N days (smoke test).
"""
import gc, json, os, sys, time
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
os.environ.setdefault('BFZ_SIP_STORE', f'{ROOT}/research/bf_zero/bars_sip.db')
os.environ['BFZ_SLIP'] = '0.0'                      # the level TOUCH is the signal; the fill is handled here
sys.path.insert(0, f'{ROOT}/research/bf_zero')
sys.path.insert(0, ROOT)

# build_candidates.py loads the whole 5M-row Databento daily panel at import and keeps it alive; that alone needs
# >3.5 GB of the node's 7.8 (build_candidates3 ran at ulimit 5.5 GB, which this node can no longer spare with a live
# trader on it). Read only the columns it actually uses, then drop the panel once the merges it feeds are done.
# Nothing about the family functions, load_bars() or the universe merge changes — parity is unaffected.
_orig_read_parquet = pd.read_parquet
_UNI_SYMS = sorted(set(pd.read_csv(f'{ROOT}/research/bf_zero/universe.csv', usecols=['symbol'], dtype=str,
                                   keep_default_na=False).symbol) | {'SPY'})


def _slim_read_parquet(path, *a, **k):
    """Only the five columns build_candidates reads, and only the symbols the universe (plus SPY) can ever ask for:
    5.05M x 8 -> 2.98M x 5. prev_close/prev_high/prev_low/high20 are per-symbol quantities, so dropping symbols the
    universe never contains cannot change a single value that is used."""
    if 'equs_daily' in str(path) and 'columns' not in k:
        import pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc
        t = pq.read_table(path, columns=['symbol', 'bar_date', 'high', 'low', 'close'])
        t = t.filter(pc.is_in(t.column('symbol'), value_set=pa.array(_UNI_SYMS)))
        return t.to_pandas()
    return _orig_read_parquet(path, *a, **k)


pd.read_parquet = _slim_read_parquet
import build_candidates as B                        # family functions, load_bars(), the universe+daily merge
pd.read_parquet = _orig_read_parquet
B.daily = B.dkey = B.spy = None                     # only B.uni (merged), B.load_bars and the fam_* functions are used
gc.collect()
from trading.orb_asset_class import classify_asset, load_class_map

D = 'research/fuckup_audit/B'
TAG = os.environ.get('B4_TAG', '')                  # smoke tests write to their own files / state
STATE = f'{D}/build4_state{TAG}.json'
OUT = f'{D}/candidates4{TAG}.csv'
MISS = f'{D}/coverage4_missing{TAG}.csv'
CAP = 0.006
OPEN_M, EOD_M = B.OPEN_M, B.EOD_M
LOCK_ARM_R, LOCK_STOP_R = 1.75, 0.5
STOPM1_PCT = 0.01                                   # H1 S4: stop = structural stop - 1% of price
RISK_USD = 100.0
SPREAD = [(10, 1.90), (20, 1.20), (50, 0.80), (100, 0.60), (1e18, 0.50)]   # score4's band table, kept for comparison
CC_CSV = 'research/lit_review_2026/cost_curve.csv'

FAMS = ([('F1', dict(P=0.12)), ('F5', dict(K=5, X=0.04)), ('F6', {})] +
        [('F8', dict(N=N)) for N in (5, 15, 30)] +
        [('F9', dict(G=0.05)), ('F10', {})] +
        [('F11', dict(base='F8', N=15)), ('F11', dict(base='F6'))] +
        [('F12', dict(base='F8', N=15)), ('F12', dict(base='F6'))] +
        [('F13', dict(K=5, X=0.04)), ('F14', dict(N=15))])
CLOSE_TRIGGERED = {'F11', 'F12', 'F13'}             # no resting order can express these

_FILL_KEYS = ['entry', 'entry_m', 'r_pct', 'rr_2r', 'why_2r', 'exit_m_2r', 'rr_hold', 'why_hold', 'exit_m_hold',
              'mae_pct', 'mfe_r', 'rr_2r_closestop', 'why_2r_closestop', 'exit_m_2r_closestop',
              'rr_2r_stopm1', 'why_2r_stopm1', 'exit_m_2r_stopm1', 'r_pct_m1', 'pnl100_2r_stopm1',
              'rr_lock', 'why_lock', 'exit_m_lock']
COLS = (['day', 'symbol', 'fam', 'cfg', 'sig_m', 'minutes_since_open', 'level', 'stop', 'price',
         'dist_open_pct', 'range_so_far_pct', 'rv_adv', 'gap_pct', 'adv20', 'spread_pct', 'spread_cc_bps',
         'sig_o', 'sig_h', 'sig_l', 'sig_c', 'sig_v',
         'n_touches', 'consol_bars', 'consol_vol_ratio', 'cum_dollar_vol', 'vwap_dist_pct', 'close_confirm',
         'pm_dollar_vol', 'prev_day_range_pct', 'prev_close', 'asset_class',
         'rest_obtain', 'rest_queue_ok'] +
        [f'{t}_{k}' for t in ('next', 'rest') for k in _FILL_KEYS])


# ------------------------------------------------------------------ cost inputs
def spread_pct(p):
    """score4's per-price-band quoted spread, % of price (kept only for backward comparison)."""
    for hi, v in SPREAD:
        if p < hi:
            return v
    return SPREAD[-1][1]


def _load_cost_curve():
    """median NBBO spread in bps of price per (price band x time-of-day band), measured on THIS population
    (research/lit_review_2026/build_cost_curve.py, 2,570 stratified SIP quote samples)."""
    d = pd.read_csv(CC_CSV, dtype={'symbol': str, 'day': str}, keep_default_na=False, na_values=[''])
    d = d[(d.n_q > 0) & d.spread.notna() & (d.price > 0)].copy()
    d['bps'] = d.spread / d.price * 1e4
    return {k: float(v) for k, v in d.groupby(['pb', 'hb']).bps.median().items()}


CC = _load_cost_curve()


def cc_bps(p, m):
    """cost-curve band lookup. Sub-$5 uses the $5-10 row (no sub-$5 quotes were ever sampled) — the scorer's
    price >= 5 gate makes that irrelevant for any scored cell, and it is flagged here so it is never forgotten."""
    pb = '$5-10' if p <= 10 else '$10-20' if p <= 20 else '$20-50' if p <= 50 else '$50-200' if p <= 200 else '$200+'
    hb = ('09:30-09:35' if m <= 575 else '09:35-10:00' if m <= 600 else '10:00-11:00' if m <= 660
          else '11:00-13:00' if m <= 780 else '13:00+')
    return CC.get((pb, hb), np.nan)


CLASS_MAP = load_class_map()


def asset_class(sym):
    """'stock' | 'wrapper' | 'unknown' from the 2026-07-11 offline dump + the leveraged-family sets. A symbol-level
    attribute, NOT point-in-time — a name that became a wrapper later is tagged wrapper for the whole window."""
    return CLASS_MAP.get(sym) or classify_asset(sym, None)


# ------------------------------------------------------------------ exit walks (vectorised, parity with build_candidates3.walk)
def _first(mask, k0):
    """first index >= k0 where mask is True, else None."""
    n = len(mask)
    if k0 >= n:
        return None
    sub = mask[k0:]
    i = int(np.argmax(sub))
    return k0 + i if bool(sub[i]) else None


def walk_2r(o, h, l, c, m, k0, stop, target, eod_idx):
    """build_candidates3.walk, verbatim semantics: for k from k0, eod (m>=EOD_M) beats stop (l<=stop) beats target
    (c>=target) within a bar; the empty range falls back to the last bar's close as 'eod'.
    Returns (px, why, exit_m, exit_idx)."""
    if k0 >= len(o):
        return float(c[-1]), 'eod', int(m[-1]), len(o) - 1
    cand = []
    if eod_idx is not None:
        # m is sorted, so the first bar at/after 15:55 in [k0, n) is max(eod_idx, k0) — a signal that is itself past
        # 15:55 exits on its own first walked bar, exactly as build_candidates3's loop does.
        cand.append((max(eod_idx, k0), 0, 'eod'))
    s = _first(l <= stop, k0)
    if s is not None:
        cand.append((s, 1, 'stop'))
    if target is not None:
        t = _first(c >= target, k0)
        if t is not None:
            cand.append((t, 2, 'target'))
    if not cand:
        return float(c[-1]), 'eod', int(m[-1]), len(o) - 1
    k, _, why = min(cand)
    px = float(o[k]) if why == 'eod' else (float(min(stop, o[k]) * 0.999) if why == 'stop' else float(target))
    return px, why, int(m[k]), k


def walk_closestop(o, h, l, c, m, k0, stop, target, eod_idx):
    """H1 S1: the stop is a bar CLOSE at/below the stop level; the engine can only act after that close, so the
    fill is the NEXT bar's open * 0.999. Target/eod unchanged."""
    if k0 >= len(o):
        return float(c[-1]), 'eod', int(m[-1]), len(o) - 1
    cand = []
    if eod_idx is not None:
        cand.append((max(eod_idx, k0), 0, 'eod'))
    cs = _first(c <= stop, k0)
    s_eff = cs + 1 if (cs is not None and cs + 1 < len(o)) else None
    if s_eff is not None:
        cand.append((s_eff, 1, 'stop'))
    if target is not None:
        t = _first(c >= target, k0)
        if t is not None:
            cand.append((t, 2, 'target'))
    if not cand:
        return float(c[-1]), 'eod', int(m[-1]), len(o) - 1
    k, _, why = min(cand)
    px = float(o[k]) if why == 'eod' else (float(o[k] * 0.999) if why == 'stop' else float(target))
    return px, why, int(m[k]), k


def walk_lock(o, h, l, c, m, k0, entry, stop, R, eod_idx):
    """ORB static lock, hold to close, no target: a CLOSED bar's high at/above entry+1.75R arms the lock effective
    from the FOLLOWING bar, moving the stop to entry+0.5R. The base stop is checked before arming within a bar."""
    px, why, xm, k1 = walk_2r(o, h, l, c, m, k0, stop, None, eod_idx)
    arm = _first(h >= entry + LOCK_ARM_R * R, k0)
    if arm is None or arm >= k1:
        return px, why, xm, k1
    lock = entry + LOCK_STOP_R * R
    px2, why2, xm2, k2 = walk_2r(o, h, l, c, m, arm + 1, lock, None, eod_idx)
    return px2, ('lock' if why2 == 'stop' else why2), xm2, k2


# ------------------------------------------------------------------ families
def fam_vwap_reclaim(h, l, c, vwap, min_below=10):
    """F10 (build_candidates2.py verbatim, SLIP=0): >= min_below consecutive closes below VWAP, then a close back
    above it; entry level = that bar's high; stop = the lowest low of the below-VWAP stretch.
    Returns (i, level, stop, lvl_idx)."""
    n = len(c)
    below = c < vwap
    run = np.zeros(n, int)
    for i in range(1, n):
        run[i] = run[i - 1] + 1 if below[i] else 0
    t = np.arange(n)
    j = np.clip(t - 1, 0, n - 1)
    k = np.clip(t - 2, 0, n - 1)
    ok = (t >= 3) & (~below[j]) & (run[k] >= min_below) & (h >= h[j])
    i = B.first_true(ok)
    if i is None:
        return None
    start = max(i - 1 - run[i - 2], 0)
    stop = float(l[start:i].min())
    level = float(h[i - 1])
    if stop >= level:
        return None
    return i, level, stop, start


def base_level(fam_base, N, h, l, prev_close, o0):
    """(level, stop, start_idx, lvl_idx) of the base family F11/F12/F14 confirm or retest. None when unavailable."""
    if fam_base == 'F8':
        if len(h) <= N:
            return None
        return float(h[:N].max()), float(l[:N].min()), N, N - 1
    if not (prev_close and prev_close == prev_close and o0 < prev_close):
        return None
    return float(prev_close), None, 1, 0            # F6's stop is the running low before the signal


def fam_close_confirm(h, l, c, level, stop, start_idx):
    """F11: the first bar at/after start_idx whose CLOSE is at/above the level (its high necessarily reached it)."""
    n = len(c)
    t = np.arange(n)
    i = B.first_true((t >= start_idx) & (c >= level) & (h >= level))
    if i is None:
        return None
    s = stop if stop is not None else float(l[:i].min())
    if s >= level:
        return None
    return i, level, s


def fam_retest(h, l, level, stop, start_idx, window=30, band=0.003):
    """F12: the base break bar b (first high >= level); then a bar d in (b, b+window] whose LOW comes within `band`
    of the level; the NEXT bar's low must hold at/above the level. The HOLD bar is the signal (the decision needs
    its close), stop = the retest bar's low. Returns (i, level, stop, lvl_idx=b)."""
    n = len(h)
    t = np.arange(n)
    b = B.first_true((t >= start_idx) & (h >= level))
    if b is None:
        return None
    lo = level * (1 - band)
    hi = level * (1 + band)
    for d in range(b + 1, min(b + window, n - 2) + 1):
        if lo <= l[d] <= hi and l[d + 1] >= level:
            s = float(l[d])
            if s >= level:
                continue
            return d + 1, float(level), s, b
    return None


def fam_sweep_reclaim(h, l, c, K, X, reclaim_bars=5, max_pierce=0.01):
    """F13: the F5 K/X consolidation low (K bars all within X of the running high) is pierced by <= max_pierce, then
    a bar CLOSES back above it within reclaim_bars. Signal = the reclaim bar; stop = the sweep low; level = the
    consolidation low (the reclaimed level). Returns (i, level, stop, lvl_idx=j)."""
    n = len(h)
    hod = np.maximum.accumulate(h)
    lo = pd.Series(l).rolling(K, min_periods=K).min().values
    t = np.arange(n)
    j = np.clip(t - 1, 0, n - 1)
    # the sweep bar p: the F5 consolidation was complete at p-1 (fam_hod's own condition) and p's low pierces its low
    cand = np.flatnonzero((t > K) & (lo[j] >= hod[j] * (1 - X)) & (lo[j] < hod[j])
                          & (l < lo[j]) & (l >= lo[j] * (1 - max_pierce)))
    for p in cand:
        p = int(p)
        cl = float(lo[p - 1])
        for q in range(p + 1, min(p + reclaim_bars, n - 1) + 1):
            if c[q] >= cl:
                s = float(l[p:q + 1].min())
                if s < cl:
                    return q, cl, s, p - 1
                break
    return None


def fam_second_break(o, h, l, c, m, N, eod_idx):
    """F14: the first F8-N break FILLED at the next open under the cap and its 2R walk ended in a STOP; the next
    break of the same level that day is the signal; stop = the lowest low from the stop bar through the signal bar."""
    if len(h) <= N:
        return None
    level = float(h[:N].max())
    stop1 = float(l[:N].min())
    t = np.arange(len(h))
    b = B.first_true((t >= N) & (h >= level))
    if b is None or b + 1 >= len(o) or stop1 >= level:
        return None
    e1 = float(o[b + 1])
    if e1 > level * (1 + CAP) or stop1 >= e1:
        return None
    _, why, _, kx = walk_2r(o, h, l, c, m, b + 2, stop1, e1 + 2 * (e1 - stop1), eod_idx)
    if why != 'stop':
        return None
    i = B.first_true((t > kx) & (h >= level))
    if i is None:
        return None
    s = float(l[kx:i + 1].min())
    if s >= level:
        return None
    return int(i), level, s, int(kx)


def detect(fam, cfg, arrays):
    """(i, level, stop, lvl_idx) for one family-config, or None. lvl_idx = the bar the level/consolidation was set."""
    o, h, l, c, v, m, o0, vwap, prev_close, pmh, gap, eod_idx = arrays
    if fam == 'F1':
        res = B.fam_flag(h, l, cfg['P'], (2, 3, 4, 5, 6), micro=False)
        if res is None:
            return None
        i, level, stop, extra = res
        return i, level, stop, max(i - 1 - int(extra['flag_len']), 0)
    if fam == 'F5':
        res = B.fam_hod(h, l, cfg['K'], cfg['X'])
        if res is None:
            return None
        i, level, stop, _ = res
        w = np.flatnonzero(h[:i] >= level - 1e-12)
        return i, level, stop, int(w[-1]) if len(w) else max(i - cfg['K'], 0)
    if fam == 'F6':
        res = B.fam_r2g(h, l, o0, prev_close if prev_close == prev_close else None)
        if res is None:
            return None
        i, level, stop, _ = res
        return i, level, stop, 0
    if fam == 'F8':
        N = cfg['N']
        res = B.fam_level(h, l, float(h[:N].max()), N, float(l[:N].min())) if len(h) > N else None
        if res is None:
            return None
        i, level, stop, _ = res
        return i, level, stop, N - 1
    if fam == 'F9':
        if not (len(h) > 5 and gap == gap and gap >= cfg['G']):
            return None
        res = B.fam_level(h, l, float(h[:5].max()), 5, float(l[:5].min()))
        if res is None:
            return None
        i, level, stop, _ = res
        return i, level, stop, 4
    if fam == 'F10':
        return fam_vwap_reclaim(h, l, c, vwap)
    if fam in ('F11', 'F12'):
        bl = base_level(cfg['base'], cfg.get('N', 0), h, l, prev_close, o0)
        if bl is None:
            return None
        level, stop, start_idx, lvl_idx = bl
        if stop is not None and stop >= level:
            return None
        if fam == 'F11':
            res = fam_close_confirm(h, l, c, level, stop, start_idx)
            return None if res is None else (res[0], res[1], res[2], lvl_idx)
        res = fam_retest(h, l, level, stop, start_idx)
        return res
    if fam == 'F13':
        return fam_sweep_reclaim(h, l, c, cfg['K'], cfg['X'])
    if fam == 'F14':
        return fam_second_break(o, h, l, c, m, cfg['N'], eod_idx)
    return None


# ------------------------------------------------------------------ the day loop
def fill_block(o, h, l, c, m, i, level, stop, price, entry, fill_idx, eod_idx):
    """Every exit column for one fill model. Exits start the bar AFTER the fill bar."""
    k0 = fill_idx + 1
    R = entry - stop
    d = {'entry': entry, 'entry_m': int(m[fill_idx]), 'r_pct': R / entry * 100.0}
    px, why, xm, kx = walk_2r(o, h, l, c, m, k0, stop, entry + 2 * R, eod_idx)
    d['rr_2r'], d['why_2r'], d['exit_m_2r'] = (px - entry) / R, why, xm
    lo_seg = float(l[k0:kx + 1].min()) if kx >= k0 else entry
    hi_seg = float(h[k0:kx + 1].max()) if kx >= k0 else entry
    d['mae_pct'] = max(0.0, (entry - lo_seg) / price * 100.0)
    d['mfe_r'] = max(0.0, (hi_seg - entry) / R)
    px, why, xm, _ = walk_2r(o, h, l, c, m, k0, stop, None, eod_idx)
    d['rr_hold'], d['why_hold'], d['exit_m_hold'] = (px - entry) / R, why, xm
    px, why, xm, _ = walk_closestop(o, h, l, c, m, k0, stop, entry + 2 * R, eod_idx)
    d['rr_2r_closestop'], d['why_2r_closestop'], d['exit_m_2r_closestop'] = (px - entry) / R, why, xm
    stop1 = stop - STOPM1_PCT * price
    R1 = entry - stop1
    px, why, xm, _ = walk_2r(o, h, l, c, m, k0, stop1, entry + 2 * R1, eod_idx)
    rr1 = (px - entry) / R1
    d['rr_2r_stopm1'], d['why_2r_stopm1'], d['exit_m_2r_stopm1'] = rr1, why, xm
    d['r_pct_m1'] = R1 / entry * 100.0
    d['pnl100_2r_stopm1'] = RISK_USD * rr1
    px, why, xm, _ = walk_lock(o, h, l, c, m, k0, entry, stop, R, eod_idx)
    d['rr_lock'], d['why_lock'], d['exit_m_lock'] = (px - entry) / R, why, xm
    return d


def build_day(day, sub):
    bars = B.load_bars(day, sub.symbol.tolist())
    rows = []
    missing = []
    for r in sub.itertuples():
        gg = bars.get(r.symbol)
        if gg is None:
            missing.append(r.symbol)
            continue
        rth = gg[(gg.m >= OPEN_M) & (gg.m < 960)].reset_index(drop=True)
        if len(rth) < 10:
            missing.append(r.symbol)
            continue
        o, h, l, c, v = (rth[k].values.astype(float) for k in ('o', 'h', 'l', 'c', 'v'))
        m = rth.m.values.astype(int)
        o0 = o[0]
        cumv = np.cumsum(v)
        cumdv = np.cumsum(c * v)
        vwap = cumdv / np.maximum(cumv, 1)
        run_hi = np.maximum.accumulate(h)
        run_lo = np.minimum.accumulate(l)
        pm = gg[gg.m < OPEN_M]
        pmh = float(pm.h.max()) if len(pm) else None
        pmdv = float((pm.c.astype(float) * pm.v.astype(float)).sum()) if len(pm) else np.nan
        prev_close = r.prev_close if r.prev_close == r.prev_close else np.nan
        gap = (o0 / prev_close - 1) if (prev_close == prev_close and prev_close > 0) else np.nan
        pdr = ((r.prev_high - r.prev_low) / r.prev_low * 100.0
               if (r.prev_high == r.prev_high and r.prev_low == r.prev_low and r.prev_low > 0) else np.nan)
        acl = asset_class(r.symbol)
        w = np.flatnonzero(m >= EOD_M)
        eod_idx = int(w[0]) if len(w) else None
        arrays = (o, h, l, c, v, m, o0, vwap, prev_close, pmh, gap, eod_idx)
        for fam, cfg in FAMS:
            res = detect(fam, cfg, arrays)
            if res is None:
                continue
            i, level, stop, lvl_idx = res
            if i < 1 or stop >= level or not (level > 0):
                continue
            price = float(level)
            nt = int(((h[:i] >= level * 0.998) & (h[:i] < level)).sum())
            cb = max(int(i - lvl_idx), 0)
            seg = v[lvl_idx:i] if i > lvl_idx else v[:i]
            mv = float(seg.mean()) if len(seg) and seg.mean() > 0 else np.nan
            d = dict(day=day, symbol=r.symbol, fam=fam, cfg=json.dumps(cfg, sort_keys=True), sig_m=int(m[i]),
                     minutes_since_open=int(m[i]) - OPEN_M, level=price, stop=float(stop), price=price,
                     dist_open_pct=(price / o0 - 1) * 100.0,
                     range_so_far_pct=(run_hi[i - 1] - run_lo[i - 1]) / o0 * 100.0,
                     rv_adv=cumv[i] / r.adv20 if (r.adv20 == r.adv20 and r.adv20 > 0) else np.nan,
                     gap_pct=gap * 100 if gap == gap else np.nan, adv20=r.adv20,
                     spread_pct=spread_pct(price), spread_cc_bps=cc_bps(price, int(m[i])),
                     sig_o=float(o[i]), sig_h=float(h[i]), sig_l=float(l[i]), sig_c=float(c[i]), sig_v=float(v[i]),
                     n_touches=nt, consol_bars=cb,
                     consol_vol_ratio=float(v[i]) / mv if mv == mv else np.nan,
                     cum_dollar_vol=float(cumdv[i]), vwap_dist_pct=(price / vwap[i - 1] - 1) * 100.0,
                     close_confirm=int(c[i] >= level), pm_dollar_vol=pmdv, prev_day_range_pct=pdr,
                     prev_close=prev_close, asset_class=acl)
            # ---- fill model 1: the engine's next-bar open under the cap
            if i + 1 < len(o):
                e = float(o[i + 1])
                if e <= level * (1 + CAP) and stop < e:
                    for k, val in fill_block(o, h, l, c, m, i, level, stop, price, e, i + 1, eod_idx).items():
                        d[f'next_{k}'] = val
            # ---- fill model 2: a resting stop-limit at the level (not available to close-triggered families)
            if fam not in CLOSE_TRIGGERED:
                e = max(float(o[i]), float(level))
                d['rest_obtain'] = int(l[i] <= e <= h[i])
                if e <= level * (1 + CAP) and stop < e:
                    shares = RISK_USD / (e - stop)
                    d['rest_queue_ok'] = int(v[i] >= 5.0 * shares)
                    for k, val in fill_block(o, h, l, c, m, i, level, stop, price, e, i, eod_idx).items():
                        d[f'rest_{k}'] = val
            rows.append(d)
    return rows, missing


def main():
    uni = B.uni
    state = json.load(open(STATE)) if os.path.exists(STATE) else {'done': []}
    done = set(state['done'])
    days = sorted(uni.bar_date.unique())
    if os.environ.get('BFZ_DAYS'):
        days = days[:int(os.environ['BFZ_DAYS'])]
    todo = [d for d in days if d not in done]
    print(f'candidates4 | days {len(days)} | done {len(done)} | todo {len(todo)} | fam-configs {len(FAMS)} | '
          f'cap {CAP:.3%} | cost-curve cells {len(CC)}', flush=True)
    t0 = time.time()
    n = 0
    for k, day in enumerate(todo):
        rows, miss = build_day(day, uni[uni.bar_date == day])
        if rows:
            pd.DataFrame(rows).reindex(columns=COLS).to_csv(OUT, mode='a', header=not os.path.exists(OUT), index=False)
        if miss:
            pd.DataFrame({'symbol': miss, 'bar_date': day}).to_csv(MISS, mode='a', header=not os.path.exists(MISS), index=False)
        n += len(rows)
        state['done'].append(day)
        json.dump(state, open(STATE, 'w'))
        el = time.time() - t0
        eta = (len(todo) - k - 1) * el / (k + 1) / 60
        print(f'{k+1}/{len(todo)} {day} rows+={len(rows)} total {n:,} | {el/60:.1f} min elapsed, '
              f'{el/(k+1):.1f} s/day, ETA {eta:.0f} min', flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
