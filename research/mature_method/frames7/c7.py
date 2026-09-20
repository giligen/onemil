#!/usr/bin/env python3
"""frames7 — shared loaders, THE THREE WALKERS, match pools and stats.

ONE definition used by F24 (the placebo ported to ORB and BF-P1), F23 (the short mirror) and
F22 (the bare geometry).  Every store is opened read-only.  Nothing outside this directory is
written.

The three walkers are written from prose, not copied:

  * `walk_orb`  — ORB's shipped static-lock geometry: stop at `r_pct` below entry, once the
                  running high reaches +1.75R the stop moves to +0.5R forever, flat at 15:45.
  * `walk_bf`   — BF P1's geometry: hard stop at `r_pct` below entry; the R-trail arms at +2R
                  and rides 1R below the running closed-bar high (`trading/bf_trail` contract:
                  the trail advances on CLOSED bars, the stop it produces is live from the NEXT
                  bar); optional 50 % partial at +2R with the stop to breakeven; flat at 15:45.
  * `walk_short`— the mirror: short entry at the next bar's open under a floor, stop ABOVE at the
                  same % distance, cover target at -2R, flat at 15:55.

All three share the fill convention of the programme: entry at the OPEN of the entry bar, exits
evaluated from the NEXT bar on, stops fill at the worse of (level, that bar's open) plus one slip.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_filter_stack')
from pass2 import load_bars                                              # noqa: E402

D7 = f'{ROOT}/research/mature_method/frames7'
OPEN_M = 570          # 09:30
ORB_RANGE_END = 575   # 09:35 — the end of the opening range
SLIP = 0.001          # the programme's stop slippage, identical to walk2.vwalk
ORB_FLAT_M = 945      # 15:45 ET, ORB's force close
BF_FLAT_M = 945       # 15:45 ET, config.yaml trading.force_close_time
HOD_FLAT_M = 955      # 15:55 ET, the HOD-break / mirror force close

SPLITS = ('TRAIN', 'VAL')
UNIV = f'{ROOT}/research/bf_zero/universe.csv'


def split_of(day: str) -> str:
    """TRAIN = 2025, VAL = 2026-01..05, TEST = 2026-06+ (SEALED — never returned as a split)."""
    if day < '2026-01-01':
        return 'TRAIN'
    if day < '2026-06-01':
        return 'VAL'
    return 'TEST'


def half_of(day: str) -> str:
    """The two TRAIN halves the rails require beside VAL."""
    return 'H1' if day < '2025-07-01' else 'H2'


# --------------------------------------------------------------------------------- bar helpers
def arrays(gg, m_lo=OPEN_M, m_hi=960):
    """(o,h,l,c,v,m) float arrays over the RTH window, or None when the day is unusable."""
    r = gg[(gg.m >= m_lo) & (gg.m < m_hi)]
    if len(r) < 10:
        return None
    o, h, l, c, v = (r[k].values.astype(float) for k in ('o', 'h', 'l', 'c', 'v'))
    return o, h, l, c, v, r.m.values.astype(int)


def idx_of_minute(m, want, exact=False):
    """Index of the first EXISTING bar at or after minute `want`, or -1.

    A missing 1-minute bar means nothing printed in that minute, so the next obtainable open is
    the next bar that exists — that, not a dropped trade, is what the engine would get.  `exact`
    restores the strict lookup for callers that need the minute itself.
    """
    k = np.where(m == int(want))[0]
    if len(k):
        return int(k[0])
    if exact:
        return -1
    k = np.where(m > int(want))[0]
    return int(k[0]) if len(k) else -1


# ------------------------------------------------------------------------------- THE WALKERS
def walk_orb(o, h, l, c, m, e, r_pct, lock_arm_r=1.75, lock_stop_r=0.5, flat_m=ORB_FLAT_M):
    """ORB static-lock bracket entered at the open of bar `e`.  Returns (rr, why, exit_m).

    stop  = E x (1 - r_pct/100); R = E - stop.  Once a CLOSED bar's high reaches E + 1.75R the
    stop moves to E + 0.5R and never moves again (the shipped `static_lock_1R`).  No target.
    Flat at `flat_m` at that bar's open.  Stops fill at min(stop, bar open) x (1 - SLIP).
    """
    n = len(o)
    if e < 0 or e + 1 >= n:
        return np.nan, '', -1
    E = float(o[e])
    R = E * (r_pct / 100.0)
    if not (R > 0) or not (E > 0):
        return np.nan, '', -1
    stop = E - R
    arm = E + lock_arm_r * R
    locked = False
    for k in range(e + 1, n):
        if m[k] >= flat_m:
            return (float(o[k]) - E) / R, 'flat', int(m[k])
        if l[k] <= stop:
            px = min(stop, float(o[k])) * (1.0 - SLIP)
            return (px - E) / R, ('lock' if locked else 'stop'), int(m[k])
        if (not locked) and h[k] >= arm:
            locked = True
            stop = E + lock_stop_r * R
    return (float(c[-1]) - E) / R, 'eod', int(m[-1])


def walk_bf(o, h, l, c, m, e, r_pct, activate_at_r=2.0, trail_r=1.0,
            partial=False, partial_r=2.0, partial_frac=0.5, flat_m=BF_FLAT_M):
    """BF P1 bracket entered at the open of bar `e`.  Returns (rr, why, exit_m).

    Hard stop at E x (1 - r_pct/100).  The R-trail arms when the running CLOSED-bar high reaches
    E + activate_at_r x R and then sits trail_r x R below that high; per `trading/bf_trail` the
    stop a bar produces is live only from the NEXT bar (check-then-ratchet).  With `partial`, the
    first closed bar whose high reaches E + partial_r x R sells `partial_frac` at that bar's CLOSE
    and moves the stop to E (true breakeven); `rr` is then the share-weighted blend.
    """
    n = len(o)
    if e < 0 or e + 1 >= n:
        return np.nan, '', -1
    E = float(o[e])
    R = E * (r_pct / 100.0)
    if not (R > 0) or not (E > 0):
        return np.nan, '', -1
    stop = E - R
    hi = E
    active = False
    pdone = False
    prr = 0.0
    frac = 1.0
    ptag = ''
    for k in range(e + 1, n):
        if m[k] >= flat_m:
            return prr + frac * (float(o[k]) - E) / R, ptag + 'flat', int(m[k])
        if l[k] <= stop:
            px = min(stop, float(o[k])) * (1.0 - SLIP)
            return prr + frac * (px - E) / R, ptag + ('trail_stop' if active else 'stop'), int(m[k])
        if partial and not pdone and h[k] >= E + partial_r * R:
            pdone = True
            prr = partial_frac * (float(c[k]) - E) / R
            frac = 1.0 - partial_frac
            stop = max(stop, E)
            ptag = 'pp+'
        hi = max(hi, float(h[k]))
        if not active and (hi - E) / R >= activate_at_r:
            active = True
        if active:
            stop = max(stop, hi - trail_r * R)
    return prr + frac * (float(c[-1]) - E) / R, ptag + 'eod', int(m[-1])


def walk_short(o, h, l, c, m, e, r_pct, floor_bps=60.0, flat_m=HOD_FLAT_M):
    """THE MIRROR — a SHORT bracket entered at the open of bar `e` under a price FLOOR.

    Never a sign flip.  Entry `E = o[e]`; the engine would rest a sell-limit at
    `o[e] x (1 - floor_bps/1e4)`, so a bar that opens BELOW the floor is a fill at the open and a
    bar that never trades down to the floor is NO FILL (returns why='nofill').  Stop is ABOVE at
    `E x (1 + r_pct/100)` (the same % distance as the long), cover target at -2R, flat at `flat_m`.
    Stop fills at max(stop, that bar's open) x (1 + SLIP) — the slip is against us on both legs.
    Returns (rr, why, exit_m) with rr in SHORT R (positive = the name fell).
    """
    n = len(o)
    if e < 1 or e + 1 >= n:
        return np.nan, 'nofill', -1
    # The mirror of the long's capped BUY: a FLOORED sell-limit is the worst price we will accept,
    # and it is set from information available at the close of bar e-1 (the engine's decision bar).
    # The bar opens at or above the floor -> our marketable sell fills at that open.  The bar gaps
    # DOWN through the floor -> no fill, $0, never a loss.
    floor = float(c[e - 1]) * (1.0 - floor_bps / 1e4)
    if float(o[e]) < floor:
        return np.nan, 'nofill', -1
    E = float(o[e])
    R = E * (r_pct / 100.0)
    if not (R > 0) or not (E > 0):
        return np.nan, 'nofill', -1
    stop = E + R
    tgt = E - 2.0 * R
    for k in range(e + 1, n):
        if m[k] >= flat_m:
            return (E - float(o[k])) / R, 'flat', int(m[k])
        if h[k] >= stop:
            px = max(stop, float(o[k])) * (1.0 + SLIP)
            return (E - px) / R, 'stop', int(m[k])
        if c[k] <= tgt:
            return (E - tgt) / R, 'target', int(m[k])
    return (E - float(c[-1])) / R, 'eod', int(m[-1])


# --------------------------------------------------------------------------------- the universe
_U = {}


def universe(days=None):
    """The point-in-time day panel with prev_close, adv20, gap and open->clock range fields."""
    if 'u' not in _U:
        u = pd.read_csv(UNIV, usecols=['symbol', 'bar_date', 'open', 'high', 'low', 'close',
                                       'volume', 'adv20'],
                        dtype={'symbol': str}, keep_default_na=False, na_values=[''])
        u = u.rename(columns={'bar_date': 'day'})
        for k in ('open', 'high', 'low', 'close', 'volume', 'adv20'):
            u[k] = pd.to_numeric(u[k], errors='coerce')
        u = u.sort_values(['symbol', 'day'], kind='mergesort')
        u['prev_close'] = u.groupby('symbol', sort=False).close.shift(1)
        u = u[u.prev_close.notna() & (u.prev_close > 0) & (u.adv20 > 0) & (u.close > 0)]
        u['gap_pct'] = (u.open - u.prev_close) / u.prev_close * 100.0
        u['advd'] = u.adv20 * u.close
        _U['u'] = u
    u = _U['u']
    return u[u.day.isin(days)] if days is not None else u


def daily_fallback(pairs):
    """prev_close / adv20 straight from `daily_bars` for (day, symbol) the PIT panel misses.

    The bf_zero PIT panel carries its own price/volume screen, so a booked trade on a name it
    screened out has no matching key.  `daily_bars` is the table live BF and live ORB themselves
    seed from, so it is the right fallback for the BOOKED side; the control POOL stays on the PIT
    panel.  Read-only.
    """
    import sqlite3
    syms = sorted({s for _, s in pairs})
    if not syms:
        return pd.DataFrame(columns=['day', 'symbol', 'prev_close', 'adv20', 'advd'])
    con = sqlite3.connect(f'file:{ROOT}/data/cache.db?mode=ro', uri=True, timeout=120)
    out = []
    for i in range(0, len(syms), 400):
        ch = syms[i:i + 400]
        q = ('select symbol, bar_date as day, close, volume from daily_bars where symbol in ('
             + ','.join('?' * len(ch)) + ')')
        out.append(pd.read_sql(q, con, params=ch))
    d = pd.concat(out, ignore_index=True).sort_values(['symbol', 'day'], kind='mergesort')
    d['prev_close'] = d.groupby('symbol', sort=False).close.shift(1)
    d['adv20'] = d.groupby('symbol', sort=False).volume.transform(
        lambda s: s.shift(1).rolling(20, min_periods=5).mean())
    d['advd'] = d.adv20 * d.close
    want = pd.DataFrame(sorted(pairs), columns=['day', 'symbol'])
    return want.merge(d[['day', 'symbol', 'prev_close', 'adv20', 'advd']],
                      on=['day', 'symbol'], how='left')


_CLS = {}


def asset_class(symbols):
    """{symbol: 'stock'|'wrapper'|'unknown'} — the shipped offline map, then the Alpaca name."""
    from trading.orb_asset_class import classify_asset, load_class_map
    if 'nm' not in _CLS:
        ass = pd.read_csv(f'{ROOT}/data/research/alpaca_assets_all_20260905.csv',
                          dtype=str, keep_default_na=False, na_values=[''])
        _CLS['nm'] = dict(zip(ass.symbol, ass.name))
        _CLS['cmap'] = load_class_map()
        _CLS['c'] = {}
    for s in set(map(str, symbols)):
        if s not in _CLS['c']:
            _CLS['c'][s] = _CLS['cmap'].get(s) or classify_asset(s, _CLS['nm'].get(s))
    return _CLS['c']


def match_pool(bk, sigset, npool=25, gap_band=False, seed=20260920):
    """For every booked trade, the `npool` nearest NON-SIGNAL symbols of the SAME session.

    Distance = |dlog(prev_close)| + |dlog(adv20)|, exact match on asset class, and — when
    `gap_band` — on the gap decile (ORB only; its population is gap-ups by construction).
    Also returns, per booked trade, `npool` RANDOM universe names of the same session (the
    universe bound), drawn without any matching at all.
    """
    rng = np.random.default_rng(seed)
    days = set(bk.day.unique())
    u = universe(days).copy()
    u = u[~pd.Series(list(zip(u.day, u.symbol)), index=u.index).isin(sigset)]
    cls = asset_class(list(u.symbol.unique()) + list(bk.symbol.unique()))
    u['cls'] = u.symbol.map(cls)
    u['lp'] = np.log(u.prev_close)
    u['la'] = np.log(u.adv20)
    bk = bk.copy()
    bk['cls'] = bk.symbol.map(cls)
    UD = {d: g.reset_index(drop=True) for d, g in u.groupby('day')}
    rows, rnd, miss = [], [], 0
    for r in bk.itertuples():
        g = UD.get(r.day)
        if g is None or len(g) < npool or not (r.prev_close > 0) or not (r.adv20 > 0):
            miss += 1
            continue
        gg = g[g.cls == r.cls]
        if gap_band and len(gg) >= npool:
            lo, hi = r.gap_pct - 5.0, r.gap_pct + 5.0
            gb = gg[(gg.gap_pct >= lo) & (gg.gap_pct <= hi)]
            if len(gb) >= npool:
                gg = gb
        if len(gg) < npool:
            gg = g
        d = np.abs(gg.lp.values - np.log(r.prev_close)) + np.abs(gg.la.values - np.log(r.adv20))
        for k in np.argsort(d, kind='mergesort')[:npool]:
            rows.append((r.day, r.symbol, int(r.entry_m), str(gg.symbol.values[k]), float(d[k])))
        for k in rng.choice(len(g), size=min(npool, len(g)), replace=False):
            rnd.append((r.day, r.symbol, int(r.entry_m), str(g.symbol.values[k])))
    P = pd.DataFrame(rows, columns=['day', 'symbol', 'entry_m', 'ctrl', 'dist'])
    U = pd.DataFrame(rnd, columns=['day', 'symbol', 'entry_m', 'ctrl'])
    return P, U, miss


# --------------------------------------------------------------------------------- statistics
def clustered_t(v, days):
    """Day-clustered t of the mean of `v` (one cluster per session)."""
    v = np.asarray(v, dtype=float)
    d = np.asarray(days)
    ok = np.isfinite(v)
    v, d = v[ok], d[ok]
    if len(v) < 3:
        return np.nan
    mu = v.mean()
    g = pd.Series(v - mu).groupby(pd.Series(d)).sum().values
    nd = len(g)
    if nd < 3:
        return np.nan
    se = np.sqrt((g ** 2).sum()) / len(v)
    return np.nan if se <= 0 else mu / se


def mde(v, power_k=2.80):
    """Two-sided 80 %-power minimum detectable effect on the mean of `v`."""
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    return np.nan if len(v) < 3 else power_k * v.std(ddof=1) / np.sqrt(len(v))


def draw_band(ctrl, keys, ndraw=200, seed=20260920):
    """200 control BOOKS: one control per booked trade, drawn at random, mean gross R each.

    `ctrl` is a DataFrame with the booked key columns plus `rr`; `keys` names those columns.
    Returns (mean of the draws, p5, p95, the draw array).
    """
    rng = np.random.default_rng(seed)
    g = ctrl.groupby(keys, sort=True).rr.apply(lambda s: s.values)
    if not len(g):
        return np.nan, np.nan, np.nan, np.array([])
    arrs = list(g.values)
    out = np.empty(ndraw)
    for i in range(ndraw):
        out[i] = np.mean([a[rng.integers(len(a))] for a in arrs])
    return float(out.mean()), float(np.percentile(out, 5)), float(np.percentile(out, 95)), out


def pctile(x, arr):
    """The observed statistic's percentile inside a draw distribution."""
    return np.nan if not len(arr) else float((arr < x).mean() * 100.0)
