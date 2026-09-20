#!/usr/bin/env python3
"""frames12 — shared exits, cost, controls and reporting.  ONE definition for F39, F38, F37.

Everything new in this pass is here; every object imports it.  The two inherited objects are
`hod_frames6.common6.walk_from` (the +2 R bracket, the pass-6 R3 rail) and
`hod_frames4.common4.book_ranked` (the slot machine, identical to `trading.hod_break.run_book`).

NEW here, all declared in `frames12/PREREG.md` §1.1 before any cell was read:
  * `walk_bare`   — the stop ridden to 15:55, no target                       (uncapped)
  * `walk_lock`   — ORB's static lock: +1.75 R arms, stop to +0.5 R forever   (uncapped)
  * `walk_moc`    — the stop honoured intraday, else the CLOSING AUCTION      (no quoted spread)
  * `walk_next`   — ... else carried overnight to the next OPENING AUCTION    (no quoted spread)
  * `attach_cost12` — the programme's cost model with the two auction legs at ratio 0.0
  * `Sheet12`     — the standard row, in R **and** in % of entry price (F31's unit)

Every store is opened READ-ONLY.  TEST (`day >= 2026-06-01`) is cut in `load_panel`.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
for _p in ('frames12', 'frames11', 'hod_frames6', 'hod_frames5', 'hod_frames4', 'hod_frames3',
           'hod_frames2'):
    sys.path.insert(0, f'{ROOT}/research/mature_method/{_p}')
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_filter_stack')
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_break')

from common6 import walk_from, bars_arrays, OPEN_M, EOD_M, SLIP           # noqa: E402,F401
from common4 import (S, S2, SPLITS, RISK, book_ranked, clustered_t, halves,   # noqa: E402,F401
                     load_breaks4)
from pass2 import load_bars                                              # noqa: E402
from trading.hod_break import profile_fraction                           # noqa: E402
from research.scripts.pit_listings import is_test_ticker                 # noqa: E402

D12 = f'{ROOT}/research/mature_method/frames12'
TEST_FROM = '2026-06-01'
CAP = 0.006                       # the engine's entry cap: limit at level x 1.006
FIRST_ENTRY_M, LAST_ENTRY_M = 577, 841      # 09:37 .. 14:01, the shipped window
LAST_BAR_M = 959                            # the last RTH 1-min bar

# the cost ratio of the EXIT leg, by exit reason.  The first three are inherited verbatim from
# `hod_break/score.RATIO`; the two auction legs are this pass's declared addition (PREREG §1.1).
RATIO12 = {'target': 0.0, 'eod': 0.412, 'stop': 0.875, 'lock': 0.875,
           'moc': 0.0, 'nextopen': 0.0}


# --------------------------------------------------------------------------- the exits
def walk_bare(o, h, l, c, m, e, stop):
    """The stop ridden to the 15:55 force close.  No target — UNCAPPED."""
    n = len(o)
    E = float(o[e]); R = E - stop
    if not (R > 0):
        return -1, np.nan, '', np.nan
    for k in range(e + 1, n):
        if int(m[k]) >= EOD_M:
            return k, float(o[k]), 'eod', (float(o[k]) - E) / R
        if l[k] <= stop:
            px = float(min(stop, o[k]) * (1.0 - SLIP))
            return k, px, 'stop', (px - E) / R
    return n - 1, float(c[-1]), 'eod', (float(c[-1]) - E) / R


def walk_lock(o, h, l, c, m, e, stop, arm=1.75, lk=0.5):
    """ORB's STATIC LOCK, verbatim: no target; once the high touches +arm R the stop moves to
    +lk R and stays there forever; flat 15:55.  UNCAPPED.

    Within a bar the stop in force at the START of the bar is the one that can fill — the arm is
    applied only after the bar has been tested against the standing stop (the conservative order).
    """
    n = len(o)
    E = float(o[e]); R = E - stop
    if not (R > 0):
        return -1, np.nan, '', np.nan
    s = float(stop); armed = False
    for k in range(e + 1, n):
        if int(m[k]) >= EOD_M:
            return k, float(o[k]), 'eod', (float(o[k]) - E) / R
        if l[k] <= s:
            px = float(min(s, o[k]) * (1.0 - SLIP))
            return k, px, ('lock' if armed else 'stop'), (px - E) / R
        if not armed and h[k] >= E + arm * R:
            armed = True; s = E + lk * R
    return n - 1, float(c[-1]), 'eod', (float(c[-1]) - E) / R


def walk_moc(o, h, l, c, m, e, stop, close_px):
    """The stop honoured intraday; anything still open is sold in the CLOSING AUCTION at the
    session's official close.  No quoted spread on that leg (RUNBOOK step 3).  UNCAPPED."""
    n = len(o)
    E = float(o[e]); R = E - stop
    if not (R > 0) or not (close_px == close_px and close_px > 0):
        return -1, np.nan, '', np.nan
    for k in range(e + 1, n):
        if l[k] <= stop:
            px = float(min(stop, o[k]) * (1.0 - SLIP))
            return k, px, 'stop', (px - E) / R
    return n - 1, float(close_px), 'moc', (float(close_px) - E) / R


def walk_next(o, h, l, c, m, e, stop, next_open_px):
    """The stop honoured intraday; anything still open is CARRIED OVERNIGHT with no overnight stop
    and sold into the next session's OPENING AUCTION.  The gap is charged in full.  UNCAPPED."""
    n = len(o)
    E = float(o[e]); R = E - stop
    if not (R > 0) or not (next_open_px == next_open_px and next_open_px > 0):
        return -1, np.nan, '', np.nan
    for k in range(e + 1, n):
        if l[k] <= stop:
            px = float(min(stop, o[k]) * (1.0 - SLIP))
            return k, px, 'stop', (px - E) / R
    return n - 1, float(next_open_px), 'nextopen', (float(next_open_px) - E) / R


EXITS = {'X2R': 'bracket', 'XBR': 'bare', 'XLK': 'lock', 'XMO': 'moc', 'XNO': 'next'}


def price_exit(kind, o, h, l, c, m, e, stop, close_px=np.nan, next_open_px=np.nan):
    """Dispatch.  Returns (exit_m, rr, why)."""
    if kind == 'bracket':
        k, px, why, rr = walk_from(o, h, l, c, m, e, stop)
    elif kind == 'bare':
        k, px, why, rr = walk_bare(o, h, l, c, m, e, stop)
    elif kind == 'lock':
        k, px, why, rr = walk_lock(o, h, l, c, m, e, stop)
    elif kind == 'moc':
        k, px, why, rr = walk_moc(o, h, l, c, m, e, stop, close_px)
    elif kind == 'next':
        k, px, why, rr = walk_next(o, h, l, c, m, e, stop, next_open_px)
    else:
        raise ValueError(kind)
    if k < 0:
        return -1, np.nan, ''
    return int(m[k]), float(rr), why


# --------------------------------------------------------------------------- the cost model
_NBBO = {}


def nbbo_table():
    """The measured NBBO fetch, keyed (day, symbol, entry_m).  Read once."""
    if not _NBBO:
        nb = pd.read_csv(f'{ROOT}/research/bf_zero/causal_filter/nbbo.csv',
                         dtype={'symbol': str, 'day': str}, keep_default_na=False,
                         na_values=['']).drop_duplicates(['day', 'symbol', 'entry_m'])
        _NBBO['t'] = nb[['day', 'symbol', 'entry_m', 'spread_mean', 'ask_dec', 'bid_dec']]
    return _NBBO['t']


def attach_cost12(x, level_col='level'):
    """The programme's cost model on an ARBITRARY signal frame.

    Requires columns: day, symbol, entry_m, next_open (the fill), stop, r_pct, rr, why.
    Adds: price, sp_pct, imputed, net, notional, obtainable, pnl_pct/net_pct (F31's unit).
    The exit leg's ratio comes from `RATIO12`, so an auction exit pays NO quoted spread.
    """
    x = x.copy()
    x['price'] = x.next_open
    nb = nbbo_table()
    x = x.merge(nb, on=['day', 'symbol', 'entry_m'], how='left')
    pb = pd.cut(x.price, S.PB_EDGES, labels=S.PB_LAB)
    hb = pd.cut(x.entry_m, S.HB_EDGES, labels=S.HB_LAB)
    sp = x.spread_mean / x.price * 100
    x['imputed'] = sp.isna()
    imp = pd.Series([S.IMPUTE.get((p, h), S.IMPUTE_GLOBAL) for p, h in zip(pb, hb)], index=x.index)
    x['sp_pct'] = sp.fillna(imp).fillna(S.IMPUTE_GLOBAL)
    bnd = pd.Series([S.BAND.get((p, h), S.BAND_GLOBAL) for p, h in zip(pb, hb)], index=x.index)
    x['sp_band'] = sp.fillna(bnd).fillna(S.BAND_GLOBAL)          # the conservative arm
    ratio = x.why.map(RATIO12).fillna(0.875)
    half = 0.5 * x.sp_pct / x.r_pct.clip(lower=0.05)
    x['cost_R'] = half + half * ratio
    x['net'] = x.rr - x.cost_R
    hb2 = 0.5 * x.sp_band / x.r_pct.clip(lower=0.05)
    x['netb'] = x.rr - hb2 - hb2 * ratio                         # `S.week_stats` reads this
    x['notional'] = RISK / (x.price - x.stop).clip(lower=1e-6) * x.price
    if level_col in x.columns:
        x['obtainable'] = np.where(x.ask_dec.notna(),
                                   x.ask_dec <= x[level_col] * (1 + CAP) * (1 + 1e-9), True)
    else:
        x['obtainable'] = True
    # F31's unit: R x (stop as % of price) IS the % of entry price.  `r_pct` is ALREADY a
    # percent, so the conversion is a plain product — dividing by 100 again would understate the
    # whole map by 100x (caught on the F38 floor, which must reproduce F34's -0.39 %).
    x['gross_pct'] = x.rr * x.r_pct                    # % of ENTRY PRICE (F31's unit)
    x['net_pct'] = x.net * x.r_pct
    x['cost_pct'] = x.cost_R * x.r_pct
    return x


def cascade(x, min_price=20.0, r_min=1.0, max_bps=100.0, max_frac_r=0.15):
    """The shipped pre-book cascade, unchanged: $20 floor, 100 bps, 15 % of R, obtainable."""
    x = x[x.r_pct.notna() & (x.r_pct >= r_min) & (x.next_open >= min_price)]
    if max_bps is not None:
        x = x[(x.sp_pct * 100) <= max_bps]
    if max_frac_r is not None:
        x = x[(x.sp_pct / x.r_pct.clip(lower=0.05)) <= max_frac_r]
    return x[x.obtainable.astype(bool)]


# --------------------------------------------------------------------------- the panel
_PANEL = {}


def load_panel():
    """The PIT daily panel, TRAIN+VAL only, with prior-session levels and next-session open.

    Columns: symbol, day, open, high, low, close, volume, adv20, prev_close, prev_low,
             h5, h20, h252 (max daily HIGH of the prior N sessions, strictly before `day`),
             next_open_d (the NEXT session's official open — the overnight exit's fill).
    Test tickers and names absent from `daily_bars` are dropped here, once.
    """
    if 'p' in _PANEL:
        return _PANEL['p']
    u = pd.read_csv(f'{ROOT}/research/bf_zero/universe.csv', dtype={'symbol': str},
                    keep_default_na=False, na_values=[''])
    u = u.rename(columns={'bar_date': 'day'})
    for k in ('open', 'high', 'low', 'close', 'volume', 'adv20'):
        u[k] = pd.to_numeric(u[k], errors='coerce')
    u = u[~u.symbol.map(lambda s: is_test_ticker(str(s)))]
    import sqlite3
    con = sqlite3.connect(f'file:{ROOT}/data/cache.db?mode=ro', uri=True, timeout=120)
    dbs = set(pd.read_sql('select distinct symbol from daily_bars', con).symbol.astype(str))
    con.close()
    u = u[u.symbol.isin(dbs)]
    u = u.sort_values(['symbol', 'day'], kind='mergesort').reset_index(drop=True)
    g = u.groupby('symbol', sort=False)
    u['prev_close'] = g.close.shift(1)
    u['prev_low'] = g.low.shift(1)
    u['next_open_d'] = g.open.shift(-1)
    ph = g.high.shift(1)
    for N, lab in ((5, 'h5'), (20, 'h20'), (252, 'h252')):
        u[lab] = ph.groupby(u.symbol, sort=False).rolling(N, min_periods=N).max() \
                   .reset_index(level=0, drop=True)
    u = u[(u.day < TEST_FROM) & ~u.day.isin(S.EARLY_CLOSE)]
    _PANEL['p'] = u
    return u


def coarse(o, h, l, c, v, m, size):
    """Clock-aligned coarse bars from 09:30.  Only COMPLETE buckets are returned."""
    b = (m - OPEN_M) // size
    keep = (m >= OPEN_M) & (m < 960)
    b = b[keep]
    o_, h_, l_, c_, v_, m_ = o[keep], h[keep], l[keep], c[keep], v[keep], m[keep]
    out_o, out_h, out_l, out_c, out_v, out_m, out_e = [], [], [], [], [], [], []
    i = 0
    n = len(b)
    while i < n:
        j = i
        while j + 1 < n and b[j + 1] == b[i]:
            j += 1
        if (j - i + 1) == size:                      # complete bucket only
            out_o.append(o_[i]); out_h.append(h_[i:j + 1].max()); out_l.append(l_[i:j + 1].min())
            out_c.append(c_[j]); out_v.append(v_[i:j + 1].sum())
            out_m.append(int(m_[i]))                 # the bucket's FIRST minute
            out_e.append(int(m_[j]))                 # the bucket's LAST minute (it closes here)
        i = j + 1
    if not out_o:
        return None
    return (np.array(out_o), np.array(out_h), np.array(out_l), np.array(out_c),
            np.array(out_v), np.array(out_m, dtype=int), np.array(out_e, dtype=int))


# --------------------------------------------------------------------------- reporting
HDR12 = ('| cell                                | split |    n |  /wk |  grossR |   costR |    netR |'
         ' gross% |  net%  |    t  |   tc  | green | red | worst $ | total $ |  MDD $ | imp% |')
SEP12 = '|' + '|'.join(['-' * 5] * 17) + '|'


class Sheet12:
    """Accumulates cells; prints R AND % of entry price on every row (PREREG §1)."""

    def __init__(self):
        self.cells, self.books, self._first = [], {}, True

    def show(self, name, b, note='', capped=False):
        self.books[name] = b
        if self._first:
            print(HDR12); print(SEP12); self._first = False
        for sp in SPLITS:
            d = b[b.split == sp]
            w = S.week_stats(b, sp)
            costR = float(d.cost_R.mean()) if len(d) else np.nan
            print(f'| {name:<35s} | {sp:5s} | {w["n"]:4d} | {w["per_wk"]:4.1f} | {w["gross"]:+7.3f} |'
                  f' {costR:+7.3f} | {w["net"]:+7.3f} | {d.gross_pct.mean() if len(d) else np.nan:+6.3f} |'
                  f' {d.net_pct.mean() if len(d) else np.nan:+6.3f} | {w["t"]:+5.2f} |'
                  f' {clustered_t(d):+5.2f} | {w["green"]:5.1f} | {w["redstreak"]:3d} |'
                  f' {w["worst"]:7.0f} | {w["total"]:7.0f} | {w["mdd"]:6.0f} | {w["imp"]:4.0f} |',
                  flush=True)
        g, n, ok = halves(b)
        for sp in SPLITS:
            d = b[b.split == sp]
            w = S.week_stats(b, sp)
            ex5 = (float(d.net[d.net <= d.net.quantile(0.95)].mean()) if len(d) > 20 else np.nan)
            w.update(cell=name, split=sp, note=note, capped=capped,
                     tc=clustered_t(d), costR=float(d.cost_R.mean()) if len(d) else np.nan,
                     gross_pct=float(d.gross_pct.mean()) if len(d) else np.nan,
                     net_pct=float(d.net_pct.mean()) if len(d) else np.nan,
                     cost_pct=float(d.cost_pct.mean()) if len(d) else np.nan,
                     r_pct=float(d.r_pct.mean()) if len(d) else np.nan,
                     mde=2.80 * float(d.net.std(ddof=1) / np.sqrt(len(d))) if len(d) > 2 else np.nan,
                     ex5_uncapped=(np.nan if capped else ex5),
                     cap_bar=(0.05 * 2.0 / 0.95 if capped else np.nan),
                     h1=g[0], h2=g[1], vgross=g[2], half_ok=ok,
                     exit_mix='|'.join(f'{k}:{v/len(b):.2f}' for k, v in
                                       b.why.value_counts().items()) if len(b) else '')
            self.cells.append(w)
        print(f'    halves H1 {g[0]:+.3f} (n {n[0]}) | H2 {g[1]:+.3f} (n {n[1]}) | VAL {g[2]:+.3f} '
              f'(n {n[2]}) -> same-signed POSITIVE in all three: {ok}'
              + ('' if capped else
                 f' | ex-top-5 % TR {self.cells[-2]["ex5_uncapped"]:+.3f} '
                 f'VAL {self.cells[-1]["ex5_uncapped"]:+.3f}'), flush=True)
        return b

    def nulls(self, path):
        rows = []
        print('\n== count-matched permutation nulls on GREEN WEEKS (2,000 draws) ==')
        print('| cell | split | green % | null mean | p5 | p95 | verdict |')
        print('|---|---|---|---|---|---|---|')
        for name, b in self.books.items():
            for sp in SPLITS:
                obs, mu, lo, hi = S.null_band(b, sp)
                if obs != obs:
                    continue
                v = 'ABOVE' if obs > hi else ('below' if obs < lo else 'inside')
                rows.append(dict(cell=name, split=sp, green=obs, null_mean=mu, p5=lo, p95=hi,
                                 verdict=v))
                print(f'| {name} | {sp} | {obs:.1f} | {mu:.1f} | {lo:.1f} | {hi:.1f} | {v} |',
                      flush=True)
        pd.DataFrame(rows).to_csv(path, index=False)
        return rows

    def dump(self, path):
        pd.DataFrame(self.cells).to_csv(path, index=False)
        print(f'\ncells -> {path}  ({len(self.cells)} rows)', flush=True)


def repro_gate():
    """PREREG §5 — asserted in code, raising, before any cell of this pass is read."""
    from common6 import base_book
    br = load_breaks4(verbose=False)
    b, _ = base_book(br, verbose=False)
    ref = {'TRAIN': (1622, -17346.0), 'VAL': (706, 893.0)}
    for sp in SPLITS:
        w = S.week_stats(b, sp); n, t = ref[sp]
        assert w['n'] == n, f'B2 repro FAIL {sp}: n {w["n"]} != {n}'
        assert abs(w['total'] - t) < 1.0, f'B2 repro FAIL {sp}: $ {w["total"]} != {t}'
    print('  GATE B2: 1,622 / -$17,346 (TRAIN), 706 / +$893 (VAL) — MATCH', flush=True)
    bk = pd.read_csv(f'{ROOT}/research/mature_method/hod_frames6/book6.csv',
                     dtype={'day': str, 'symbol': str})
    bk = bk[bk.split.isin(SPLITS)].sample(120, random_state=12)
    worst = 0.0
    for day, gg in bk.groupby('day'):
        bars = load_bars(day, sorted(gg.symbol.unique()))
        for r in gg.itertuples():
            a = bars.get(r.symbol)
            if a is None:
                continue
            arr = bars_arrays(a)
            if arr is None:
                continue
            o, h, l, c, v, m = arr
            idx = np.flatnonzero(m == int(r.entry_m))
            if not len(idx):
                continue
            _, _, _, rr = walk_from(o, h, l, c, m, int(idx[0]), float(r.stop))
            if rr == rr:
                worst = max(worst, abs(rr - float(r.rr)))
    print(f'  GATE walk_from vs book6.rr on 120 booked trades: max |diff| = {worst:.2e}', flush=True)
    assert worst < 1e-9, f'walker parity FAIL ({worst})'
    return b


def build_cost_model():
    """`S.IMPUTE` must exist before `attach_cost12` is called."""
    S.build_impute(S2.load_pop())
