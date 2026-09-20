#!/usr/bin/env python3
"""F40 — THE OVERNIGHT FLOOR.  Close-to-next-open over the whole PIT daily panel.

  python3 _year.py <year>   # the per-year builder -> on_<year>.parquet (one short process each)
  python3 f40.py score      # the 16 cells + the map rows + the gap-risk distribution

Everything is declared in `frames13/PREREG.md` §1 BEFORE any cell was read.
Stores are opened READ-ONLY; nothing is written outside `frames13/`.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
from trading.regime_helpers import compute_regime_features, classify_regime   # noqa: E402

D13 = f'{ROOT}/research/mature_method/frames13'
PX = f'{ROOT}/research/multiday/data/prices_by_year'
TEST_FROM = '2026-06-01'
YEARS = list(range(2016, 2027))

PB_EDGES = [0, 5, 20, 50, 200, np.inf]
PB_LAB = ['<5', '5-20', '20-50', '50-200', '>=200']
AB_EDGES = [0, 1e6, 1e7, 1e8, np.inf]
AB_LAB = ['<1M', '1-10M', '10-100M', '>=100M']
APRS = (6.0, 7.0, 8.0)
APR_MID = 7.0


def margin_pct(nights, apr):
    """% of position financed overnight at Reg-T 2x — half the position is borrowed."""
    return apr / 360.0 * 0.5 * np.asarray(nights, dtype=float)


ERA_EDGES = [('PRE', '2025-01-01'), ('2025H1', '2025-07-01'), ('2025H2', '2026-01-01'),
             ('2026', TEST_FROM)]


def _dnum(s):
    return int(np.datetime64(s).astype('datetime64[D]').astype(np.int32))


def load(drop_ca=True):
    """The panel as a DICT OF NUMPY ARRAYS — no pandas frame, no python strings.

    17.4M rows must fit the node's 3 GB virtual cap: a pandas frame consolidates its float blocks
    with a vstack and dies, so the columns live in memory-mapped `col_*.npy` files written by
    `_merge.py` and every cell is a boolean mask over them.
    """
    C = {k: np.load(f'{D13}/col_{k}.npy', mmap_mode='r')
         for k in ('sid', 'd', 'close', 'adv20', 'on_pct', 'ca', 'nights')}
    sm = pd.read_csv(f'{D13}/symbols.csv', keep_default_na=False, na_values=[''])
    kinds = pd.Index(['common', 'fund', 'wrapper'])
    kmap = np.full(int(sm.sid.max()) + 1, -1, np.int8)
    kmap[sm.sid.values] = kinds.get_indexer(pd.Index(sm['kind'].astype(str))).astype(np.int8)

    # ONE boolean mask, built column by column off the memmaps, then applied once each.
    sid_all = C['sid'][:]
    m = kmap[sid_all] >= 0                          # test tickers / non-universe names out
    m &= C['d'][:] < _dnum(TEST_FROM)
    if drop_ca:
        m &= ~C['ca'][:]
    x = dict(sid=sid_all[m])
    del sid_all
    x['kind_i'] = kmap[x['sid']]
    for k, dt in (('d', np.int32), ('close', np.float64), ('adv20', np.float64),
                  ('on_pct', np.float64), ('nights', np.float64)):
        x[k] = C[k][m].astype(dt, copy=False)
    del C, m
    era = np.full(len(x['d']), 3, np.int8)
    era[x['d'] < _dnum('2026-01-01')] = 2
    era[x['d'] < _dnum('2025-07-01')] = 1
    era[x['d'] < _dnum('2025-01-01')] = 0
    x['era_i'] = era
    x['pb_i'] = np.searchsorted(np.array(PB_EDGES[1:-1], float), x['close'], 'right').astype(np.int8)
    x['ab_i'] = np.searchsorted(np.array(AB_EDGES[1:-1], float), x['adv20'], 'right').astype(np.int8)
    x['dow'] = ((x['d'] + 4) % 7).astype(np.int8)   # 1970-01-01 was a Thursday
    return x


def sub(x, m):
    """A cell — the same dict, masked."""
    return {k: v[m] for k, v in x.items()}


def date_of(dnum):
    return str(np.array(dnum, dtype='int32').astype('datetime64[D]'))


_SM = None


def sym_of(sid):
    global _SM
    if _SM is None:
        _SM = pd.read_csv(f'{D13}/symbols.csv', keep_default_na=False,
                          na_values=['']).set_index('sid').symbol
    return str(_SM.get(int(sid), '?'))


def stat(x, spread_pct=0.0, apr=APR_MID, ex_top5=False, label=''):
    """One scored row.  `spread_pct` is the ROUND-TRIP quoted spread charged (0 for auction legs)."""
    on = x['on_pct']
    if not len(on):
        return None
    v = on - spread_pct - margin_pct(x['nights'], apr)
    if ex_top5 and len(v) > 40:
        v = v[v <= np.quantile(v, 0.95)]
    n = len(v)
    m = float(np.mean(v))
    se = float(np.std(v, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    i = int(np.argmin(on))
    r = dict(cell=label, n=n, mean_pct=m, t=(m / se if se else np.nan),
             p1=float(np.percentile(on, 1)), p5=float(np.percentile(on, 5)),
             p50=float(np.percentile(on, 50)), p95=float(np.percentile(on, 95)),
             p99=float(np.percentile(on, 99)),
             lt_m5=float((on <= -5).mean() * 100), lt_m10=float((on <= -10).mean() * 100),
             worst=float(on[i]),
             worst_key=f"{sym_of(x['sid'][i])} {date_of(x['d'][i])}")
    for ei, e in enumerate(('PRE', '2025H1', '2025H2', '2026')):
        s = x['era_i'] == ei
        if s.sum():
            vv = on[s] - spread_pct - margin_pct(x['nights'][s], apr)
            if ex_top5 and len(vv) > 40:           # the cap rule applies INSIDE each era too
                vv = vv[vv <= np.quantile(vv, 0.95)]
            r[f'net_{e}'] = float(np.mean(vv))
            r[f'n_{e}'] = int(s.sum())
        else:
            r[f'net_{e}'] = np.nan
            r[f'n_{e}'] = 0
    return r


def spy_regime():
    """SPY A/B/C1/C2 per session from features strictly BEFORE it (the shipped engine rule)."""
    s = pd.read_csv(f'{D13}/spy_daily.csv', dtype={'date': str})
    f = compute_regime_features(s.rename(columns={'date': 'bar_date'}))
    f['regime'] = [classify_regime(v, a, sl) for v, a, sl in
                   zip(f.vol_20_ann, f.above_sma_50, f.sma_50_slope_10d)]
    f['day'] = f.bar_date.shift(-1)
    return dict(zip(f.day.dropna(), f.regime))


def score():
    x = load()
    print(f"panel {len(x['d']):,} name-nights (corporate actions dropped), "
          f"{len(np.unique(x['d'])):,} sessions {date_of(x['d'].min())} .. "
          f"{date_of(x['d'].max())}", flush=True)
    ca = np.load(f'{D13}/col_ca.npy', mmap_mode='r')[:]
    dd = np.load(f'{D13}/col_d.npy', mmap_mode='r')[:] < _dnum(TEST_FROM)
    onx = np.load(f'{D13}/col_on_pct.npy', mmap_mode='r')
    n_all = int((dd).sum())
    mean_all = float(onx[dd].mean())
    del ca, dd, onx
    print(f"  with corporate actions {n_all:,} "
          f"({(n_all - len(x['d'])) / n_all * 100:.2f} % dropped, incl. the non-universe names); "
          f"unconditional mean with them {mean_all:+.4f} % vs {x['on_pct'].mean():+.4f} % "
          f"— the price-scale rail", flush=True)

    sp = 0.0
    try:
        q = pd.read_csv(f'{D13}/openspread.csv')
        q = q[q.rt_pct.notna()]
        # A14 is A13's cell, so its cost is the A13-like slice of the stratified sample — the
        # whole sample over-weights the microcap strata by construction.
        qa = q[(q.close >= 20) & q.ab.isin(['10-100M', '>=100M'])]
        sp = float(qa.rt_pct.median())
        print(f'  MEASURED SIP NBBO, {len(q)} sampled name-nights (0 errors): round-trip '
              f'marketable cost (15:55 + 09:31) median {q.rt_pct.median():.4f} % of price over the '
              f'whole stratified sample, {sp:.4f} % on the {len(qa)} A13-like rows; the OPENING '
              f'MINUTE (09:30-09:31) quotes {q.open_cross_pct.median():.4f} % / '
              f'{qa.open_cross_pct.median():.4f} % of price — the opening cross\'s own liquidity '
              f'risk, and it is NOT free', flush=True)
    except Exception as e:
        print(f'  WARNING leg-(b) spread sample missing ({e}) — A14 printed GROSS', flush=True)

    KC, KW = 0, 2                       # the `kinds` index in load(): common, fund, wrapper
    # the cells are built ONE AT A TIME — holding all sixteen masked copies of a 16.5M-row panel
    # at once is what the node's 3 GB cap forbids.
    A13m = (x['kind_i'] == KC) & (x['close'] >= 20) & (x['adv20'] >= 1e7)
    cells = [
        ('A1 whole panel', lambda: None, 0.0, False),
        ('A2 price <5', lambda: x['pb_i'] == 0, 0.0, False),
        ('A3 price 5-20', lambda: x['pb_i'] == 1, 0.0, False),
        ('A4 price 20-50', lambda: x['pb_i'] == 2, 0.0, False),
        ('A5 price 50-200', lambda: x['pb_i'] == 3, 0.0, False),
        ('A6 price >=200', lambda: x['pb_i'] == 4, 0.0, False),
        ('A7 ADV$ <1M', lambda: x['ab_i'] == 0, 0.0, False),
        ('A8 ADV$ 1-10M', lambda: x['ab_i'] == 1, 0.0, False),
        ('A9 ADV$ 10-100M', lambda: x['ab_i'] == 2, 0.0, False),
        ('A10 ADV$ >=100M', lambda: x['ab_i'] == 3, 0.0, False),
        ('A11 wrapper', lambda: x['kind_i'] == KW, 0.0, False),
        ('A12 common', lambda: x['kind_i'] == KC, 0.0, False),
        ('A13 common,>=$20,ADV$>=10M', lambda: A13m, 0.0, False),
        ('A14 A13 leg(b) marketable', lambda: A13m, sp, False),
        ('A15 A13 ex-top5%', lambda: A13m, 0.0, True),
        ('A16 A13 2016-2024', lambda: A13m & (x['era_i'] == 0), 0.0, False),
        # diagnostics, NOT scored cells: the cap rule on the broader cells
        ('d1 A1 ex-top5%', lambda: None, 0.0, True),
        ('d2 A2 ex-top5%', lambda: x['pb_i'] == 0, 0.0, True),
        ('d3 A7 ex-top5%', lambda: x['ab_i'] == 0, 0.0, True),
        ('d4 A12 ex-top5%', lambda: x['kind_i'] == KC, 0.0, True),
    ]
    rows = []
    for lab, mf, spc, ex in cells:
        m = mf()
        c = x if m is None else sub(x, m)
        r = stat(c, spread_pct=spc, ex_top5=ex, label=lab)
        if r:
            rows.append(r)
        del c, m
    A13 = sub(x, A13m)
    t = pd.DataFrame(rows)
    t.to_csv(f'{D13}/cells40.csv', index=False)
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 40)
    print('\n== THE 16 SCORED CELLS — net of margin @ APR 7.0 %, in % of price ==')
    print(t[['cell', 'n', 'mean_pct', 't', 'net_PRE', 'net_2025H1', 'net_2025H2', 'net_2026']]
          .to_string(index=False, float_format=lambda v: f'{v:+.4f}'))
    print('\n== GAP RISK per cell (% of price, on the GROSS overnight move) ==')
    print(t[['cell', 'p1', 'p5', 'p50', 'p95', 'p99', 'lt_m5', 'lt_m10', 'worst', 'worst_key']]
          .to_string(index=False))

    print('\n== MARGIN SENSITIVITY on A13 ==')
    for apr in APRS:
        mg = margin_pct(A13['nights'], apr)
        print(f"  APR {apr:.1f} %: net {A13['on_pct'].mean() - mg.mean():+.4f} % | "
              f'mean charge {mg.mean():.4f} % | median charge {np.median(mg):.4f} %')

    dates = pd.to_datetime(A13['d'].astype('datetime64[D]')).strftime('%Y-%m-%d')
    A = pd.DataFrame(dict(on=A13['on_pct'], nights=A13['nights'], era=A13['era_i'],
                          dow=A13['dow'], date=dates))
    A['net'] = A.on - margin_pct(A.nights.values, APR_MID)
    A['regime'] = A.date.map(spy_regime())
    print('\n== MAP ROWS (diagnostics on A13, not scored cells) ==')
    for key, name in (('dow', 'weekday 0=Mon'), ('regime', 'SPY regime')):
        g = A.groupby(key, observed=True).agg(n=('on', 'size'), gross=('on', 'mean'),
                                              median=('on', 'median'), net=('net', 'mean'))
        print(f'\n-- {name} --')
        print(g.to_string(float_format=lambda v: f'{v:+.4f}'))
    A['mon'] = A.date.str[5:7]
    print('\n-- month --')
    print(A.groupby('mon').agg(n=('on', 'size'), gross=('on', 'mean'),
                               net=('net', 'mean')).to_string(float_format=lambda v: f'{v:+.4f}'))
    A['yr'] = A.date.str[:4]
    print('\n== ERA STABILITY of A13 by calendar year ==')
    g = A.groupby('yr').agg(n=('on', 'size'), gross=('on', 'mean'), net=('net', 'mean'))
    g['ex_top5'] = A.groupby('yr').net.apply(
        lambda v: v[v <= np.quantile(v, 0.95)].mean() if len(v) > 40 else np.nan)
    print(g.to_string(float_format=lambda v: f'{v:+.4f}'))
    g.to_csv(f'{D13}/a13_by_year.csv')

    print('\n== A13 ex-top-5 % by era (F35 cap rule) ==')
    for ei, e in enumerate(('PRE', '2025H1', '2025H2', '2026')):
        v = A[A.era == ei].net.values
        if len(v) > 40:
            print(f'  {e}: n {len(v):,} net {v.mean():+.4f} % | '
                  f'ex-top5% {v[v <= np.quantile(v, 0.95)].mean():+.4f} %')


if __name__ == '__main__':
    {'score': score}[sys.argv[1]]()
