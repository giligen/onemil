#!/usr/bin/env python3
"""frames15 ARM B (multi-day) — the volume fields as a STANDALONE detector, auction entry leg.

Cells B1..B8 of PREREG §2. Population: the dense point-in-time daily panel, test tickers and names
absent from `daily_bars` removed, close >= $5, ADV$ >= $1M, the corporate-action rail applied.
Entry = the decision session's CLOSING AUCTION (no quoted spread, frames13 F42), exit = the closing
auction h sessions later, financing charged at 7.0 % APR per night. Results in % of price first,
R second (R = the declared 2 % stop). TEST never opened: a cell drops any decision session whose
EXIT session would land on/after 2026-06-01.

Also prints the placebo decomposition (D1 the universe bound), the wrapper-enrichment check, the
base rates, ex-top-5 % and the MDE.

Memory: the big frame never holds an object column (symbol/day are categories, the session index
`di` is an int32) — the 3 GB `ulimit -v` rail.
"""
import sqlite3
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '/home/ec2-user/onemil/research/mature_method/frames15')
from common15 import (D, ROOT, STOP_PCT, clust_t1, mde_pct,               # noqa: E402
                      attach_instrument, week_shape, null_green)
sys.path.insert(0, ROOT)
from research.scripts.pit_listings import is_test_ticker                  # noqa: E402

APR = 0.07
TEST_FROM = '2026-06-01'
COLS = ['symbol', 'date', 'close', 'adv20d', 'rvd', 'interest5', 'ret5', 'weak',
        'fwd1', 'fwd2', 'fwd5', 'ca_bad']
SP = {'TRAIN': 0, 'VAL': 1}

CELLS = [
    ('B1', 'interest5 >= 3', 1, lambda d: d.interest5 >= 3),
    ('B2', 'interest5 >= 3', 2, lambda d: d.interest5 >= 3),
    ('B3', 'interest5 >= 3', 5, lambda d: d.interest5 >= 3),
    ('B4', 'interest5 >= 4', 5, lambda d: d.interest5 >= 4),
    ('B5', 'V3xV4 interest5>=3 & |ret5|<=3%', 5,
     lambda d: (d.interest5 >= 3) & (d.ret5.abs() <= 0.03)),
    ('B6', 'V3xV4 interest5>=3 & |ret5|<=3%', 2,
     lambda d: (d.interest5 >= 3) & (d.ret5.abs() <= 0.03)),
    ('B7', 'V5 accumulation on weakness', 2, lambda d: d.weak.astype(bool)),
    ('B8', 'CONTROL rvd >= 3 single-session spike', 1, lambda d: d.rvd >= 3),
]


def load_universe():
    parts = []
    for y in ('2025', '2026'):
        t = pd.read_parquet(f'{D}/daily15_{y}.parquet', columns=COLS)
        t = t[(t.close >= 5.0) & (t.adv20d >= 1e6) & (~t.ca_bad.astype(bool))]
        parts.append(t)
        del t
    d = pd.concat(parts, ignore_index=True)
    del parts
    d['symbol'] = d.symbol.astype(str).astype('category')
    d['day'] = d.date.astype(str).astype('category')
    d = d.drop(columns=['date', 'ca_bad'])

    con = sqlite3.connect(f'file:{ROOT}/data/cache.db?mode=ro', uri=True, timeout=120)
    dbs = set(pd.read_sql('select distinct symbol from daily_bars', con).symbol.astype(str))
    con.close()
    cats = list(d.symbol.cat.categories)
    ok = {c for c in cats if c in dbs and not is_test_ticker(str(c))}
    n0 = len(d)
    d = d[d.symbol.isin(ok)].copy()
    d['symbol'] = d.symbol.cat.remove_unused_categories()
    d['day'] = d.day.cat.remove_unused_categories()
    print(f'[B] universe {n0:,} -> ex-(non-daily_bars, test tickers) {len(d):,} rows, '
          f'{len(d.symbol.cat.categories):,} symbols, {len(d.day.cat.categories)} sessions',
          flush=True)

    sessions = np.array(d.day.cat.categories)
    d['di'] = d.day.cat.codes.astype(np.int32)
    d['split'] = np.where(sessions[d.di.values] < '2026-01-01', 0, 1).astype(np.int8)
    ac = attach_instrument(pd.DataFrame({'symbol': list(d.symbol.cat.categories),
                                         'day': '2025-01-02'}))
    wrapset = set(ac.symbol[ac.asset_class.astype(str) == 'wrapper'].astype(str))
    d['wrap'] = d.symbol.isin(wrapset).values
    return d, sessions


def cell_stats(d, mask, h, tag, ok_di, mid):
    fin = APR / 365.0 * h
    m = mask.values & d[f'fwd{h}'].notna().values & ok_di[d.di.values]
    dd = d[m]
    out = {'cell': tag, 'h': h, 'n': len(dd)}
    for sp, code in SP.items():
        z = dd[dd.split == code]
        p = z[f'fwd{h}'].values.astype(float) - fin
        mu, t = clust_t1(p, z.di.values)
        out[f'{sp}_n'] = len(z)
        out[f'{sp}_pct'] = mu * 100
        out[f'{sp}_R'] = mu / STOP_PCT
        out[f'{sp}_t'] = t
        out[f'{sp}_mde'] = mde_pct(p, z.di.values) * 100
        out[f'{sp}_ex5'] = (float(np.nanmean(p[p <= np.nanquantile(p, 0.95)])) * 100
                            if len(p) > 20 else np.nan)
        out[f'{sp}_wrap'] = float(z.wrap.mean()) if len(z) else np.nan
    tr = dd[dd.split == 0]
    out['h1'] = float(np.nanmean(tr[tr.di < mid][f'fwd{h}'].values.astype(float) - fin)) * 100
    out['h2'] = float(np.nanmean(tr[tr.di >= mid][f'fwd{h}'].values.astype(float) - fin)) * 100
    return out, dd


def book(dd, h, nday=12, nconc=4):
    """First-come slot rule: <= nday new positions a session, <= nconc concurrent, hold h."""
    z = dd.sort_values(['di', 'interest5', 'symbol'], ascending=[True, False, True],
                       kind='mergesort')
    di = z.di.values
    keep = np.zeros(len(z), dtype=bool)
    free, cur, cnt = [], -1, 0
    for j in range(len(z)):
        i = di[j]
        if i != cur:
            cur, cnt = i, 0
            free = [x for x in free if x > i]
        if cnt >= nday or len(free) >= nconc:
            continue
        cnt += 1
        free.append(i + h)
        keep[j] = True
    return z[keep]


def main():
    d, sess = load_universe()
    n_sess = len(sess)
    mid = int(np.searchsorted(sess, '2025-07-01'))
    test_start = int(np.searchsorted(sess, TEST_FROM))

    print(f'[B] base rates: interest5>=3 {float((d.interest5 >= 3).mean())*100:.2f}% of universe '
          f'symbol-days, >=4 {float((d.interest5 >= 4).mean())*100:.2f}%, '
          f'rvd>=1.5 {float((d.rvd >= 1.5).mean())*100:.2f}%, '
          f'weak {float(d.weak.astype(bool).mean())*100:.2f}%', flush=True)
    print(f'[B] wrapper share of the universe: {float(d.wrap.mean())*100:.1f}%', flush=True)
    fire = d[d.interest5 >= 3].groupby('di', observed=True).size()
    alln = d.groupby('di', observed=True).size()
    print(f'[B] names firing interest5>=3 per session: median {fire.median():.0f} of '
          f'{alln.median():.0f} eligible ({fire.median()/alln.median()*100:.1f}%)', flush=True)

    print('\n== D1 the universe bound: the unconditional auction-to-auction hold ==', flush=True)
    dec = {}
    for h in (1, 2, 5):
        ok_di = np.array([(i + h < n_sess) and (i + h < test_start) for i in range(n_sess)])
        fin = APR / 365.0 * h
        for sp, code in SP.items():
            z = d[(d.split == code) & d[f'fwd{h}'].notna() & ok_di[d.di.values]]
            mu, t = clust_t1(z[f'fwd{h}'].values.astype(float) - fin, z.di.values)
            dec[(h, sp)] = mu
            print(f'  D1 h={h} {sp:5s} n={len(z):>9,} {mu*100:+.4f}% ({mu/STOP_PCT:+.3f} R) '
                  f't={t:+.2f}', flush=True)

    rows = []
    print('\n== B cells ==', flush=True)
    for tag, name, h, fn in CELLS:
        ok_di = np.array([(i + h < n_sess) and (i + h < test_start) for i in range(n_sess)])
        mask = fn(d).fillna(False)
        st, dd = cell_stats(d, mask, h, tag, ok_di, mid)
        st['name'] = name
        fin = APR / 365.0 * h
        for sp in SP:
            st[f'{sp}_excess'] = st[f'{sp}_pct'] - dec[(h, sp)] * 100
        print(f'  {tag} h={h} {name:34s} n={st["n"]:>7,}  '
              f'TRAIN {st["TRAIN_pct"]:+.4f}% (exc {st["TRAIN_excess"]:+.4f}, t {st["TRAIN_t"]:+.2f}, '
              f'mde {st["TRAIN_mde"]:.4f})  VAL {st["VAL_pct"]:+.4f}% (exc {st["VAL_excess"]:+.4f}, '
              f't {st["VAL_t"]:+.2f})  halves {st["h1"]:+.3f}/{st["h2"]:+.3f}  '
              f'wrap {st["TRAIN_wrap"]*100:.0f}%  ex5 {st["TRAIN_ex5"]:+.4f}', flush=True)

        bk = book(dd, h)
        bk = bk.assign(rr=(bk[f'fwd{h}'].values.astype(float) - fin) / STOP_PCT,
                       day=bk.day.astype(str),
                       split=np.where(bk.split.values == 0, 'TRAIN', 'VAL'))
        for sp in SP:
            w = week_shape(bk, sp)
            nl = null_green(bk, sp)
            st[f'bk_{sp}_n'] = w['n']
            st[f'bk_{sp}_wk'] = w['per_wk']
            st[f'bk_{sp}_green'] = w['green']
            st[f'bk_{sp}_total'] = w['total']
            st[f'bk_{sp}_worst'] = w['worst']
            st[f'bk_{sp}_streak'] = w['redstreak']
            st[f'bk_{sp}_null95'] = nl[2]
            print(f'      book {sp:5s} n={w["n"]:4d} ({w["per_wk"]:4.1f}/wk) green={w["green"]:5.1f}% '
                  f'(null p95 {nl[2]:5.1f}) ${w["total"]:+,.0f} wk ${w["wk_mean"]:+,.0f} '
                  f'worst ${w["worst"]:+,.0f} streak {w["redstreak"]}', flush=True)
        rows.append(st)
    pd.DataFrame(rows).to_csv(f'{D}/cellsB_multi.csv', index=False)
    print(f'\n[B] wrote cellsB_multi.csv ({len(rows)} cells)', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
