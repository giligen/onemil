#!/usr/bin/env python3
"""hod_fresh — supplementary diagnostics (NOT cells).  Read-only.  TEST sealed."""
import os, sqlite3, sys
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT); sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_break')
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_filter_stack')
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_fresh')
import score as S                                              # noqa: E402
import score2 as S2                                            # noqa: E402
import score4 as S4                                            # noqa: E402
from research.scripts.pit_listings import is_test_ticker       # noqa: E402

D = f'{ROOT}/research/mature_method/hod_fresh'
RD = dict(dtype={'symbol': str, 'day': str}, keep_default_na=False, na_values=[''])
S.SPLITS = S2.SPLITS = S4.SPLITS = ('TRAIN', 'VAL')


def load():
    pop = S2.load_pop()
    S.build_impute(pop)
    p = pd.read_csv(f'{D}/sig3.csv', **RD)
    p = p[~p.day.isin(S4.EARLY_CLOSE)]
    p = p[~p.symbol.map(is_test_ticker)]
    con = sqlite3.connect(f'file:{ROOT}/data/cache.db?mode=ro', uri=True, timeout=120)
    dbs = set(pd.read_sql('select distinct symbol from daily_bars', con).symbol.astype(str))
    con.close()
    p = p[p.symbol.isin(dbs)].reset_index(drop=True)
    p['split'] = S.split_of(p.day.values)
    p['wk'] = pd.to_datetime(p.day).dt.to_period('W-FRI').astype(str)
    nb = pd.read_csv(f'{ROOT}/research/bf_zero/causal_filter/nbbo.csv', **RD
                     ).drop_duplicates(['day', 'symbol', 'entry_m'])
    p = p.merge(nb[['day', 'symbol', 'entry_m', 'spread_mean', 'ask_dec', 'bid_dec', 'n_sig']],
                on=['day', 'symbol', 'entry_m'], how='left')
    df = pd.read_csv(f'{ROOT}/research/mature_method/hod_preopen_regime/day_fields.csv',
                     dtype={'day': str}, keep_default_na=False, na_values=[''])
    p = p.merge(df[['day', 'spy_r5_pct']], on='day', how='left')
    u = pd.read_csv(f'{ROOT}/research/bf_zero/universe.csv',
                    usecols=['symbol', 'bar_date', 'open', 'high', 'low'],
                    dtype={'symbol': str, 'bar_date': str}, keep_default_na=False, na_values=[''])
    u = u.rename(columns={'bar_date': 'day'})
    for k in ('open', 'high', 'low'):
        u[k] = pd.to_numeric(u[k], errors='coerce')
    u['day_range_pct'] = (u.high - u.low) / u.open * 100.0
    p = p.merge(u[['symbol', 'day', 'day_range_pct']].drop_duplicates(['symbol', 'day']),
                on=['symbol', 'day'], how='left')
    return pop, p.sort_values(['day', 'symbol', 'entry_m'], kind='mergesort').reset_index(drop=True)


def main():
    pop, p = load()

    # ---- S1: WHERE THE +0.2151 R LIVES -------------------------------------------------------
    print('\n== S1 — the +0.2151 R constant, located ==')
    print('cost in R per trade = rr - net = half x (1 + exit ratio); the gates that remove the')
    print('expensive trades are part of the shipped book, so the BOOKED cost is far below the')
    print('POPULATION cost. Both are printed; only the booked one is a break-even for a cell.')
    for lab, kw in (('B0 pre-book, NO cost gates', dict(max_bps=None, max_frac_r=None, obtain=False)),
                    ('B0 pre-book, +100bps gate', dict(max_bps=100.0, max_frac_r=None, obtain=False)),
                    ('B0 pre-book, +15%-of-R gate', dict(max_bps=100.0, max_frac_r=0.15, obtain=False)),
                    ('B0 pre-book, shipped (+obtain)', dict(max_bps=100.0, max_frac_r=0.15, obtain=True))):
        x = S2.sig_set(pop, **{**S2.BASES['B0'], **kw})
        for sp in ('TRAIN', 'VAL'):
            d = x[x.split == sp]
            print(f'  {lab:<32s} {sp:5s} n {len(d):6d}  cost/R {float((d.rr-d.net).mean()):.4f}  '
                  f'median R% {float(d.r_pct.median()):.2f}  gross {d.rr.mean():+.4f}')

    # ---- S2: the A-ladder at the SIGNAL level (frequency frontier) ---------------------------
    print('\n== S2 — admission ladder, pre-book signals (the frequency the ADMISSION creates) ==')
    for r in S4.RUNGS:
        row = []
        for sp in ('TRAIN', 'VAL'):
            pre = S4.sig_set4(p, rung=r, stop='n5')
            d = pre[pre.split == sp]
            nwk = S.NW[sp]
            row.append(f'{sp} n {len(d):5d} ({len(d)/nwk:5.1f}/wk sig) gross {d.rr.mean():+.4f}')
        print(f'  {r:6s} ' + ' | '.join(row))

    # ---- S3: B-i == B-ii5 on the ge20 rung, asserted ----------------------------------------
    g = p[p.first_ge20 == 1]
    same = (g.stop_b.round(6) == g.stop_n5.round(6)) | (g.stop_b.isna() & g.stop_n5.isna())
    print(f'\n== S3 — on the ge20 rung the SHIPPED consolidation low IS the last-5-bar low on '
          f'{same.mean():.4%} of rows ==')
    print('  (consol_bars >= 20 means the last 20 bar lows are all within 4% of the level, so the')
    print('   shipped K5/X4% proximity test is satisfied by construction. B-i and B-ii5 are the')
    print('   same cell on this rung — not a bug, a consequence of the admission.)')

    # ---- S4: the selected cell C1, in detail -------------------------------------------------
    base = S4.sig_set4(p, rung='ge20', stop='n5')
    c1 = S.apply_book(base[(base.spy_r5_pct > 0).fillna(False)], 12, 4)
    print('\n== S4 — C1 (ge20 x last-5-bar low x SPY 09:35 up): weekly dollars at $100 risk ==')
    for sp in ('TRAIN', 'VAL'):
        d = c1[c1.split == sp]
        w = d.groupby('wk').pnl.sum().reindex(S.ALL_WEEKS[sp]).fillna(0.0)
        n = d.groupby('wk').size().reindex(S.ALL_WEEKS[sp]).fillna(0).astype(int)
        print(f'  {sp}:  ' + ' '.join(f'{v:+.0f}({c})' for v, c in zip(w.values, n.values)))
        print(f'    green {int((w>0).sum())}/{len(w)}  total {w.sum():+.0f}  worst {w.min():+.0f}  '
              f'best {w.max():+.0f}')
    print('\n== S4b — C1 ex-tail (diagnostic, never a rejection reason) ==')
    for sp in ('TRAIN', 'VAL'):
        d = c1[c1.split == sp]
        e1 = d.net[d.net <= d.net.quantile(0.99)].mean()
        e5 = d.net[d.net <= d.net.quantile(0.95)].mean()
        print(f'  {sp} net {d.net.mean():+.4f} -> ex-top-1% {e1:+.4f} -> ex-top-5% {e5:+.4f}')

    # ---- S5: the C1 gate's own separation, iid vs day-clustered ------------------------------
    print('\n== S5 — the D2 gate on the ge20 x n5 population: kept minus rejected ==')
    for sp in ('TRAIN', 'VAL'):
        d = base[base.split == sp]
        k = d[(d.spy_r5_pct > 0).fillna(False)]; r = d[~(d.spy_r5_pct > 0).fillna(False)]
        dm = float(k.rr.mean() - r.rr.mean())
        se = float(np.sqrt(k.rr.var(ddof=1) / len(k) + r.rr.var(ddof=1) / len(r)))
        # cluster-robust on the difference: day-level means of the demeaned contributions
        z = pd.concat([k.assign(w=1.0 / len(k)), r.assign(w=-1.0 / len(r))])
        gsum = (z.rr * z.w * len(z)).groupby(z.day.values).sum()
        se_c = float(np.sqrt((gsum ** 2).sum())) / len(z)
        print(f'  {sp}: kept n {len(k)} gross {k.rr.mean():+.4f} | rejected n {len(r)} '
              f'gross {r.rr.mean():+.4f} | sep {dm:+.4f}  iid t {dm/se:+.2f}  clustered t '
              f'{dm/se_c:+.2f}')

    # ---- S6: green-week resolution ----------------------------------------------------------
    print('\n== S6 — green-week MDE (95% half-width of a binomial share) ==')
    for sp in ('TRAIN', 'VAL'):
        n = S.NW[sp]
        print(f'  {sp}: {n} market weeks -> +/- {1.96*np.sqrt(0.25/n)*100:.1f} pp')
    print('\nDONE')


if __name__ == '__main__':
    main()
