#!/usr/bin/env python3
"""frames16 ARM 3 — MEASURE the NBBO on the mirror-short population's OWN minutes.

The deciding caveat of arm 3: the cost charged in `score3.py` is the frames14 F45 minute-of-day
MEDIAN, measured on 210 symbol-days of the **HOD-break** population. The mirror-short population is
a different set of names (already +5 % on the day, 3x their own hourly volume, a >2 % hour), and if
its true spread is wider the cell's net evaporates. F45 was a proxy; this measures the real thing.

Alpaca SIP consolidated quotes, mean and median (ask - bid) over the one-minute window, the
Stage-P / frames13 / F45 convention — same code shape as `frames14/f45_fetch.py`, so the two numbers
are directly comparable. A stratified random sample of the cell's own filled trades, both legs.

Also prints the (a)-check: `frames15` B12's LONG gross restricted to this arm's exact universe
(gate5, price >= $5, ex-wrapper), so the short can be compared with its own mirror rather than with
a number computed on a different population.
"""
import glob
import os
import sys
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/research/mature_method/frames15')
from common15 import attach_instrument, clust_t1                      # noqa: E402
from scoreB_intra import cost_at, cost_table                          # noqa: E402
from alpaca.data.historical import StockHistoricalDataClient          # noqa: E402
from alpaca.data.requests import StockQuotesRequest                   # noqa: E402
from alpaca.data.enums import DataFeed                                # noqa: E402
from config import Config                                             # noqa: E402

D = f'{ROOT}/research/mature_method/frames16'
D15 = f'{ROOT}/research/mature_method/frames15'
OUT = f'{D}/cost_check.csv'
ET = ZoneInfo('America/New_York')
SEED = 16
N_SAMPLE = int(os.environ.get('F16_COST_N', '350'))


def minute_spread(cl, sym, day, m):
    t0 = datetime.strptime(day, '%Y-%m-%d').replace(hour=int(m) // 60, minute=int(m) % 60,
                                                    tzinfo=ET)
    try:
        q = cl.get_stock_quotes(StockQuotesRequest(
            symbol_or_symbols=sym, start=t0, end=t0 + timedelta(minutes=1),
            feed=DataFeed.SIP, limit=6000)).data.get(sym, [])
    except Exception as e:
        return np.nan, np.nan, 0, str(e)[:40]
    sp, mid = [], []
    for x in q:
        if x.ask_price and x.bid_price and x.ask_price > x.bid_price:
            sp.append(float(x.ask_price) - float(x.bid_price))
            mid.append((float(x.ask_price) + float(x.bid_price)) / 2.0)
    if not sp:
        return np.nan, np.nan, 0, ''
    return float(np.mean(sp)), float(np.median(sp)), len(sp), ''


def the_a_check():
    """frames15 B12 LONG gross on THIS arm's exact universe."""
    fs = sorted(glob.glob(f'{D15}/intra_*.csv'))
    d = pd.concat([pd.read_csv(f, dtype={'symbol': str, 'day': str}) for f in fs],
                  ignore_index=True)
    d = d[(d.day < '2026-06-01') & d.gate5.astype(bool) & d.f_mir.astype(bool)]
    d = attach_instrument(d)
    d['split'] = np.where(d.day < '2026-01-01', 'TRAIN', 'VAL')
    for lab, z in (('B12 as published (all prices, wrappers in)', d),
                   ('B12 on the short arm universe (>=$5, ex-wrapper)',
                    d[(d.price >= 5) & (d.asset_class != 'wrapper')])):
        line = f'  {lab:52s}'
        for sp in ('TRAIN', 'VAL'):
            x = z[z.split == sp]
            mu, t = clust_t1(x.rr_bare.values, x.day.values)
            line += f' | {sp} n={len(x):5,} {mu:+.3f} R (t {t:+.2f})'
        print(line, flush=True)


def main():
    print('== (a) the LONG mirror on the short arm\'s own universe ==', flush=True)
    the_a_check()

    fs = sorted(glob.glob(f'{D}/sw_*.csv'))
    d = pd.concat([pd.read_csv(f, dtype={'symbol': str, 'day': str},
                               keep_default_na=False, na_values=['']) for f in fs],
                  ignore_index=True)
    d = d[(d.day < '2026-06-01') & d.gate5.astype(bool) & (d.price >= 5)
          & d.f_up2.astype(bool)].copy()
    d = attach_instrument(d)
    d = d[d.asset_class != 'wrapper']
    d['split'] = np.where(d.day < '2026-01-01', 'TRAIN', 'VAL')
    rng = np.random.default_rng(SEED)
    take = []
    for sp in ('TRAIN', 'VAL'):
        z = d[d.split == sp]
        k = min(N_SAMPLE // 2, len(z))
        take.append(z.iloc[rng.choice(len(z), size=k, replace=False)])
    s = pd.concat(take).reset_index(drop=True)
    print(f'\n== (b) measured NBBO on {len(s)} sampled S3 trades x 2 legs ==', flush=True)

    done = set()
    if os.path.exists(OUT):
        d0 = pd.read_csv(OUT, dtype={'day': str, 'symbol': str})
        done = set(zip(d0.day, d0.symbol, d0.leg_m))
    cl = StockHistoricalDataClient(Config().alpaca_api_key, Config().alpaca_api_secret)
    rows = []
    for i, r in enumerate(s.itertuples()):
        for leg, m in (('entry', int(r.entry_m)), ('exit', int(r.exitm_a_bare))):
            if (r.day, r.symbol, m) in done:
                continue
            mean_sp, med_sp, n, err = minute_spread(cl, r.symbol, r.day, m)
            rows.append(dict(day=r.day, symbol=r.symbol, split=r.split, leg=leg, leg_m=m,
                             price=r.price, rpct=r.rpct_a, sp_mean=mean_sp, sp_med=med_sp,
                             n_q=n, err=err))
        if len(rows) >= 40:
            pd.DataFrame(rows).to_csv(OUT, mode='a', header=not os.path.exists(OUT), index=False)
            rows = []
        if i % 25 == 0:
            print(f'  {i+1}/{len(s)}', flush=True)
            time.sleep(0.2)
    if rows:
        pd.DataFrame(rows).to_csv(OUT, mode='a', header=not os.path.exists(OUT), index=False)

    q = pd.read_csv(OUT, dtype={'day': str, 'symbol': str})
    q = q[q.n_q > 0].copy()
    q['sp_pct'] = q.sp_med / q.price * 100.0
    ct = cost_table()
    q['imp_pct'] = cost_at(ct, q.leg_m.values) * 100.0
    print(f'\n  coverage {len(q):,} of {len(q)+int((pd.read_csv(OUT).n_q==0).sum()):,} legs '
          f'fetched with quotes', flush=True)
    for leg in ('entry', 'exit'):
        z = q[q.leg == leg]
        print(f'  {leg:5s} n={len(z):5,}  MEASURED median spread {z.sp_pct.median():.3f} % of '
              f'price (mean {z.sp_pct.mean():.3f})  vs F45 IMPUTED {z.imp_pct.median():.3f} %  '
              f'-> ratio {z.sp_pct.median()/z.imp_pct.median():.2f}x', flush=True)
    for sp in ('TRAIN', 'VAL'):
        z = q[q.split == sp]
        print(f'  {sp:5s} measured/imputed median ratio '
              f'{(z.sp_pct/z.imp_pct).median():.2f}x', flush=True)
    # the cost in R on the sampled trades, measured vs imputed
    piv = q.pivot_table(index=['day', 'symbol'], columns='leg',
                        values=['sp_pct', 'imp_pct', 'rpct'], aggfunc='first').dropna()
    meas_R = 0.5 * (piv[('sp_pct', 'entry')] + piv[('sp_pct', 'exit')]) / 100.0 / \
        piv[('rpct', 'entry')]
    imp_R = 0.5 * (piv[('imp_pct', 'entry')] + piv[('imp_pct', 'exit')]) / 100.0 / \
        piv[('rpct', 'entry')]
    print(f'\n  cost in R on the sampled trades: MEASURED mean {meas_R.mean():.3f} '
          f'(median {meas_R.median():.3f})  vs IMPUTED mean {imp_R.mean():.3f} '
          f'(median {imp_R.median():.3f})  -> ratio {meas_R.mean()/imp_R.mean():.2f}x', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
