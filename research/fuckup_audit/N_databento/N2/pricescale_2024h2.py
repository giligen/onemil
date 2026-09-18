#!/usr/bin/env python3
"""Stage N2 step 1c — price-scale check of the NEW 2024H2 Databento daily file vs Alpaca.

200 random (symbol, day) keys present in BOTH `data/research/databento/equs_daily_2024H2.parquet`
and `data/cache.db::daily_bars` (read-only), compared on OHLC + volume.  A Databento daily file that
is split/dividend ADJUSTED while the Alpaca bars are RAW would silently fabricate 52-week highs.
Reports the share of keys off by more than 0.5% (the threshold named in the N2 task).
"""
from __future__ import annotations

import os
import random
import re
import sqlite3
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, '/home/ec2-user/onemil')
os.chdir('/home/ec2-user/onemil')

N2 = 'research/fuckup_audit/N_databento/N2'
NEW = 'data/research/databento/equs_daily_2024H2.parquet'
CACHE = 'data/cache.db'
TEST_TICKER = re.compile(r'^Z[A-Z]ZZT$')
N_KEYS, SEED = 200, 17


def log(m):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {m}', flush=True)


def main() -> int:
    d = pd.read_parquet(NEW)
    d = d[d.symbol.notna()]
    log(f'{len(d):,} rows, {d.symbol.nunique():,} symbols')
    con = sqlite3.connect(f'file:{CACHE}?mode=ro', uri=True, timeout=180)
    n_cache = con.execute("select count(*) from daily_bars where bar_date between "
                          "'2024-07-01' and '2024-12-31'").fetchone()[0]
    log(f'cache.db daily_bars rows in 2024H2: {n_cache:,}')
    random.seed(SEED)
    idx = list(range(len(d)))
    random.shuffle(idx)
    sym = d.symbol.to_numpy()
    bd = d.bar_date.to_numpy()
    o, h, l, c, v = (d[k].to_numpy() for k in ('open', 'high', 'low', 'close', 'volume'))
    rows, tried = [], 0
    for i in idx:
        if len(rows) >= N_KEYS:
            break
        s = str(sym[i])
        if TEST_TICKER.match(s):
            continue
        tried += 1
        r = con.execute('select open, high, low, close, volume from daily_bars '
                        'where symbol=? and bar_date=?', (s, str(bd[i]))).fetchone()
        if not r:
            continue
        rows.append(dict(symbol=s, bar_date=str(bd[i]), p_open=float(o[i]), p_high=float(h[i]),
                         p_low=float(l[i]), p_close=float(c[i]), p_vol=float(v[i]),
                         a_open=float(r[0]), a_high=float(r[1]), a_low=float(r[2]),
                         a_close=float(r[3]), a_vol=float(r[4])))
    con.close()
    ps = pd.DataFrame(rows)
    for k in ('open', 'high', 'low', 'close'):
        ps[f'd_{k}'] = (ps[f'p_{k}'] / ps[f'a_{k}'] - 1.0) * 100.0
    ps['d_vol'] = (ps.p_vol / ps.a_vol - 1.0) * 100.0
    ps.to_csv(f'{N2}/pricescale_2024h2.csv', index=False)

    L = ['# N2 step 1c — price-scale check, 2024H2 Databento daily vs Alpaca `daily_bars`', '',
         f'Generated {datetime.now(timezone.utc).isoformat(timespec="seconds")} by '
         '`research/fuckup_audit/N_databento/N2/pricescale_2024h2.py`.', '',
         f'{len(ps)} keys matched out of {tried:,} sampled panel rows (cache.db holds '
         f'{n_cache:,} daily rows in 2024H2 — the bull-flag universe only, so most panel rows have '
         'no Alpaca counterpart). `diff = databento / alpaca - 1`, percent.', '',
         '| field | within 0.01% | within 0.5% | within 1% | median abs | p95 abs | max abs | '
         '**off by > 0.5%** |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for k in ('open', 'high', 'low', 'close', 'vol'):
        g = ps[f'd_{k}'].replace([np.inf, -np.inf], np.nan).dropna()
        if not len(g):
            continue
        L.append(f'| {k} | {(g.abs() <= 0.01).mean() * 100:.1f}% | {(g.abs() <= 0.5).mean() * 100:.1f}% '
                 f'| {(g.abs() <= 1.0).mean() * 100:.1f}% | {g.abs().median():.4f}% | '
                 f'{g.abs().quantile(0.95):.4f}% | {g.abs().max():.3f}% | '
                 f'**{(g.abs() > 0.5).mean() * 100:.1f}%** |')
    bad = ps[ps[['d_open', 'd_high', 'd_low', 'd_close']].abs().max(axis=1) > 0.5]
    L += ['', f'Keys off by more than 0.5% on ANY of OHLC: **{len(bad)} of {len(ps)} '
              f'({len(bad) / max(len(ps), 1) * 100:.1f}%)**.']
    if len(bad):
        L += ['', '| symbol | day | db close | alpaca close | diff % |', '|---|---|---:|---:|---:|']
        for _, r in bad.head(20).iterrows():
            L.append(f'| {r.symbol} | {r.bar_date} | {r.p_close:.4f} | {r.a_close:.4f} | '
                     f'{r.d_close:+.3f}% |')
    L += ['', 'Rows: `pricescale_2024h2.csv`.']
    open(f'{N2}/pricescale_2024h2.md', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
