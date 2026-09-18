#!/usr/bin/env python3
"""Stage N3 diagnostic — is the daily-file ATR the paper's ATR?

XNAS.ITCH ohlcv-1d aggregates the whole feed day (04:00-20:00 ET), so its high/low include
extended-hours prints.  The paper's ATR14 comes from a daily bar file (IQFeed).  If ours is wider,
R = 0.10 x ATR14 is too big and every R-denominated number shrinks toward zero.

Measured directly on the names we actually trade: daily-file range vs the 09:30-15:59 range from
the 1-min tape, same symbol-days.  Prints the median ratio -> sim.py --atr-scale.
"""
from __future__ import annotations

import os
import sqlite3
import sys

import pandas as pd
import pyarrow.parquet as pqf

sys.path.insert(0, '/home/ec2-user/onemil')
os.chdir('/home/ec2-user/onemil')

N3 = 'research/fuckup_audit/N_databento/N3'


def main() -> int:
    con = sqlite3.connect(f'file:{N3}/tape.db?mode=ro', uri=True)
    rth = pd.read_sql('select day, symbol, max(h) hi, min(l) lo, count(*) n from bars '
                      'where m between 570 and 959 group by day, symbol', con)
    con.close()
    rth['symbol'] = rth.symbol.astype(str)
    keys = set(zip(rth.day, rth.symbol))
    print(f'tape symbol-days: {len(rth):,}', flush=True)

    rows = []
    pf = pqf.ParquetFile(f'{N3}/xnas_daily.parquet')
    for b in pf.iter_batches(batch_size=1_000_000,
                             columns=['bar_date', 'symbol', 'high', 'low', 'close']):
        d = b.to_pandas()
        d['symbol'] = d.symbol.astype(str)
        d = d[[k in keys for k in zip(d.bar_date, d.symbol)]]
        if len(d):
            rows.append(d)
    day = pd.concat(rows, ignore_index=True)
    j = rth.merge(day, left_on=['day', 'symbol'], right_on=['bar_date', 'symbol'], how='inner')
    j = j[(j.hi > j.lo) & (j.high > j.low)]
    j['ratio'] = (j.high - j.low) / (j.hi - j.lo)
    q = j.ratio.quantile([.25, .5, .75])
    lines = ['# daily-file range vs RTH range on the traded names', '',
             f'matched symbol-days: {len(j):,}',
             f'median (feed-day range / RTH range) = **{q[.5]:.3f}**  '
             f'(p25 {q[.25]:.3f}, p75 {q[.75]:.3f})',
             '',
             f'-> ATR14 correction for an RTH-only ATR: `--atr-scale {1 / q[.5]:.3f}`']
    open(f'{N3}/atr_scale.md', 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
