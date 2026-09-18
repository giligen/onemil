#!/usr/bin/env python3
"""Stage P step 5 — where the measured cost sits (entry vs exit), and the
tail/cap checks PLAN §1 requires on every number that reaches the owner.

Usage: python3 decompose.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/research/fuckup_audit/P_cost')
from trading.orb_csv import read_orb_csv      # noqa: E402
from rescore import range_lookup, split_of, _old_pos, EXIT_SLIP, ARMS  # noqa: E402

P = f'{ROOT}/research/fuckup_audit/P_cost'
OLD_POS = _old_pos()


def main() -> int:
    sp = pd.read_parquet(f'{P}/spreads.parquet')
    sp['key'] = sp.symbol + '|' + sp.date
    b = read_orb_csv(f'{P}/book_asis_n8.csv')
    b['date'] = pd.to_datetime(b['date']).dt.strftime('%Y-%m-%d')
    b['key'] = b.symbol + '|' + b.date
    f = b[b.entered.astype(float) != 0].merge(sp, on='key', how='left',
                                              suffixes=('', '_sp'))
    print(f'book fills {len(f)}, spread rows joined {int(f.entry_spread.notna().sum())}')

    sh = np.maximum(1, (OLD_POS / f.entry_price).astype(int))
    scale = f['_rp_position'] / OLD_POS            # the book's own $ scaling
    cap = f.entry_price
    ask = f.entry_ask
    # entry concession vs the shipped assumption (positive = measured CHEAPER)
    ent_gain = (cap - np.minimum(ask, cap)) * sh * scale
    lvl = np.where(f.exit_reason.isin(['stop', 'lock', 'scale_stop', 'scale_lock']),
                   f.exit_price / (1 - EXIT_SLIP), np.nan)
    # exit concession vs the shipped assumption (positive = measured WORSE)
    ex_loss = (f.exit_price - f.exit_bid) * sh * scale
    ex_loss = ex_loss.where(~f.exit_reason.astype(str).str.startswith('scale_'))
    print(f'\nmeasured vs shipped, book fills (8 slots, $ at book sizing):')
    print(f'  entry  measured cheaper by  ${np.nansum(ent_gain):,.0f} '
          f'(median {np.nanmedian(ent_gain):.1f}/trade; ask <= cap on '
          f'{float((ask <= cap).mean()) * 100:.1f}% of fills)')
    print(f'  exit   measured worse by    ${np.nansum(ex_loss):,.0f} '
          f'(non-scale legs only)')

    # ---- tails, per arm ---------------------------------------------------
    rl = range_lookup()
    rows = []
    for arm in ARMS:
        for n in (8, 3):
            p = f'{P}/book_{arm}_n{n}.csv'
            if not os.path.exists(p):
                continue
            x = read_orb_csv(p)
            x['date'] = pd.to_datetime(x['date']).dt.strftime('%Y-%m-%d')
            rng = np.array([rl.get((s, d), np.nan) for s, d in zip(x.symbol, x.date)])
            R = np.where(rng > 0, x['_sized_pnl'] / (x['_rp_position'] / x.entry_price * rng), 0.0)
            v = np.sort(R)
            k1 = max(1, int(round(0.01 * len(v))))
            k5 = max(1, int(round(0.05 * len(v))))
            rows.append(dict(arm=arm, slots=n, n=len(v),
                             meanR=round(float(R.mean()), 3),
                             ex_top1=round(float(v[:-k1].mean()), 3),
                             ex_top5=round(float(v[:-k5].mean()), 3),
                             cap3R=round(float(np.minimum(R, 3.0).mean()), 3),
                             t=round(float(R.mean() / (R.std(ddof=1) / np.sqrt(len(R)))), 2)))
    t = pd.DataFrame(rows)
    t.to_csv(f'{P}/tails.csv', index=False)
    print('\ntail dependence (R per pick):')
    print(t.to_string(index=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
