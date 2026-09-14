#!/usr/bin/env python3
"""Exit variants under real spreads (pre-registered 2026-09-14, owner: "if spread eats 20%, move the 2R above it?").

On the spread-study sample (`spread_study.csv`: ~7,000 spec signals with the NBBO at the signal minute),
re-walk each trade from the bars under:
  V0 baseline      entry = next open (≤ cap), stop = consolidation low, target = entry + 2R
  V1 target+spread stop unchanged, target = entry + 2R + spread
  V2 both+spread   stop = consolidation low − spread, target = entry + 2·(R + spread)
Costs charged identically to all: entry at the ask = next open + spread/2 (the bar's open is a mid-ish
print), stop exits at the bid = fill − spread/2, target exits at the limit (no extra). R for the R-multiple
is the VARIANT's own risk, dollars are per $100 of that risk. Splits fixed; TEST read once.
Output: spread_exit_variants.md
"""
import os, sys
import numpy as np, pandas as pd
ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT); sys.path.insert(0, ROOT); sys.path.insert(0, f'{ROOT}/research/bf_zero')
import build_candidates as B                      # bar loader over cache + side DBs
from trading.hod_break import HodBreakParams, detect, OPEN_MINUTE, FLAT_MINUTE
D = 'research/bf_zero'; P = HodBreakParams()
d = pd.read_csv(f'{D}/spread_study.csv', dtype={'symbol': str}); d = d[d.n_quotes > 0].copy()
d['split'] = np.where(d.day < '2026-01-01', 'TRAIN', np.where(d.day < '2026-06-01', 'VAL', 'TEST'))
print('quoted signals', len(d), flush=True)


def walk(o, h, l, c, m, i, entry, stop, target):
    for k in range(i + 1, len(o)):
        if int(m[k]) >= FLAT_MINUTE: return float(o[k]), 'eod'
        if l[k] <= stop: return float(min(stop, o[k])), 'stop'
        if c[k] >= target: return float(target), 'target'
    return float(c[-1]), 'eod'


rows = []
for n, (day, sub) in enumerate(d.groupby('day')):
    bars = B.load_bars(day, sub.symbol.tolist())
    for r in sub.itertuples():
        gg = bars.get(r.symbol)
        if gg is None: continue
        rth = gg[(gg.m >= OPEN_MINUTE) & (gg.m < 960)].reset_index(drop=True)
        o, h, l, c, v = (rth[k].values.astype(float) for k in ('o', 'h', 'l', 'c', 'v')); m = rth.m.values.astype(int)
        sig = detect(o, h, l, v, m, 0.0, HodBreakParams(rv_lo=0.0, rv_hi=1e9))   # rv already satisfied by construction; re-find the same break
        if sig is None or sig.bar_idx + 1 >= len(o): continue
        i = sig.bar_idx + 1; nxt = float(o[i]); sp = float(r.spread_last)
        if nxt > sig.level * (1 + P.cap): continue
        fill = nxt + sp / 2                                          # pay the ask
        for name, stop, target in (('V0', sig.stop, fill + 2 * (fill - sig.stop)),
                                   ('V1', sig.stop, fill + 2 * (fill - sig.stop) + sp),
                                   ('V2', sig.stop - sp, fill + 2 * (fill - (sig.stop - sp)))):
            R = fill - stop
            if R <= 0: continue
            px, why = walk(o, h, l, c, m, i, fill, stop, target)
            if why != 'target': px -= sp / 2                         # exits at the bid
            rows.append(dict(day=day, symbol=r.symbol, split=r.split, variant=name, R=R, rr=(px - fill) / R, usd_per_100=(px - fill) / R * 100, why=why, spread_frac_r=sp / R))
    if n % 40 == 0: print(f'{n} days {len(rows)} rows', flush=True)
T = pd.DataFrame(rows); T.to_csv(f'{D}/spread_exit_variants_rows.csv', index=False)
lines = [f'# Exit variants under real spreads — {T.symbol.nunique():,} signals, costs charged identically', '']
for v in ('V0', 'V1', 'V2'):
    x = T[T.variant == v]
    g = x.groupby('split').agg(n=('rr', 'size'), meanR=('rr', 'mean'), WR=('rr', lambda s: (s > 0).mean() * 100), target_rate=('why', lambda s: (s == 'target').mean() * 100), stop_rate=('why', lambda s: (s == 'stop').mean() * 100), usd_per_100=('usd_per_100', 'mean')).round(3).reindex(['TRAIN', 'VAL', 'TEST'])
    lines += [f'## {v}', g.to_markdown(), '']
lines += ['Note: V2 changes R itself (stop below the structure), so its R-multiples are on a larger unit — compare usd_per_100 (same $100 risk) across variants.']
open(f'{D}/spread_exit_variants.md', 'w').write('\n'.join(lines)); print('\n'.join(lines)); print('DONE', flush=True)
