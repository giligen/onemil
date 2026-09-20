#!/usr/bin/env python3
"""hod_frames6 / F19 stage A — the point-in-time listing sets, written to `pitsets6.csv`.

Separate process: the 17 monthly definition parquets and the 1.19M-row break stream do not fit in
the same 3 GB budget, and a stage boundary is cheaper than a memory trick.
"""
import os, sys
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT); sys.path.insert(0, ROOT)
from research.scripts.pit_listings import PitListings   # noqa: E402

D6 = f'{ROOT}/research/mature_method/hod_frames6'
ERAS = {'H1-2025': ('2025-01', '2025-06'), 'H2-2025': ('2025-07', '2025-12'),
        'VAL': ('2026-01', '2026-05')}

pit = PitListings()
print(f'PIT definition coverage {pit.coverage[0]}..{pit.coverage[1]} — the whole TRAIN+VAL window '
      f'(202501..202605) is INSIDE it; the XNAS.ITCH 2018-2024 fallback is not needed, not used.',
      flush=True)
L = {}
for e, (a, b) in ERAS.items():
    sets = []
    for m in [str(p) for p in pd.period_range(a, b, freq='M')]:
        sets.append(pit.listed_symbols(f'{m}-15'))
        PitListings._month.cache_clear()
    L[e] = frozenset.intersection(*sets)
    print(f'  {e}: listed throughout = {len(L[e])} symbols '
          f'(monthly sets {min(len(x) for x in sets)}..{max(len(x) for x in sets)})', flush=True)
allsym = sorted(L['H1-2025'] | L['H2-2025'] | L['VAL'])
df = pd.DataFrame({'symbol': allsym})
for e in ERAS:
    df[e] = df.symbol.isin(L[e])
df['INTERSECT'] = df[list(ERAS)].all(axis=1)
df.to_csv(f'{D6}/pitsets6.csv', index=False)
print(f'  INTERSECT = {int(df.INTERSECT.sum())} symbols; pitsets6.csv {len(df)} rows', flush=True)
