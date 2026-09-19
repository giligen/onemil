#!/usr/bin/env python3
"""hod_frames5 / F18 — build the dedicated NBBO fetch list (PREREG §4.3).

The union of
  (a) every pre-book row of the $20-floor **gates-OFF** first-break set — so the spread-gate ladder
      can be scored on MEASURED cost including the rows the gates currently remove, and
  (b) every row BOOKED by any declared F18 cell (the floor ladder on mixed and on wrappers, the
      spread ladder),
minus everything already measured in `research/bf_zero/causal_filter/nbbo.csv`.
Writes `nbbo5_todo.csv`.  Scores nothing.
"""
import sys
import numpy as np, pandas as pd

sys.path.insert(0, '/home/ec2-user/onemil/research/mature_method/hod_frames5')
from common5 import (ROOT, D5, S, S2, load_breaks4, sigset5, admit, book_ranked,   # noqa: E402
                     attach_instrument)

br = load_breaks4()
S.build_impute(S2.load_pop())
FIRST = admit(br, pd.Series(True, index=br.index))

key = ['day', 'symbol', 'entry_m']
want = []

# (a) the $20-floor gates-OFF pre-book set
a = sigset5(FIRST, min_price=20.0, max_bps=None, max_frac_r=None)
want.append(a[key])
print(f'(a) $20 gates-OFF pre-book rows: {len(a)}', flush=True)

# (b) every booked row of every declared F18 cell
cells = []
for fl in (5.0, 10.0, 20.0, 30.0, 50.0):
    cells.append(('all', fl, 0.15, 100.0))
    cells.append(('wrap', fl, 0.15, 100.0))
for fr, bps in ((0.08, 100.0), (0.25, 100.0), (0.40, 100.0), (None, None)):
    cells.append(('all', 20.0, fr, bps))
for pop, fl, fr, bps in cells:
    s = sigset5(FIRST, min_price=fl, max_frac_r=fr, max_bps=bps)
    if pop == 'wrap':
        s = attach_instrument(s)
        s = s[s.asset_class == 'wrapper']
    b = book_ranked(s, 12, 4)
    want.append(b[key])
    print(f'(b) {pop:5s} floor {fl:5.0f} frac_r {str(fr):5s} bps {str(bps):6s} -> booked {len(b)}',
          flush=True)

w = pd.concat(want, ignore_index=True).drop_duplicates(key)
nb = pd.read_csv(f'{ROOT}/research/bf_zero/causal_filter/nbbo.csv',
                 dtype={'symbol': str, 'day': str}, keep_default_na=False,
                 na_values=['']).drop_duplicates(key)
have = set(zip(nb.day, nb.symbol, nb.entry_m.astype(int)))
w['_k'] = list(zip(w.day, w.symbol, w.entry_m.astype(int)))
todo = w[~w._k.isin(have)].drop(columns=['_k'])
todo = todo.sort_values(key).reset_index(drop=True)
todo.to_csv(f'{D5}/nbbo5_todo.csv', index=False)
print(f'\nunion {len(w)} | already measured {len(w) - len(todo)} | TODO {len(todo)} quote-minutes '
      f'(~{len(todo)/340:.0f} min at the pass-3 fetch rate)', flush=True)
