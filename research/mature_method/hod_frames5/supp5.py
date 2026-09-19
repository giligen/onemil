#!/usr/bin/env python3
"""hod_frames5 — supplementary: DECOMPOSE what the dedicated NBBO fetch changed.

PREREG §4.3 declared that measuring the spread changes MEMBERSHIP in both directions.  The
measured arm's B2 is -$12,627 / +$2,427 against the as-is reference -$17,346 / +$893, and it would
be dishonest to present that as "the cost was overstated" without saying how much of it is
re-pricing and how much is a different set of trades.  Three terms:
  (i)  rows the measured `ask_dec` makes UNOBTAINABLE that a missing quote let through,
  (ii) rows the spread gates now reject / now admit,
  (iii) re-pricing of the rows common to both arms.
No cell is scored here; this is an accounting of a declared sensitivity arm.
"""
import sys
import numpy as np, pandas as pd

sys.path.insert(0, '/home/ec2-user/onemil/research/mature_method/hod_frames5')
from common5 import (ROOT, D5, S, S2, SPLITS, RISK, load_breaks5, sigset5, admit,   # noqa: E402
                     book_ranked)

K = ['day', 'symbol', 'entry_m']


def arm(extra):
    br = load_breaks5(extra_nbbo=extra, verbose=False)
    S.build_impute(S2.load_pop())
    F = admit(br, pd.Series(True, index=br.index))
    pre_all = sigset5(F, obtain=False, max_bps=None, max_frac_r=None)
    pre = sigset5(F)
    return pre_all, pre, book_ranked(pre, 12, 4)


a_all, a_pre, a_b = arm(())
m_all, m_pre, m_b = arm((f'{D5}/nbbo5.csv',))

print('== what the dedicated fetch changed (PREREG §4.3) ==')
print(f'  pre-book signals: as-is {len(a_pre)} -> measured {len(m_pre)}')
ka = set(map(tuple, a_pre[K].values)); km = set(map(tuple, m_pre[K].values))
print(f'    dropped by measurement {len(ka - km)} | added by measurement {len(km - ka)} | '
      f'common {len(ka & km)}')

# (i) obtainability
for lab, d in (('as-is', a_all), ('measured', m_all)):
    print(f'  {lab}: of {len(d)} gate-free rows, obtainable {int(d.obtainable.sum())} '
          f'({d.obtainable.mean():.1%}); ask_dec present {d.ask_dec.notna().mean():.1%}')
ao = a_all.set_index(K).obtainable
mo = m_all.set_index(K).obtainable
j = pd.concat([ao.rename('a'), mo.rename('m')], axis=1).dropna()
print(f'  (i)  obtainability flips: True->False {int(((j.a) & (~j.m.astype(bool))).sum())} | '
      f'False->True {int(((~j.a.astype(bool)) & (j.m)).sum())}')

# (ii) spread-gate flips on the rows both arms priced
af = a_all.set_index(K); mf = m_all.set_index(K)
com = af.index.intersection(mf.index)
gate_a = ((af.loc[com].sp_pct * 100) <= 100) & \
         ((af.loc[com].sp_pct / af.loc[com].r_pct.clip(lower=0.05)) <= 0.15)
gate_m = ((mf.loc[com].sp_pct * 100) <= 100) & \
         ((mf.loc[com].sp_pct / mf.loc[com].r_pct.clip(lower=0.05)) <= 0.15)
print(f'  (ii) spread-gate flips: pass->fail {int((gate_a & ~gate_m).sum())} | '
      f'fail->pass {int((~gate_a & gate_m).sum())}')

# (iii) re-pricing on the common booked rows
kb_a = set(map(tuple, a_b[K].values)); kb_m = set(map(tuple, m_b[K].values))
both = kb_a & kb_m
ai = a_b.set_index(K); mi = m_b.set_index(K)
bi = pd.Index(list(both))
d_cost = (ai.loc[bi].rr - ai.loc[bi].net).mean() - (mi.loc[bi].rr - mi.loc[bi].net).mean()
print(f'\n  booked rows: as-is {len(a_b)} | measured {len(m_b)} | in both {len(both)}')
print(f'  (iii) on the {len(both)} rows in BOTH books, mean cost R falls by {d_cost:+.4f} '
      f'(as-is {(ai.loc[bi].rr - ai.loc[bi].net).mean():+.4f} -> measured '
      f'{(mi.loc[bi].rr - mi.loc[bi].net).mean():+.4f}); gross is unchanged by construction '
      f'({ai.loc[bi].rr.mean():+.4f} vs {mi.loc[bi].rr.mean():+.4f})')
for sp in SPLITS:
    aa = ai.loc[bi][ai.loc[bi].split == sp]; mm = mi.loc[bi][mi.loc[bi].split == sp]
    d_only_a = ai.loc[pd.Index(list(kb_a - both))]
    d_only_m = mi.loc[pd.Index(list(kb_m - both))]
    print(f'   {sp}: common rows $ {aa.pnl.sum():+,.0f} -> {mm.pnl.sum():+,.0f} '
          f'(repricing {mm.pnl.sum() - aa.pnl.sum():+,.0f}) | as-is-only rows '
          f'{d_only_a[d_only_a.split == sp].pnl.sum():+,.0f} ({(d_only_a.split == sp).sum()} tr) | '
          f'measured-only rows {d_only_m[d_only_m.split == sp].pnl.sum():+,.0f} '
          f'({(d_only_m.split == sp).sum()} tr)')
print('\nDONE supp5')
