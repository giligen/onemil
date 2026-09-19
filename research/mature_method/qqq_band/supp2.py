#!/usr/bin/env python3
"""Additivity for the cell that passed the claim bar (H2), beside B0, and the stacked weekly table."""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '/home/ec2-user/onemil/research/fuckup_audit/Q')
sys.path.insert(0, '/home/ec2-user/onemil/research/mature_method/qqq_band')
import zsim as Z                                                         # noqa: E402
import score as S                                                        # noqa: E402

data = Z.load_symbol('QQQ')
COST = float(pd.read_csv(S.HERE + 'cost_nbbo.csv')['half_bp'].mean())
UB, LB = Z.bands(data, vm=1.0)
books = {'B0': S.day_frame(data, S.sim3(data, UB, LB, slip_bp=0.0), COST),
         'H2': S.day_frame(data, S.sim3(data, UB, LB, slip_bp=0.0, stop_mode='none'), COST),
         'T2': S.day_frame(data, S.sim3(data, UB, LB, slip_bp=0.0,
                                        checks=list(range(30, 331, 60))), COST)}

orb = pd.read_csv('/home/ec2-user/onemil/research/orb_gates2/book_G3_meas.csv',
                  usecols=['date', '_sized_pnl'])
orb['date'] = pd.to_datetime(orb['date'])
bf = pd.read_csv('/home/ec2-user/onemil/research/bf_frequency/runs/VOL_OFF.csv',
                 usecols=['date', 'pnl'])
bf['date'] = pd.to_datetime(bf['date']); bf['pnl'] *= 0.075
lo, hi = pd.Timestamp('2025-01-02'), pd.Timestamp('2026-09-15')


def wkmetrics(v):
    c = np.cumsum(v)
    mdd = (np.maximum.accumulate(np.r_[0, c]) - np.r_[0, c]).max()
    best = cur = 0
    for x in v:
        cur = cur + 1 if x <= 0 else 0
        best = max(best, cur)
    return dict(n=len(v), green=100 * (v > 0).mean(), total=v.sum(), worst=v.min(),
                best_wk=v.max(), streak=best, mdd=mdd)


cal = books['B0'][['date']].copy()
cal = cal[(cal['date'] >= lo) & (cal['date'] <= hi)].copy()
cal['wk'] = cal['date'].dt.to_period('W-FRI')
W = pd.DataFrame({'wk': sorted(cal['wk'].unique())})
for nm, src, col in (('orb', orb, '_sized_pnl'), ('bf', bf, 'pnl')):
    s = src[(src['date'] >= lo) & (src['date'] <= hi)].copy()
    s['wk'] = s['date'].dt.to_period('W-FRI')
    W = W.merge(s.groupby('wk')[col].sum().rename(nm).reset_index(), on='wk', how='left')
W[['orb', 'bf']] = W[['orb', 'bf']].fillna(0.0)
W['live'] = W['orb'] + W['bf']
for cid, d in books.items():
    x = d[(d['date'] >= lo) & (d['date'] <= hi)].copy()
    x['wk'] = x['date'].dt.to_period('W-FRI')
    W = W.merge((x.groupby('wk')['r1x'].sum() * S.AUM).rename(cid).reset_index(), on='wk', how='left')
    W[cid] = W[cid].fillna(0.0)
W.to_csv(S.HERE + 'additivity_weeks2.csv', index=False)

print(f'{"book":22s} {"n":>3s} {"green%":>7s} {"total$":>9s} {"worst$":>8s} {"best$":>8s} '
      f'{"redstk":>6s} {"mdd$":>8s}')


def line(nm, v):
    m = wkmetrics(np.asarray(v, float))
    print(f'{nm:22s} {m["n"]:3d} {m["green"]:7.1f} {m["total"]:9,.0f} {m["worst"]:8,.0f} '
          f'{m["best_wk"]:8,.0f} {m["streak"]:6d} {m["mdd"]:8,.0f}')


line('ORB (G3, stage size)', W['orb'])
line('BF (VOL_OFF x0.075)', W['bf'])
line('LIVE = ORB + BF', W['live'])
for cid in books:
    for lev, mult in (('1x', 1.0), ('2x', 2.0)):
        line(f'{cid} sleeve {lev}', W[cid] * mult)
        line(f'LIVE + {cid} {lev}', W['live'] + W[cid] * mult)
print()
for cid in books:
    print(f'corr(week $) {cid}~ORB {W[cid].corr(W["orb"]):+.3f}  {cid}~BF {W[cid].corr(W["bf"]):+.3f}  '
          f'{cid}~LIVE {W[cid].corr(W["live"]):+.3f}')
    red = W[W['live'] <= 0]; grn = W[W['live'] > 0]
    for lev, mult in (('1x', 1.0), ('2x', 2.0)):
        print(f'  {cid} {lev}: of {len(red)} live-RED weeks {int((red["live"] + red[cid] * mult > 0).sum())} '
              f'turn green; of {len(grn)} live-GREEN weeks '
              f'{int((grn["live"] + grn[cid] * mult <= 0).sum())} turn red')
print(f'\ncorr ORB~BF {W["orb"].corr(W["bf"]):+.3f}')
print('\nStacked weekly $ (LIVE / LIVE+H2 1x), last 26 weeks:')
t = W.tail(26)
print('  LIVE    : ' + ' '.join(f'{v:+.0f}' for v in t['live']))
print('  +H2 1x  : ' + ' '.join(f'{v:+.0f}' for v in (t['live'] + t['H2'])))
