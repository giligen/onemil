#!/usr/bin/env python3
"""R5 - cause attribution. Re-derive every TEST symbol-day where A's and B's CANDIDATE sets disagree
(and 100 random shared ones with a field difference) from the raw bars under both conventions, then flip
one switch at a time from A's config toward B's to find which convention is responsible.
"""
import os, sys, random
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT); sys.path.insert(0, f'{ROOT}/research/fuckup_audit/H/F6_reconcile')
from pipeline import Bars, run_day, A_CFG, B_CFG

D = 'research/fuckup_audit'; OUT = f'{D}/H/F6_reconcile'
RD = lambda p, **k: pd.read_csv(p, keep_default_na=False, na_values=[''], dtype={'symbol': str, 'day': str}, **k)
log = lambda *a: (print(*a), sys.stdout.flush())

SW = ['bars', 'level_mult', 'floor_den', 'stop', 'price_on', 'day_open5', 'min_bars']

prev = RD(f'{OUT}/prev_table.csv').set_index(['day', 'symbol'])
cj = RD(f'{OUT}/cand_join.csv')
cj = cj[cj.split == 'TEST']
div = cj[cj.in_A != cj.in_B].copy()
log('TEST divergent candidate keys:', len(div))

# booked-trade net R (hold), for the "net R of those trades" column
jb = RD(f'{OUT}/join_hold.csv'); jb = jb[jb.split == 'TEST']
booked = jb.set_index(['day', 'symbol'])[['in_A', 'in_B', 'A_net', 'B_net']]

bars = Bars()


def prow(day, sym, which):
    try:
        r = prev.loc[(day, sym)]
    except KeyError:
        return dict(close=np.nan, high=np.nan, low=np.nan), np.nan
    if isinstance(r, pd.DataFrame):
        r = r.iloc[0]
    t = 'panel' if which == 'panel' else 'uni'
    return dict(close=float(r[f'prev_close_{t}']), high=float(r[f'prev_high_{t}']), low=float(r[f'prev_low_{t}'])), float(r['day_open'])


def evaluate(day, sym, cfg):
    p, dop = prow(day, sym, cfg['prev'])
    return run_day(bars, sym, day, p, cfg, day_open=dop)


recs = []
for i, r in enumerate(div.itertuples(index=False)):
    if i % 100 == 0:
        log('  %d/%d' % (i, len(div)))
    ra = evaluate(r.day, r.symbol, A_CFG)
    rb = evaluate(r.day, r.symbol, B_CFG)
    rec = dict(day=r.day, symbol=r.symbol, side='A_only' if r.in_A else 'B_only',
               repro_A=int(ra['ok']), repro_B=int(rb['ok']), reason_A=ra['reason'], reason_B=rb['reason'],
               src_A=ra['src'], src_B=rb['src'])
    flips = []
    for s in SW:
        cfg = dict(A_CFG); cfg[s] = B_CFG[s]
        rr = evaluate(r.day, r.symbol, cfg)
        if int(rr['ok']) != int(ra['ok']):
            flips.append(s)
        rec['flipA_' + s] = int(rr['ok'])
        rec['reasonA_' + s] = rr['reason']
    rec['flips'] = '|'.join(flips) if flips else 'none_single'
    recs.append(rec)

f = pd.DataFrame(recs)
f = f.merge(booked.reset_index()[['day', 'symbol', 'in_A', 'in_B', 'A_net', 'B_net']], on=['day', 'symbol'], how='left')
f['booked_net'] = np.where(f.side == 'A_only', f.A_net, f.B_net)
f.to_csv(f'{OUT}/causes_test.csv', index=False)

L = ['# R5 cause attribution -- TEST, pre-book candidate divergences', '']
L.append('reproduction of the divergence by the parameterised pipeline:')
L.append(pd.crosstab([f.side], [f.repro_A, f.repro_B]).to_string())
L.append('')
L.append('first-gate reasons (A convention x B convention):')
L.append(pd.crosstab(f.reason_A, f.reason_B).to_string())
L.append('')
g = f.groupby(['side', 'flips']).agg(n=('day', 'size'), booked=('booked_net', lambda s: int(s.notna().sum())),
                                     net_R=('booked_net', lambda s: round(float(s.sum()), 2))).reset_index()
L.append('cause (set of single switch flips that change the verdict) -> trades -> booked net R:')
L.append(g.sort_values('n', ascending=False).to_string(index=False))
open(f'{OUT}/r5_causes.md', 'w').write('\n'.join(L) + '\n')
log('\n'.join(L))
