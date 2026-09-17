#!/usr/bin/env python3
"""R7 - 100 random SHARED booked TEST hold trades that differ in any field: re-derive under both conventions,
check the parameterised pipeline reproduces each side exactly, and find the single switch that reconciles them."""
import os, sys, random
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT); sys.path.insert(0, f'{ROOT}/research/fuckup_audit/H/F6_reconcile')
from pipeline import Bars, run_day, A_CFG, B_CFG

OUT = 'research/fuckup_audit/H/F6_reconcile'
RD = lambda p, **k: pd.read_csv(p, keep_default_na=False, na_values=[''], dtype={'symbol': str, 'day': str}, **k)
log = lambda *a: (print(*a), sys.stdout.flush())
SW = ['bars', 'level_mult', 'floor_den', 'stop', 'price_on', 'day_open5', 'min_bars']

s = RD(f'{OUT}/join_hold_shared.csv')
s = s[s.split == 'TEST']
diff = s[(s.d_entry_m.abs() > 0) | (s.d_entry.abs() > 1e-9) | (s.d_stop.abs() > 1e-9) | (s.d_exit_m.abs() > 0)]
log('shared TEST booked hold trades with any field difference:', len(diff), 'of', len(s))
random.seed(20260917)
sample = diff.sample(min(100, len(diff)), random_state=20260917)

prev = RD(f'{OUT}/prev_table.csv').set_index(['day', 'symbol'])
bars = Bars()


def ev(day, sym, cfg):
    try:
        r = prev.loc[(day, sym)]
    except KeyError:
        return None
    if isinstance(r, pd.DataFrame):
        r = r.iloc[0]
    t = 'panel' if cfg['prev'] == 'panel' else 'uni'
    p = dict(close=float(r[f'prev_close_{t}']), high=float(r[f'prev_high_{t}']), low=float(r[f'prev_low_{t}']))
    return run_day(bars, sym, day, p, cfg, day_open=float(r['day_open']))


def same(res, em, en, st, xm):
    return (res is not None and res['ok'] and int(res['entry_m']) == int(em) and abs(res['entry'] - en) < 1e-6
            and abs(res['stop'] - st) < 1e-6 and int(res['hold_exit_m']) == int(xm))


rec = []
for r in sample.itertuples(index=False):
    ra = ev(r.day, r.symbol, A_CFG); rb = ev(r.day, r.symbol, B_CFG)
    d = dict(day=r.day, symbol=r.symbol, A_sig=r.A_sig_m, B_sig=r.B_sig_m,
             repro_A=int(same(ra, r.A_entry_m, r.A_entry, r.A_stop, r.A_exit_m)),
             repro_B=int(same(rb, r.B_entry_m, r.B_entry, r.B_stop, r.B_exit_m)))
    fix = []
    for sw in SW:
        cfg = dict(A_CFG); cfg[sw] = B_CFG[sw]
        rr = ev(r.day, r.symbol, cfg)
        if same(rr, r.B_entry_m, r.B_entry, r.B_stop, r.B_exit_m):
            fix.append(sw)
    d['switch_that_turns_A_into_B'] = '|'.join(fix) if fix else 'none_single'
    rec.append(d)
f = pd.DataFrame(rec)
f.to_csv(f'{OUT}/shared_sample_test.csv', index=False)
L = ['# R7 shared TEST trades with field differences -- 100-trade random sample', '',
     f'shared booked hold trades in TEST: {len(s)};  with any field difference: {len(diff)}',
     f'sample: {len(f)}',
     f'pipeline reproduces A exactly: {int(f.repro_A.sum())}/{len(f)}',
     f'pipeline reproduces B exactly: {int(f.repro_B.sum())}/{len(f)}', '',
     'single switch flipped on A that reproduces B:',
     f.switch_that_turns_A_into_B.value_counts().to_string()]
open(f'{OUT}/r7_shared.md', 'w').write('\n'.join(L) + '\n')
log('\n'.join(L))
