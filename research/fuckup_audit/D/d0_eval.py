#!/usr/bin/env python3
"""D0 step 5 — evaluation. `python3 d0_eval.py PERIOD [PERIOD ...]`

PERIOD in {trainpred, val, test}. TEST is a separate invocation on purpose: it is run ONCE, after the
VAL table is written into REPORT.md and the selection rule is frozen there.

Selection rules (pre-registered, §0.4):
  S1  per day, top 12 by the predicted value (tie: symbol), then run_book(12, 4)   [diagnostic ceiling]
  S2  per-row gate (reg > 0 / clf P > 0.5 / baseline score >= TRAIN median), then run_book(12, 4)  [live rule]
Controls: FCFS (no selection) and RAND12 (12 random candidates a day, mean of 20 seeds).
Tapes: real / rev (Nagel reversed) / shuf (targets permuted within day, scored on the TRUE outcome).
"""
import os, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr

ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT)
sys.path.insert(0, f'{ROOT}/research/fuckup_audit/D')
import d0_core as K

D = 'research/fuckup_audit/D'
PERIODS = {'trainpred': lambda p: p.month.between('2025-10', '2025-12'),
           'val': lambda p: p.split == 'VAL',
           'test': lambda p: p.split == 'TEST'}
MODELS = [('reg', 'HGB regressor'), ('clf', 'HGB classifier'), ('base', 'transparent baseline')]
TAPES = ['real', 'rev', 'shuf']


def target(p, tape):
    return -p.net if tape == 'rev' else p.net          # shuf is scored on the TRUE outcome


def bookit(x, tape):
    y = x.copy()
    y['net'] = target(y, tape)
    return K.book(y)


def sel_s1(x, col):
    o = x.sort_values([col, 'symbol'], ascending=[False, True])
    return o.groupby('day', sort=False).head(12)


def sel_s2(x, mod, tape):
    if mod == 'reg':
        return x[x[f'reg_{tape}'] > 0]
    if mod == 'clf':
        return x[x[f'clf_{tape}'] > 0.5]
    return x[x[f'base_{tape}'] >= x[f'base_{tape}_thr']]


def row(key, model, sel, tape, t, weeks, extra=None):
    if t is None or not len(t):
        return dict(key=key, model=model, sel=sel, tape=tape, n=0)
    r = dict(key=key, model=model, sel=sel, tape=tape)
    r.update(K.stats(t, weeks))
    if extra:
        r.update(extra)
    return r


def main(periods):
    P = pd.read_csv(f'{D}/preds.csv', dtype={'symbol': str, 'day': str, 'key': str, 'split': str,
                                             'month': str, 'wk': str, 'why_hold': str},
                    keep_default_na=False, na_values=[''])
    for per in periods:
        p = P[PERIODS[per](P)].copy()
        weeks = sorted(p.wk.unique())
        print(f'== {per}: {len(p)} candidates, {p.day.nunique()} days, {len(weeks)} weeks', flush=True)
        out, mono, booked = [], [], []
        for key in K.KEYS:
            x = p[p.key == key]
            if not len(x):
                continue
            # ---- controls
            t = K.book(x)
            out.append(row(key, 'FCFS', '-', 'real', t, weeks, K.tail_stats(t, weeks)))
            rs = np.random.RandomState(7)
            acc = []
            for s in range(20):
                x2 = x.assign(_r=rs.rand(len(x)))
                pick = x2.sort_values('_r').groupby('day', sort=False).head(12)
                tb = K.book(pick)
                acc.append(K.stats(tb, weeks) if tb is not None else None)
            acc = [a for a in acc if a]
            out.append(dict(key=key, model='RAND12', sel='-', tape='real',
                            n=int(np.mean([a['n'] for a in acc])), tpw=round(np.mean([a['tpw'] for a in acc]), 1),
                            meanR=round(float(np.mean([a['meanR'] for a in acc])), 4),
                            se=round(float(np.std([a['meanR'] for a in acc], ddof=1)), 4),
                            t=np.nan, mde=np.nan, WR=round(float(np.mean([a['WR'] for a in acc])), 1),
                            wkR=round(float(np.mean([a['wkR'] for a in acc])), 2),
                            green=round(float(np.mean([a['green'] for a in acc])), 2),
                            worst=round(float(np.mean([a['worst'] for a in acc])), 1),
                            mix_stop=np.nan, mix_eod=np.nan, mix_target=np.nan))
            # ---- the 18 declared cells (x3 tapes)
            for mod, _lab in MODELS:
                for tape in TAPES:
                    col = f'{mod}_{tape}'
                    t1 = bookit(sel_s1(x, col), tape)
                    t2 = bookit(sel_s2(x, mod, tape), tape)
                    ex1 = K.tail_stats(t1, weeks) if (t1 is not None and tape == 'real') else None
                    ex2 = K.tail_stats(t2, weeks) if (t2 is not None and tape == 'real') else None
                    out.append(row(key, mod, 'S1', tape, t1, weeks, ex1))
                    out.append(row(key, mod, 'S2', tape, t2, weeks, ex2))
                    if tape == 'real':
                        for t_, s_ in ((t1, 'S1'), (t2, 'S2')):
                            if t_ is not None:
                                z = t_.copy(); z['key'] = key; z['model'] = mod; z['sel'] = s_; z['period'] = per
                                booked.append(z)
                # ---- decile calibration of the real-tape prediction over the whole candidate pool
                v = x[f'{mod}_real'].values
                q = np.unique(np.nanquantile(v, np.linspace(0, 1, 11)))
                if len(q) >= 4:
                    d = np.digitize(v, q[1:-1])
                    g = pd.DataFrame({'d': d, 'net': x.net.values}).groupby('d').net.agg(['mean', 'size'])
                    rho = spearmanr(g.index.values, g['mean'].values).correlation
                    for di, rr in g.iterrows():
                        mono.append(dict(key=key, model=mod, period=per, decile=int(di),
                                         meanR=round(float(rr['mean']), 4), n=int(rr['size']),
                                         rho=round(float(rho), 3)))
        pd.DataFrame(out).to_csv(f'{D}/cells_{per}.csv', index=False)
        pd.DataFrame(mono).to_csv(f'{D}/calib_{per}.csv', index=False)
        if booked:
            pd.concat(booked, ignore_index=True).to_csv(f'{D}/booked_{per}.csv', index=False)
        print(pd.DataFrame(out)[['key', 'model', 'sel', 'tape', 'n', 'tpw', 'meanR', 'se', 't', 'mde',
                                 'WR', 'wkR', 'green', 'worst']].to_string(index=False), flush=True)


if __name__ == '__main__':
    pd.set_option('display.width', 250); pd.set_option('display.max_rows', 400)
    main(sys.argv[1:] or ['trainpred', 'val'])
