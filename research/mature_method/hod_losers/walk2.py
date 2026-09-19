#!/usr/bin/env python3
"""hod_losers — the EXIT-VARIANT walk (PREREG §2A).  For every B0 u B2 pre-book signal on
TRAIN+VAL, re-walk the tape under each declared post-fill exit and emit its rr / exit minute /
exit reason.  `base` reproduces the shipped walk and is the parity gate.  TEST days never read.
"""
import json, os, sys
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT); sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_break')
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_filter_stack')
import score as S            # noqa: E402
import score2 as S2          # noqa: E402
from pass2 import load_bars  # noqa: E402

D = f'{ROOT}/research/mature_method/hod_losers'
OUTC, STATE = f'{D}/exits.csv', f'{D}/walk2_state.json'
EOD_M, SLIP = 955, 0.001
VARIANTS = ['base', 't10', 't5', 'ruleD', 'shape', 'be', 't10be']
COLS = ['day', 'symbol', 'entry_m'] + [f'{k}_{v}' for v in VARIANTS
                                       for k in ('rr', 'exit_m', 'why')]


def walk(o, h, l, c, m, e, E, ST, Rd, tmin=None, tthr=0.25, be_r=None):
    """The spec's walk from bar e+1, with the declared modifications.

    tmin  -- conditional time stop: at the close of bar e+tmin-1, if the running max high since
             the fill has not reached E + tthr*Rd, sell that close.
    be_r  -- once the running max high (closed bars only) reaches E + be_r*Rd, the stop moves to E.
    """
    tgt = E + 2.0 * Rd
    stop = ST
    runmax = float(h[e])
    j_t = (e + tmin - 1) if tmin else None
    if be_r is not None and runmax >= E + be_r * Rd:
        stop = max(stop, E)
    for k in range(e + 1, len(o)):
        if int(m[k]) >= EOD_M:
            return k, float(o[k]), 'eod'
        if l[k] <= stop:
            return k, float(min(stop, o[k]) * (1.0 - SLIP)), ('bestop' if stop > ST else 'stop')
        if c[k] >= tgt:
            return k, float(tgt), 'target'
        runmax = max(runmax, float(h[k]))
        if j_t is not None and k == j_t and runmax < E + tthr * Rd:
            return k, float(c[k]), 'timestop'
        if be_r is not None and runmax >= E + be_r * Rd:
            stop = max(stop, E)
    return len(o) - 1, float(c[-1]), 'eod'


def main():
    pop = S2.load_pop(); S.build_impute(pop)
    keys = []
    for nm, kw in (('B0', S2.BASES['B0']), ('B2', S2.BASES['B2'])):
        sg = S2.sig_set(pop, **kw)
        sg = sg[sg.split.isin(('TRAIN', 'VAL'))]
        keys.append(sg[['day', 'symbol', 'entry_m', 'break_m', 'next_open', 'stop']])
    K = pd.concat(keys).drop_duplicates(['day', 'symbol', 'entry_m']).reset_index(drop=True)
    del pop, keys, sg
    print(f'signals {len(K)} days {K.day.nunique()}', flush=True)
    state = json.load(open(STATE)) if os.path.exists(STATE) else {'done': []}
    done = set(state['done'])
    days = [d for d in sorted(K.day.unique()) if d not in done]
    for nd, day in enumerate(days):
        sub = K[K.day == day]
        bars = load_bars(day, sorted(sub.symbol.unique()))
        rows = []
        for r in sub.itertuples():
            gg = bars.get(r.symbol)
            if gg is None:
                continue
            rth = gg[(gg.m >= 570) & (gg.m < 960)].reset_index(drop=True)
            if len(rth) < 10:
                continue
            o, h, l, c = (rth[k].values.astype(float) for k in ('o', 'h', 'l', 'c'))
            m = rth.m.values.astype(int)
            ii = np.where(m == int(r.break_m))[0]
            if not len(ii):
                continue
            i = int(ii[0]); e = i + 1
            if e >= len(o):
                continue
            E, ST = float(r.next_open), float(r.stop)
            Rd = E - ST
            if not (Rd > 0):
                continue
            row = dict(day=day, symbol=r.symbol, entry_m=int(r.entry_m))
            specs = dict(base={}, t10=dict(tmin=10), t5=dict(tmin=5), be=dict(be_r=0.5),
                         t10be=dict(tmin=10, be_r=0.5))
            for v, kw in specs.items():
                k, px, why = walk(o, h, l, c, m, e, E, ST, Rd, **kw)
                row[f'rr_{v}'] = (px - E) / Rd
                row[f'exit_m_{v}'] = int(m[k]); row[f'why_{v}'] = why
            # Rule D -- acts at the close of the fill bar, before the spec walk starts
            if l[e] <= E - 0.75 * Rd:
                row['rr_ruleD'] = -0.5; row['exit_m_ruleD'] = int(m[e]); row['why_ruleD'] = 'ruleD'
            else:
                for k in ('rr', 'exit_m', 'why'):
                    row[f'{k}_ruleD'] = row[f'{k}_base']
            # fill-bar shape -- sells the NEXT bar's open if the fill bar closed in its bottom half
            rng = h[e] - l[e]
            if e + 1 < len(o) and rng > 0 and (c[e] - l[e]) / rng < 0.5 and int(m[e + 1]) < EOD_M:
                row['rr_shape'] = (float(o[e + 1]) - E) / Rd
                row['exit_m_shape'] = int(m[e + 1]); row['why_shape'] = 'shape'
            else:
                for k in ('rr', 'exit_m', 'why'):
                    row[f'{k}_shape'] = row[f'{k}_base']
            rows.append(row)
        if rows:
            pd.DataFrame(rows)[COLS].to_csv(OUTC, mode='a', header=not os.path.exists(OUTC),
                                            index=False)
        state['done'].append(day); json.dump(state, open(STATE, 'w'))
        if nd % 40 == 0:
            print(f'{nd + 1}/{len(days)} {day} +{len(rows)}', flush=True)
    print('WALK2 DONE', flush=True)


if __name__ == '__main__':
    main()
