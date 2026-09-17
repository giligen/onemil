#!/usr/bin/env python3
"""H/QQQ step 2b — drop-bucket era test for F6/F7, overlap of the survivors, rule-valid stacks.
TRAIN only."""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/ec2-user/onemil/research/fuckup_audit/H/QQQ')
sys.path.insert(0, '/home/ec2-user/onemil/research/fuckup_audit/Q')
import zsim as Z                                                    # noqa: E402
import filters as F                                                 # noqa: E402

H = F.H
TR = F.SPLITS['TRAIN']
rows = []


def drop_bucket(base, dates_ok, tag):
    d = base[~base['date'].isin(dates_ok)]
    d = d[(d['date'] >= TR[0]) & (d['date'] <= TR[1]) & (d['ntr'] > 0)]
    h1 = d[d['date'] <= F.H1[1]]; h2 = d[d['date'] >= F.H2[0]]
    print(f'  drop[{tag:28s}] n={len(d):4d} mean={d["bps"].mean():+7.2f} '
          f'| H1 n={len(h1):4d} {h1["bps"].mean():+7.2f} | H2 n={len(h2):4d} {h2["bps"].mean():+7.2f}'
          f'   {"BOTH NEG" if h1["bps"].mean() < 0 and h2["bps"].mean() < 0 else ""}')
    return h1['bps'].mean() < 0 and h2['bps'].mean() < 0


def rep(name, d):
    print(F.line(name, d, *TR))
    print('    ' + F.line('  H1', d, *F.H1))
    print('    ' + F.line('  H2', d, *F.H2))
    a = F.stats(d, *TR); a1 = F.stats(d, *F.H1); a2 = F.stats(d, *F.H2)
    rows.append(dict(cell=name, TRAIN_n=a['traded'], TRAIN_bps=a['bps'], TRAIN_t=a['t'],
                     TRAIN_sum=a['sum'], TRAIN_green=a['green'], TRAIN_sr=a['sharpe'],
                     TRAIN_mdd=a['mdd'], H1_bps=a1['bps'], H2_bps=a2['bps']))


def main():
    data = Z.load_symbol('QQQ')
    UB, LB = Z.bands(data, 1.0)
    fx = F.extra_features(data, UB, LB)
    base = F.day_df(data, F.simulate2(data, UB, LB))
    D = len(data['days'])

    print('\n== drop-bucket era test (rule 1a: the vetoed bucket must be negative in BOTH halves) ==')
    tr_mask = (fx['date'] >= TR[0]) & (fx['date'] <= TR[1])
    cands = {
        'F1 band_w>=0.5': (fx['band_w'] >= 0.5).values,
        'F1 band_w>=0.6': (fx['band_w'] >= 0.6).values,
        'F1 band_w>=0.67': (fx['band_w'] >= 0.67).values,
        'F1 band_w>=0.7': (fx['band_w'] >= 0.7).values,
        'F1 band_w>=0.8': (fx['band_w'] >= 0.8).values,
        'F2 gap/band>=0.15': (fx['gap_over_band'] >= 0.15).values,
        'F2 gap/band>=0.20': (fx['gap_over_band'] >= 0.20).values,
        'F2 gap/band>=0.25': (fx['gap_over_band'] >= 0.25).values,
        'F2 gap/band>=0.30': (fx['gap_over_band'] >= 0.30).values,
        'F6 rv20>=0.70': (fx['rv20'] >= 0.70).values,
        'F6 rv20>=1.00': (fx['rv20'] >= 1.00).values,
        'F7 absmove5>=2.93 (T1)': (fx['absmove5'] >= fx.loc[tr_mask, 'absmove5'].quantile(1 / 3)).values,
        'F7 absmove5>=2.47 (Q1)': (fx['absmove5'] >= fx.loc[tr_mask, 'absmove5'].quantile(0.25)).values,
    }
    valid = {}
    for tag, ok in cands.items():
        if drop_bucket(base, fx['date'][ok], tag):
            valid[tag] = ok
    print(f'\n  rule-1a survivors: {list(valid)}')

    print('\n== overlap of the survivors on TRAIN traded days ==')
    keys = list(valid)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = valid[keys[i]], valid[keys[j]]
            m = tr_mask.values
            print(f'  {keys[i]:24s} x {keys[j]:24s} '
                  f'both-keep={100*(a&b)[m].mean():.1f}% jaccard='
                  f'{100*(a & b)[m].sum() / max((a | b)[m].sum(), 1):.1f}%')

    print('\n== rule-valid singles and stacks ==')
    rep('BASE unfiltered', base)
    combos = {}
    for tag, ok in valid.items():
        combos[tag] = ok
    ks = list(valid)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            combos[f'{ks[i]} + {ks[j]}'] = valid[ks[i]] & valid[ks[j]]
    if len(ks) >= 3:
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                for k in range(j + 1, len(ks)):
                    combos[f'{ks[i]} + {ks[j]} + {ks[k]}'] = valid[ks[i]] & valid[ks[j]] & valid[ks[k]]
    for nm, ok in combos.items():
        d = F.day_df(data, F.simulate2(data, UB, LB, day_ok=ok))
        rep(nm, d)

    pd.DataFrame(rows).to_csv(H + 'step2b_stacks.csv', index=False)
    print(f'\ncells in this file: {len(rows)}')


if __name__ == '__main__':
    main()
