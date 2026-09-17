#!/usr/bin/env python3
"""H/QQQ step 2/3 — the candidate-filter grid, TRAIN ONLY (2016-01..2022-12).

Prints one line per cell: TRAIN, TRAIN-H1, TRAIN-H2, plus the removed bucket's own mean.
Writes H/QQQ/step2_grid.csv.  VAL and TEST are not touched.
"""
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


def report(name, df, base_df=None, kind='shape'):
    r = dict(cell=name, kind=kind)
    for lab, (lo, hi) in (('TRAIN', TR), ('H1', F.H1), ('H2', F.H2)):
        a = F.stats(df, lo, hi)
        r[f'{lab}_n'] = a['traded']; r[f'{lab}_bps'] = a['bps']; r[f'{lab}_t'] = a['t']
        r[f'{lab}_sum'] = a['sum']; r[f'{lab}_green'] = a['green']
    r['TRAIN_mdd'] = F.stats(df, *TR)['mdd']; r['TRAIN_sr'] = F.stats(df, *TR)['sharpe']
    rows.append(r)
    print(F.line(name, df, *TR))
    print('    ' + F.line('  H1', df, *F.H1))
    print('    ' + F.line('  H2', df, *F.H2))
    return r


def main():
    data = Z.load_symbol('QQQ')
    UB, LB = Z.bands(data, 1.0)
    fx = F.extra_features(data, UB, LB)
    D = len(data['days'])

    base_tr = F.simulate2(data, UB, LB)
    base = F.day_df(data, base_tr)
    print('\n===== BASELINE (unfiltered, scenario C) =====')
    report('BASE unfiltered', base, kind='base')

    print('\n===== F1 band-width floor =====')
    for c in (0.5, 0.6, 0.67, 0.7, 0.8):
        ok = (fx['band_w'] >= c).values
        d = F.day_df(data, F.simulate2(data, UB, LB, day_ok=ok))
        report(f'F1 band_w >= {c}%', d, kind='day')
        drop = base[~base['date'].isin(fx['date'][ok])]
        dtr = drop[(drop['date'] >= TR[0]) & (drop['date'] <= TR[1]) & (drop['ntr'] > 0)]
        d1 = dtr[dtr['date'] <= F.H1[1]]; d2 = dtr[dtr['date'] >= F.H2[0]]
        print(f'      dropped bucket: n={len(dtr)} mean={dtr["bps"].mean():+.2f} '
              f'H1 {d1["bps"].mean():+.2f} (n={len(d1)}) H2 {d2["bps"].mean():+.2f} (n={len(d2)})')
        rows[-1]['drop_n'] = len(dtr); rows[-1]['drop_bps'] = dtr['bps'].mean()
        rows[-1]['drop_h1'] = d1['bps'].mean(); rows[-1]['drop_h2'] = d2['bps'].mean()

    print('\n===== F2 gap/band floor =====')
    for c in (0.15, 0.20, 0.25, 0.30):
        ok = (fx['gap_over_band'] >= c).values
        d = F.day_df(data, F.simulate2(data, UB, LB, day_ok=ok))
        report(f'F2 |gap|/band >= {c}', d, kind='day')
        drop = base[~base['date'].isin(fx['date'][ok])]
        dtr = drop[(drop['date'] >= TR[0]) & (drop['date'] <= TR[1]) & (drop['ntr'] > 0)]
        d1 = dtr[dtr['date'] <= F.H1[1]]; d2 = dtr[dtr['date'] >= F.H2[0]]
        print(f'      dropped bucket: n={len(dtr)} mean={dtr["bps"].mean():+.2f} '
              f'H1 {d1["bps"].mean():+.2f} H2 {d2["bps"].mean():+.2f}')
        rows[-1]['drop_n'] = len(dtr); rows[-1]['drop_bps'] = dtr['bps'].mean()
        rows[-1]['drop_h1'] = d1['bps'].mean(); rows[-1]['drop_h2'] = d2['bps'].mean()

    print('\n===== F3 entry-count cap =====')
    for me in (1, 2):
        d = F.day_df(data, F.simulate2(data, UB, LB, max_entries=me))
        report(f'F3 max {me} entries/day', d, kind='shape')

    print('\n===== F4 side gate on the trailing trend =====')
    for nm, col in (('MA20', 'above_ma20'), ('MA5', 'above_ma5')):
        up = fx[col].values.astype(bool)
        d = F.day_df(data, F.simulate2(data, UB, LB, long_ok=up, short_ok=~up))
        report(f'F4 long only if prev_close>{nm}, short if <', d, kind='shape')

    print('\n===== F5 last check for a NEW entry =====')
    for lc, nm in ((150, '12:00'), (210, '13:00')):
        d = F.day_df(data, F.simulate2(data, UB, LB, last_check=lc))
        report(f'F5 entries only up to {nm} (k<={lc})', d, kind='shape')

    print('\n===== F6 realised-vol floor (rv20, causal) =====')
    for c in (0.70, 1.00):
        ok = (fx['rv20'] >= c).values
        d = F.day_df(data, F.simulate2(data, UB, LB, day_ok=ok))
        report(f'F6 rv20 >= {c}%', d, kind='day')

    print('\n===== F7 prior-5-session absolute movement floor =====')
    tr_mask = (fx['date'] >= TR[0]) & (fx['date'] <= TR[1])
    for p, nm in ((1 / 3, 'bottom tercile'), (0.25, 'bottom quartile')):
        c = fx.loc[tr_mask, 'absmove5'].quantile(p)
        ok = (fx['absmove5'] >= c).values
        d = F.day_df(data, F.simulate2(data, UB, LB, day_ok=ok))
        report(f'F7 absmove5 >= {c:.2f}% ({nm})', d, kind='day')

    print('\n===== STACKS (<=3 filters) =====')
    ok1 = (fx['band_w'] >= 0.67).values
    ok2 = (fx['gap_over_band'] >= 0.20).values
    for nm, ok, me in (
            ('S1 = F1(0.67)', ok1, 99),
            ('S2 = F2(0.20)', ok2, 99),
            ('S3 = F1(0.67) + F2(0.20)', ok1 & ok2, 99),
            ('S4 = F1(0.67) + F3(max1)', ok1, 1),
            ('S5 = F1(0.67) + F2(0.20) + F3(max1)', ok1 & ok2, 1),
            ('S6 = F2(0.20) + F3(max1)', ok2, 1),
            ('S7 = F3(max1) only', np.ones(D, bool), 1),
            ('S8 = F1(0.6) + F2(0.20) + F3(max1)',
             (fx['band_w'] >= 0.6).values & ok2, 1),
    ):
        d = F.day_df(data, F.simulate2(data, UB, LB, day_ok=ok, max_entries=me))
        report(nm, d, kind='stack')
        d.to_csv(H + f'stack_{nm.split()[0]}_days.csv', index=False)

    pd.DataFrame(rows).to_csv(H + 'step2_grid.csv', index=False)
    print(f'\ncells reported: {len(rows)}')


if __name__ == '__main__':
    main()
