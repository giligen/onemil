#!/usr/bin/env python3
"""H/QQQ step 3 — TRAIN robustness of the candidate stack: monthly, tails, caps, weeks.
Also writes the day-level book CSVs for BASE and for the stack. TRAIN only."""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/ec2-user/onemil/research/fuckup_audit/H/QQQ')
sys.path.insert(0, '/home/ec2-user/onemil/research/fuckup_audit/Q')
import zsim as Z                                                    # noqa: E402
import filters as F                                                 # noqa: E402

H = F.H
TR = F.SPLITS['TRAIN']


def robust(d, lo, hi, tag):
    x = d[(d['date'] >= lo) & (d['date'] <= hi)].copy()
    td = x[x['ntr'] > 0]
    x['ym'] = pd.to_datetime(x['date']).dt.to_period('M').astype(str)
    mo = x.groupby('ym')['bps'].sum()
    out = dict(tag=tag, days=len(x), traded=len(td), bps=td['bps'].mean(),
               t=td['bps'].mean() / td['bps'].std(ddof=1) * np.sqrt(len(td)),
               cal=x['bps'].mean(), sum=x['bps'].sum(),
               sharpe=Z.metrics(x, 'r1x')['sharpe'], mdd=Z.metrics(x, 'r1x')['mdd'],
               green_day=100 * (td['bps'] > 0).mean(),
               months=len(mo), green_month=100 * (mo > 0).mean(),
               worst_month=mo.min(), best_month=mo.max())
    s = td['bps']
    cut1 = s.quantile(0.99); cut5 = s.quantile(0.95)
    out['ex_top1_bps'] = s[s < cut1].mean()
    out['ex_top5_bps'] = s[s < cut5].mean()
    out['cap_1pct_bps'] = s.clip(upper=100).mean()
    out['cap_05pct_bps'] = s.clip(upper=50).mean()
    xw = x.copy(); xw['wk'] = pd.to_datetime(xw['date']).dt.to_period('W').astype(str)
    wk = xw.groupby('wk').agg(b=('bps', 'sum'), n=('ntr', 'sum'))
    wk = wk[wk['n'] > 0]
    out['weeks'] = len(wk); out['green_week'] = 100 * (wk['b'] > 0).mean()
    out['worst_week'] = wk['b'].min()
    return out


def main():
    data = Z.load_symbol('QQQ')
    UB, LB = Z.bands(data, 1.0)
    fx = F.extra_features(data, UB, LB)
    f1 = (fx['band_w'] >= 0.8).values
    f2 = (fx['gap_over_band'] >= 0.20).values

    books = {
        'BASE': F.day_df(data, F.simulate2(data, UB, LB)),
        'F1 only': F.day_df(data, F.simulate2(data, UB, LB, day_ok=f1)),
        'F2 only': F.day_df(data, F.simulate2(data, UB, LB, day_ok=f2)),
        'STACK F1+F2': F.day_df(data, F.simulate2(data, UB, LB, day_ok=f1 & f2)),
    }
    rows = []
    for nm, d in books.items():
        for lab, (lo, hi) in (('TRAIN', TR), ('TRAIN-H1', F.H1), ('TRAIN-H2', F.H2)):
            rows.append(robust(d, lo, hi, f'{nm} | {lab}'))
    out = pd.DataFrame(rows)
    out.to_csv(H + 'step3_train_robust.csv', index=False)
    pd.set_option('display.width', 250)
    print(out.to_string(index=False, float_format=lambda v: f'{v:,.2f}'))

    # monthly table for the stack, TRAIN
    d = books['STACK F1+F2']
    x = d[(d['date'] >= TR[0]) & (d['date'] <= TR[1])].copy()
    x['ym'] = pd.to_datetime(x['date']).dt.to_period('M').astype(str)
    mo = x.groupby('ym')['bps'].sum()
    print(f'\nTRAIN stack months: {len(mo)} green {100*(mo>0).mean():.0f}% '
          f'worst {mo.min():.0f} best {mo.max():.0f}')

    # full-sample day books for the record (split column added by base.py convention)
    for nm, d in books.items():
        d2 = d.copy()
        d2['book'] = nm
        d2.to_csv(H + f'book_{nm.replace(" ", "_").replace("+", "")}.csv', index=False)


if __name__ == '__main__':
    main()
