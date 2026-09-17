#!/usr/bin/env python3
"""H/QQQ step 4 — the FROZEN stack applied to VAL (2023-01..2024-06). Read once.
Stack: F1 band_w >= 0.8% AND F2 |gap|/band >= 0.20.  Per-filter contribution reported."""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/ec2-user/onemil/research/fuckup_audit/H/QQQ')
sys.path.insert(0, '/home/ec2-user/onemil/research/fuckup_audit/Q')
import zsim as Z                                                    # noqa: E402
import filters as F                                                 # noqa: E402

H = F.H
VA = F.SPLITS['VAL']


def main():
    data = Z.load_symbol('QQQ')
    UB, LB = Z.bands(data, 1.0)
    fx = F.extra_features(data, UB, LB)
    f1 = (fx['band_w'] >= 0.8).values
    f2 = (fx['gap_over_band'] >= 0.20).values

    books = {'BASE': None, 'F1 only': f1, 'F2 only': f2, 'STACK F1+F2': f1 & f2}
    rows = []
    for nm, ok in books.items():
        d = F.day_df(data, F.simulate2(data, UB, LB, day_ok=ok))
        print(F.line(nm, d, *VA))
        g, nw, worst, mean = F.weekly_green(d, *VA)
        print(f'      weeks={nw} green={g:.1f}% worst_week={worst:.0f} mean_week={mean:+.1f}')
        a = F.stats(d, *VA)
        x = d[(d['date'] >= VA[0]) & (d['date'] <= VA[1])]
        td = x[x['ntr'] > 0]
        s = td['bps']
        rows.append(dict(book=nm, days=a['days'], traded=a['traded'], bps=a['bps'], t=a['t'],
                         sum=a['sum'], green_day=a['green'], cal=a['cal'], sharpe=a['sharpe'],
                         mdd=a['mdd'], weeks=nw, green_week=g, worst_week=worst,
                         ex_top5=s[s < s.quantile(0.95)].mean(), cap1=s.clip(upper=100).mean()))
        if nm == 'BASE':
            base = d

    # the vetoed buckets' own VAL means (rule 3b)
    print('\n-- vetoed buckets on VAL --')
    x = base[(base['date'] >= VA[0]) & (base['date'] <= VA[1]) & (base['ntr'] > 0)]
    for nm, ok in (('F1 dropped (band_w<0.8)', ~f1), ('F2 dropped (gap/band<0.20)', ~f2),
                   ('STACK dropped (either)', ~(f1 & f2))):
        dates = set(fx['date'][ok])
        sub = x[x['date'].isin(dates)]
        print(f'  {nm:30s} n={len(sub):4d} mean={sub["bps"].mean():+7.2f} '
              f'sum={sub["bps"].sum():+8.0f}')
        rows.append(dict(book=nm, traded=len(sub), bps=sub['bps'].mean(), sum=sub['bps'].sum()))

    pd.DataFrame(rows).to_csv(H + 'step4_val.csv', index=False)


if __name__ == '__main__':
    main()
