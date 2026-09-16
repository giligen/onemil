#!/usr/bin/env python3
"""H-B2: is the opening-range-break / HOD-break P&L monotone in relative volume (Zarattini's core claim: RV<1 −0.02R, RV>=1 +0.08R,
RV>30x +0.38R) on OUR population (>=5%-range days, price >= 5, SIP tape)? Reads bf_zero2/candidates_full.csv for F8 N=5 and
F5 K=8 X=0.02, hold-to-close exit (rr_e4, −1R stop) and the live exit (rr_e1c); buckets of rv_profile; gross R per trade."""
import numpy as np, pandas as pd
c = pd.read_csv('research/bf_zero2/candidates_full.csv', usecols=['fam', 'cfg', 'day', 'price', 'entry_m', 'rv_profile', 'rv_clock', 'rr_e4', 'rr_e1c', 'r_pct', 'dist_open_pct'],
                dtype={'fam': 'category', 'cfg': 'category', 'day': str}, keep_default_na=False, na_values=[''])
c = c[c.fam.isin(['F8', 'F5']) & (c.price >= 5) & (c.entry_m <= 841)]
c = c[((c.fam == 'F8') & (c.cfg == '{"N": 5}')) | ((c.fam == 'F5') & (c.cfg == '{"K": 8, "X": 0.02}'))]
c['split'] = np.where(c.day < '2026-01-01', 'TRAIN', np.where(c.day < '2026-06-01', 'VAL', 'TEST'))
pd.set_option('display.width', 220)
for fam in ('F8', 'F5'):
    x = c[c.fam == fam]
    for floor_name, xx in (('all', x), ('causal floor dist>=5', x[x.dist_open_pct >= 5])):
        xx = xx.assign(rv=pd.cut(xx.rv_profile, [0, 0.5, 1, 2, 5, 10, 30, 1e9], labels=['<0.5', '0.5-1', '1-2', '2-5', '5-10', '10-30', '>30']))
        t = xx.groupby(['split', 'rv'], observed=True).agg(n=('rr_e4', 'size'), hold_R=('rr_e4', 'mean'), live_R=('rr_e1c', 'mean'), r_pct=('r_pct', 'median')).round(3)
        print(f'\n## {fam} {"ORB-5" if fam == "F8" else "HOD K8"} — {floor_name}'); print(t.unstack('split').to_string())
