"""Diagnostic: why do the builds pick different names on a few Mondays? Computes A-style (pandas, float32 returns, 273-bar guard window)
and B-style (cumsum, float64, 272-bar guard window) signal/guard/ADV for the one-sided names on their signal Fridays."""
import numpy as np, pandas as pd, pyarrow.parquet as pq
H = '/home/ec2-user/onemil/research/momentum_weekly/'
cases = {'2020-08-21': ['DKNG', 'LVGO', 'FSLY', 'JD'], '2020-10-23': ['PTON', 'PLUG'], '2018-10-26': ['ROKU', 'ESRX']}
syms = sorted({s for v in cases.values() for s in v})
t = pq.read_table(H + 'panel_2016_2026.parquet', columns=['symbol', 'bar_date', 'open', 'high', 'low', 'close', 'volume'], filters=[('symbol', 'in', syms)]).to_pandas()
t['bar_date'] = pd.to_datetime(t.bar_date)
for sig, ss in cases.items():
    for s in ss:
        d = t[t.symbol == s].sort_values('bar_date').reset_index(drop=True)
        d = d[~d.duplicated('bar_date', keep='last')].reset_index(drop=True)
        p = d.index[d.bar_date == pd.Timestamp(sig)]
        if len(p) == 0: print(sig, s, 'NO BAR', len(d), flush=True); continue
        p = int(p[0]); c32 = d.close.astype('float32'); c = c32.astype('float64').values
        ret32 = c32.pct_change(); vol_a = ret32.rolling(252, min_periods=252).std().iloc[p]
        r = np.zeros(len(c)); r[1:] = c[1:] / c[:-1] - 1; vol_b = r[p - 251:p + 1].std(ddof=1)
        sg_a = (c32.shift(21) / c32.shift(252) - 1).iloc[p] / vol_a; sg_b = (c[p - 21] / c[p - 252] - 1) / vol_b
        mv = np.abs(r); bad = np.flatnonzero((r > 2.0) | (r < -0.75)); gap = np.flatnonzero(np.r_[0, np.diff(d.bar_date.values).astype('timedelta64[D]').astype(float)] > 10)
        dv = (d.close.astype('float32') * d.volume.astype('float32')).astype('float64')
        print(f'{sig} {s}: bars {len(d)} p={p} sigA {sg_a:.5f} sigB {sg_b:.5f} volA {vol_a:.6f} volB {vol_b:.6f} adv20 {dv.iloc[p-19:p+1].mean():,.0f} '
              f'badmove idx {[(int(i), p - int(i)) for i in bad[-3:]]} gaps {[(int(i), p - int(i)) for i in gap[-3:]]}', flush=True)
