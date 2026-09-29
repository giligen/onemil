#!/usr/bin/env python3
"""Refuter follow-ups for cell 1,625: tau-interpretation sensitivity (report-only, not a tuning pass),
exact-minute obtainability of the SPY/IWM entry/exit bars, arm_m-vs-fill lag, post-10:00 cadence."""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, '/home/ec2-user/onemil/research/hod_entry/review')
import importlib.util
spec = importlib.util.spec_from_file_location('chk', '/home/ec2-user/onemil/research/hod_entry/review/1623_index_refuter_chk.py')
chk = importlib.util.module_from_spec(spec); spec.loader.exec_module(chk)
from zoneinfo import ZoneInfo
H = chk.HERE
sig = pd.read_csv(f'{H}/cell_1625_signals.csv')
fills = chk.load_fills()
day_hold = fills.drop_duplicates('day').set_index('day').holdout.to_dict(); days = sorted(day_hold)
lag = fills.fill_min - fills.arm_m
print('lag<1 share', round(float((lag < 1).mean()), 4), 'arm_m==floor(fill_min) share', round(float((fills.arm_m == np.floor(fills.fill_min)).mean()), 4))
bars = pd.read_parquet(f'{H}/index_bars_1625.parquet', columns=['symbol', 'day', 't', 'o', 'c', 'v', 'source'])
ts = pd.to_datetime(bars['t'], utc=True).dt.tz_convert(ZoneInfo('America/New_York'))
bars['m'] = ts.dt.hour * 60 + ts.dt.minute; bars['d_et'] = ts.dt.strftime('%Y-%m-%d')
have = set(zip(bars.symbol, bars.d_et, bars.m))
for s in ('SPY', 'IWM'):
    miss_e = sum((s, d, m + 1) not in have for d, m in zip(sig.day, sig.m_star))
    miss_x = sum((s, d, min(m + 61, 955)) not in have for d, m in zip(sig.day, sig.m_star))
    print(s, 'entry minute missing (tolerance used):', miss_e, 'exit minute missing:', miss_x)
O = chk.open_matrix(bars, 'SPY', days); R60 = chk.ret_matrix(O, 61)
# tau interpretations: (i) builder full-session grid p90 = 7; (ii) p90 over 09:30-14:30 minutes only (the arming
# window + 30 min; after it B30 is 0 by construction); (iii) the rebuild's event-pooled tau = 23
by_day = {}
for d, sub in fills.groupby('day'):
    grid, cnt = chk.grid_counts(np.floor(sub.arm_m.to_numpy()).astype(int)); by_day[d] = cnt
win = (grid >= 570) & (grid <= 870)
tau_win = float(np.percentile(np.concatenate([v[win] for d, v in by_day.items() if day_hold[d] == 'TRAIN-H2']), 90))
for name, tau in (('builder_7', 7.0), ('window_0930_1430', tau_win), ('rebuild_event_23', 23.0)):
    rows = []
    for d in days:
        hit = np.nonzero(by_day[d] >= tau)[0]
        if len(hit):
            m = int(grid[hit[0]]); rows.append((d, day_hold[d], m, R60.at[d, m]))
    df = pd.DataFrame(rows, columns=['day', 'split', 'm', 'r'])
    out = {h: dict(chk.stats(df[df.split == h].r), share_pre10=round(float((df[df.split == h].m < 600).mean()), 2))
           for h in ('TRAIN-H2', 'VAL')}
    print(name, 'tau', tau, json.dumps(out))
v = sig[(sig.split == 'VAL') & (sig.m_star >= 600)]
print('VAL post-10 signals', len(v), 'weeks spanned', round((pd.to_datetime(sig[sig.split=='VAL'].day).max() - pd.to_datetime(sig[sig.split=='VAL'].day).min()).days / 7 + 1, 1),
      'post-10 days in VAL top5 SPY:', sorted(set(v.day) & {'2026-04-02', '2026-03-03', '2026-03-09', '2026-03-10', '2026-02-18'}))
vv = v.spy_ret_bps_60.sort_values(ascending=False); print('VAL post-10 SPY ex-top-2 mean', round(float(vv.iloc[2:].mean()), 2), 'top2', [round(x, 1) for x in vv.iloc[:2]])
