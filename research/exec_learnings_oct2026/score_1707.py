#!/usr/bin/env python3
"""Cell 1,707: marketable-at-submit vs resting split on the ORB tick-replay population (read-only).

Submit instant = 09:35:05 ET (t_sec = 5). MARKETABLE = last XNAS print at/before the submit instant
>= trigger*(1+x). Fill/pnl = replay_one at delay 5 (engine chase rule), results.csv. No new data.
"""
import math, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, '/home/ec2-user/onemil/research/orb_latency_bt')
import replay as rp

OUT = '/home/ec2-user/onemil/research/exec_learnings_oct2026'
SUBMIT = 5.0
R_UNIT = 375.0
pop = pd.read_csv(rp.POP_CSV)
amap = pd.read_csv('/home/ec2-user/onemil/data/research/orb_asset_class_map_20260711.csv').drop_duplicates('symbol').set_index('symbol')['asset_class']
rows = []
for r in pop.to_dict('records'):
    path = f"{rp.RAW_DIR}/{r['date']}__{r['symbol']}.parquet"
    base = dict(symbol=r['symbol'], date=r['date'], period=r['period'], exit_reason=r['exit_reason'],
                trigger=r['trigger'], limit=r['limit'], pnl_bt=r['_sized_pnl'], cls=amap.get(r['symbol'], 'unknown'))
    if not os.path.exists(path):
        rows.append({**base, 'covered': False}); continue
    raw = pd.read_parquet(path)
    tr = raw[raw.schema == 'trades'].copy(); mb = raw[raw.schema == 'mbp-1'].copy()
    tr['t_sec'] = rp.to_sec_since_0935(tr.ts_event.values, r['date']); mb['t_sec'] = rp.to_sec_since_0935(mb.ts_event.values, r['date'])
    tr = tr.sort_values('t_sec'); mb = mb.sort_values('t_sec')
    pre = tr[tr.t_sec <= SUBMIT]
    last = float(pre.price.iloc[-1]) if len(pre) else np.nan
    res = rp.replay_one(tr, mb, r['trigger'], r['limit'], SUBMIT)
    pnl = np.nan
    if res['status'] == 'filled':
        sh = 0.0 if pd.isna(r['shares']) else r['shares']
        pnl = r['_sized_pnl'] + sh * (r['entry_price'] - res['fill_price'])
    ts = res['t_star']
    ttf = (max(ts, SUBMIT) - SUBMIT) if ts is not None else np.nan
    rows.append({**base, 'covered': True, 'last_at_submit': last, 'status': res['status'], 't_star': ts,
                 'ttf': ttf, 'fill_price': res['fill_price'], 'pnl_replay': pnl})
df = pd.DataFrame(rows)
for x in (0, 10, 30):
    df[f'mkt{x}'] = df.last_at_submit >= df.trigger * (1 + x / 1e4)
df['noprint'] = df.covered & df.last_at_submit.isna()
df['bucket'] = pd.cut(df.ttf, [-1, 5, 34, 240, 1e9], labels=['<=5s', '5-34s', '34s-4m', '>4m'])
df['R'] = df.pnl_replay / R_UNIT; df['R_bt'] = df.pnl_bt / R_UNIT
df['stop'] = df.exit_reason == 'stop'
df['seg'] = np.where(df.date >= '2025-01-01', 'in', 'out')
df.to_csv(f'{OUT}/fills_1707.csv', index=False)

def stat(d, col='R'):
    """n, mean, day-clustered 95% CI, ex-top5%, cap3, stop share, wrapper share."""
    d = d.dropna(subset=[col]); n = len(d)
    if n == 0: return 'n=0'
    x = d[col].values; m = x.mean()
    g = d.assign(e=x - m).groupby('date').e.sum()
    se = math.sqrt((g ** 2).sum()) / n
    k = max(1, math.ceil(0.05 * n)); ex = np.sort(x)[:-k].mean() if n > k else float('nan')
    cap = np.minimum(x, 3).mean()
    return (f'n={n:3d} R={m:+.3f} CI[{m-1.96*se:+.3f},{m+1.96*se:+.3f}] ex5%={ex:+.3f} cap3={cap:+.3f} '
            f'stop={d.stop.mean():.0%} wrap={(d.cls=="wrapper").mean():.0%} sumR={x.sum():+.1f}')

cov = df[df.covered]
print(f'population {len(df)} covered {len(cov)} uncovered {len(df)-len(cov)}; noprint(<=5s lookback) {int(cov.noprint.sum())}')
print('status counts d=5:', cov.status.value_counts().to_dict())
disc = cov[(cov.t_star.notna()) & (cov.t_star < SUBMIT) & (~cov.mkt0)]
print(f'discordant: t*<submit but last<trigger at submit: {len(disc)}')
fl = cov[cov.status == 'filled'].copy()
segs = {'ALL': fl, 'in-regime': fl[fl.seg == 'in'], '2025H1': fl[fl.period == '2025H1'], '2025H2+26': fl[fl.period == '2025H2-2026-09'],
        'out 2023-24': fl[fl.seg == 'out'], 'out 23..24H1': fl[(fl.seg == 'out') & (fl.date < '2024-07-01')], 'out 2024H2': fl[(fl.seg == 'out') & (fl.date >= '2024-07-01')]}
for x in (0, 10, 30):
    print(f'\n=== A: replay-filled at d=5, marketable = last >= trigger*(1+{x}bp) (noprint -> RESTING) ===')
    for name, s in segs.items():
        print(f'-- {name}')
        print('  MKT  ', stat(s[s[f'mkt{x}']])); print('  REST ', stat(s[~s[f'mkt{x}']])); print('  ALL  ', stat(s))
print('\n=== buckets (time submit->fill), replay-filled ===')
for name in ('ALL', 'in-regime', 'out 2023-24'):
    for b in ['<=5s', '5-34s', '34s-4m', '>4m']:
        print(f'{name:12s} {b:7s}', stat(segs[name][segs[name].bucket == b]))
print('\n=== B: all covered BT-entered fills with BT pnl, classified at +0bp ===')
cb = cov.copy(); cb['R'] = cb.R_bt
for name, mask in {'in-regime': cb.seg == 'in', '2025H1': cb.period == '2025H1', '2025H2+26': cb.period == '2025H2-2026-09', 'out': cb.seg == 'out'}.items():
    s = cb[mask]; print(f'-- {name}'); print('  MKT ', stat(s[s.mkt0])); print('  REST', stat(s[~s.mkt0]))
wk = (pd.Timestamp('2026-09-23') - pd.Timestamp('2025-01-01')).days / 7
inr = fl[fl.seg == 'in']; print(f'\nweeks in-regime {wk:.1f}; replay-filled RESTING/wk {(~inr.mkt0).sum()/wk:.2f}; MKT/wk {(inr.mkt0).sum()/wk:.2f}; all {len(inr)/wk:.2f}')
print(f'BT-entered (covered) in-regime REST/wk {(~cb[cb.seg=="in"].mkt0).sum()/wk:.2f} all {len(cb[cb.seg=="in"])/wk:.2f}')
print('stop share by bucket in replay-filled in-regime:', fl[fl.seg=='in'].groupby('bucket', observed=True).stop.mean().round(3).to_dict())
print('BT stop share all entered:', pop.exit_reason.eq('stop').mean().round(3))
