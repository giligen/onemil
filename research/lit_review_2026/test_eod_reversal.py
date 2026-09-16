#!/usr/bin/env python3
"""H-B5 (Baltussen-Da-Soebhag 2024, end-of-day reversal): names whose prior-close→15:00 return is <= -8% are bought at the
15:30 bar's open and sold at the 15:59 close (MOC). Symmetric winner tail (+8%) as the control. Universe: point-in-time daily
panel, close >= $5, 20-day dollar volume >= $2M; bars from cache.db then bars_sip.db (SIP). Costs: half a 40-bps spread on the
entry, the close fill at the auction (0). Every input known at 15:30. Splits fixed. Output eod_reversal.md"""
import os, sqlite3, numpy as np, pandas as pd
ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT)
d = pd.read_parquet('research/lit_review_2026/daily_panel.parquet', columns=['symbol', 'bar_date', 'prev_close', 'low', 'high', 'close', 'dvol20'])
d = d[(d.close >= 5) & (d.dvol20 >= 2e6) & (d.prev_close > 0)]
import re
bad = [c for c in d.symbol.cat.categories if re.match(r'^Z[VWX]ZZ|^ZZ', str(c))] if hasattr(d.symbol, 'cat') else []
d = d[~d.symbol.isin(bad)] if bad else d
cand = d[(d.low <= d.prev_close * 0.92) | (d.high >= d.prev_close * 1.08)]           # superset: the 15:00 return is checked on the bars
print('candidate symbol-days', len(cand), flush=True)
import sys
if '--summary' in sys.argv:
    rows = pd.read_csv('research/lit_review_2026/eod_reversal_trades.csv', dtype={'symbol': str}).to_dict('records'); cand = cand.iloc[0:0]
cache = sqlite3.connect('file:data/cache.db?mode=ro', uri=True); sip = sqlite3.connect('file:research/bf_zero/bars_sip.db?mode=ro', uri=True)
rows = []
for day, g in cand.groupby('bar_date'):
    syms = g.symbol.astype(str).tolist(); pc = dict(zip(g.symbol.astype(str), g.prev_close))
    q = f"select symbol, timestamp as t, open as o, close as c from intraday_bars_1min where bar_date=? and symbol in ({','.join('?' * len(syms))})"
    b = pd.read_sql(q, cache, params=[day] + syms)
    left = [s for s in syms if s not in set(b.symbol)]
    if left:
        b2 = pd.read_sql("select symbol, t, o, c from bars where day=?", sip, params=[day]); b = pd.concat([b, b2[b2.symbol.isin(left)]], ignore_index=True)
    if not len(b): continue
    ts = pd.to_datetime(b.t, utc=True).dt.tz_convert('America/New_York'); b['m'] = ts.dt.hour * 60 + ts.dt.minute
    for s, gg in b.groupby('symbol'):
        gg = gg[(gg.m >= 570) & (gg.m < 960)].sort_values('m')
        if len(gg) < 30: continue
        at15 = gg[gg.m <= 900]; entry = gg[gg.m >= 930]; last = gg[gg.m <= 959]
        if not len(at15) or not len(entry) or not len(last): continue
        p15 = float(at15.c.iloc[-1]); e = float(entry.o.iloc[0]); x = float(last.c.iloc[-1])
        if e <= 0 or p15 <= 0 or pc[s] <= 0: continue
        r15 = p15 / pc[s] - 1
        if r15 <= -0.08 or r15 >= 0.08:
            rows.append(dict(day=day, symbol=s, r15=r15, side='loser' if r15 <= -0.08 else 'winner', entry=e, exit=x, ret=x / e - 1))
T = pd.DataFrame(rows); T['split'] = np.where(T.day < '2026-01-01', 'TRAIN', np.where(T.day < '2026-06-01', 'VAL', 'TEST'))
T['net'] = T.ret - 0.0020; T['wk'] = pd.to_datetime(T.day).dt.to_period('W-FRI').astype(str)
T.to_csv('research/lit_review_2026/eod_reversal_trades.csv', index=False)
T = T[T.ret.abs() <= 0.5]                                                          # bad prints / halts reopening: not a fill we would get
def tstat(v): return v.mean() / (v.std() / np.sqrt(len(v))) if len(v) > 1 and v.std() > 0 else float('nan')
lines = ['# H-B5 end-of-day reversal — 15:30 open → 15:59 close, prior-close→15:00 return tails', '']
for side in ('loser', 'winner'):
    for sp in ('TRAIN', 'VAL', 'TEST'):
        x = T[(T.side == side) & (T.split == sp)]
        if not len(x): lines.append(f'{side} {sp}: none'); continue
        w = x.groupby('wk').net.sum()
        lines.append(f"{side:6s} {sp:5s} n {len(x):5d} ({len(x) / x.day.nunique():.1f}/day) gross {x.ret.mean() * 1e4:+6.1f} bps median {x.ret.median() * 1e4:+6.1f} net {x.net.mean() * 1e4:+6.1f} t {tstat(x.net):+5.2f} hit {(x.net > 0).mean() * 100:4.1f}% | weeks green {int((w > 0).sum())}/{w.size} worst {w.min() * 100:+.1f}%")
    lines.append('')
for lo, hi in ((-0.30, -0.15), (-0.15, -0.10), (-0.10, -0.08)):
    x = T[(T.side == 'loser') & (T.r15 > lo) & (T.r15 <= hi)]
    if len(x): lines.append(f'loser bucket {lo:+.2f}..{hi:+.2f}: n {len(x)} gross {x.ret.mean() * 1e4:+.1f} bps median {x.ret.median() * 1e4:+.1f} hit {(x.ret > 0).mean() * 100:.1f}%')
open('research/lit_review_2026/eod_reversal.md', 'w').write('\n'.join(lines)); print('\n'.join(lines), flush=True); print('DONE')
