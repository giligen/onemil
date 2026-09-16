#!/usr/bin/env python3
"""H-B1: Zarattini-Barbon-Aziz 'A Profitable Day Trading Strategy for the U.S. Equity Market' — OUT-OF-SAMPLE replication
2025-01 → 2026-09 on the liquid universe (prior close > $5, ADV14 >= 1M, ATR14 > $0.50). At 09:35: RV = 09:30-09:34 volume ÷ the
mean of the same window over the prior 14 days; top-20 by RV with RV >= 1. Direction = the sign of the first 5-min candle;
entry = stop order at the 5-min high (long) / low (short), filled when touched (+ the paper's 0 slippage; we also report at our
half-spread); stop = 0.1 × ATR14 from entry; no target; exit at the 15:59 close. Paper: +0.08R/trade at RV >= 1, +0.38R at RV > 30x,
IRR 41.6%, Sharpe 2.81 (2016-2023). Costs: paper $0.0035/share; here 1 bp/leg (their liquid names) AND a 20-bps round trip."""
import os, sqlite3, numpy as np, pandas as pd
ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT)
con = sqlite3.connect('file:research/lit_review_2026/liquid_open.db?mode=ro', uri=True)
top = pd.read_csv('research/lit_review_2026/liquid_top20.csv', dtype={'symbol': str})
d = pd.read_parquet('research/lit_review_2026/daily_panel.parquet', columns=['symbol', 'bar_date', 'high', 'low', 'prev_close'])
d['symbol'] = d.symbol.astype(str); d = d.sort_values(['symbol', 'bar_date'])
tr = np.maximum(d.high - d.low, np.maximum((d.high - d.prev_close).abs(), (d.low - d.prev_close).abs()))
d['atr14'] = tr.groupby(d.symbol).transform(lambda s: s.shift(1).rolling(14, min_periods=14).mean())
atr = {(s, b): a for s, b, a in zip(d.symbol, d.bar_date, d.atr14)}
rows = []
for day, g in top.groupby('day'):
    b = pd.read_sql("select symbol, t, o, h, l, c, v from days where day=?", con, params=(day,))
    if not len(b): continue
    ts = pd.to_datetime(b.t, utc=True).dt.tz_convert('America/New_York'); b['m'] = ts.dt.hour * 60 + ts.dt.minute
    for r in g.itertuples():
        x = b[(b.symbol == r.symbol) & (b.m >= 570) & (b.m < 960)].sort_values('m')
        a = atr.get((r.symbol, day))
        if len(x) < 60 or not a or np.isnan(a): continue
        f = x[x.m < 575]; rest = x[x.m >= 575]
        if len(f) < 3 or not len(rest): continue
        o5, c5, hi, lo = f.o.iloc[0], f.c.iloc[-1], f.h.max(), f.l.min()
        if c5 == o5: continue
        side = 1 if c5 > o5 else -1; level = hi if side > 0 else lo
        hit = rest[(rest.h >= level) if side > 0 else (rest.l <= level)]
        if not len(hit): rows.append(dict(day=day, symbol=r.symbol, rv=r.rv, side=side, filled=0, rr=np.nan, why='no_fill', px=np.nan, entry=level)); continue
        walk = rest.loc[hit.index[0]:]; entry = level; stop = entry - side * 0.1 * a; px = None; why = 'close'
        for hh, ll in zip(walk.h.values, walk.l.values):
            if (side > 0 and ll <= stop) or (side < 0 and hh >= stop): px = stop; why = 'stop'; break
        if px is None: px = float(walk.c.iloc[-1])
        R = 0.1 * a
        rows.append(dict(day=day, symbol=r.symbol, rv=r.rv, side=side, filled=1, entry=entry, px=px, why=why, rr=side * (px - entry) / R, r_pct=R / entry * 100))
T = pd.DataFrame(rows); T['split'] = np.where(T.day < '2026-01-01', 'TRAIN', np.where(T.day < '2026-06-01', 'VAL', 'TEST'))
T['wk'] = pd.to_datetime(T.day).dt.to_period('W-FRI').astype(str)
F = T[T.filled == 1].copy()
F['cost_1bp_R'] = 0.0002 / (F.r_pct / 100); F['cost_20bp_R'] = 0.0020 / (F.r_pct / 100)          # round-trip cost in R units
F['net_1bp'] = F.rr - F.cost_1bp_R; F['net_20bp'] = F.rr - F.cost_20bp_R
F.to_csv('research/lit_review_2026/stocks_in_play_trades.csv', index=False)
pd.set_option('display.width', 250)
lines = [f'# H-B1 stocks-in-play ORB, OOS 2025-26 | picks {len(T)} | filled {int(T.filled.sum())} ({T.filled.mean()*100:.0f}%) | median R {F.r_pct.median():.2f}% of price', '']
for sp in ('TRAIN', 'VAL', 'TEST', 'ALL'):
    x = F if sp == 'ALL' else F[F.split == sp]
    if not len(x): continue
    w = x.groupby('wk').net_1bp.sum(); nw = x.wk.nunique()
    lines.append(f"{sp:5s} n {len(x):5d} ({len(x)/x.day.nunique():.1f}/day) gross {x.rr.mean():+.3f}R hit {(x.rr>0).mean()*100:.1f}% stop-rate {(x.why=='stop').mean()*100:.0f}% | net@1bp {x.net_1bp.mean():+.3f}R net@20bp {x.net_20bp.mean():+.3f}R | weekly net@1bp {w.sum()/nw:+.1f}R green {int((w>0).sum())}/{nw} worst {w.min():+.1f}")
lines.append('\nby RV bucket (paper: monotone, +0.08R at >=1, +0.38R at >30x):')
F['rvb'] = pd.cut(F.rv, [1, 2, 5, 10, 30, 1e9], labels=['1-2', '2-5', '5-10', '10-30', '>30'])
lines.append(F.groupby(['split', 'rvb'], observed=True).agg(n=('rr', 'size'), gross_R=('rr', 'mean'), net_1bp=('net_1bp', 'mean'), hit=('rr', lambda s: (s > 0).mean())).round(3).unstack('split').to_string())
lines.append('\nlong vs short:'); lines.append(F.groupby(['split', 'side']).agg(n=('rr', 'size'), gross_R=('rr', 'mean'), net_1bp=('net_1bp', 'mean')).round(3).to_string())
open('research/lit_review_2026/stocks_in_play.md', 'w').write('\n'.join(lines)); print('\n'.join(lines)); print('DONE')
