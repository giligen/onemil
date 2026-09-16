#!/usr/bin/env python3
"""RUNBOOK rows 11, 12, 13 on the ETF 1-minute store (2016-01 → 2026-09, SIP).
 M11  SPY overnight premium: buy the 15:59 close, sell the 09:30 open, every day. 1 bp/leg.
 M12  conditional overnight reversal: the same trade only after a bottom-quintile open→close day (quintile cut on the IS years);
      plus the regression of the next overnight return on the prior last-30-min return (slope expected negative).
 M5   volatility-gated last-30-min timing: the four A1 variants restricted to days whose SPY 09:30–10:00 realized volatility
      (sum of squared 1-min returns) is in the TOP TERCILE of the trailing 250 sessions (causal).
 M10  QQQ VWAP flip: long above the running RTH VWAP, short below, flip on every 1-min close crossing, flat 16:00, 1 bp/leg.
IS = 2016–2023, OOS = 2024-01 → today. Output etf_queue.md."""
import os, sqlite3, numpy as np, pandas as pd
ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT)
con = sqlite3.connect('file:research/lit_review_2026/etf_1min.db?mode=ro', uri=True)

def load(sym):
    b = pd.read_sql("select t, o, h, l, c, v from bars where symbol=? order by t", con, params=(sym,))
    ts = pd.to_datetime(b.t, utc=True).dt.tz_convert('America/New_York')
    b['day'] = ts.dt.strftime('%Y-%m-%d'); b['m'] = ts.dt.hour * 60 + ts.dt.minute
    return b[(b.m >= 570) & (b.m < 960)]

def rep(x, col, name):
    if len(x) < 30: return f'{name}: n {len(x)} — underpowered'
    v = x[col].dropna()
    return (f"{name}: n {len(v):5d} mean {v.mean()*1e4:+6.2f} bps t {v.mean()/(v.std()/np.sqrt(len(v))):+5.2f} hit {(v>0).mean()*100:4.1f}% "
            f"ann {v.mean()*252*100:+5.1f}% SR {v.mean()/v.std()*np.sqrt(252):+5.2f}")

L = ['# RUNBOOK rows 11–13 — SPY overnight, volatility-gated last-30-min, QQQ VWAP flip', '']
spy = load('SPY'); rows = []
for day, g in spy.groupby('day'):
    if len(g) < 300: continue
    first30 = g[g.m < 600]                                        # 09:30-09:59 closes
    r = np.diff(np.log(first30.c.values)) if len(first30) >= 20 else np.array([np.nan])
    rows.append(dict(day=day, o=g.o.iloc[0], c=g.c.iloc[-1], p1000=first30.c.iloc[-1], p1500=g[g.m <= 899].c.iloc[-1], p1530=g[g.m <= 929].c.iloc[-1],
                     rv30=float(np.nansum(r ** 2)) if np.isfinite(r).any() else np.nan))
D = pd.DataFrame(rows).sort_values('day').reset_index(drop=True)
D['prev_c'] = D.c.shift(1); D['on'] = D.o / D.prev_c - 1; D['id'] = D.c / D.o - 1
D['r_first'] = D.p1000 / D.prev_c - 1; D['r_last'] = D.c / D.p1530 - 1; D['on_next'] = D.on.shift(-1)
D['oos'] = D.day >= '2024-01-01'
# M11 / M12
for label, m in (('IS 2016-23', ~D.oos), ('OOS 2024-26', D.oos)):
    x = D[m & D.on.notna()].copy(); x['net'] = x.on - 0.0002
    L.append(rep(x, 'net', f'M11 SPY buy close sell open [{label}]'))
    q = D[~D.oos]['id'].quantile(0.2)
    y = D[m & (D['id'].shift(1) <= q) & D.on.notna()].copy(); y['net'] = y.on - 0.0002
    L.append(rep(y, 'net', f'M12 overnight after a bottom-quintile intraday day, id<={q*100:.2f}% [{label}]'))
    z = D[m & D.r_last.notna() & D.on_next.notna()]
    if len(z) > 50:
        b = np.polyfit(z.r_last, z.on_next, 1)[0]; c = np.corrcoef(z.r_last, z.on_next)[0, 1]
        L.append(f'M12b slope of next overnight on the last-30-min return [{label}]: {b:+.4f} (corr {c:+.3f}, n {len(z)}) — negative = reversal')
# M5 volatility-gated last-30-min
D['rv_thr'] = D.rv30.shift(1).rolling(250, min_periods=100).quantile(2/3)
gated = D[D.rv30.notna() & D.rv_thr.notna() & (D.rv30 >= D.rv_thr)]
L.append('')
for label, m in (('IS 2016-23', ~gated.oos), ('OOS 2024-26', gated.oos)):
    x = gated[m & gated.r_first.notna()].copy()
    for nm, sig in (('first-half-hour sign', np.sign(x.r_first)), ('15:00→15:30 sign', np.sign(x.p1530 / x.p1500 - 1))):
        y = x.copy(); y['net'] = sig * y.r_last - 0.0002
        L.append(rep(y, 'net', f'M5 high-vol days only ({len(x)} of the split), {nm} [{label}]'))
# M10 QQQ VWAP flip
qqq = load('QQQ'); rows = []
for day, g in qqq.groupby('day'):
    if len(g) < 300: continue
    cum_v = g.v.cumsum().values; vwap = (g.c.values * g.v.values).cumsum() / np.maximum(cum_v, 1)
    c = g.c.values; pos = 0; entry = None; pnl = 0.0; legs = 0
    for i in range(1, len(c)):
        want = 1 if c[i] > vwap[i] else -1
        if want != pos:
            if pos != 0: pnl += pos * (c[i] - entry) / entry; legs += 1
            pos, entry = want, c[i]; legs += 1
    if pos != 0: pnl += pos * (c[-1] - entry) / entry; legs += 1
    rows.append(dict(day=day, net=pnl - legs * 0.0001, legs=legs))
V = pd.DataFrame(rows); V['oos'] = V.day >= '2024-01-01'
L.append('')
for label, m in (('IS 2016-23', ~V.oos), ('OOS 2024-26', V.oos)):
    x = V[m]; L.append(rep(x, 'net', f'M10 QQQ VWAP flip, 1bp/leg [{label}], flips/day {x.legs.mean():.1f}'))
open('research/lit_review_2026/etf_queue.md', 'w').write('\n'.join(L)); print('\n'.join(L)); print('DONE', flush=True)
