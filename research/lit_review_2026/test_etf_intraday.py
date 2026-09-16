#!/usr/bin/env python3
"""Index-ETF intraday hypotheses on ten years of SIP 1-minute bars (research/lit_review_2026/etf_1min.db):
 A1 market intraday momentum (Gao-Han-Li-Zhou 2018): sign of the first-half-hour return (09:30→10:00, incl. overnight from
    prev close) predicts the last-half-hour return (15:30→16:00). Trade: at 15:30 long/short SPY in the sign of r_first, exit at
    the close. Report by year: mean bps/day, t, hit, annual Sharpe, net of 1 bp per leg.
 A1b same signal, long-only (buy when r_first > 0), and the second-to-last half-hour variant (15:00→15:30 sign).
 A2 Zarattini-Aziz SPY intraday momentum (2024): trend bands = open ± vol-scaled move (they use the average absolute move
    from the open at each minute over the last 14 days); long when price crosses above the upper band, short below the lower, flip on
    the opposite cross, exit at the close; sized by target vol. Simplified here to the band rule with 1 bp/leg cost.
 A3 Zarattini-Barbon-Aziz 5-min ORB on TQQQ/QQQ (2023): at 09:35 buy above the first 5-min high / short below the low in the
    direction of the first candle; stop = 0.1 × ATR14 (daily); no target; exit at the close; cost 1 bp/leg.
Splits: 2016-2023 = the papers' in-sample; 2024-01 → today = true out of sample (report both)."""
import os, sqlite3, numpy as np, pandas as pd
ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT)
con = sqlite3.connect('file:research/lit_review_2026/etf_1min.db?mode=ro', uri=True)

def load(sym):
    b = pd.read_sql("select t, o, h, l, c, v from bars where symbol=? order by t", con, params=(sym,))
    ts = pd.to_datetime(b.t, utc=True).dt.tz_convert('America/New_York'); b['day'] = ts.dt.strftime('%Y-%m-%d'); b['m'] = ts.dt.hour * 60 + ts.dt.minute
    return b[(b.m >= 570) & (b.m < 960)]

def stats(x, name):
    if not len(x): return dict(hyp=name, n=0)
    yrs = x.groupby(x.day.str[:4]).net.agg(['mean', 'count', 'std'])
    ann = x.net.mean() / x.net.std() * np.sqrt(252) if x.net.std() > 0 else np.nan
    return dict(hyp=name, n=len(x), bps=round(x.net.mean() * 1e4, 2), t=round(x.net.mean() / (x.net.std() / np.sqrt(len(x))), 2), hit=round((x.net > 0).mean() * 100, 1), sharpe=round(ann, 2),
                by_year=' '.join(f"{y}:{r['mean'] * 1e4:+.1f}" for y, r in yrs.iterrows()))

res = []
spy = load('SPY'); days = []
for day, g in spy.groupby('day'):
    if len(g) < 300: continue
    o930 = g.o.iloc[0]; p1000 = g[g.m <= 599].c.iloc[-1] if (g.m <= 599).any() else np.nan
    p1500 = g[g.m <= 899].c.iloc[-1]; p1530 = g[g.m <= 929].c.iloc[-1]; close = g.c.iloc[-1]
    days.append(dict(day=day, o=o930, p1000=p1000, p1500=p1500, p1530=p1530, close=close))
D = pd.DataFrame(days); D['prev_close'] = D.close.shift(1)
D['r_first'] = D.p1000 / D.prev_close - 1; D['r_first_open'] = D.p1000 / D.o - 1; D['r_last'] = D.close / D.p1530 - 1; D['r_1500'] = D.p1530 / D.p1500 - 1
D['oos'] = D.day >= '2024-01-01'
for label, mask in (('IS 2016-23', ~D.oos), ('OOS 2024-26', D.oos)):
    x = D[mask & D.r_first.notna()].copy()
    x['net'] = np.sign(x.r_first) * x.r_last - 0.0002; res.append(stats(x, f'A1 SPY first-half-hour sign → last-half-hour, L/S [{label}]'))
    x2 = x[x.r_first > 0].copy(); x2['net'] = x2.r_last - 0.0002; res.append(stats(x2, f'A1b long-only when first half hour up [{label}]'))
    x3 = x.copy(); x3['net'] = np.sign(x3.r_1500) * x3.r_last - 0.0002; res.append(stats(x3, f'A1c 15:00→15:30 sign → last half hour L/S [{label}]'))
    x4 = x.copy(); x4['net'] = np.sign(x4.r_first_open) * x4.r_last - 0.0002; res.append(stats(x4, f'A1d first half hour FROM THE OPEN sign → last half hour [{label}]'))
# A2 simplified band rule on SPY
spy = spy.copy(); spy['m_idx'] = spy.m - 570
prof = {}
rows = []
for day, g in spy.groupby('day'):
    if len(g) < 300: continue
    o = g.o.iloc[0]; mv = (g.c / o - 1).abs().values; prof[day] = pd.Series(mv, index=g.m.values)
daylist = sorted(prof)
for i, day in enumerate(daylist):
    if i < 14: continue
    base = pd.concat([prof[d] for d in daylist[i - 14:i]], axis=1).mean(axis=1)
    g = spy[spy.day == day]; o = g.o.iloc[0]; pos = 0; pnl = 0.0; entry = None; legs = 0
    for m, c in zip(g.m.values, g.c.values):
        band = base.get(m, np.nan)
        if np.isnan(band): continue
        up, lo = o * (1 + band), o * (1 - band)
        if pos <= 0 and c > up: 
            if pos < 0: pnl += (entry - c) / entry; legs += 1
            pos, entry = 1, c; legs += 1
        elif pos >= 0 and c < lo:
            if pos > 0: pnl += (c - entry) / entry; legs += 1
            pos, entry = -1, c; legs += 1
    if pos != 0: pnl += (g.c.iloc[-1] - entry) / entry * pos; legs += 1
    rows.append(dict(day=day, net=pnl - legs * 0.0001, legs=legs))
A2 = pd.DataFrame(rows)
for label, mask in (('IS 2016-23', A2.day < '2024-01-01'), ('OOS 2024-26', A2.day >= '2024-01-01')):
    res.append(stats(A2[mask], f'A2 SPY vol-band intraday momentum, flip, close exit, 1bp/leg [{label}]'))
# A3 5-min ORB on TQQQ and QQQ
for sym in ('TQQQ', 'QQQ'):
    b = load(sym); dly = b.groupby('day').agg(h=('h', 'max'), l=('l', 'min'), c=('c', 'last')).reset_index(); dly['pc'] = dly.c.shift(1)
    tr = np.maximum(dly.h - dly.l, np.maximum((dly.h - dly.pc).abs(), (dly.l - dly.pc).abs())); dly['atr14'] = tr.shift(1).rolling(14).mean(); atr = dict(zip(dly.day, dly.atr14))
    rows = []
    for day, g in b.groupby('day'):
        if len(g) < 300 or not atr.get(day) or np.isnan(atr[day]): continue
        f = g[g.m < 575]; rest = g[g.m >= 575]
        if len(f) < 3 or not len(rest): continue
        hi, lo, o5, c5 = f.h.max(), f.l.min(), f.o.iloc[0], f.c.iloc[-1]
        if c5 == o5: continue
        side = 1 if c5 > o5 else -1; level = hi if side > 0 else lo; stop_d = 0.1 * atr[day]
        hit = rest[(rest.h >= level) if side > 0 else (rest.l <= level)]
        if not len(hit): continue
        k = hit.index[0]; entry = level; walk = rest.loc[k:]
        stop = entry - side * stop_d; out = None
        for m, hh, ll, cc in zip(walk.m.values, walk.h.values, walk.l.values, walk.c.values):
            if (side > 0 and ll <= stop) or (side < 0 and hh >= stop): out = stop; break
        px = out if out is not None else walk.c.iloc[-1]
        rows.append(dict(day=day, net=side * (px - entry) / entry - 0.0002, stopped=out is not None))
    A3 = pd.DataFrame(rows)
    for label, mask in (('IS 2016-23', A3.day < '2024-01-01'), ('OOS 2024-26', A3.day >= '2024-01-01')):
        r = stats(A3[mask], f'A3 5-min ORB {sym}, stop 0.1 ATR, hold to close [{label}]'); r['stopped_pct'] = round(A3[mask].stopped.mean() * 100, 1); res.append(r)
R = pd.DataFrame(res); pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 120)
print(R.to_string(index=False)); R.to_csv('research/lit_review_2026/etf_intraday_results.csv', index=False)
