#!/usr/bin/env python3
"""ATTACK: the F6 fill. fam_r2g excludes bar 0 (`t >= 1`) but keeps the level = prev_close x 1.003.
If the 09:30 bar already traded through the level, the "first cross" is detected on a LATER bar and
filled at a stale price the market has left behind — possibly BELOW that bar's low (an impossible fill).
Re-derives every F6 eligible candidate from the bars, flags impossible fills, and re-books under
corrected fill models. Read-only; writes audit_data/*.csv."""
import os, sqlite3, sys
import numpy as np, pandas as pd
ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT); sys.path.insert(0, ROOT)
A = 'research/bf_zero2/audit_data'
SLIP = 0.003; EOD_M = 955; OPEN_M = 570

cache = sqlite3.connect('file:data/cache.db?mode=ro', uri=True, timeout=180)
sip = sqlite3.connect('file:research/bf_zero/bars_sip.db?mode=ro', uri=True, timeout=180)

daily = pd.read_parquet('data/research/databento/equs_daily_2025_2026.parquet',
                        columns=['bar_date', 'symbol', 'open', 'high', 'low', 'close'])
daily['bar_date'] = daily.bar_date.astype(str).str[:10]
daily = daily[daily.symbol.notna()].sort_values(['symbol', 'bar_date']).reset_index(drop=True)
g = daily.groupby('symbol')
daily['prev_close'] = g.close.shift(1)
daily['prev_date'] = g.bar_date.shift(1)
PC = daily.set_index(['symbol', 'bar_date'])[['prev_close', 'prev_date']]

e = pd.read_csv(f'{A}/f6_eligible.csv', dtype={'day': str, 'symbol': str, 'why_e1c': str}, keep_default_na=False, na_values=[''])
print('eligible', len(e), flush=True)


def load_bars(day, syms):
    out = {}
    q = ("select symbol, timestamp as t, open as o, high as h, low as l, close as c, volume as v "
         f"from intraday_bars_1min where bar_date=? and symbol in ({','.join('?' * len(syms))})")
    for s, gg in pd.read_sql(q, cache, params=[day] + list(syms)).groupby('symbol'): out[s] = gg
    left = [s for s in syms if s not in out]
    if left:
        t = pd.read_sql("select symbol, t, o, h, l, c, v from bars where day=?", sip, params=[day])
        for s, gg in t[t.symbol.isin(left)].groupby('symbol'): out[s] = gg
    res = {}
    for s, gg in out.items():
        ts = pd.to_datetime(gg.t, utc=True).dt.tz_convert('America/New_York')
        gg = gg.assign(m=(ts.dt.hour * 60 + ts.dt.minute).values).sort_values('m').drop_duplicates('m')
        res[s] = gg[['m', 'o', 'h', 'l', 'c', 'v']].reset_index(drop=True)
    return res


def exit_close_fill(o, h, l, c, m, i, entry, stop, mult=2.0):
    Rd = entry - stop
    oo, hh, ll, cc, mm = o[i + 1:], h[i + 1:], l[i + 1:], c[i + 1:], m[i + 1:]
    if len(oo) == 0 or Rd <= 0: return 0.0, 'none', int(m[i])
    tgt = entry + mult * Rd
    for k in range(len(oo)):
        if mm[k] >= EOD_M: return (oo[k] - entry) / Rd, 'eod', int(mm[k])
        if ll[k] <= stop: return (min(stop, oo[k]) * 0.999 - entry) / Rd, 'stop', int(mm[k])
        if cc[k] >= tgt: return mult, 'target', int(mm[k])
    return (cc[-1] - entry) / Rd, 'eod', int(mm[-1])


rows = []
days = sorted(e.day.unique())
for n, day in enumerate(days):
    sub = e[e.day == day]
    bars = load_bars(day, sub.symbol.unique().tolist())
    for r in sub.itertuples():
        gg = bars.get(r.symbol)
        if gg is None: continue
        rth = gg[(gg.m >= OPEN_M) & (gg.m < 960)].reset_index(drop=True)
        if len(rth) < 10: continue
        o, h, l, c = (rth[k].values.astype(float) for k in ('o', 'h', 'l', 'c'))
        m = rth.m.values.astype(int)
        try:
            pc = float(PC.loc[(r.symbol, day), 'prev_close'])
        except KeyError:
            continue
        if not (pc == pc and pc > 0 and o[0] < pc): continue
        level = pc; lvl = level * (1 + SLIP)
        idx = np.flatnonzero((np.arange(len(h)) >= 1) & (h >= lvl))
        if not len(idx): continue
        i = int(idx[0])
        lo = np.minimum.accumulate(l)
        if lo[i - 1] >= level: continue
        stop = float(lo[i - 1]); entry = lvl
        if stop >= entry or i + 1 >= len(o): continue
        d = dict(day=day, symbol=r.symbol, entry_m=int(m[i]), i=i, entry=entry, stop=stop,
                 o0=float(o[0]), prev_close=pc, bar0_high=float(h[0]), bar0_low=float(l[0]),
                 ei_o=float(o[i]), ei_h=float(h[i]), ei_l=float(l[i]), ei_c=float(c[i]),
                 nxt_o=float(o[i + 1]), nbars=len(rth),
                 crossed_bar0=int(h[0] >= lvl), fill_below_bar_low=int(entry < l[i] - 1e-9),
                 r_pct=(entry - stop) / entry * 100,
                 range_so_far_excl_entry=(h[:i].max() - l[:i].min()) / o[0] * 100 if i > 0 else 0.0,
                 range_so_far_incl=(h[:i + 1].max() - l[:i + 1].min()) / o[0] * 100)
        rr, why, xm = exit_close_fill(o, h, l, c, m, i, entry, stop)
        d.update(rr_asis=rr, why_asis=why, xm_asis=xm)
        # V1: cannot be filled below the signal bar's low
        e1 = max(entry, float(l[i]))
        rr1, why1, xm1 = exit_close_fill(o, h, l, c, m, i, e1, stop)
        d.update(entry_v1=e1, rr_v1=rr1, why_v1=why1, xm_v1=xm1, r_pct_v1=(e1 - stop) / e1 * 100)
        # V3: pre-registered fill = next bar's open, capped at level x 1.006
        no = float(o[i + 1])
        ok3 = no <= level * 1.006 and no > stop and i + 2 < len(o)
        if ok3:
            rr3, why3, xm3 = exit_close_fill(o, h, l, c, m, i + 1, no, stop)
        else:
            rr3, why3, xm3 = np.nan, 'skip', 0
        d.update(entry_v3=no if ok3 else np.nan, rr_v3=rr3, why_v3=why3, xm_v3=xm3,
                 r_pct_v3=(no - stop) / no * 100 if ok3 else np.nan)
        rows.append(d)
    if n % 25 == 0:
        print(f'{n}/{len(days)} {day} rows {len(rows)}', flush=True)

R = pd.DataFrame(rows)
R.to_csv(f'{A}/f6_fill_audit.csv', index=False)
print('rederived', len(R), 'of', len(e), flush=True)
print('crossed in bar 0 (level already traded through before the signal bar): %.1f%%' % (100 * R.crossed_bar0.mean()), flush=True)
print('fill BELOW the signal bar low (impossible): %.1f%%' % (100 * R.fill_below_bar_low.mean()), flush=True)
print('by entry minute bucket:', flush=True)
b = pd.cut(R.entry_m, [569, 572, 575, 580, 600, 660, 842])
print(R.groupby(b, observed=True)[['crossed_bar0', 'fill_below_bar_low']].mean().round(3).to_string(), flush=True)
print('\ngap below the bar low, when impossible (as %% of entry):', flush=True)
z = R[R.fill_below_bar_low == 1]
print(((z.ei_l - z.entry) / z.entry * 100).describe(percentiles=[.5, .9, .99]).round(3).to_string(), flush=True)
print('DONE', flush=True)
