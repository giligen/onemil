#!/usr/bin/env python3
"""Stage H/F6_sizing step 2 — per-trade liquidity at the fill bar, for the WHOLE F6 population.

Bar source = EXACTLY the builder's precedence (research/bf_zero/build_candidates.py::load_bars):
data/cache.db intraday_bars_1min first, then research/bf_zero/bars_sip.db.  Both opened read-only.
(bars_sip.db alone covers only 41% of these symbol-days — it is the side store, not the primary.)

For every population row the fill bar is the 1-min bar at ET minute `next_entry_m` (the next-open fill).
Emits that bar's dollar volume (close x volume) and the dollar volume of the 5 bars ENDING at it
(minutes next_entry_m-4 .. next_entry_m; minutes with no print contribute $0).
"""
import os, sqlite3, time
import pandas as pd

ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT)
D = 'research/fuckup_audit/H/F6_sizing'
OUT = f'{D}/pop_f6_liq.csv'

d = pd.read_csv(f'{D}/pop_f6.csv', dtype={'symbol': str, 'day': str}, keep_default_na=False, na_values=[''])
cache = sqlite3.connect(f'file:{ROOT}/data/cache.db?mode=ro', uri=True, timeout=120)
side = sqlite3.connect(f'file:{ROOT}/research/bf_zero/bars_sip.db?mode=ro', uri=True, timeout=120)


def et_min(series):
    ts = pd.to_datetime(series, utc=True, format='mixed').dt.tz_convert('America/New_York')
    return (ts.dt.hour * 60 + ts.dt.minute).values


def day_bars(day, syms):
    """{symbol: {et_minute: (close, vol, high, low)}} with the builder's source precedence."""
    out = {}
    ph = ','.join('?' * len(syms))
    q = (f'select symbol, timestamp as t, high as h, low as l, close as c, volume as v '
         f'from intraday_bars_1min where bar_date=? and symbol in ({ph})')
    got = pd.read_sql(q, cache, params=[day] + list(syms))
    for s, gg in got.groupby('symbol'):
        out[s] = gg
    left = [s for s in syms if s not in out]
    if left:
        ph2 = ','.join('?' * len(left))
        t = pd.read_sql(f'select symbol, t, h, l, c, v from bars where day=? and symbol in ({ph2})',
                        side, params=[day] + list(left))
        for s, gg in t.groupby('symbol'):
            out[s] = gg
    res = {}
    for s, gg in out.items():
        m = et_min(gg.t)
        res[s] = {int(mm): (c, v, h, l) for mm, c, v, h, l in zip(m, gg.c, gg.v, gg.h, gg.l)}
    return res


fill_dv, five_dv, fill_v, n_bars5, sanity, src_ok = [], [], [], [], [], []
t0 = time.time()
idx = d.index.to_list()
order = []
for day, gg in d.groupby('day'):
    bars = day_bars(day, sorted(gg.symbol.unique()))
    for r in gg.itertuples():
        order.append(r.Index)
        m2 = bars.get(r.symbol, {})
        fm = int(r.next_entry_m)
        b = m2.get(fm)
        src_ok.append(1 if m2 else 0)
        fill_dv.append(b[0] * b[1] if b else float('nan'))
        fill_v.append(b[1] if b else float('nan'))
        s = 0.0; nb = 0
        for m in range(fm - 4, fm + 1):
            x = m2.get(m)
            if x:
                s += x[0] * x[1]; nb += 1
        five_dv.append(s if b else float('nan')); n_bars5.append(nb)
        sanity.append(1 if (b and b[3] - 1e-9 <= r.next_entry <= b[2] + 1e-9) else 0)
    if len(order) % 2000 < 40:
        print(f'{len(order):,}/{len(d):,}  {(time.time()-t0)/60:.1f}m', flush=True)

o = pd.DataFrame({'fill_bar_dollar_vol': fill_dv, 'fill_bar_vol': fill_v, 'five_min_dollar_vol': five_dv,
                  'n_bars_in_5': n_bars5, 'fill_in_bar': sanity, 'sym_day_found': src_ok}, index=order)
d = d.join(o)
d.to_csv(OUT, index=False)
print(f'DONE {len(d):,}  symbol-day found {d.sym_day_found.mean():.4f}  '
      f'fill bar present {d.fill_bar_dollar_vol.notna().mean():.4f}  '
      f'fill inside bar {d.fill_in_bar.mean():.4f}  {(time.time()-t0)/60:.1f}m -> {OUT}', flush=True)
