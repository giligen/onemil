#!/usr/bin/env python3
"""The stale-fill probe on the 1,676 claimed trades only (no parquet, low memory): how far had the
market ALREADY traded above the fill price before the 'entry' bar, and was the +2R target already
exceeded before entry? Read-only."""
import os, sqlite3
import numpy as np, pandas as pd
ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT)
A = 'research/bf_zero2/audit_data'
cache = sqlite3.connect('file:data/cache.db?mode=ro', uri=True, timeout=180)
sip = sqlite3.connect('file:research/bf_zero/bars_sip.db?mode=ro', uri=True, timeout=180)
b = pd.read_csv('research/bf_zero2/f6_2r_book.csv', dtype={'symbol': str, 'day': str}, keep_default_na=False)
b['entry'] = b.price; b['stop'] = b.price * (1 - b.r_pct / 100.0)


def load(day, syms):
    out = {}
    q = ("select symbol, timestamp as t, open as o, high as h, low as l, close as c from intraday_bars_1min "
         f"where bar_date=? and symbol in ({','.join('?' * len(syms))})")
    for s, gg in pd.read_sql(q, cache, params=[day] + list(syms)).groupby('symbol'): out[s] = gg
    left = [s for s in syms if s not in out]
    if left:
        t = pd.read_sql("select symbol, t, o, h, l, c from bars where day=?", sip, params=[day])
        for s, gg in t[t.symbol.isin(left)].groupby('symbol'): out[s] = gg
    res = {}
    for s, gg in out.items():
        ts = pd.to_datetime(gg.t, utc=True).dt.tz_convert('America/New_York')
        gg = gg.assign(m=(ts.dt.hour * 60 + ts.dt.minute).values).sort_values('m').drop_duplicates('m')
        res[s] = gg[['m', 'o', 'h', 'l', 'c']].reset_index(drop=True)
    return res


rows = []
for day, sub in b.groupby('day'):
    bars = load(day, sub.symbol.unique().tolist())
    for r in sub.itertuples():
        gg = bars.get(r.symbol)
        if gg is None: continue
        rth = gg[(gg.m >= 570) & (gg.m < 960)].reset_index(drop=True)
        w = np.flatnonzero(rth.m.values == r.em)
        if not len(w): continue
        i = int(w[0])
        h = rth.h.values.astype(float); l = rth.l.values.astype(float); o = rth.o.values.astype(float)
        R = r.entry - r.stop; tgt = r.entry + 2 * R
        rows.append(dict(day=day, symbol=r.symbol, em=r.em, net=r.net, why=r.why_e1c, split=r.split, i=i,
                         entry=r.entry, stop=r.stop, r_pct=r.r_pct,
                         hmax_before=float(h[:i].max()) if i else np.nan,
                         hmax_incl=float(h[:i + 1].max()),
                         ei_l=float(l[i]), ei_o=float(o[i]), ei_h=float(h[i]),
                         nxt_o=float(o[i + 1]) if i + 1 < len(o) else np.nan,
                         crossed_before=int(i > 0 and h[:i].max() >= r.entry),
                         fill_below_bar_low=int(r.entry < l[i] - 1e-9),
                         target_already_exceeded=int(h[:i + 1].max() >= tgt),
                         target_exceeded_before=int(i > 0 and h[:i].max() >= tgt)))
R = pd.DataFrame(rows)
R.to_csv(f'{A}/stale_fill_book.csv', index=False)
print('rows', len(R), flush=True)
R['early'] = R.em <= 572
for tag, x in (('ALL', R), ('early em<=572', R[R.early]), ('late em>572', R[~R.early])):
    print(f'\n== {tag}  n={len(x)}')
    print('  level already traded through BEFORE the signal bar: %.1f%%' % (100 * x.crossed_before.mean()))
    print('  fill price BELOW the signal bar low (impossible): %.1f%%' % (100 * x.fill_below_bar_low.mean()))
    print('  +2R target already exceeded by the signal bar:      %.1f%%' % (100 * x.target_already_exceeded.mean()))
    print('  +2R target already exceeded BEFORE the signal bar:  %.1f%%' % (100 * x.target_exceeded_before.mean()))
    print('  market already above the fill by (pct of entry), before the signal bar:')
    q = ((x.hmax_before / x.entry - 1) * 100).describe(percentiles=[.25, .5, .75, .9]).round(2).to_dict()
    print('   ', {k: round(v, 2) for k, v in q.items()})
print('\nP&L split by the impossible-fill flag:')
print(R.groupby(['split', 'fill_below_bar_low']).net.agg(['count', 'mean', 'sum']).round(3).to_string())
print('\nP&L split by target-already-exceeded:')
print(R.groupby(['split', 'target_already_exceeded']).net.agg(['count', 'mean', 'sum']).round(3).to_string())
print('\nP&L split by crossed_before:')
print(R.groupby(['split', 'crossed_before']).net.agg(['count', 'mean', 'sum']).round(3).to_string())
print('\nwhy_e1c x target_already_exceeded:')
print(pd.crosstab(R.target_already_exceeded, R.why).to_string())
print('DONE', flush=True)
