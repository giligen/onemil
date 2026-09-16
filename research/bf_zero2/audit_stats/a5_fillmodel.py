#!/usr/bin/env python3
"""a4 — follow-ups the six questions imply:
  (1) the honest (bootstrap, non-normal) per-split p-value of the book's t;
  (2) the first-come effect measured WITHIN day (day fixed effects) and within R-size strata —
      is "the first four" a signal or a proxy for a tight stop?
  (3) how much of the TRAIN->VAL->TEST rise is the changing wrapper mix;
  (4) cost / stop-slippage stress in bps of PRICE (the book lives in tight-R trades, where a fixed
      % slip is a large number of R);
  (5) how many target->stop conversions zero out each split;
  (6) the fill model: pass 1 fills at the touch inside the signal bar. The live spec (research/bf_zero
      REPORT 6) is a capped limit filled at the NEXT bar's open. Re-simulated from the real bars.
"""
import os, sys, sqlite3
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT); sys.path.insert(0, ROOT)
A = 'research/bf_zero2/audit_stats'
rng = np.random.default_rng(31337)
OUT = []
def P(*a):
    s = ' '.join(str(x) for x in a); OUT.append(s); print(s, flush=True)

book = pd.read_csv('research/bf_zero2/f6_2r_book.csv', keep_default_na=False, na_values=[''])
book['day'] = book.day.astype(str); book['symbol'] = book.symbol.astype(str)
pool = pd.read_parquet(f'{A}/pool_f6.parquet'); pool['day'] = pool.day.astype(str); pool['symbol'] = pool.symbol.astype(str)
SPL = ('TRAIN', 'VAL', 'TEST')
alldays = {s: sorted(pool.day[pool.split == s].unique()) for s in SPL}

# ---------------------------------------------------------------- 6
P('\n\n# 6 — the fill model: touch-inside-the-signal-bar vs the live capped limit at the NEXT bar open\n')
cache = sqlite3.connect('file:data/cache.db?mode=ro', uri=True)
side = sqlite3.connect('file:research/bf_zero/bars_sip.db?mode=ro', uri=True)

def bars_for(day, syms):
    out = {}
    q = ("select symbol, timestamp as t, open as o, high as h, low as l, close as c from intraday_bars_1min "
         f"where bar_date=? and symbol in ({','.join('?'*len(syms))})")
    try:
        t = pd.read_sql(q, cache, params=[day] + list(syms))
        for sy, g in t.groupby('symbol'): out[sy] = g
    except Exception as e:
        pass
    left = [s for s in syms if s not in out]
    if left:
        t = pd.read_sql(f"select symbol, t, o, h, l, c from bars where day=? and symbol in ({','.join('?'*len(left))})",
                        side, params=[day] + left)
        for sy, g in t.groupby('symbol'): out[sy] = g
    res = {}
    for sy, g in out.items():
        ts = pd.to_datetime(g.t, utc=True).dt.tz_convert('America/New_York')
        g = g.assign(m=(ts.dt.hour * 60 + ts.dt.minute).values).sort_values('m').drop_duplicates('m')
        res[sy] = g[['m', 'o', 'h', 'l', 'c']].reset_index(drop=True)
    return res

EOD = 955
rows = []
miss = 0
for day, gg in book.groupby('day'):
    bb = bars_for(day, sorted(gg.symbol.unique()))
    for r in gg.itertuples():
        d = bb.get(r.symbol)
        if d is None: miss += 1; continue
        d = d[(d.m >= 570) & (d.m < 960)].reset_index(drop=True)
        w = np.where(d.m.values == r.em)[0]
        if not len(w): miss += 1; continue
        i = int(w[0])
        level = r.price / 1.003                      # pass 1 entry = level x 1.003
        stop = r.price * (1 - r.r_pct / 100.0)
        o, h, l, c, m = (d[k].values.astype(float) for k in ('o', 'h', 'l', 'c')), None, None, None, None
        o = d.o.values.astype(float); h = d.h.values.astype(float); l = d.l.values.astype(float)
        c = d.c.values.astype(float); m = d.m.values.astype(int)
        rec = dict(day=day, symbol=r.symbol, split=r.split, wk=r.wk, as_is=r.net, r_pct=r.r_pct,
                   why=r.why_e1c, price=r.price)
        for cap, tag in ((0.006, 'cap60'), (0.010, 'cap100')):
            if i + 1 >= len(o): rec[tag] = np.nan; continue
            nxt = o[i + 1]
            if nxt > level * (1 + cap):
                rec[tag] = 0.0; rec[tag + '_fill'] = 0   # no fill: the slot is spent, zero P&L
                continue
            entry = max(nxt, level * 1.000)              # a limit never fills better than the market open here
            Rd = entry - stop
            if Rd <= 0: rec[tag] = np.nan; continue
            tgt = entry + 2.0 * Rd
            rr, why = None, None
            for k in range(i + 2, len(o)):
                if m[k] >= EOD: rr, why = (o[k] - entry) / Rd, 'eod'; break
                if l[k] <= stop: rr, why = (min(stop, o[k]) * 0.999 - entry) / Rd, 'stop'; break
                if c[k] >= tgt: rr, why = 2.0, 'target'; break
            if rr is None: rr, why = (c[-1] - entry) / Rd, 'eod'
            rpct2 = Rd / entry * 100
            rec[tag] = rr - (0.0 if why == 'target' else 0.5 * 0.40 / max(rpct2, 0.05))
            rec[tag + '_fill'] = 1
        rows.append(rec)
F = pd.DataFrame(rows)
F.to_csv(f'{A}/fill_model_check.csv', index=False)
P(f'book trades re-simulated: {len(F)} of {len(book)} (bars missing for {miss})')
P(f"{'split':6} {'n':>5} {'as-is meanR':>12} {'as-is R/wk':>11} | {'next-open cap 0.6% meanR':>25} {'R/wk':>7} {'fill rate':>10} | {'cap 1.0% meanR':>15} {'R/wk':>7}")
for s in SPL:
    x = F[F.split == s]
    nwk = pool[pool.split == s].wk.nunique()
    P(f'{s:6} {len(x):5d} {x.as_is.mean():12.3f} {x.as_is.sum()/nwk:11.2f} | '
      f'{x.cap60.mean():25.3f} {x.cap60.sum()/nwk:7.2f} {x.cap60_fill.mean():10.1%} | '
      f'{x.cap100.mean():15.3f} {x.cap100.sum()/nwk:7.2f}')
P('\n  ("no fill" is scored 0R and still consumes the slot, which is what the live book does.)')
P('  t of the next-open-fill book: ' + ' '.join(
    f'{s} {F[F.split==s].cap60.mean()/(F[F.split==s].cap60.std()/np.sqrt((F.split==s).sum())):.2f}' for s in SPL))

open(f'{A}/fillmodel.md', 'w').write('\n'.join(OUT))
print('DONE', flush=True)
