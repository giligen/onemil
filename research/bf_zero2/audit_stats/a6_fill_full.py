#!/usr/bin/env python3
"""a6 — the fill model, done properly: re-simulate the WHOLE F6 candidate pool under three entry
conventions and run the 4-concurrent first-come book on each, so an unfilled signal frees its slot
for the next candidate instead of being scored as a zero.

  (A) as-is (pass 1): the order fills at level x 1.003 the instant the signal bar's high touches it.
  (B) stop order, next print: always fills, at the NEXT bar's open (never better than level x 1.003).
  (C) live capped limit (research/bf_zero REPORT 6, the spec the HOD engine ships): fills at the next
      bar's open only if that open <= level x (1 + cap); otherwise no trade at all.

Exits identical in all three: stop first (filled min(stop, open) x 0.999), +2R on a bar CLOSE,
flat 15:55, 20 bps half-spread on non-target exits. Exits walk from the bar AFTER the entry bar.
"""
import os, sys, sqlite3
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT); sys.path.insert(0, ROOT)
from trading.hod_break import run_book
A = 'research/bf_zero2/audit_stats'
OUT = []
def P(*a):
    s = ' '.join(str(x) for x in a); OUT.append(s); print(s, flush=True)

pool = pd.read_parquet(f'{A}/pool_f6.parquet')
pool['day'] = pool.day.astype(str); pool['symbol'] = pool.symbol.astype(str)
pool['entry_m'] = pool.entry_m.astype(int)
SPL = ('TRAIN', 'VAL', 'TEST')
NWK = {s: pool[pool.split == s].wk.nunique() for s in SPL}
EOD = 955
cache = sqlite3.connect('file:data/cache.db?mode=ro', uri=True)
side = sqlite3.connect('file:research/bf_zero/bars_sip.db?mode=ro', uri=True)


def bars_for(day, syms):
    out = {}
    q = ("select symbol, timestamp as t, open as o, high as h, low as l, close as c from intraday_bars_1min "
         f"where bar_date=? and symbol in ({','.join('?'*len(syms))})")
    t = pd.read_sql(q, cache, params=[day] + list(syms))
    for sy, g in t.groupby('symbol'): out[sy] = g
    left = [s for s in syms if s not in out]
    if left:
        t = pd.read_sql(f"select symbol, t, o, h, l, c from bars where day=? and symbol in ({','.join('?'*len(left))})",
                        side, params=[day] + left)
        for sy, g in t.groupby('symbol'): out[sy] = g
    res = {}
    for sy, g in out.items():
        ts = pd.to_datetime(g.t, utc=True).dt.tz_convert('America/New_York')
        g = g.assign(m=(ts.dt.hour * 60 + ts.dt.minute).values).sort_values('m').drop_duplicates('m')
        g = g[(g.m >= 570) & (g.m < 960)]
        res[sy] = (g.m.values.astype(int), g.o.values.astype(float), g.h.values.astype(float),
                   g.l.values.astype(float), g.c.values.astype(float))
    return res


def walk(m, o, h, l, c, k0, entry, stop):
    """exits from bar index k0 onwards. returns (net R, why, exit minute)"""
    Rd = entry - stop
    if Rd <= 0: return None
    tgt = entry + 2.0 * Rd
    rr, why, xm = None, None, EOD
    for k in range(k0, len(o)):
        if m[k] >= EOD: rr, why, xm = (o[k] - entry) / Rd, 'eod', int(m[k]); break
        if l[k] <= stop: rr, why, xm = (min(stop, o[k]) * 0.999 - entry) / Rd, 'stop', int(m[k]); break
        if c[k] >= tgt: rr, why, xm = 2.0, 'target', int(m[k]); break
    if rr is None: rr, why, xm = (c[-1] - entry) / Rd, 'eod', int(m[-1])
    rp = Rd / entry * 100
    return rr - (0.0 if why == 'target' else 0.5 * 0.40 / max(rp, 0.05)), why, xm


CAP = 0.006
rows = []
nday = 0
for day, gg in pool.groupby('day', observed=True):
    nday += 1
    if nday % 50 == 0: print(f'  day {nday} rows {len(rows):,}', flush=True)
    bb = bars_for(day, sorted(gg.symbol.unique()))
    for r in gg.itertuples():
        d = bb.get(r.symbol)
        if d is None: continue
        m, o, h, l, c = d
        w = np.where(m == r.entry_m)[0]
        if not len(w): continue
        i = int(w[0])
        if i + 1 >= len(o): continue
        level = r.price / 1.003
        stop = r.price * (1 - r.r_pct / 100.0)
        nxt = o[i + 1]
        rec = dict(day=day, symbol=r.symbol, split=r.split, wk=str(r.wk), entry_m=r.entry_m,
                   r_pct=r.r_pct, price=r.price, slip_bps=(nxt / level - 1) * 1e4)
        # A — as-is, exits from bar i+1 (the pass-1 convention)
        a = walk(m, o, h, l, c, i + 1, r.price, stop)
        rec['A_net'], rec['A_why'], rec['A_xm'] = a if a else (np.nan, '', EOD)
        # B — stop order, filled at the next bar's open, never better than the trigger
        eB = max(nxt, level * 1.003)
        b = walk(m, o, h, l, c, i + 2, eB, stop)
        rec['B_net'], rec['B_why'], rec['B_xm'] = b if b else (np.nan, '', EOD)
        # C — live capped limit
        if nxt <= level * (1 + CAP):
            cc = walk(m, o, h, l, c, i + 2, max(nxt, level), stop)
            rec['C_net'], rec['C_why'], rec['C_xm'], rec['C_fill'] = (*(cc if cc else (np.nan, '', EOD)), 1)
        else:
            rec['C_net'], rec['C_why'], rec['C_xm'], rec['C_fill'] = np.nan, 'nofill', EOD, 0
        rows.append(rec)
F = pd.DataFrame(rows)
F.to_csv(f'{A}/pool_refill.csv', index=False)
P(f'# the fill model, re-simulated on the whole F6 candidate pool ({len(F):,} of {len(pool):,} candidates had bars)\n')

P('## how far above the trigger level is the price one minute later? (all candidates)')
P(f"{'split':6} {'median bps':>11} {'mean bps':>9} {'p75':>7} {'p90':>7} {'share <= +60 bps':>17}")
for s in SPL:
    x = F[F.split == s].slip_bps
    P(f'{s:6} {x.median():11.0f} {x.mean():9.0f} {np.percentile(x,75):7.0f} {np.percentile(x,90):7.0f} {(x<=60).mean():17.1%}')

P('\n## the 4-concurrent first-come book under each entry convention')
P(f"{'convention':38} " + ' '.join(f'{s:>26}' for s in SPL))
res = {}
for tag, label in (('A', 'A  fill at the touch (THE CLAIM)'),
                   ('B', 'B  stop order, next bar open'),
                   ('C', 'C  live capped limit, no chase')):
    cells = []
    for s in SPL:
        x = F[(F.split == s) & F[f'{tag}_net'].notna()]
        rws = [(r.day, int(r.entry_m), int(getattr(r, f'{tag}_xm')), r.symbol, getattr(r, f'{tag}_net'), r.wk)
               for r in x.itertuples()]
        t = pd.DataFrame(run_book(rws, 4, 4), columns=['day', 'em', 'xm', 'sym', 'net', 'wk'])
        tt = t.net.mean() / (t.net.std() / np.sqrt(len(t)))
        cells.append(f'{t.net.mean():+.3f}R {t.net.sum()/NWK[s]:+5.2f}/wk t{tt:+5.2f} n{len(t)}')
        res[(tag, s)] = t
    P(f'{label:38} ' + ' '.join(f'{c:>26}' for c in cells))

P('\n## C, in detail — the live spec: what fraction of signals are even obtainable?')
for s in SPL:
    x = F[F.split == s]
    P(f'  {s:6} candidates {len(x):6d}  fillable at <= +0.6% {x.C_fill.mean():5.1%}  '
      f'filled-only meanR {x.C_net[x.C_fill==1].mean():+.3f}  book n {len(res[("C",s)])}')

P('\n## the same book, weeks green')
for tag in ('A', 'B', 'C'):
    line = []
    for s in SPL:
        t = res[(tag, s)]
        w = t.groupby('wk').net.sum().reindex(sorted(pool[pool.split == s].wk.unique())).fillna(0)
        line.append(f'{s} {int((w>0).sum())}/{len(w)}')
    P(f'  {tag}: ' + ' | '.join(line))

open(f'{A}/fill_full.md', 'w').write('\n'.join(OUT))
print('DONE', flush=True)
