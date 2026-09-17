#!/usr/bin/env python3
"""R8 - implementation E: the F6-PDR book re-scored under EXACTLY the LIVE ENGINE's conventions.

E = trading/red_to_green.py (the shipped spec module) + trading/hod_break.{entry_fill, walk_exit, run_book}:
  prior day      prior row of the daily panel (the study stand-in for the engine's daily_bars prior row;
                 R4 proved universe.csv and the panel agree on prior-day OHLC to 1e-6 on all 644,580 keys)
  precondition   o[0] of the FIRST RTH bar < prior close   AND   (prev_high-prev_low)/prev_low >= 8%
  level          prior close x 1.003                       (red_to_green.level_for, level_buffer=0.003)
  floor          (run_hi[i-1]-run_lo[i-1])/run_lo[i-1] >= 5%   -- denominator is the RUNNING LOW
  scan           the first bar i>=1 at which the floor holds AND h[i] >= level AND run_lo[i] < level;
                 bars that fail the floor are SKIPPED, not fatal (red_to_green.detect's `continue`)
  cut            m[i] > 840 on the SIGNAL bar aborts the day (detect returns None)
  stop           run_lo[i], lowest low 09:30 THROUGH the signal bar
  fill           the next bar that prints, at its open, iff open <= level x 1.006 (hod_break.entry_fill)
  floors         entry >= $5 (engine min_price), (entry-stop)/entry >= 1% (red_to_green.r_ok)
  exits          hod_break.walk_exit from the bar AFTER the fill bar: eod(m>=955)@open beats stop(l<=stop)
                 @min(stop,open)*0.999 beats target(c>=target)@target;  partial = half at +2R then stop->entry
  book           trading.hod_break.run_book(rows, 12, 4)
  cost           contract (c): half = 0.5*(spread_cc_bps/100)/max(r_pct,0.05); entry 0.25*half;
                 exit legs stop 0.875 / eod 0.412 / target 0.875 x half   (cc band on the ENTRY price/minute)
"""
import os, sys, time
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT); sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/research/fuckup_audit/H/F6_reconcile')
from pipeline import Bars, run_day, E_CFG
from trading.hod_break import run_book

OUT = 'research/fuckup_audit/H/F6_reconcile'
CC_CSV = 'research/lit_review_2026/cost_curve.csv'
RD = lambda p, **k: pd.read_csv(p, keep_default_na=False, na_values=[''], dtype={'symbol': str, 'day': str}, **k)
log = lambda *a: (print(*a), sys.stdout.flush())

# ---------------------------------------------------------------- cost curve (build_candidates4._load_cost_curve)
d = pd.read_csv(CC_CSV, dtype={'symbol': str, 'day': str}, keep_default_na=False, na_values=[''])
d = d[(d.n_q > 0) & d.spread.notna() & (d.price > 0)].copy()
d['bps'] = d.spread / d.price * 1e4
CC = {k: float(v) for k, v in d.groupby(['pb', 'hb']).bps.median().items()}
del d


def cc_bps(p, m):
    pb = '$5-10' if p <= 10 else '$10-20' if p <= 20 else '$20-50' if p <= 50 else '$50-200' if p <= 200 else '$200+'
    hb = ('09:30-09:35' if m <= 575 else '09:35-10:00' if m <= 600 else '10:00-11:00' if m <= 660
          else '11:00-13:00' if m <= 780 else '13:00+')
    return CC.get((pb, hb), np.nan)


# ---------------------------------------------------------------- the walk
prev = RD(f'{OUT}/prev_table.csv')
prev = prev[prev.pdr_panel >= 8.0].reset_index(drop=True)
log('E population: universe symbol-days with a prior panel row and PDR >= 8:', len(prev))
bars = Bars()
rows = []
reasons = {}
t0 = time.time()
for i, r in enumerate(prev.itertuples(index=False)):
    if i % 10000 == 0:
        log('  %d/%d  taken=%d  %.1f min' % (i, len(prev), len(rows), (time.time() - t0) / 60))
    res = run_day(bars, r.symbol, r.day, dict(close=r.prev_close_panel, high=r.prev_high_panel,
                                              low=r.prev_low_panel), E_CFG, day_open=r.day_open)
    reasons[res['reason']] = reasons.get(res['reason'], 0) + 1
    if not res['ok']:
        continue
    rows.append(dict(day=r.day, symbol=r.symbol, src=res['src'], sig_m=res['sig_m'], entry_m=res['entry_m'],
                     entry=res['entry'], stop=res['stop'], R=res['R'], r_pct=res['r_pct'], level=res['level'],
                     pdr=res['pdr'],
                     **{k: res[k] for k in res if k.startswith(('hold_', 'r2_', 'partial_'))}))
log('reasons:', reasons)
c = pd.DataFrame(rows)
c.to_csv(f'{OUT}/e_cands.csv', index=False)
log('E candidates:', len(c))

# ---------------------------------------------------------------- costs
c['cc_bps'] = [cc_bps(p, m) for p, m in zip(c.entry, c.entry_m)]
c['half'] = 0.5 * (c.cc_bps / 100.0) / c.r_pct.clip(lower=0.05)
K = {'stop': 0.875, 'eod': 0.412, 'target': 0.875}
for mode in ('hold', 'r2', 'partial'):
    cost = 0.25 * c.half
    legcost = []
    for legs, h in zip(c[f'{mode}_legs'], c.half):
        s = 0.0
        for leg in str(legs).split(';'):
            w, t = leg.split(':'); s += float(w) * h * K[t]
        legcost.append(s)
    c[f'net_{mode}'] = c[f'{mode}_grossR'] - cost - np.array(legcost)
c['split'] = np.where(c.day < '2026-01-01', 'TRAIN', np.where(c.day < '2026-06-01', 'VAL', 'TEST'))
c['wk'] = pd.to_datetime(c.day).dt.to_period('W-FRI').astype(str)
c['mo'] = c.day.str[:7]
WEEKS = {s: sorted(c[c.split == s].wk.unique()) for s in ('TRAIN', 'VAL', 'TEST')}


def book(x, mode):
    rr = [(r.day, int(r.entry_m), int(getattr(r, f'{mode}_exit_m')), r.symbol, getattr(r, f'net_{mode}'),
           getattr(r, f'{mode}_grossR'), getattr(r, f'{mode}_exit_type'), r.wk, r.mo, r.entry, r.stop,
           int(r.sig_m), r.src, r.r_pct) for r in x.itertuples()]
    t = run_book(rr, 12, 4)
    return pd.DataFrame(t, columns=['day', 'entry_m', 'exit_m', 'symbol', 'net', 'gross', 'why', 'wk', 'mo',
                                    'entry', 'stop', 'sig_m', 'src', 'r_pct'])


def stats(t, sp):
    v = t.net.values; n = len(v)
    se = v.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    w = t.groupby('wk').net.sum().reindex(WEEKS[sp]).fillna(0)
    s = np.sort(v)
    return dict(split=sp, n=n, tpw=round(n / len(WEEKS[sp]), 1), meanR=round(v.mean(), 4),
                gross=round(t.gross.mean(), 4), t=round(v.mean() / se, 2),
                WR=round((v > 0).mean() * 100, 1), stopP=round(t.why.str.contains('stop').mean() * 100, 1),
                wkR=round(w.mean(), 2), green=round((w > 0).mean(), 2), worst=round(w.min(), 1),
                ex1=round(s[:max(int(round(n * 0.99)), 1)].mean(), 4),
                ex5=round(s[:max(int(round(n * 0.95)), 1)].mean(), 4),
                cap3=round(np.minimum(v, 3).mean(), 4), totR=round(v.sum(), 1),
                mdd=round(float((w.cumsum() - w.cumsum().cummax()).min()), 1))


L = ['# R8 - implementation E: F6-PDR under the LIVE ENGINE conventions', '',
     open(__file__).read().split('"""')[1], '',
     f'E candidates (pre-book): {len(c)}   population: {len(prev)} PDR>=8 symbol-days', '']
allrows = []
for mode in ('hold', 'r2', 'partial'):
    st = []
    for sp in ('TRAIN', 'VAL', 'TEST'):
        t = book(c[c.split == sp], mode)
        t.to_csv(f'{OUT}/e_trades_{mode}_{sp}.csv', index=False)
        st.append(stats(t, sp))
        allrows.append(t.assign(mode=mode))
    L += [f'## exit {mode}', pd.DataFrame(st).to_string(index=False), '']
    at = pd.concat([x for x in allrows if x['mode'].iloc[0] == mode])
    m = at.groupby('mo').net.agg(['sum', 'count'])
    L += ['monthly net R: ' + ' | '.join(f"{i} {r['sum']:+.1f} ({int(r['count'])})" for i, r in m.iterrows()),
          f"months green {int((m['sum'] > 0).sum())}/{len(m)}", '']
open(f'{OUT}/r8_engine_book.md', 'w').write('\n'.join(L) + '\n')
log('\n'.join(L))
