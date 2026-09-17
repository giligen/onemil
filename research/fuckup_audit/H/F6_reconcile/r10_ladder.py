#!/usr/bin/env python3
"""R10 - the B -> E ladder in ONE tape walk. Four configs, each differing from the previous by exactly one
convention, so the step that moves the book is visible:

  L0  = B's rule set, on E's data plumbing (panel prior day, cache-first bars)   [level 1.003, floor/runlo,
        stop incl, price on entry, day-open >= 5, 14:01 on the FILL bar, first break THEN floor]
  L1  = L0 without B's day-level `universe open >= 5` prefilter
  L2  = L1 with the engine's cut: 14:00 on the SIGNAL bar (detect returns None past it)
  L3  = L2 with the engine's SCAN: bars failing the floor are skipped, not fatal   == implementation E
"""
import os, sys, time
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT); sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/research/fuckup_audit/H/F6_reconcile')
import pipeline
from pipeline import Bars, run_day, E_CFG
from trading.hod_break import run_book

OUT = 'research/fuckup_audit/H/F6_reconcile'
CC_CSV = 'research/lit_review_2026/cost_curve.csv'
RD = lambda p, **k: pd.read_csv(p, keep_default_na=False, na_values=[''], dtype={'symbol': str, 'day': str}, **k)
log = lambda *a: (print(*a), sys.stdout.flush())

L0 = dict(E_CFG); L0.update(day_open5=True, late_on='fill', late_m=841, scan='break_then_floor')
L1 = dict(L0); L1['day_open5'] = False
L2 = dict(L1); L2.update(late_on='signal', late_m=840)
L3 = dict(E_CFG)
CFGS = [('L0_B_rules', L0), ('L1_no_dayopen5', L1), ('L2_engine_cut', L2), ('L3_engine_scan_E', L3)]


class CachedBars(Bars):
    """One-entry cache: the four configs ask for the same (symbol, day) back to back."""
    _k = None; _v = None

    def get(self, sym, day, order):
        k = (sym, day, order)
        if k != self._k:
            self._k = k; self._v = Bars.get(self, sym, day, order)
        return self._v


prev = RD(f'{OUT}/prev_table.csv')
prev = prev[prev.pdr_panel >= 8.0].reset_index(drop=True)
log('population:', len(prev))
bars = CachedBars()
rows = {k: [] for k, _ in CFGS}
t0 = time.time()
for i, r in enumerate(prev.itertuples(index=False)):
    if i % 20000 == 0:
        log('  %d/%d  %.1f min  %s' % (i, len(prev), (time.time() - t0) / 60, {k: len(v) for k, v in rows.items()}))
    p = dict(close=r.prev_close_panel, high=r.prev_high_panel, low=r.prev_low_panel)
    for lab, cfg in CFGS:
        res = run_day(bars, r.symbol, r.day, p, cfg, day_open=r.day_open)
        if res['ok']:
            rows[lab].append(dict(day=r.day, symbol=r.symbol, sig_m=res['sig_m'], entry_m=res['entry_m'],
                                  entry=res['entry'], stop=res['stop'], R=res['R'], r_pct=res['r_pct'],
                                  **{k: res[k] for k in res if k.startswith(('hold_', 'r2_', 'partial_'))}))

d = pd.read_csv(CC_CSV, dtype={'symbol': str, 'day': str}, keep_default_na=False, na_values=[''])
d = d[(d.n_q > 0) & d.spread.notna() & (d.price > 0)].copy(); d['bps'] = d.spread / d.price * 1e4
CC = {k: float(v) for k, v in d.groupby(['pb', 'hb']).bps.median().items()}; del d
K = {'stop': 0.875, 'eod': 0.412, 'target': 0.875}


def cc_bps(p, m):
    pb = '$5-10' if p <= 10 else '$10-20' if p <= 20 else '$20-50' if p <= 50 else '$50-200' if p <= 200 else '$200+'
    hb = ('09:30-09:35' if m <= 575 else '09:35-10:00' if m <= 600 else '10:00-11:00' if m <= 660
          else '11:00-13:00' if m <= 780 else '13:00+')
    return CC.get((pb, hb), np.nan)


res = []
for lab, _ in CFGS:
    c = pd.DataFrame(rows[lab])
    c.to_csv(f'{OUT}/ladder_{lab}.csv', index=False)
    c['half'] = 0.5 * (np.array([cc_bps(p, m) for p, m in zip(c.entry, c.entry_m)]) / 100.0) / c.r_pct.clip(lower=0.05)
    c['split'] = np.where(c.day < '2026-01-01', 'TRAIN', np.where(c.day < '2026-06-01', 'VAL', 'TEST'))
    c['wk'] = pd.to_datetime(c.day).dt.to_period('W-FRI').astype(str)
    weeks = {s: sorted(c[c.split == s].wk.unique()) for s in ('TRAIN', 'VAL', 'TEST')}
    for mode in ('hold', 'r2', 'partial'):
        lc = [sum(float(w) * h * K[t] for w, t in (x.split(':') for x in str(lg).split(';')))
              for lg, h in zip(c[f'{mode}_legs'], c.half)]
        c['net'] = c[f'{mode}_grossR'] - 0.25 * c.half - np.array(lc)
        for sp in ('TRAIN', 'VAL', 'TEST'):
            x = c[c.split == sp]
            rr = [(r.day, int(r.entry_m), int(getattr(r, f'{mode}_exit_m')), r.symbol, r.net, r.wk)
                  for r in x.itertuples()]
            t = pd.DataFrame(run_book(rr, 12, 4), columns=['day', 'em', 'xm', 'symbol', 'net', 'wk'])
            v = t.net.values
            w = t.groupby('wk').net.sum().reindex(weeks[sp]).fillna(0)
            res.append(dict(variant=lab, exit=mode, split=sp, n=len(v), meanR=round(v.mean(), 4),
                            t=round(v.mean() / (v.std(ddof=1) / np.sqrt(len(v))), 2), totR=round(v.sum(), 1),
                            green=round((w > 0).mean(), 2), worst=round(w.min(), 1)))
st = pd.DataFrame(res)
st.to_csv(f'{OUT}/ladder_stats.csv', index=False)
L = ['# R10 - the B -> E convention ladder', '', open(__file__).read().split('"""')[1], '',
     'pre-book candidates: ' + str({k: len(v) for k, v in rows.items()}), '']
for mode in ('hold', 'r2', 'partial'):
    L += [f'## exit {mode}', st[st.exit == mode].drop(columns=['exit']).pivot(index='variant', columns='split').to_string(), '']
open(f'{OUT}/r10_ladder.md', 'w').write('\n'.join(L) + '\n')
log('\n'.join(L))
