#!/usr/bin/env python3
"""bf_zero §7 — the unbiased check: a random 10% of tradable symbols with ALL their days (no range gate).

Reads sample_candidates.csv (same scanner, same families) and reports the HOD-break family
(F5 K=5 X=0.04, fixed 1:2 exit) three ways:
  A. raw population on the sample, NO causal floor — includes days the 5%-range universe never saw
  B. with the causal floor (entry >= 5% above the open) — should match the main study's population
  C. the relative-volume band on top (the live spec's filters, approximately: rv_profile 1-5, r_pct >= 1)
plus coverage (symbol-days with bars vs requested) so the reader can judge the missing third.
"""
import os
import numpy as np, pandas as pd
os.chdir('/home/ec2-user/onemil'); D = 'research/bf_zero'
c = pd.read_csv(f'{D}/sample_candidates.csv', low_memory=False, dtype={'symbol': str, 'fam': 'category', 'cfg': 'category'}, keep_default_na=False, na_values=[''])
for k in ('entry_m', 'dist_open_pct', 'r_pct', 'rv_adv', 'adv20', 'rr_e1', 'rr_e2', 'price'): c[k] = pd.to_numeric(c[k], errors='coerce')
u = pd.read_csv(f'{D}/sample_universe.csv', dtype={'symbol': str})
miss = pd.read_csv(f'{D}/sample_missing.csv', dtype=str) if os.path.exists(f'{D}/sample_missing.csv') else pd.DataFrame(columns=['symbol', 'bar_date'])
print(f"sample universe {len(u):,} symbol-days ({u.symbol.nunique()} symbols) | missing bars {len(miss):,} ({len(miss) / len(u):.0%}) | candidate rows {len(c):,}")
# profile-adjusted rv as in pass2 (same checkpoint fractions)
VP = {575: 0.02, 585: 0.051, 600: 0.097, 630: 0.188, 660: 0.269, 720: 0.411, 780: 0.529, 840: 0.642, 900: 0.76}
cks = np.array(sorted(VP)); idx = np.searchsorted(cks, c.entry_m.values, side='right') - 1
c['rv_profile'] = c.rv_adv / pd.Series(cks[np.clip(idx, 0, len(cks) - 1)]).map(VP).values
c['split'] = np.where(c.day < '2026-01-01', 'TRAIN', np.where(c.day < '2026-06-01', 'VAL', 'TEST')); c['wk'] = pd.to_datetime(c.day).dt.to_period('W-FRI').astype(str)
NW = {'TRAIN': 53, 'VAL': 22, 'TEST': 14}
f5 = c[(c.fam == 'F5') & (c.cfg == '{"K": 5, "X": 0.04}')]


def st(d, s):
    r = d.rr_e1.dropna(); w = d.groupby('wk').rr_e1.sum(); nw = NW[s]
    if not len(r): return 'n=0'
    return f"n={len(r):6d} ({len(r) / nw:5.1f}/wk) meanR {r.mean():+.3f} WR {(r > 0).mean() * 100:4.1f} | weekly R {w.sum() / nw:+.1f} green {(w > 0).sum()}/{nw} worst {w.min():+.1f}"


for title, m in (('A. raw, no floor (days the universe rule never saw included)', pd.Series(True, index=f5.index)),
                 ('B. causal floor: entry >= 5% above open', f5.dist_open_pct >= 5),
                 ('C. floor + rv_profile 1-5 + r_pct >= 1 (the live filters)', (f5.dist_open_pct >= 5) & (f5.rv_profile >= 1) & (f5.rv_profile < 5) & (f5.r_pct >= 1)),
                 ('C5. C + price >= $5 (the live cost rule)', (f5.dist_open_pct >= 5) & (f5.rv_profile >= 1) & (f5.rv_profile < 5) & (f5.r_pct >= 1) & (f5.price >= 5))):
    print(f"\n## {title}")
    for s in ('TRAIN', 'VAL', 'TEST'):
        print(f"  {s:5s} {st(f5[m & (f5.split == s)], s)}")
print('\n## A by distance from open (the look-ahead check on an unbiased population):')
print(f5.groupby(pd.cut(f5.dist_open_pct, [-1e9, 0, 2, 5, 10, 20, 1e9]), observed=True).rr_e1.agg(['mean', 'count']).round(3).to_string())
print('\n## sample days with range < 5% (never in the main universe): F5 rows and their E1 meanR:')
u['range'] = (u.high - u.low) / u.low; small = set(zip(u[u.range < 0.05].symbol, u[u.range < 0.05].bar_date))
sm = f5[[(s, d) in small for s, d in zip(f5.symbol, f5.day)]]
print(f"  rows {len(sm)} meanR {sm.rr_e1.mean():+.3f} WR {(sm.rr_e1 > 0).mean() * 100:.1f} | with floor: n={int((sm.dist_open_pct >= 5).sum())} (a >= 5%-above-open entry needs a >= 5% day, so this must be ~0)")
