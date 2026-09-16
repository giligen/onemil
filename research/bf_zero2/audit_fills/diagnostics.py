#!/usr/bin/env python3
"""Per-item diagnostics for the fill audit: how often each optimistic assumption bites, on the POOL and on
the 1,676 trades the book actually takes.  → audit_fills/diagnostics.md
"""
import os, sys
import numpy as np, pandas as pd
ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT); sys.path.insert(0, ROOT)
from trading.hod_break import run_book
A = 'research/bf_zero2/audit_fills'
R = pd.read_csv(f'{A}/pool_rewalk.csv', dtype={'day': str, 'symbol': str}, keep_default_na=False, na_values=[''])
R['split'] = np.where(R.day < '2026-01-01', 'TRAIN', np.where(R.day < '2026-06-01', 'VAL', 'TEST'))
half = 0.5 * 0.40 / R.r_pct.clip(lower=0.05)
R['net_base'] = R.rr_base - np.where(R.why_base == 'target', 0.0, half)
rows = [(r.day, int(r.entry_m), int(r.xm_base), r.symbol, r.Index, r.wk) for r in R.itertuples()]
taken = set(x[4] for x in run_book(rows, 4, 4))
R['in_book'] = R.index.isin(taken)
B = R[R.in_book]
L = [f'# Fill-audit diagnostics — pool {len(R):,} candidates, book {len(B):,} trades', '']


def frac(mask, d, what):
    return f'{what}: {int(mask.sum()):,} of {len(d):,} ({mask.mean() * 100:.1f}%)'


L += ['## 1 — entry bar OPENS above the fill level (a stop-buy cannot fill below the open)']
for name, d in (('pool', R), ('book', B)):
    m = d.gap_through == 1
    L.append('  ' + frac(m, d, name) + f' | median gap-through {d.loc[m, "gap_through_pct"].median():.2f}% of the entry price, '
             f'p90 {d.loc[m, "gap_through_pct"].quantile(0.9):.2f}%, max {d.loc[m, "gap_through_pct"].max():.2f}%')
L.append('  book gap-through by split: ' + ' | '.join(f'{s} {(B[B.split == s].gap_through == 1).mean() * 100:.1f}%' for s in ('TRAIN', 'VAL', 'TEST')))
L.append('  book mean net R, gapped vs not: ' + ' | '.join(
    f'{s} gapped {B[(B.split == s) & (B.gap_through == 1)].net_base.mean():+.3f} (n {int(((B.split == s) & (B.gap_through == 1)).sum())}) '
    f'clean {B[(B.split == s) & (B.gap_through == 0)].net_base.mean():+.3f}' for s in ('TRAIN', 'VAL', 'TEST')))

L += ['', '## 2 — the entry bar itself trades through the stop (the sim only looks from the next bar)']
for name, d in (('pool', R), ('book', B)):
    L.append('  ' + frac(d.entry_bar_stop_hit == 1, d, name))
m = B.entry_bar_stop_hit == 1
L.append(f'  of those book trades, the study booked: ' + B.loc[m, 'why_base'].value_counts().to_dict().__str__())
L.append(f'  their mean net R as booked: {B.loc[m, "net_base"].mean():+.3f} vs {-1.0 - float((0.5 * 0.40 / B.loc[m, "r_pct"].clip(lower=0.05)).mean()):+.3f} if stopped at the entry bar')

L += ['', '## 3 — stop fills']
st = B[B.why_base == 'stop']
gapdn = st.copy()
L.append(f'  stop exits in the book: {len(st):,} ({len(st) / len(B) * 100:.0f}%), mean booked R {st.net_base.mean():+.3f} (a clean −1R exit would be −1.0 minus cost)')
L.append(f'  mean rr_base on stop exits {st.rr_base.mean():+.3f} → the min(stop, open) rule already books '
         f'{(st.rr_base + 1).mean() * 100:+.1f}% of R beyond −1R on average (gap-downs ARE handled)')
for tag, bps in (('s25', 25), ('s50', 50), ('s100', 100)):
    L.append(f'  extra slip {bps} bps → mean rr on those trades {B.loc[B.why_base == "stop", f"rr_{tag}"].mean():+.3f}')

L += ['', '## 4 — target fills (conservative check)']
tg = R[R.why_base == 'target']
Rd = R.entry - R.stop
bad = int((tg.h_i.notna() & False).sum())
L.append(f'  target exits in the pool {len(tg):,}; the rule needs a bar CLOSE >= entry+2R, so the bar traded through the fill price by construction (0 impossible fills).')
L.append('  ' + frac(B.limit_touch_before_exit == 1, B, 'book trades whose HIGH reached +2R before the booked exit (a resting limit would have filled; the study did not)'))
lt = B[(B.limit_touch_before_exit == 1) & (B.why_base != 'target')]
L.append(f'  of those, {len(lt)} exited at something other than the target, booking a mean {lt.net_base.mean():+.3f}R instead of +2R → the close-fill rule is CONSERVATIVE by ~{(2.0 - lt.net_base.mean()) * len(lt) / len(B):+.3f}R per book trade')

L += ['', '## 5 — the 15:55 exit']
eo = B[B.why_base == 'eod']
L.append(f'  eod exits {len(eo):,} ({len(eo) / len(B) * 100:.0f}%), mean net R {eo.net_base.mean():+.3f}; the half-spread IS charged on them (score3.py line 40: cost unless why == target)')
L.append(f'  fill = the OPEN of the first bar >= 15:55; mean(open − close of that bar)/R over eod exits is reported in the variants table (eod10 = 10 bps worse)')

L += ['', '## 6 — liquidity at $100 of risk per trade']
for name, d in (('pool', R), ('book', B)):
    L.append(f'  {name}: median notional ${d.notional_100r.median():,.0f} | median $vol in the 5 min after entry ${d.dv5.median():,.0f} | '
             f'median position as a share of it {d.notional_frac_dv5.median() * 100:.2f}%')
for thr in (0.01, 0.02, 0.05):
    L.append('  ' + frac(B.notional_frac_dv5 > thr, B, f'book trades above {thr * 100:.0f}% of the 5-min $ volume') +
             f' | their mean net R {B.loc[B.notional_frac_dv5 > thr, "net_base"].mean():+.3f} vs {B.loc[B.notional_frac_dv5 <= thr, "net_base"].mean():+.3f} for the rest')
L.append(f'  NOTE: at $100 risk/trade the book is tiny. Scaling: the same trades at $2,000 risk multiply the position by 20 → '
         f'{(B.notional_frac_dv5 * 20 > 0.01).mean() * 100:.0f}% of book trades would then exceed 1% of the 5-min $ volume.')

L += ['', '## 7 — tape that ends before 15:55 (halts / no prints)']
for name, d in (('pool', R), ('book', B)):
    m = d.why_base == 'eod_notape'
    L.append('  ' + frac(m, d, f'{name}: exits taken at the LAST bar because the tape ends before 15:55'))
m = B.why_base == 'eod_notape'
L.append(f'  their mean net R as booked {B.loc[m, "net_base"].mean():+.3f} (n {int(m.sum())}), contribution {B.loc[m, "net_base"].sum():+.1f}R of the book total {B.net_base.sum():+.1f}R')
L.append('  ' + frac(B.max_gap_min >= 5, B, 'book trades with a >= 5-minute hole in the tape between entry and exit'))
L.append('  ' + frac(B.max_gap_min >= 15, B, 'book trades with a >= 15-minute hole (a plausible halt)'))
hm = B.max_gap_min >= 15
L.append(f'  the >=15-min-hole trades book {B.loc[hm, "net_base"].mean():+.3f}R mean, {B.loc[hm, "net_base"].sum():+.1f}R total')
L.append('  ' + frac(B.last_m < 955, B, 'book trades whose symbol has no bar at or after 15:55 at all'))

L += ['', '## 8 — costs', f'  the study charges 0.5 x 40 bps / r_pct in R units on every non-target exit;',
      f'  median r_pct in the book {B.r_pct.median():.2f}% → median charge {float((0.5 * 0.40 / B.r_pct.clip(lower=0.05)).median()):.3f}R, '
      f'mean charge {float((0.5 * 0.40 / B.r_pct.clip(lower=0.05)).mean()):.3f}R']
if os.path.exists(f'{A}/book_spreads.csv'):
    s = pd.read_csv(f'{A}/book_spreads.csv')
    s = s[s.n_quotes > 0].copy(); s['sp_pct'] = s.spread_med / s.price * 100; s['spm_pct'] = s.spread_mean / s.price * 100
    L.append(f'  REAL NBBO at the book\'s own fill minute (n {len(s)}): median full spread {s.sp_pct.median():.3f}% of price, '
             f'mean {s.sp_pct.mean():.3f}%, p75 {s.sp_pct.quantile(.75):.3f}%, p90 {s.sp_pct.quantile(.9):.3f}%')
    L.append('  by price band: ' + ' | '.join(f'{i} med {g.sp_pct.median():.3f}% mean {g.sp_pct.mean():.3f}% (n {len(g)})'
                                              for i, g in s.groupby(pd.cut(s.price, [5, 10, 20, 50, 1e9]), observed=True)))
    L.append(f'  the study assumes a 40 bps full spread → half 20 bps. Half the REAL median is {s.sp_pct.median() / 2 * 100:.0f} bps, '
             f'half the real MEAN is {s.sp_pct.mean() / 2 * 100:.0f} bps.')
    L.append(f'  entry: the study fills 30 bps through the level. Half the real spread exceeds 30 bps on '
             f'{(s.sp_pct / 2 * 100 > 30).mean() * 100:.0f}% of sampled trades → the 30 bps entry slip does NOT double-count the exit half-spread; '
             f'on those trades it UNDER-charges the entry.')
open(f'{A}/diagnostics.md', 'w').write('\n'.join(L))
print('\n'.join(L)); print('DONE', flush=True)
