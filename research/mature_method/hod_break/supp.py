#!/usr/bin/env python3
"""Supplementary rows the main map could not resolve: Ge (r_min) and Gg (100 bps) are SUBSUMED by
the 15%-of-R cap at the shipped stack, so they are re-measured with that cap OFF -- which is also
the engine's own cascade position (r_min and the bps ceiling are both checked BEFORE spread_r)."""
import sys
sys.argv = ['x']
sys.path.insert(0, '/home/ec2-user/onemil/research/mature_method/hod_break')
import numpy as np, score as S

bk = S.load_breaks(); S.build_impute(bk)
print(S.SEP_HDR)
rows = []
r = S.stack(bk, r_min=0.0, max_frac_r=0.0)
rows.append(S.sep_row(r[r.r_pct >= 1.0], r[r.r_pct < 1.0], 'Ge stop >=1% (15%-cap OFF)', '5'))
print(f'   r_min reject rate at the engine position: {(r.r_pct < 1.0).mean():.1%} of {len(r)} signals '
      f'(EOD-check item 16 flagged 28% live as WATCH)')
g = S.stack(bk, max_bps=0.0, max_frac_r=0.0)
rows.append(S.sep_row(g[g.sp_pct * 100 <= 100], g[g.sp_pct * 100 > 100], 'Gg spread <=100 bps (15%-cap OFF)', '7'))
print(f'   100 bps ceiling reject rate with the 15%-cap off: {(g.sp_pct*100 > 100).mean():.1%} of {len(g)}')
b = S.stack(bk)
fr = b.sp_pct / b.r_pct.clip(lower=0.05)
print(f'   at the SHIPPED stack the 15%-of-R cap leaves {(b.r_pct < 1.0).sum()} sub-1% stops and '
      f'{(b.sp_pct*100 > 100).sum()} >100 bps signals -> Ge and Gg are INERT (F-e and F-g are byte-equal to B0)')
for o in rows:
    print(S.fmt_sep(o))
# ex-tail diagnostics + green-week MDE on the shipped book
b0 = S.apply_book(b, 12, 4)
for sp in S.SPLITS:
    d = b0[b0.split == sp]
    q99, q95 = d.net.quantile(0.99), d.net.quantile(0.95)
    p = (d.groupby('wk').pnl.sum().reindex(S.ALL_WEEKS[sp]).fillna(0) > 0).mean()
    se = np.sqrt(p * (1 - p) / S.NW[sp]) * 100
    print(f'  {sp}: B0 net {d.net.mean():+.3f} | ex-top-1% {d.net[d.net <= q99].mean():+.3f} | '
          f'ex-top-5% {d.net[d.net <= q95].mean():+.3f} | green {p*100:.1f}% +/- {se:.1f}pp '
          f'(MDE80 {2.8*se:.1f}pp over {S.NW[sp]} weeks)')
