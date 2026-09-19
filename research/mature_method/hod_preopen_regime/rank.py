#!/usr/bin/env python3
"""hod_preopen_regime — apply PREREG §5's two bars to the scored cells, plus the T-curve,
the count-matched null on every cell, and the MDE. No new cells are created here."""
import os, sys
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT); sys.path.insert(0, ROOT)
D = f'{ROOT}/research/mature_method/hod_preopen_regime'
c = pd.read_csv(f'{D}/cells.csv')
c['cell'] = c.cell.str.strip()
c = c.drop_duplicates(['cell', 'split'])
p = c.pivot(index='cell', columns='split')
t, g, w, n, gr, wo, md, rs = (p['total'], p['green'], p['per_wk'], p['net'], p['gross'],
                              p['worst'], p['mdd'], p['redstreak'])
Z = pd.concat([t, g, w, n, gr, wo, md, rs], axis=1,
              keys=['tot', 'grn', 'wk', 'net', 'gross', 'worst', 'mdd', 'rs'])

print('== LIVE-EXPLORATION BAR: total $ > 0 on BOTH splits AND >= 10 trades/wk on both ==')
ok = (t.TRAIN > 0) & (t.VAL > 0) & (w.TRAIN >= 10) & (w.VAL >= 10)
print(f'  cells passing: {int(ok.sum())} of {len(t)}')
if ok.any():
    print(Z[ok].sort_values(('tot', 'VAL'), ascending=False).round(3).to_string())

print('\n== CLAIM BAR G1: TRAIN net R > 0, t >= 2.0, >= 10/wk, TRAIN gross >= +0.25 ==')
g1 = (n.TRAIN > 0) & (p['t'].TRAIN >= 2.0) & (w.TRAIN >= 10) & (gr.TRAIN >= 0.25)
print(f'  cells passing G1: {int(g1.sum())}')
print(f'  max TRAIN gross over all cells: {gr.TRAIN.max():+.3f} '
      f'({gr.TRAIN.idxmax()});  max TRAIN net t: {p["t"].TRAIN.max():+.2f}')

print('\n== best on the PRIMARY metric (green weeks), >= 10/wk both splits ==')
m = (w.TRAIN >= 10) & (w.VAL >= 10)
print(Z[m].assign(grn_sum=g.TRAIN[m] + g.VAL[m]).sort_values('grn_sum', ascending=False)
      .head(15).round(2).to_string())

print('\n== top 15 by VAL dollars (>=10/wk both) ==')
print(Z[m].sort_values(('tot', 'VAL'), ascending=False).head(15).round(3).to_string())
print('\n== bottom 5 by VAL dollars (>=10/wk both) ==')
print(Z[m].sort_values(('tot', 'VAL')).head(5).round(3).to_string())

# ---------------------------------------------------------------- the T curve
cv = pd.read_csv(f'{D}/curve.csv')
print('\n== THE T-CURVE — separation and trades kept as a function of gate time ==')
for bn in cv.base.unique():
    for idx in cv.idx.unique():
        s = cv[(cv.base == bn) & (cv.idx == idx)]
        if not len(s):
            continue
        print(f'\n-- {idx.upper()} on {bn} --')
        print('| gate | thr |  TR sep g |  TR t | TR /wk | TR grn | TR $     | '
              'VA sep g |  VA t | VA /wk | VA grn | VA $     |')
        for r in s.itertuples():
            print(f'| {r.gate} | {r.thr} | {r.TRAIN_sepg:+9.3f} | {r.TRAIN_sept:+5.2f} | '
                  f'{r.TRAIN_perwk:6.1f} | {r.TRAIN_green:6.1f} | {r.TRAIN_total:+8.0f} | '
                  f'{r.VAL_sepg:+8.3f} | {r.VAL_sept:+5.2f} | {r.VAL_perwk:6.1f} | '
                  f'{r.VAL_green:6.1f} | {r.VAL_total:+8.0f} |')
cand = cv[(cv.TRAIN_perwk >= 10) & (cv.VAL_perwk >= 10)]
if len(cand):
    b = cand.loc[cand.VAL_total.idxmax()]
    print(f'\n  rung maximising VAL weekly $ at >= 10/wk: {b.idx.upper()} {b.gate} {b.thr} on {b.base} '
          f'-> VAL ${b.VAL_total:+,.0f} at {b.VAL_perwk:.1f}/wk (TRAIN ${b.TRAIN_total:+,.0f} '
          f'at {b.TRAIN_perwk:.1f}/wk)')
    b2 = cand.loc[(cand.TRAIN_total + cand.VAL_total).idxmax()]
    print(f'  rung maximising TRAIN+VAL $: {b2.idx.upper()} {b2.gate} {b2.thr} on {b2.base} '
          f'-> TRAIN ${b2.TRAIN_total:+,.0f}, VAL ${b2.VAL_total:+,.0f}')

# ---------------------------------------------------------------- MDE
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_break')
import score as S                                                    # noqa: E402
print('\n== MDE ==')
for sp in ('TRAIN', 'VAL'):
    N = S.NW[sp]
    hw = 1.96 * np.sqrt(0.25 / N) * 100
    print(f'  {sp}: {N} W-FRI weeks -> 95% half-width on a green-week share at 50% = +/-{hw:.1f} pp '
          f'(a share must exceed {50+hw:.1f}% to be separable from a coin flip on that split alone)')
b2n = c[(c.cell == 'B2')].set_index('split')
for sp in ('TRAIN', 'VAL'):
    nn = b2n.loc[sp, 'n']
    print(f'  B2 {sp}: n={int(nn)} trades -> per-trade MDE (2.8 x SE of a 0.60 R sd) '
          f'= {2.8 * 0.60 / np.sqrt(nn):.3f} R against the +0.2151 R the book must clear')
