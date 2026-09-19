#!/usr/bin/env python3
"""Supplements: the reproduction gate, trades/week on the frontier, the quarter-resolution power
statement, and the ONE TEST read that PREREG section 5 authorises (G1 and G2 both passed).
"""
import math
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '/home/ec2-user/onemil/research/fuckup_audit/Q')
sys.path.insert(0, '/home/ec2-user/onemil/research/mature_method/qqq_band')
import zsim as Z                                                         # noqa: E402
import score as S                                                        # noqa: E402

data = Z.load_symbol('QQQ')

# ---------------------------------------------------------------- 1. reproduction gate
print('\n=== REPRODUCTION GATE ===')
UB, LB = Z.bands(data, vm=1.0)
tr_paper, _ = Z.simulate(data, UB, LB, fill='close', eod='moc', slip_bp=0.0)
d = Z.daily_returns(data, tr_paper, cost='paper')
i, o = Z.split(d)
for nm, x in (('IS 2016-2023', i), ('OOS 2024->', o)):
    m = Z.metrics(x, 'rdyn'); m1 = Z.metrics(x, 'r1x')
    print(f'  RESULTS row 1b {nm:14s} dyn {m["bps"]:.2f} bps/traded day  t {m["t"]:.2f}  '
          f'1x SR {m1["sharpe"]:.3f}  1x ann {m1["ann"]:.2f}%  trades/day {m["trades_day"]:.2f}')
tr_live, _ = Z.simulate(data, UB, LB, fill='next_open', eod='moc', slip_bp=0.5)
d2 = Z.daily_returns(data, tr_live, cost='none')
i2, o2 = Z.split(d2)
for nm, x in (('IS', i2), ('OOS', o2)):
    m = Z.metrics(x, 'r1x')
    print(f'  Q scenario C   {nm:14s} 1x {m["bps"]:.2f} bps/traded day  t {m["t"]:.2f}  '
          f'SR {m["sharpe"]:.3f}  ann {m["ann"]:.2f}%  MDD {m["mdd"]:.1f}%')
print(f'  trade count (paper convention): {len(tr_paper)}   (live convention): {len(tr_live)}')

# ---------------------------------------------------------------- 2. frontier in TRADES
print('\n=== FREQUENCY FRONTIER — trades, not just traded days ===')
COST = float(pd.read_csv(S.HERE + 'cost_nbbo.csv')['half_bp'].mean())
rows = []
for cid, label, kw, bkw in S.cells(data):
    U, L = Z.bands(data, vm=bkw.get('vm', 1.0), anchor=bkw.get('anchor', 'open_prevclose'))
    trs = S.sim3(data, U, L, slip_bp=0.0, **kw)
    dn = S.day_frame(data, trs, COST)
    days = pd.to_datetime([data['days'][t['d']] for t in trs])
    for sp, (a, b) in (('TRAIN', S.SPLITS['TRAIN']), ('VAL', S.SPLITS['VAL'])):
        m = (dn['date'] >= a) & (dn['date'] <= b)
        nwk = len(S.weekly(dn[m.values]))
        ntr = int(((days >= a) & (days <= b)).sum())
        rows.append(dict(cell=cid, split=sp, trades=ntr, trades_wk=ntr / nwk,
                         traded_days_wk=S.cell_stats(dn[m.values])['tr_wk']))
F = pd.DataFrame(rows)
print(F.pivot(index='cell', columns='split', values=['trades_wk', 'traded_days_wk'])
      .round(2).to_string())

# ---------------------------------------------------------------- 3. power / resolution
print('\n=== POWER ===')
UB0, LB0 = Z.bands(data, vm=1.0)
for cid, kw, bkw in (('B0', {}, {}), ('H2', dict(stop_mode='none'), {})):
    U, L = Z.bands(data, vm=bkw.get('vm', 1.0))
    dn = S.day_frame(data, S.sim3(data, U, L, slip_bp=0.0, **kw), COST)
    for sp in ('TRAIN', 'VAL'):
        a, b = S.SPLITS[sp]
        x = dn[(dn['date'] >= a) & (dn['date'] <= b)]
        r = x['r1x'].values * 1e4
        tr = r[x['ntr'].values > 0]
        sdc, sdt = r.std(ddof=1), tr.std(ddof=1)
        print(f'  {cid} {sp:5s} calendar sd {sdc:.1f} bps  traded sd {sdt:.1f}  '
              f'MDE80 traded {2.80 * sdt / math.sqrt(len(tr)):.2f} vs observed {tr.mean():.2f}')
    # how long to resolve the point estimate at 80% power?
    a, b = S.SPLITS['VAL']
    x = dn[(dn['date'] >= a) & (dn['date'] <= b)]
    r = x['r1x'].values * 1e4
    eff = r.mean()
    if eff > 0:
        n_need = (2.80 * r.std(ddof=1) / eff) ** 2
        print(f'  {cid} calendar days needed to resolve its own VAL point estimate '
              f'({eff:.2f} bps/day) at 80% power: {n_need:,.0f} sessions = {n_need / 252:,.1f} years')

# ---------------------------------------------------------------- 4. the ONE authorised TEST read
print('\n=== TEST (PREREG section 5: opened only for cells passing G1 and G2 — that is H2 only) ===')
C = pd.read_csv(S.HERE + 'cells.csv')
print(C[(C.cell == 'H2') & (C.split == 'TEST')][
    ['cell', 'split', 'traded', 'tr_wk', 'gross_bps', 'net_bps', 't', 'mde', 'green_wk',
     'streak', 'worst_wk', 'total', 'mdd', 'green_mo']].round(3).to_string(index=False))
dn = S.day_frame(data, S.sim3(data, UB0, LB0, slip_bp=0.0, stop_mode='none'), COST)
a, b = S.SPLITS['TEST']
x = dn[(dn['date'] >= a) & (dn['date'] <= b)]
w = S.weekly(x)
print('  H2 TEST weekly $ at $60K 1x: ' + ' '.join(f'{v:+.0f}' for v in w['usd'].values))
mu, p5, p95 = S.null_band(x)
print(f'  H2 TEST null: obs {S.cell_stats(x)["green_wk"]:.1f}%  null {mu:.1f}% [{p5:.1f}, {p95:.1f}]')
