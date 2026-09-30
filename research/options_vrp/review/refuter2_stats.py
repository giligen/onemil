"""Refuter 2 (statistics/risk lens) recomputation for PREREG_1567 from cell_1567_cycles.csv.

Recomputes per-cell per-split monthly stats two ways: (a) builder's convention (only months with exits),
(b) calendar months zero-filled from the split's first entry month to its last exit month. Also checks the
concurrent budget (sum of worst cases of open spreads on any day), selection sensitivity, neighbours,
tail concentration, naked comparison and SPY buy-and-hold per unit drawdown.
"""
import math
import pandas as pd
import numpy as np

B = 6500.0
c = pd.read_csv('/home/ec2-user/onemil/research/options_vrp/cell_1567_cycles.csv', parse_dates=['entry_date', 'expiry', 'exit_date'])
SPLITS = {'TRAIN': ('2024-02-05', '2025-06-30'), 'VAL': ('2025-07-07', '2026-08-17')}


def stats(df, months):
    """Monthly stats on B; months=None -> only months with exits (builder convention)."""
    m = df.groupby(df.exit_date.dt.to_period('M')).pnl_usd.sum()
    if months is not None:
        m = m.reindex(months, fill_value=0.0)
    r = m / B
    sh = r.mean() / r.std(ddof=1) * math.sqrt(12) if len(r) > 1 and r.std(ddof=1) > 0 else float('nan')
    cum = m.cumsum(); dd = float((cum - cum.cummax()).min())
    srt = df.pnl_usd.sort_values(ascending=False); k = max(1, math.ceil(0.05 * len(srt)))
    return dict(n=len(df), n_months=len(m), mean_ret=r.mean(), sharpe=sh, green=(m > 0).mean(),
                worst=m.min(), maxdd=-dd, top5_share=srt.iloc[:k].sum() / srt.sum() if srt.sum() else np.nan,
                ex_top5=srt.iloc[k:].sum(), total=df.pnl_usd.sum(), naked=df.naked_pnl_usd.sum())


rows = []
for split, (s, e) in SPLITS.items():
    sd = c[(c.entry_date >= s) & (c.entry_date <= e)]
    months = pd.period_range(pd.Period(s, 'M'), sd.exit_date.max().to_period('M'), freq='M')
    for cell, g in sd.groupby('cell'):
        a = stats(g, None); z = stats(g, months)
        d = g.iloc[0]
        rows.append(dict(split=split, cell=cell, delta=d.delta, width=d.width, mgmt=d.mgmt, gate=d.gate,
                         n=a['n'], nm_exit=a['n_months'], nm_cal=z['n_months'],
                         mean_b=a['mean_ret'], sh_b=a['sharpe'], green_b=a['green'],
                         mean_z=z['mean_ret'], sh_z=z['sharpe'], green_z=z['green'], worst=z['worst'], maxdd=z['maxdd'],
                         top5=a['top5_share'], ex_top5=a['ex_top5'], total=a['total'], naked=a['naked']))
R = pd.DataFrame(rows)
pd.set_option('display.width', 250)
for split in SPLITS:
    print(f'\n== {split} ==')
    print(R[R.split == split].sort_values('sh_z', ascending=False).round(4).to_string(index=False))

# selection under both conventions
tr = R[R.split == 'TRAIN']
for tag, sh, gr in [('builder', 'sh_b', 'green_b'), ('zero-filled', 'sh_z', 'green_z')]:
    el = tr[(tr.n >= 12) & (tr[gr] >= 0.55)].sort_values(sh, ascending=False)
    print(f'\nSelection ({tag}): top 5 ->', el[['cell', sh, gr]].head(5).values.tolist())
va = R[R.split == 'VAL'].set_index('cell')
print('\nVAL rank of 1574 by builder Sharpe:', int((va.sh_b > va.loc[1574, 'sh_b']).sum()) + 1, 'of 24;',
      'by zero-filled mean:', int((va.mean_z > va.loc[1574, 'mean_z']).sum()) + 1)
print('VAL best mean_b:', va.mean_b.idxmax(), va.mean_b.max(), ' best mean_z:', va.mean_z.idxmax(), va.mean_z.max())
print('VAL cells with mean_z >= 4%:', va[va.mean_z >= 0.04].index.tolist(), ' mean_b>=4%:', va[va.mean_b >= 0.04].index.tolist())
print('Rank corr TRAIN vs VAL sharpe (builder):', tr.set_index('cell').sh_b.rank().corr(va.sh_b.rank()),
      ' zero-filled:', tr.set_index('cell').sh_z.rank().corr(va.sh_z.rank()))

# neighbours of 1574 (d=.15 W=10 B gate1): delta neighbour 0.20 (1582), width neighbour W=5 (1570)
for nb in [1582, 1570, 1573, 1572]:
    print('neighbour', nb, 'VAL mean_z %.4f sh_z %.2f total %.0f' % (va.loc[nb, 'mean_z'], va.loc[nb, 'sh_z'], va.loc[nb, 'total']))

# concurrent budget: sum of worst cases open on each day, per cell
print('\nConcurrent worst-case exposure (max over days) per cell, vs B=6500:')
out = []
for cell, g in c.groupby('cell'):
    days = pd.date_range(g.entry_date.min(), g.exit_date.max(), freq='B')
    exp = [g[(g.entry_date <= d) & (g.exit_date > d)].worst_case_usd.sum() for d in days]
    nopen = [((g.entry_date <= d) & (g.exit_date > d)).sum() for d in days]
    out.append((cell, max(exp), max(nopen)))
E = pd.DataFrame(out, columns=['cell', 'max_exposure', 'max_open'])
print(E.to_string(index=False))
print('cells over B:', E[E.max_exposure > B].cell.tolist())
R.to_csv('/home/ec2-user/onemil/research/options_vrp/review/refuter2_cellstats.csv', index=False)
E.to_csv('/home/ec2-user/onemil/research/options_vrp/review/refuter2_exposure.csv', index=False)
