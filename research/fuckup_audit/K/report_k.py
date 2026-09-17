#!/usr/bin/env python3
"""Stage K — turn the cell table into the pre-registered gate verdicts and the report tables.

`python3 report_k.py A`  -> K/phaseA.md  (TRAIN + VAL only; the gate that decides the freeze)
`python3 report_k.py B`  -> K/phaseB.md  (adds the TEST column for the frozen survivors)

Gates are PLAN.md §1 verbatim:
  G1 TRAIN : mean net > 0, t >= 2.0, >= 5 trades/week
  G2 VAL   : mean net > 0, t >= 1.0, >= 55% of weeks green, and the weekly mean must clear
             (number of G1 survivors // 10) standard errors of the weekly series
  G3 TEST  : read once, reported whatever it says
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
K = f'{ROOT}/research/fuckup_audit/K'


def fmt(v, n=1, suf=''):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return '—'
    return f'{v:,.{n}f}{suf}'


def g1_pass(r):
    return bool(r.n and r.net_bps > 0 and np.isfinite(r.t_trade) and r.t_trade >= 2.0
                and r.trades_per_week >= 5.0)


def g2_pass(r, extra_se):
    if not r.n or not np.isfinite(r.t_trade):
        return False
    se = r.mde_week_bps / 2.0 if np.isfinite(r.mde_week_bps) else np.nan
    bar = extra_se * se if np.isfinite(se) else 0.0
    return bool(r.net_bps > 0 and r.t_trade >= 1.0 and r.wk_green >= 55.0
                and r.wk_mean_bps > bar)


def main(phase):
    cells = pd.read_csv(f'{K}/cells_{"trainval" if phase == "A" else "all"}.csv',
                        keep_default_na=False, na_values=[''])
    perm = pd.read_csv(f'{K}/perm_p.csv', keep_default_na=False, na_values=[''])
    pmap = {(r.cell, r.split): r.p_adj for r in perm.itertuples()}
    sig = pd.read_csv(f'{K}/signal_counts.csv', keep_default_na=False, na_values=[''])
    prim = cells[cells.universe == 'primary']
    tr = prim[prim.split == 'TRAIN'].set_index('cell')
    va = prim[prim.split == 'VAL'].set_index('cell')
    order = [c for c in prim.cell.unique()]

    g1 = {c: g1_pass(tr.loc[c]) for c in order if c in tr.index}
    n_g1 = sum(g1.values())
    extra_se = n_g1 // 10
    g2 = {c: (g1.get(c, False) and c in va.index and g2_pass(va.loc[c], extra_se))
          for c in order}

    L = [f'# Stage K — phase {phase} tables', '']
    L.append('## Signal frequency (pre-book, primary universe)')
    L.append('')
    L.append('| family | TRAIN signals | /week | VAL | TEST |')
    L.append('|---|---:|---:|---:|---:|')
    for r in sig[sig.universe == 'primary'].itertuples():
        L.append(f'| {r.fam} | {r.train:,} | {r.train_per_week:.1f} | {r.val:,} | {r.test:,} |')
    L.append('')

    splits = ('TRAIN', 'VAL') if phase == 'A' else ('TRAIN', 'VAL', 'TEST')
    for sp in splits:
        x = prim[prim.split == sp].set_index('cell')
        L.append(f'## {sp} — the 20 pre-registered cells (primary universe, PREREG cost model)')
        L.append('')
        L.append('| cell | n | tr/wk | gross bps | **net bps** | t | WR% | wk mean bps | wk green% '
                 '| stop% | net R | MDE/trade bps | MDE/week bps | p_adj |')
        L.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
        for c in order:
            if c not in x.index:
                continue
            r = x.loc[c]
            L.append(f'| `{c}` | {int(r.n):,} | {fmt(r.trades_per_week)} | {fmt(r.gross_bps)} | '
                     f'**{fmt(r.net_bps)}** | {fmt(r.t_trade, 2)} | {fmt(r.wr)} | '
                     f'{fmt(r.wk_mean_bps)} | {fmt(r.wk_green)} | {fmt(r.stop_rate)} | '
                     f'{fmt(r.net_R, 3)} | {fmt(r.mde_trade_bps)} | {fmt(r.mde_week_bps)} | '
                     f'{fmt(pmap.get((c, sp)), 3)} |')
        L.append('')

    L.append('## Gate verdicts')
    L.append('')
    L.append('G1 (TRAIN): mean net > 0, t >= 2.0, >= 5 trades/week. '
             f'G2 (VAL): mean net > 0, t >= 1.0, >= 55% weeks green, weekly mean above '
             f'{extra_se} SE ({n_g1} cells passed G1 -> {n_g1}//10 = {extra_se}).')
    L.append('')
    L.append('| cell | G1 | G2 |')
    L.append('|---|---|---|')
    for c in order:
        L.append(f'| `{c}` | {"PASS" if g1.get(c) else "fail"} | '
                 f'{"PASS" if g2.get(c) else "fail"} |')
    L.append('')
    surv = [c for c in order if g2.get(c)]
    L.append(f'**G1 survivors: {n_g1} of {len(order)}. G2 survivors: {len(surv)}'
             + (f' — {", ".join(surv)}.' if surv else '.') + '**')
    L.append('')

    L.append('## Tails and the cost model (primary universe)')
    L.append('')
    for sp in splits:
        x = prim[prim.split == sp].set_index('cell')
        L.append(f'### {sp}')
        L.append('')
        L.append('| cell | net bps | no top 1% | no top 5% | winners capped | net bps (daily-band '
                 'costs) | net bps (auction costs) | gross bps |')
        L.append('|---|---:|---:|---:|---:|---:|---:|---:|')
        for c in order:
            if c not in x.index:
                continue
            r = x.loc[c]
            L.append(f'| `{c}` | {fmt(r.net_bps)} | {fmt(r.net_bps_no_top1)} | '
                     f'{fmt(r.net_bps_no_top5)} | {fmt(r.net_bps_wincap)} | '
                     f'{fmt(r.net_daily_bps)} | {fmt(r.net_auction_bps)} | {fmt(r.gross_bps)} |')
        L.append('')

    L.append('## Capacity (1% of the 20-day median dollar volume per position)')
    L.append('')
    L.append('One position size for the whole book (PREREG sizes equal-$ per position): the median '
             'over the cell\'s trades of 1% of the name\'s 20-day median dollar volume.')
    L.append('')
    L.append('| cell | split | position $ (median) | position $ (p25) | book $ at N slots | $/month '
             '(mean) | worst month $ | %book/month | months |')
    L.append('|---|---|---:|---:|---:|---:|---:|---:|---:|')
    for sp in splits:
        x = prim[prim.split == sp].set_index('cell')
        for c in order:
            if c not in x.index:
                continue
            r = x.loc[c]
            L.append(f'| `{c}` | {sp} | {fmt(r.pos_usd_med, 0)} | {fmt(r.pos_usd_p25, 0)} | '
                     f'{fmt(r.book_usd, 0)} | {fmt(r.usd_per_month, 0)} | '
                     f'{fmt(r.usd_worst_month, 0)} | {fmt(r.pct_book_per_month, 2, "%")} | '
                     f'{int(r.n_months) if np.isfinite(r.n_months) else 0} |')
    L.append('')

    sec = cells[cells.universe == 'secondary']
    if len(sec):
        L.append('## Survivorship control — secondary universe (stock OR not-in-map, i.e. only '
                 'positively identified wrappers dropped), declared hold, 10 slots')
        L.append('')
        L.append('| cell | split | n | net bps | t | wk green% |')
        L.append('|---|---|---:|---:|---:|---:|')
        for r in sec.itertuples():
            L.append(f'| `{r.cell}` | {r.split} | {int(r.n):,} | {fmt(r.net_bps)} | '
                     f'{fmt(r.t_trade, 2)} | {fmt(r.wk_green)} |')
        L.append('')

    with open(f'{K}/phase{phase}.md', 'w') as f:
        f.write('\n'.join(L) + '\n')
    print('\n'.join(L))
    print(f'\nwrote {K}/phase{phase}.md', flush=True)
    return surv


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'A')
