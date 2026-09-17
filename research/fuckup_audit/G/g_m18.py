#!/usr/bin/env python3
"""Stage G correction — the pure M18 spec (no stop) measured on the RIGHT population.

`score_short.py`'s M18 block ran on the SCOREABLE set, which carries `r_pct >= 1.0` — a
stop-distance filter that, for S5, keeps only names whose 09:30-09:34 high is at least 1% above the
09:35 open, i.e. names that had ALREADY fallen in the first five minutes. M18 has no stop and no such
filter. This script recomputes the block on the full S5 signal population (the borrowable attention
and control names with a 09:35 bar) and prints the R-filtered version beside it so the size of that
selection is visible.

Also reports the same quantity split by the M18 dollar-volume bands so it can be laid against
`research/lit_review_2026/open_fade.md` directly.

Writes ONLY research/fuckup_audit/G/m18_check.md.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
G = f'{ROOT}/research/fuckup_audit/G'
ENTRY_MULT, COVER_RATIO, BORROW_BPS = 0.25, 0.412, 5.274
L = []


def log(m):
    print(m, flush=True)
    L.append(m)


def tab(y, name):
    rows = []
    for grp in ('attention', 'control'):
        for hz, col in (('09:35->10:30', 'ret_1030_bps'), ('09:35->close', 'ret_eod_bps')):
            for sp in ('TRAIN', 'VAL'):
                z = y[(y.attn_grp == grp) & (y.split == sp) & y[col].notna()]
                if len(z) < 20:
                    continue
                cost = (ENTRY_MULT + COVER_RATIO) * z.spread_cc_bps / 2.0
                gross, net = z[col], z[col] - cost
                se = net.std(ddof=1) / np.sqrt(len(net))
                rows.append(dict(pop=name, group=grp, horizon=hz, split=sp, n=len(z),
                                 gross_bps=round(float(gross.mean()), 1),
                                 net_bps=round(float(net.mean()), 1),
                                 t=round(float(net.mean() / se), 2) if se > 0 else 0.0,
                                 net_borrow_bps=round(float(net.mean() - BORROW_BPS), 1),
                                 MDE_bps=round(2.80 * float(se), 1)))
    return pd.DataFrame(rows)


def main():
    d = pd.read_csv(f'{G}/candidates_short.csv',
                    usecols=['day', 'symbol', 'fam', 'split', 'attn_grp', 'ssr', 'entry', 'r_pct',
                             'spread_cc_bps', 'ret_1030_bps', 'ret_eod_bps', 'dvol20_med'],
                    dtype={'fam': str, 'split': str, 'attn_grp': str, 'symbol': str, 'day': str},
                    keep_default_na=False, na_values=[''])
    d = d[(d.fam == 'S5') & d.entry.notna() & (d.entry >= 10) & (d.ssr == 0)].copy()
    log('# Stage G — the pure M18 spec on the right population')
    log('')
    log(f'- S5 filled rows, borrowable, SSR excluded: **{len(d):,}** '
        f'(attention {int((d.attn_grp == "attention").sum()):,} / '
        f'control {int((d.attn_grp == "control").sum()):,})')
    r1 = d[d.r_pct >= 1.0]
    log(f'- of which `r_pct >= 1.0` keeps {len(r1):,} ({len(r1)/len(d)*100:.1f}%) — the filter that '
        f'contaminated the first version of this table: for S5 it means "the 09:30-09:34 high is at '
        f'least 1% above the 09:35 open", i.e. the name had ALREADY dropped in the first five '
        f'minutes. M18 has no such filter.')
    log('')
    log('## A. The M18 spec as published: every attention/control name, no stop, no R filter')
    log('')
    log(tab(d, 'ALL (correct for M18)').to_string(index=False))
    log('')
    log('## B. The same on the r_pct >= 1.0 subset (what score_short.py reported — kept for the record)')
    log('')
    log(tab(r1, 'r_pct>=1 subset').to_string(index=False))
    log('')
    log('## C. By M18 dollar-volume band, population A, open(09:35)->10:30, in bps of the SHORT')
    log('')
    bands = [('$10-50M', 10e6, 50e6), ('>$50M', 50e6, 1e18)]
    rows = []
    for bl, lo, hi in bands:
        for grp in ('attention', 'control'):
            for sp in ('TRAIN', 'VAL'):
                z = d[(d.dvol20_med >= lo) & (d.dvol20_med < hi) & (d.attn_grp == grp)
                      & (d.split == sp) & d.ret_1030_bps.notna()]
                if len(z) < 20:
                    continue
                rows.append(dict(band=bl, group=grp, split=sp, n=len(z),
                                 short_gross_bps=round(float(z.ret_1030_bps.mean()), 1),
                                 stock_move_bps=round(-float(z.ret_1030_bps.mean()), 1)))
    log(pd.DataFrame(rows).to_string(index=False))
    log('')
    log('`short_gross_bps` > 0 = the stock FELL (the fade M18 measured); `stock_move_bps` is the '
        'same number with M18\'s sign (the stock\'s own return), so it can be laid directly against '
        '`research/lit_review_2026/open_fade.md` — remembering that M18 measures from the 09:30 '
        'OPEN and this measures from 09:35, so the first five minutes of the fade '
        '(M18 TRAIN r0935: -37 bps at $10-50M, -14 at >$50M) is NOT in these numbers.')
    open(f'{G}/m18_check.md', 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
