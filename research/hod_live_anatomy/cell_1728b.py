#!/usr/bin/env python3
"""Cell 1,728b (EXPLORATORY follow-up to cell_1728.py): score the shortlist candidates (prior-day return, Monday)
with the same kept-set statistics, add winner-capped means, and audit live rows that sit outside the live gates
(level < $20, level-basis R % < 1.5). Reads cell_1728_bt.csv / cell_1728_live.csv / fills_1721.csv only.
Output: research/hod_live_anatomy/cell_1728b_out.txt"""
import os

import numpy as np
import pandas as pd

from cell_1728 import dclust_t, ex_top5

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = open(os.path.join(HERE, 'cell_1728b_out.txt'), 'w')


def log(*a):
    """Print flushed and mirror to the output file."""
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    OUT.write(s + '\n')
    OUT.flush()


def stats(k: pd.DataFrame, col: str) -> str:
    """n / mean / day-t / ex-top-5 % / winner-capped (+1.5 R) / alternating-week halves A|B of one kept set."""
    if len(k) == 0:
        return 'n 0'
    return (f"n {len(k)} mean {k[col].mean():+.3f} t {dclust_t(k, col):+.2f} ex5 {ex_top5(k[col]):+.3f} "
            f"cap1.5 {k[col].clip(upper=1.5).mean():+.3f} A|B {k[k.halfAB == 'A'][col].mean():+.3f}|{k[k.halfAB == 'B'][col].mean():+.3f}")


def main():
    """Shortlist scoring and live-gate audit."""
    b = pd.read_csv(os.path.join(HERE, 'cell_1728_bt.csv'), low_memory=False)
    L = pd.read_csv(os.path.join(HERE, 'cell_1728_live.csv'))
    L['dow'] = pd.to_datetime(L.date).dt.dayofweek
    rules = {
        'prev_ret >= 0': lambda x: x.prev_ret >= 0,
        'prev_ret >= 3': lambda x: x.prev_ret >= 3,
        'Monday': lambda x: x.dow == 0,
        'prev_ret >= 3 OR Monday': lambda x: (x.prev_ret >= 3) | (x.dow == 0),
    }
    for lab, f in rules.items():
        log(f'\n### {lab}')
        for s in ('TRAIN-H2', 'VAL'):
            m = f(b).fillna(False) & (b.split == s)
            log(f'  BT {s} kept: net {stats(b[m], "net_R")} | gross_lvl {b[m].gross_lvl.mean():+.3f}')
            log(f'  BT {s} drop: net {b[~f(b).fillna(False) & (b.split == s)].net_R.mean():+.3f}')
        ml = f(L).fillna(False)
        log(f'  LIVE kept {ml.mean():.0%}: gross {stats(L[ml], "gross_R")} | net {L[ml].net_R.mean():+.3f} | dropped gross {L[~ml].gross_R.mean():+.3f}')
    f = pd.read_csv(os.path.join(HERE, 'fills_1721.csv'), low_memory=False)
    f = f[f.valid_r.astype(str) == 'True']
    lo_px = f[f.level < 20]
    log(f'\n## live gate audit: level < $20: {len(lo_px)} fills, dates {sorted(lo_px.trade_date.astype(str).unique())[:8]} ... '
        f'gross {lo_px.gross_R.mean():+.3f}')
    rl = (f.level - f.consol_low) / f.level * 100
    rr = f.risk_per_share / f.entry_price * 100
    log(f'level-basis R% < 1.5: {(rl < 1.5).sum()} fills; risk_per_share/entry R% < 1.5: {(rr < 1.5).sum()}; '
        f'dates of rr<1.5: {sorted(f[rr < 1.5].trade_date.astype(str).unique())[:8]}; gross {f[rr < 1.5].gross_R.mean():+.3f}')
    log(f'live date range {f.trade_date.min()} .. {f.trade_date.max()}; weeks {pd.to_datetime(f.trade_date).dt.isocalendar().week.nunique()}')


if __name__ == '__main__':
    main()
