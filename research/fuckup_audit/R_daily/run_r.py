#!/usr/bin/env python3
"""Stage R_daily step 3 — the five K families on the 2018-2026 Nasdaq-listed daily panel.

This does NOT reimplement anything.  It imports `research/fuckup_audit/K/build_k.py` and
`report_k.py` — the frozen Stage-K code, unmodified on disk — and patches module globals:

    PANEL   -> R_daily/daily_panel_2018_2026.parquet
    K       -> R_daily/                      (outputs land here; K/ and N2/ are not touched)
    SPLITS  -> TRAIN 2019-01-01..2023-12-31 | VAL 2024-01-01..2025-06-30 | TEST 2025-07-01..2026-09-04
    EARLY_CLOSES -> the 1 p.m. sessions of 2018..2026 (K's list only covered 2025-26)
    build_universe -> point-in-time Nasdaq-listed common stock (see below) instead of the 2026 class map

Universe (PREREG §1): base = valid prices, non-test ticker, 20-day median dollar volume >= $10M,
close >= $5, not an early close.  Membership:
    primary   — EQUS era (>= 2024-07-01): `exchange == XNAS and security_type == C` in THAT month;
                ITCH era: the union of the bought months (no point-in-time source before 2024-07).
    secondary — the union rule on both sides (the control for the per-month listing test).
Signals are suppressed on the 20 sessions from 2024-07-01, the only days whose 20-day volume window
straddles the ITCH/EQUS seam.

    python3 run_r.py A     # TRAIN + VAL only (TEST is never computed)
    python3 run_r.py B     # adds TEST, for cells frozen in R_daily/FREEZE.md
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

os.environ.setdefault('ARROW_DEFAULT_MEMORY_POOL', 'system')
ROOT = '/home/ec2-user/onemil'
sys.path.insert(0, f'{ROOT}/research/fuckup_audit/K')
os.chdir(ROOT)

import build_k as bk        # noqa: E402  (the Stage-K implementation, unmodified)
import report_k as rk       # noqa: E402

R = f'{ROOT}/research/fuckup_audit/R_daily'
SEAM = '2024-07-01'
SEAM_BLACKOUT = 20          # sessions whose 20-day volume window straddles the seam

bk.PANEL = f'{R}/daily_panel_2018_2026.parquet'
bk.K = R
rk.K = R
bk.SPLITS = [('TRAIN', '2019-01-01', '2023-12-31'),
             ('VAL', '2024-01-01', '2025-06-30'),
             ('TEST', '2025-07-01', '2026-09-04')]
# NYSE/Nasdaq 1 p.m. sessions 2018..2026 (a shortened session is never a signal day, PLAN §1)
bk.EARLY_CLOSES = {'2018-07-03', '2018-11-23', '2018-12-24',
                   '2019-07-03', '2019-11-29', '2019-12-24',
                   '2020-11-27', '2020-12-24',
                   '2021-11-26',
                   '2022-11-25',
                   '2023-07-03', '2023-11-24',
                   '2024-07-03', '2024-11-29', '2024-12-24',
                   '2025-07-03', '2025-11-28', '2025-12-24',
                   '2026-11-27', '2026-12-24'}


def build_universe(f):
    """Point-in-time Nasdaq-listed common stock, plus Stage-K's own base gates."""
    ps = pd.read_csv(f'{R}/pit_xnas_common.csv', keep_default_na=False, na_values=[''])
    ps['month'] = ps.month.astype(str)
    months = sorted(ps.month.unique())
    midx = {m: i for i, m in enumerate(months)}
    sidx = {str(s): i for i, s in enumerate(f['syms'])}
    member = np.zeros((len(months), len(f['syms'])), dtype=bool)
    for m, sym in zip(ps.month.to_numpy(), ps.symbol.to_numpy()):
        j = sidx.get(str(sym))
        if j is not None:
            member[midx[m], j] = True
    union = member.any(axis=0)

    day_month = np.array([str(d)[:4] + str(d)[5:7] for d in f['days']], dtype=object)
    day_mrow = np.array([midx.get(m, -1) for m in day_month], dtype='int32')
    is_equs = np.array([str(d) >= SEAM for d in f['days']])

    rows_day = f['day']
    rows_equs = is_equs[rows_day]
    mrow = day_mrow[rows_day]
    permonth = np.zeros(len(rows_day), dtype=bool)
    ok = mrow >= 0
    permonth[ok] = member[mrow[ok], f['sym'][ok]]
    row_union = union[f['sym']]

    is_test = np.array([bool(bk.TEST_TICKER.match(str(s))) for s in f['syms']])[f['sym']]
    base = (~f['bad']) & (~is_test) & np.isfinite(f['dvol20_med']) \
        & (f['dvol20_med'] >= bk.MIN_DVOL20_MED) & (f['close'] >= bk.MIN_PRICE)

    # seam blackout: the first SEAM_BLACKOUT sessions from the seam are not signal days
    days = np.array([str(d) for d in f['days']])
    seam_i = int(np.searchsorted(days, SEAM))
    blackout = np.zeros(len(days), dtype=bool)
    blackout[seam_i:seam_i + SEAM_BLACKOUT] = True
    not_black = ~blackout[rows_day]

    u_prim = base & not_black & np.where(rows_equs, permonth, row_union)
    u_sec = base & not_black & row_union
    bk.log(f'universe rows: base {int(base.sum()):,}  primary {int(u_prim.sum()):,}  '
           f'secondary(union rule both eras) {int(u_sec.sum()):,}  '
           f'seam blackout drops {int((base & ~not_black).sum()):,}')
    cls_of = np.array(['stock'] * len(f['syms']), dtype=object)
    return u_prim, u_sec, cls_of


bk.build_universe = build_universe


def split_control(phase):
    """Control book: every trade whose hold window contains a suspected split is removed.

    The daily files are UNADJUSTED (seam.md §4), so a 2:1 split inside a hold fabricates a -50%
    exit and a split on the signal day fabricates the signal itself.  This reports the same cells
    with those trades dropped; it never edits the primary numbers.
    """
    sc = pd.read_csv(f'{R}/split_candidates.csv', keep_default_na=False, na_values=[''])
    bad = set(zip(sc.symbol.astype(str), sc.bar_date.astype(str)))
    rows = []
    for fn in sorted(os.listdir(f'{R}/trades')):
        if not fn.endswith('.csv') or fn.endswith('_sec.csv'):
            continue
        t = pd.read_csv(f'{R}/trades/{fn}', keep_default_na=False, na_values=[''])
        if not len(t):
            continue
        hit = np.zeros(len(t), dtype=bool)
        for i, (sym, a, b) in enumerate(zip(t.symbol.astype(str), t.entry_date.astype(str),
                                            t.exit_date.astype(str))):
            for s, d in bad:
                if s == sym and a <= d <= b:
                    hit[i] = True
                    break
        for sp in (('TRAIN', 'VAL') if phase == 'A' else ('TRAIN', 'VAL', 'TEST')):
            x = t[(t.split == sp)]
            y = t[(t.split == sp) & ~hit]
            if not len(x):
                continue
            rows.append(dict(cell=fn[:-4], split=sp, n=len(x), n_dropped=len(x) - len(y),
                             net_bps=x.net.mean() * 1e4, net_bps_nosplit=y.net.mean() * 1e4,
                             t_nosplit=bk.tstat(y.net)))
    out = pd.DataFrame(rows)
    out.to_csv(f'{R}/split_control.csv', index=False, float_format='%.6g')
    bk.log(f'split control: {len(out)} cell-splits, '
           f'{int(out.n_dropped.sum()) if len(out) else 0} trade-splits removed in total')
    return out


if __name__ == '__main__':
    phase = sys.argv[1] if len(sys.argv) > 1 else 'A'
    bk.main(phase)
    rk.main(phase)
    split_control(phase)
