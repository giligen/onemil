#!/usr/bin/env python3
"""PART 3 — slots on the GREEN-WEEK metric, plus the buying-power bind rate.

D1 measured slots on total P&L (edge gone by rank 9).  This re-reads the same
lever on the owner's metric, and prices the capital it needs.
"""
from __future__ import annotations

import sys

import pandas as pd

D = '/home/ec2-user/onemil/research/orb_frequency'
sys.path.insert(0, D)
from score import score_book, load_book, split_of  # noqa: E402

RISK = 375.0
MIN_STOP = 1.0
LIVE_ACCOUNT = 66_000.0      # the live account, ~$66K
ORB_BUDGET = 26_666.67       # orb.yaml account_budget_usd today


def bind(book: pd.DataFrame, per_pos: float) -> float:
    stop = book['range_size_pct'].clip(lower=MIN_STOP)
    uncapped = RISK / (stop / 100.0)
    return 100.0 * float((uncapped > per_pos).mean())


def main():
    rows = []
    for cfg, label in (('F0', 'F0 shipped B+'),
                       ('Fcat', 'catalyst veto OFF'),
                       ('F4', 'F4 = -range-size -G1 -PDR')):
        for n in (3, 8, 12, 16):
            p = f'{D}/book_S{n}_{cfg}_meas.csv'
            b = load_book(p)
            s = score_book(p, f'{cfg}_N{n}')
            tr = s[s.split == 'TRAIN'].iloc[0]
            va = s[s.split == 'VAL'].iloc[0]
            maxday = int(b.groupby('date').size().max())
            rows.append({
                'config': label, 'N': n,
                'picks': len(b[b.date.map(split_of).isin(['TRAIN', 'VAL'])]),
                'TR_pk_wk': round(tr.picks_per_wk, 2), 'VA_pk_wk': round(va.picks_per_wk, 2),
                'TR_green%': round(tr.green_wk_pct, 1), 'VA_green%': round(va.green_wk_pct, 1),
                'TR_flat%': round(tr.flat_wk_pct, 1), 'VA_flat%': round(va.flat_wk_pct, 1),
                'TR_streak': tr.red_streak, 'VA_streak': va.red_streak,
                'TR_worst_wk': round(tr.worst_wk), 'VA_worst_wk': round(va.worst_wk),
                'TR_mdd': round(tr.mdd), 'VA_mdd': round(va.mdd),
                'TR_$': round(tr.pnl), 'VA_$': round(va.pnl),
                'TR_R/pk': round(tr.R_pick, 3), 'VA_R/pk': round(va.R_pick, 3),
                'max_day': maxday,
                'bind%_inv3333': round(bind(b, 3333.333333333333), 1),
                'bind%_live66k': round(bind(b, LIVE_ACCOUNT / n), 1),
                'bind%_orb26k': round(bind(b, ORB_BUDGET / n), 1),
                'cap_live66k': round(LIVE_ACCOUNT / n),
                'capital_used': round(maxday * min(LIVE_ACCOUNT / n, 3333.33)),
            })
    t = pd.DataFrame(rows)
    t.to_csv(f'{D}/slots.csv', index=False)
    pd.set_option('display.width', 320)
    print(t.to_string(index=False))


if __name__ == '__main__':
    main()
