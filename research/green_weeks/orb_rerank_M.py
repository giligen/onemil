#!/usr/bin/env python3
"""green_weeks — re-rank Stage M's EXISTING ORB exit cells on week shape.

PREREG §4: X1..X5 are re-ranked, not re-run.  Stage M ranked them on total P&L
and max drawdown and adopted none; this asks the different question.
No backtest is executed here — it reads Stage M's committed book CSVs.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, '/home/ec2-user/onemil/research/green_weeks')
os.chdir('/home/ec2-user/onemil')
import weekshape as W                                       # noqa: E402

M = 'research/fuckup_audit/M'
OUT = 'research/green_weeks'
TEST_END = '2026-09-16'
SHAPES = ['X0', 'X1', 'X2', 'X3', 'X4', 'X5']
LABEL = {
    'X0': 'E0 shipped (static lock 1.75R->+0.5R)',
    'X1': 'M-X1  X0 + 10-min time stop < +0.25R',
    'X2': 'M-X2  X0 + 5-min time stop < 0R',
    'X3': 'M-X3  no lock, hold to 15:45',
    'X4': 'M-X4  no lock + 10-min time stop',
    'X5': 'M-X5  no lock + 10-min TS + breakeven at +1R',
}


def load(shape, n=8):
    d = pd.read_csv(f'{M}/book_{shape}_n{n}.csv', keep_default_na=False,
                    na_values=[''], dtype={'symbol': str})
    d['day'] = d['date'].astype(str).str[:10]
    return d


def main():
    reveal = set(sys.argv[1:])
    rows, wks = [], {}
    for s in SHAPES:
        d = load(s)
        r = W.score_all(d, TEST_END, value='_sized_pnl',
                        reveal_test=bool(reveal & {s, 'ALL'}))
        for split, m in r.items():
            wks[(s, split)] = m.pop('_wk')
            rows.append({'cell': s, 'label': LABEL[s], **m})
    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT}/orb_M_rerank.csv', index=False)

    for split in ('TRAIN', 'VAL'):
        sub = df[df.split == split].sort_values('green_pct', ascending=False)
        print(f"\n=== ORB N=8 · {split} "
              f"({int(sub.n_weeks.iloc[0])} weeks) — ranked on GREEN WEEKS ===")
        print(f"{'cell':4s} {'green%':>7s} {'flat%':>6s} {'red%':>6s} "
              f"{'redstk':>6s} {'worstwk':>9s} {'mo_grn%':>7s} {'mdd':>9s} "
              f"{'pnl':>10s} {'disc':>5s}")
        for _, r in sub.iterrows():
            dsc = W.discordant(wks[('X0', split)], wks[(r.cell, split)])
            print(f"{r.cell:4s} {r.green_pct:7.1f} {r.flat_pct:6.1f} "
                  f"{r.red_pct:6.1f} {int(r.red_streak):6d} {r.worst_wk:9.0f} "
                  f"{r.mo_green_pct:7.1f} {r.mdd:9.0f} {r.pnl:10.0f} "
                  f"{dsc[0]:5d}")
    print("\n(disc = weeks green in exactly one of {this cell, X0} — PREREG §7)")


if __name__ == '__main__':
    main()
