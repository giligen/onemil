#!/usr/bin/env python3
"""hod_frames4 — build the compact pre-book signal set ONCE (F13/F14/F15 all read it).

`sig4.csv` = the B2 pre-book signal set (the shipped cascade, one row per symbol-day: the FIRST
qualifying break, cost attached, obtainable only) — exactly the object `hod_frames3` books with
`S.apply_book(..., 12, 4)`.  Also asserts that this pass's generalised slot machine reproduces
`trading.hod_break.run_book` EXACTLY when no ranking score is passed.  Read-only.  One process.
"""
import sys
import numpy as np, pandas as pd

sys.path.insert(0, '/home/ec2-user/onemil/research/mature_method/hod_frames4')
from common4 import ROOT, D4, S, S2, SPLITS, sigset, admit, load_breaks4, book_ranked  # noqa: E402

COLS = ['day', 'symbol', 'wk', 'split', 'entry_m', 'break_m', 'exit_m', 'level', 'price',
        'next_open', 'dist_open_pct', 'rv_profile', 'adv20', 'cumv', 'cum_dollar', 'bar_vol',
        'dollar_frac', 'adv_dollar', 'med_rng', 'day_range_pct', 'rng_sig', 'rng_day',
        'consol_bars', 'hod_age_bars', 'rr', 'net', 'netb', 'sp_pct', 'r_pct', 'stop',
        'notional', 'why', 'imputed', 'spy_r5_pct']


def line(tag, b):
    for sp in SPLITS:
        w = S.week_stats(b, sp)
        print(f'  {tag:28s} {sp:5s} n {w["n"]:5d} /wk {w["per_wk"]:5.1f} gross {w["gross"]:+.3f} '
              f'net {w["net"]:+.3f} green {w["green"]:5.1f}% $ {w["total"]:+9.0f}', flush=True)


def main():
    pop = S2.load_pop(); S.build_impute(pop)
    sig = S2.sig_set(pop, **S2.BASES['B2'])
    b2 = S.apply_book(sig, 12, 4)
    print('== REPRODUCTION GATE (ref B2 TRAIN 1,622/30.6/-0.039/-0.107/32.1%/-17,346 | '
          'VAL 706/30.7/+0.083/+0.013/43.5%/+893) ==', flush=True)
    line('R1/R2 B2 (pop.csv)', b2)

    # --- slot-machine parity: book_ranked(score=None) MUST equal run_book, row for row ----------
    b2b = book_ranked(sig, 12, 4)
    same = (len(b2) == len(b2b)) and set(b2.index) == set(b2b.index)
    print(f'\n  SLOT-MACHINE PARITY vs trading.hod_break.run_book: {len(b2)} vs {len(b2b)} rows, '
          f'identical set = {same}', flush=True)
    assert same, 'book_ranked(score=None) does not reproduce run_book'

    br = load_breaks4(use_nbbo3=False)
    base = sigset(admit(br, pd.Series(True, index=br.index)))
    print('\n== R2b REBUILT from breaks2 (must match R1/R2) ==', flush=True)
    line('R2b B2 (breaks2, no nbbo3)', S.apply_book(base, 12, 4))

    out = base[[c for c in COLS if c in base.columns]].copy()
    out.to_csv(f'{D4}/sig4.csv', index=False)
    print(f'\nsig4.csv rows {len(out)} | days {out.day.nunique()} | symbols {out.symbol.nunique()}',
          flush=True)

    print('\n== STRUCTURE of the slot rule (counts only, no P&L) ==', flush=True)
    for sp in SPLITS:
        d = out[out.split == sp]
        per_day = d.groupby('day').size()
        sim = d.groupby(['day', 'entry_m']).size()
        bk = S.apply_book(d, 12, 4)
        print(f'  {sp}: signals {len(d)} -> booked {len(bk)} ({len(bk)/len(d):.1%}) | days '
              f'{d.day.nunique()} | signals/day med {per_day.median():.0f} mean {per_day.mean():.1f}'
              f' p90 {per_day.quantile(.9):.0f}')
        print(f'         minutes with >=2 simultaneous signals: {(sim >= 2).sum()} of {len(sim)} '
              f'({(sim>=2).mean():.1%}); signals sharing a minute with another: '
              f'{sim[sim>=2].sum()} of {len(d)} ({sim[sim>=2].sum()/len(d):.1%})')
    print('\nDONE', flush=True)


if __name__ == '__main__':
    main()
