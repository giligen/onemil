#!/usr/bin/env python3
"""hod_frames4 / F14 supplementary — NO decision attached.

(S1) why F14-d6 (breadth at 09:35) is VOID by construction; (S2) the same breadth field at 10:00 and
10:30 as a descriptive; (S3) the H1-2025 mirror of the d7 search (the question `FRAMES.md` F14 asks,
beside the PREREG's H2 one); (S4) the day-level power statement on the D2 gate.
"""
import sys
import numpy as np, pandas as pd

sys.path.insert(0, '/home/ec2-user/onemil/research/mature_method/hod_frames4')
from common4 import D4, S, SPLITS, book_ranked   # noqa: E402
from score14 import day_table                    # noqa: E402


def main():
    s = pd.read_csv(f'{D4}/sig4.csv', dtype={'symbol': str, 'day': str, 'wk': str, 'split': str,
                                             'why': str}, keep_default_na=False, na_values=[''])
    bk = book_ranked(s, 12, 4)
    days = day_table(s)
    days['y'] = days.day.map(bk.groupby('day').pnl.sum()).fillna(0.0)

    print('== S1 — why F14-d6 (breadth at 09:35) is VOID ==')
    print(f'   pre-book signals with entry_m <= 576 (09:36): {int((s.entry_m<=576).sum())} of '
          f'{len(s)}; MIN entry_m in the population = {int(s.entry_m.min())} '
          f'({int(s.entry_m.min())//60:02d}:{int(s.entry_m.min())%60:02d} ET).')
    print('   The rule needs >= 5 closed bars holding within 4 % of a high that is already >= 5 % '
          'above the 09:30 open, so a break BEFORE 09:36 is structurally impossible. The field is '
          'identically 0 on every session and its terciles are 0/0. **F14-d6 is VOID by '
          'construction, not by data.**\n', flush=True)

    print('== S2 — the same breadth field at 10:00 and 10:30 (SUPPLEMENTARY, no decision) ==')
    print('| clock | split | T1 day $ | T2 day $ | T3 day $ | T3 H1 | T3 H2 |')
    print('|---|---|---|---|---|---|---|')
    for m, lab in ((601, '10:00'), (631, '10:30')):
        days[f'b{m}'] = days.day.map(s[s.entry_m <= m].groupby('day').size()).fillna(0)
        T = days[days.split == 'TRAIN']
        q = [float(T[f'b{m}'].quantile(x)) for x in (1 / 3, 2 / 3)]
        for sp in SPLITS:
            d = days[days.split == sp]
            t1 = d[d[f'b{m}'] <= q[0]].y.mean(); t2 = d[(d[f'b{m}'] > q[0]) & (d[f'b{m}'] <= q[1])].y.mean()
            t3d = d[d[f'b{m}'] > q[1]]
            h1 = t3d[t3d.h == 'H1'].y.mean(); h2 = t3d[t3d.h == 'H2'].y.mean()
            print(f'| {lab} (cuts {q[0]:.0f}/{q[1]:.0f}) | {sp} | {t1:+.1f} | {t2:+.1f} | '
                  f'{t3d.y.mean():+.1f} | {h1:+.1f} | {h2:+.1f} |')

    print('\n== S3 — the H1-2025 mirror of the d7 search (FRAMES.md F14 s own question) ==')
    T = days[days.split == 'TRAIN']
    q = [float(T.spy_r5_pct.quantile(x)) for x in (1 / 3, 2 / 3)]
    r = [float(T.spy_rng5_atr.quantile(x)) for x in (1 / 3, 2 / 3)]
    states = {
        'spy_r5>0': days.spy_r5_pct > 0,
        'spy_r5 T3 (most up)': days.spy_r5_pct > q[1],
        'qqq_r5>0': days.qqq_r5_pct > 0,
        'spy_r5>0 AND qqq_r5>0': (days.spy_r5_pct > 0) & (days.qqq_r5_pct > 0),
        'spy_rng5_atr T3 (wild)': days.spy_rng5_atr > r[1],
        'spy_rng5_atr T1 (quiet)': days.spy_rng5_atr <= r[0],
        'spy_gap>0': days.spy_gap_pct > 0,
        'spy_gap>0 AND spy_r5>0': (days.spy_gap_pct > 0) & (days.spy_r5_pct > 0),
    }
    print('| state | H1 days | H1 day $ | H1 t | H1 green d % | H1 total $ | H2 day $ | VAL day $ |')
    print('|---|---|---|---|---|---|---|---|')
    for nm, m in states.items():
        d1 = days[(days.h == 'H1') & m.reindex(days.index).fillna(False)]
        y = d1.y.values
        se = y.std(ddof=1) / np.sqrt(len(y)) if len(y) > 2 else np.nan
        d2 = days[(days.h == 'H2') & m.reindex(days.index).fillna(False)]
        dv = days[(days.h == 'VAL') & m.reindex(days.index).fillna(False)]
        print(f'| {nm} | {len(y)} | {y.mean():+.1f} | {y.mean()/se if se else np.nan:+.2f} | '
              f'{(y>0).mean()*100:.1f} | {y.sum():+.0f} | {d2.y.mean():+.1f} | {dv.y.mean():+.1f} |')

    print('\n== S4 — the day-level power statement on the D2 gate ==')
    for sp, hh in (('TRAIN', None), ('VAL', None)):
        d = days[(days.split == sp) & (days.spy_r5_pct > 0)]
        y = d.y.values; se = y.std(ddof=1) / np.sqrt(len(y))
        print(f'   {sp}: {len(y)} gated days, day mean {y.mean():+.1f}, SE {se:.1f}, '
              f'day-level t {y.mean()/se:+.2f}, 80%-power MDE {2.80*se:+.0f} $/day '
              f'-> the point estimate is {"BELOW" if abs(y.mean())<2.80*se else "above"} the MDE.')
    print('\nDONE', flush=True)


if __name__ == '__main__':
    main()
