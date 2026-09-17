#!/usr/bin/env python3
"""Stage H / F6 — step 0: the PARITY ANCHOR.  Nothing else in the stage runs until this reproduces.

Three anchors, all computed with `h_core.py` (which re-states Stage C's contract in code):

  A1  E/REPORT.md §7-8.8   F6 {} x next x hold x range_so_far_pct >= 5, causal universe Q
                           expected TRAIN +0.0896 (n 1034, t 2.03, 20.3/wk), VAL +0.2088 (n 512, t 2.80)
  A2  C/REPORT.md §4.1     F6 {} x next x hold, all-day, universe P
                           expected TRAIN +0.051 (t 1.34), VAL +0.163 (t 2.46)
  A3  C/score5_results.csv F6 {} x next x '2R stop-1%', all-day, universe P  (the secondary variant)

Plus the no-floor twin on Q (E/REPORT.md §8.4: floor-passing +0.090 / below-floor -0.114 on TRAIN,
+0.209 / -0.046 on VAL) so the floor's own contribution is on the record.

Usage: ulimit -v 1500000; nice -n 10 python3 research/fuckup_audit/H/F6/h0_parity.py
"""
import os, sys, time
import pandas as pd

sys.path.insert(0, '/home/ec2-user/onemil/research/fuckup_audit/H/F6')
import h_core as C

H = C.H
pd.set_option('display.width', 240)


def line(tag, st):
    if st is None:
        return f'| {tag} | (no book) | | | | | | |'
    return (f'| {tag} | {st["n"]} | {st["tpw"]} | **{st["meanR"]:+.4f}** | {st["grossR"]:+.4f} | '
            f'{st["t"]:.2f} | {st["WR"]:.1f} | {st["green"]:.2f} |')


def main():
    L = ['# Stage H / F6 — step 0: parity anchors', '', f'generated {time.strftime("%Y-%m-%d %H:%M:%S")}', '']
    exp = {
        ('Q', 'hold', True, 'TRAIN'): (1034, 0.0896, 2.03),
        ('Q', 'hold', True, 'VAL'): (512, 0.2088, 2.80),
        ('P', 'hold', True, 'TRAIN'): (None, 0.051, 1.34),
        ('P', 'hold', True, 'VAL'): (None, 0.163, 2.46),
    }
    rows = []
    for pop in ('P', 'Q'):
        d = C.load(pop, with_features=False)
        wk = C.weeks_of(pop)
        L += [f'## universe {pop} — {len(d):,} F6 rows, weeks '
              + ' '.join(f'{s}={len(wk[s])}' for s in ('TRAIN', 'VAL', 'TEST')), '',
              '| cell | n | tr/wk | net R | gross R | t | WR | green |', '|---|---:|---:|---:|---:|---:|---:|---:|']
        for variant in ('hold', 'stopm1'):
            for floor in (True, False):
                x = C.scoreable(d, variant, floor)
                for sp in ('TRAIN', 'VAL'):
                    st, _ = C.book_stats(x, sp, wk, variant)
                    tag = f'{variant} floor={int(floor)} {sp}'
                    L.append(line(tag, st))
                    rows.append(dict(pop=pop, variant=variant, floor=int(floor), split=sp,
                                     **(st or {})))
                    e = exp.get((pop, variant, floor, sp))
                    if e and st:
                        dn = '' if e[0] is None else f'  dn={st["n"]-e[0]:+d}'
                        L.append(f'| &nbsp;&nbsp;expected | {e[0] or ""} | | {e[1]:+.4f} | | {e[2]:.2f} | | |'
                                 f'{dn}')
        # the no-floor twin, split in two halves (E/REPORT.md §8.4)
        if pop == 'Q':
            L += ['', '### the floor twin — each half booked on its own (E §8.4)', '',
                  '| half | split | n | net R | t | stop% |', '|---|---|---:|---:|---:|---:|']
            for nm, m in (('passed floor', d.range_so_far_pct >= 5), ('below floor', d.range_so_far_pct < 5)):
                x = C.scoreable(d[m], 'hold', floor=False)
                for sp in ('TRAIN', 'VAL'):
                    st, _ = C.book_stats(x, sp, wk, 'hold')
                    if st:
                        L.append(f'| {nm} | {sp} | {st["n"]} | **{st["meanR"]:+.4f}** | {st["t"]:.2f} | '
                                 f'{st["stopP"]:.1f} |')
        L.append('')
    pd.DataFrame(rows).to_csv(f'{H}/h0_parity.csv', index=False)
    open(f'{H}/h0_parity.md', 'w').write('\n'.join(L))
    print('\n'.join(L))


if __name__ == '__main__':
    main()
