#!/usr/bin/env python3
"""hod_filter_stack — the reproduction gate and the independent-pass check.

(1) B0 built from THIS study's bar pass (`sig2.csv`) vs `hod_break/REPORT.md` §6's printed B0 row.
(2) Trade-by-trade against the SEPARATE bar pass `hod_break/breaks.csv` on the shared population:
    share of identical (day, symbol, entry_m) signals and max |delta rr|.
Read-only. One process.
"""
import os, sys
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT); sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_break')
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_filter_stack')
import score as S            # noqa: E402
import score2 as S2          # noqa: E402

REF = {'TRAIN': dict(n=1688, per_wk=31.8, gross=-0.027, net=-0.088, green=41.5, total=-14835),
       'VAL': dict(n=820, per_wk=35.7, gross=+0.016, net=-0.050, green=43.5, total=-4128)}


def main():
    p = S2.load_pop()
    S.build_impute(p)
    b0 = S2.sig_set(p, tag='b', band=1)
    bk = S.apply_book(b0, 12, 4)
    print('== REPRODUCTION GATE — B0 from THIS pass vs hod_break/REPORT.md §6 ==')
    print(S2.HDR)
    print(S.fmt_cell('B0 (this pass)', bk))
    ok = True
    for sp, r in REF.items():
        w = S.week_stats(bk, sp)
        d = [abs(w['n'] - r['n']), abs(w['gross'] - r['gross']), abs(w['net'] - r['net']),
             abs(w['green'] - r['green']), abs(w['total'] - r['total'])]
        good = d[0] <= max(5, 0.01 * r['n']) and d[1] < 0.01 and d[2] < 0.01 and d[3] < 2.0 and d[4] < 500
        ok &= good
        print(f'  {sp}: ref n={r["n"]} gross={r["gross"]:+.3f} net={r["net"]:+.3f} '
              f'green={r["green"]:.1f} total={r["total"]:+.0f}  ->  '
              f'{"MATCH" if good else "MISMATCH"} (dn {d[0]}, dgross {d[1]:.4f}, dnet {d[2]:.4f}, '
              f'dgreen {d[3]:.1f}, dtotal {d[4]:.0f})')
    print(f'REPRODUCTION GATE: {"PASS" if ok else "FAIL"}')

    print('\n== INDEPENDENT-PASS CHECK — sig2.csv vs hod_break/breaks.csv (separate bar passes) ==')
    ob = S.load_breaks()
    o_sig = S.signals(ob, 'b')
    o_sig = o_sig[(o_sig.fill_capped == 1)]
    raw_new = S2.sig_set(p, tag='b', band=1, max_bps=0.0, max_frac_r=0.0, obtain=False)
    a = raw_new[['day', 'symbol', 'entry_m', 'rr']].rename(columns={'rr': 'rr_new'})
    # both sides carry min_price/r_min/fill-cap; the cost gates are OFF on both
    b = o_sig[['day', 'symbol', 'entry_m', 'rr_b']].rename(columns={'rr_b': 'rr_old'})
    sd_new = set(map(tuple, a[['day', 'symbol']].values))
    sd_old = set(map(tuple, b[['day', 'symbol']].values))
    shared = sd_new & sd_old
    print(f'  symbol-days: new {len(sd_new)}  old {len(sd_old)}  shared {len(shared)} '
          f'({len(shared)/max(len(sd_new),1):.2%} of new)')
    m = a.merge(b, on=['day', 'symbol'], suffixes=('_n', '_o'))
    same_min = (m.entry_m_n == m.entry_m_o)
    print(f'  identical entry minute on shared symbol-days: {same_min.mean():.4%}')
    mm = m[same_min]
    dr = (mm.rr_new - mm.rr_old).abs()
    print(f'  max |delta rr| on those: {dr.max():.3e}   (n={len(mm)})')


if __name__ == '__main__':
    main()
