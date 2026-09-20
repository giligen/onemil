#!/usr/bin/env python3
"""F35 SUPPLEMENT (declared and counted 2026-09-20, after the capped book returned no tail).

The B2 book is +2 R CAPPED: its "top 5 %" is 81 of the 330 trades that hit the cap, every one at
rr = +2.000 exactly, so there is no tail to identify and the net label ranks inside a point mass by
COST. The three objects the ex-top-5 % clause actually killed (`hod_fresh` C1, pass-9 SUPP A,
pass-10 F32-L) live on G3 — the BARE STOP ridden to 15:55, **uncapped** — which does have a real
right tail. The frame's question is therefore re-asked, unchanged, on the uncapped book.

Same 11 fields, same label rule (top 5 % by net within split), same four-part decision rule, plus
the same GROSS-label circularity check. 11 more cells, counted.

  python3 s35c.py           # writes cells35c.csv
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
for _p in ('frames11', 'frames10', 'frames9', 'frames8', 'frames7', 'hod_frames6',
           'hod_frames5', 'hod_frames4', 'hod_frames3', 'hod_frames2'):
    sys.path.insert(0, f'{ROOT}/research/mature_method/{_p}')
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_filter_stack')
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_break')

import s35                                                        # noqa: E402
from common6 import base_book, mde                                # noqa: E402
from common4 import book_ranked                                   # noqa: E402

D8 = f'{ROOT}/research/mature_method/frames8'
D11 = f'{ROOT}/research/mature_method/frames11'
SPLITS = ('TRAIN', 'VAL')
GEOM = 'G3'


def main() -> int:
    rng = np.random.default_rng(20260923)
    print(f'F35 SUPPLEMENT — the same question on the UNCAPPED book ({GEOM}, bare stop to 15:55)',
          flush=True)
    b0, sig = base_book(verbose=False)
    s35.repro_gate(b0[b0.split.isin(SPLITS)])

    w = pd.read_csv(f'{D8}/w_sig.csv', dtype={'day': str, 'symbol': str},
                    usecols=['day', 'symbol', 'entry_m', f'rr_{GEOM}', f'why_{GEOM}',
                             f'xm_{GEOM}'])
    s = sig.merge(w, on=['day', 'symbol', 'entry_m'], how='inner')
    cov = len(s) / len(sig)
    print(f'  {GEOM} walk joined on {len(s):,} of {len(sig):,} admitted signals '
          f'({cov*100:.1f} % — availability rail 80 %)', flush=True)
    assert cov >= 0.80, f'{GEOM} availability {cov:.2f} below the 80 % rail'

    # the geometry's OWN exit minute decides slot occupancy (F25's re-booking rule)
    s['rr'] = s[f'rr_{GEOM}']
    s['why'] = s[f'why_{GEOM}']
    s['exit_m'] = s[f'xm_{GEOM}']
    s = s[s.rr.notna()]
    s['net'] = s.rr - (s.netb if 'netb' in s.columns else 0.0) * 0  # cost re-attached below
    # the programme books cost as (rr - net) on the shipped book; re-apply the SAME per-trade cost
    cost = (sig.rr - sig.net).reindex(s.index) if 'net' in sig.columns else None
    ck = sig[['day', 'symbol', 'entry_m']].copy()
    ck['cost'] = sig.rr - sig.net
    s = s.merge(ck, on=['day', 'symbol', 'entry_m'], how='left')
    s['net'] = s.rr - s.cost
    s['pnl'] = s.net * 100.0            # the programme's $100-risk unit, as book6.pnl
    b = book_ranked(s, 12, 4)
    b = b[b.split.isin(SPLITS)].copy()
    for sp in SPLITS:
        d = b[b.split == sp]
        print(f'  {GEOM} book {sp:5s} n={len(d):5d} gross={d.rr.mean():+.3f} net={d.net.mean():+.3f} '
              f'| rr max {d.rr.max():+.2f} | share at +2R {100*(d.rr>1.99).mean():.1f} %', flush=True)

    import c9
    b = b.merge(c9.panel()[['day', 'symbol', 'gap_pct']], on=['day', 'symbol'], how='left')
    b['spread_over_r'] = b.sp_pct / b.r_pct.clip(lower=0.05)

    bn = s35.label_tail(b)
    bg = b.copy()
    bg['is_tail'] = False
    for sp in SPLITS:
        d = bg[bg.split == sp]
        k = max(1, int(round(len(d) * 0.05)))
        bg.loc[d.rr.sort_values(ascending=False).index[:k], 'is_tail'] = True
    print(f'  net-label / gross-label tails overlap {100*(bn.is_tail & bg.is_tail).sum()/max(bn.is_tail.sum(),1):.0f} %',
          flush=True)
    for sp in SPLITS:
        d = bn[bn.split == sp]
        t = d[d.is_tail]
        print(f'  {sp}: tail n={len(t)} mean net {t.net.mean():+.3f} R (range {t.net.min():+.2f}..'
              f'{t.net.max():+.2f}) | rest {d[~d.is_tail].net.mean():+.3f} | book {d.net.mean():+.3f}'
              f' | ex-top-5 % {d[~d.is_tail].net.mean():+.3f}', flush=True)

    bn, cont = s35.build_fields(bn)
    bg, _ = s35.build_fields(bg)
    rows = []
    for cid, (fld, prov) in cont.items():
        r = dict(cell=f'{cid}G', field=fld, provenance=prov)
        for lab, frame in (('net', bn), ('gross', bg)):
            for sp in SPLITS:
                mt, mr, df_, t, nt, nr = s35.diff_t(frame[frame.split == sp], fld)
                r[f'{lab}_{sp}_diff'], r[f'{lab}_{sp}_t'] = df_, t
        pv, p95 = s35.null_p(bn[bn.split == 'TRAIN'], fld, rng)
        pv_v, _ = s35.null_p(bn[bn.split == 'VAL'], fld, rng)
        r['p_null_train'], r['p_null_val'] = pv, pv_v
        r['avail_train'] = float(bn.loc[bn.split == 'TRAIN', fld].notna().mean())
        r['avail_val'] = float(bn.loc[bn.split == 'VAL', fld].notna().mean())
        ok_av = r['avail_train'] >= 0.80 and r['avail_val'] >= 0.80
        sep_net = bool(r['net_TRAIN_diff'] == r['net_TRAIN_diff']
                       and np.sign(r['net_TRAIN_diff']) == np.sign(r['net_VAL_diff'])
                       and abs(r['net_TRAIN_t']) >= 2.0 and abs(r['net_VAL_t']) >= 1.0
                       and pv <= 0.05 and ok_av)
        sep_gr = bool(r['gross_TRAIN_diff'] == r['gross_TRAIN_diff']
                      and np.sign(r['gross_TRAIN_diff']) == np.sign(r['gross_VAL_diff'])
                      and abs(r['gross_TRAIN_t']) >= 2.0 and abs(r['gross_VAL_t']) >= 1.0 and ok_av)
        r['SEPARATES'] = sep_net
        r['separates_on_gross'] = sep_gr
        rows.append(r)
        print(f'  {cid+"G":5s} {fld:16s} NET TR {r["net_TRAIN_diff"]:+8.3f} (t {r["net_TRAIN_t"]:+5.2f}) '
              f'VAL {r["net_VAL_diff"]:+8.3f} (t {r["net_VAL_t"]:+5.2f}) p {pv:.3f} | '
              f'GROSS TR {r["gross_TRAIN_diff"]:+8.3f} (t {r["gross_TRAIN_t"]:+5.2f}) '
              f'VAL {r["gross_VAL_diff"]:+8.3f} (t {r["gross_VAL_t"]:+5.2f}) | '
              f'avail {r["avail_train"]*100:.0f}/{r["avail_val"]*100:.0f}% | '
              f'{"SEPARATES" if sep_net else "no"}{" (gross too)" if sep_gr else ""}', flush=True)
    pd.DataFrame(rows).to_csv(f'{D11}/cells35c.csv', index=False)
    n = int(pd.DataFrame(rows).SEPARATES.sum())
    ng = int(pd.DataFrame(rows).separates_on_gross.sum())
    print(f'\n  11 supplement cells: {n} separate on the net label, {ng} on the gross label',
          flush=True)
    for sp in SPLITS:
        print(f'  MDE {sp}: {mde(bn[bn.split == sp]):.3f} R', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
