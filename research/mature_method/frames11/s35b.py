#!/usr/bin/env python3
"""F35 stage 2 — the circularity check and the two admission cells (PREREG §F35 AMENDMENT).

  (1) the tail label re-derived on GROSS `rr` (no cost term), every one of the 11 cells re-scored;
  (2) one admission cell per separating field: the parameter-free MEDIAN cut on the TRAIN book,
      keep the tail's side, re-book the FULL admitted signal set with run_book(12, 4).

  python3 s35b.py           # writes cells35b.csv, admit35.csv

Reads only; TEST is never loaded.
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
from common6 import base_book, mde, S                             # noqa: E402
from common4 import book_ranked                                   # noqa: E402
from common3 import clustered_t                                   # noqa: E402

D11 = f'{ROOT}/research/mature_method/frames11'
SPLITS = ('TRAIN', 'VAL')
NDRAW = 2000
SEED = 20260922
WKS = {'H1': 26, 'H2': 27, 'VAL': 23}


def week_row(bk, sp, tag):
    """The programme's week stats for one book on one split, plus the dollar path."""
    w = S.week_stats(bk, sp)
    d = bk[bk.split == sp]
    return dict(cell=tag, split=sp, n=w['n'], per_wk=w['per_wk'], gross=w['gross'],
                net=w['net'], green=w['green'], total=w['total'],
                t=clustered_t(d, 'net') if len(d) > 3 else np.nan,
                mde=mde(d) if len(d) > 5 else np.nan)


def green_null(bk, sp, rng, n=NDRAW):
    """Count-matched null of the green-week share: shuffle this cell's own P&L across its weeks."""
    d = bk[bk.split == sp]
    if not len(d):
        return np.nan
    wk = d.groupby('wk').pnl.sum()
    nweeks = WKS.get(sp if sp == 'VAL' else 'H1', len(wk))
    pnl = d.pnl.values
    cnt = d.groupby('wk').size().values
    out = np.empty(n)
    for k in range(n):
        p = rng.permutation(pnl)
        i, tot = 0, []
        for c in cnt:
            tot.append(p[i:i + c].sum())
            i += c
        out[k] = 100.0 * (np.array(tot) > 0).sum() / max(nweeks, 1)
    return float(np.percentile(out, 95))


def ex_top5(bk, sp):
    d = bk[bk.split == sp]
    if len(d) < 20:
        return np.nan
    k = max(1, int(round(len(d) * 0.05)))
    keep = d.net.sort_values(ascending=False).index[k:]
    return float(d.loc[keep, 'net'].mean())


def main() -> int:
    rng = np.random.default_rng(SEED)
    print('F35 stage 2 — circularity check + admission cells (PREREG AMENDMENT)', flush=True)
    b0, sig = base_book(verbose=False)
    b0 = b0[b0.split.isin(SPLITS)].copy()
    s35.repro_gate(b0)
    import c9
    pan = c9.panel()[['day', 'symbol', 'gap_pct']]
    b0 = b0.merge(pan, on=['day', 'symbol'], how='left')
    b0['spread_over_r'] = b0.sp_pct / b0.r_pct.clip(lower=0.05)

    # ---------------------------------------------------------------- (1) the circularity check
    print('\n  (1) THE CIRCULARITY CHECK — the tail re-labelled on GROSS rr (no cost term):',
          flush=True)
    bn = s35.label_tail(b0)                      # the declared label, on net
    bg = b0.copy()
    bg['is_tail'] = False
    for sp in SPLITS:
        d = bg[bg.split == sp]
        k = max(1, int(round(len(d) * 0.05)))
        bg.loc[d.rr.sort_values(ascending=False).index[:k], 'is_tail'] = True
    overlap = (bn.is_tail & bg.is_tail).sum() / max(bn.is_tail.sum(), 1)
    print(f'  net-label and gross-label tails overlap on {overlap*100:.0f} % of trades', flush=True)
    bn, cont = s35.build_fields(bn)
    bg, _ = s35.build_fields(bg)
    rows = []
    for cid, (fld, prov) in cont.items():
        r = dict(cell=cid, field=fld)
        for lab, frame in (('net', bn), ('gross', bg)):
            for sp in SPLITS:
                mt, mr, df_, t, nt, nr = s35.diff_t(frame[frame.split == sp], fld)
                r[f'{lab}_{sp}_diff'] = df_
                r[f'{lab}_{sp}_t'] = t
        same = (r['gross_TRAIN_diff'] == r['gross_TRAIN_diff']
                and np.sign(r['gross_TRAIN_diff']) == np.sign(r['gross_VAL_diff'])
                and abs(r['gross_TRAIN_t']) >= 2.0 and abs(r['gross_VAL_t']) >= 1.0)
        r['separates_on_gross'] = bool(same)
        rows.append(r)
        print(f'  {cid:4s} {fld:16s} NET diff TR {r["net_TRAIN_diff"]:+8.3f} (t {r["net_TRAIN_t"]:+5.2f}) '
              f'VAL {r["net_VAL_diff"]:+8.3f} (t {r["net_VAL_t"]:+5.2f}) | '
              f'GROSS diff TR {r["gross_TRAIN_diff"]:+8.3f} (t {r["gross_TRAIN_t"]:+5.2f}) '
              f'VAL {r["gross_VAL_diff"]:+8.3f} (t {r["gross_VAL_t"]:+5.2f}) | '
              f'{"SEPARATES on gross" if same else "no"}', flush=True)
    pd.DataFrame(rows).to_csv(f'{D11}/cells35b.csv', index=False)

    # ---------------------------------------------------------------- (2) the admission cells
    print('\n  (2) THE ADMISSION CELLS — median cut on the TRAIN book, keep the tail\'s side:',
          flush=True)
    s = sig.copy()
    s['spread_over_r'] = s.sp_pct / s.r_pct.clip(lower=0.05)
    out = []
    base = book_ranked(s, 12, 4)
    base = base[base.split.isin(SPLITS)]
    for sp in SPLITS:
        r = week_row(base, sp, 'B2 (reference)')
        r['null_green_p95'] = green_null(base, sp, rng)
        r['ex_top5'] = ex_top5(base, sp)
        out.append(r)
        print(f'  B2      {sp:5s} n={r["n"]:5d} /wk={r["per_wk"]:5.1f} gross={r["gross"]:+.3f} '
              f'net={r["net"]:+.3f} green={r["green"]:5.1f} (null p95 {r["null_green_p95"]:5.1f}) '
              f'${r["total"]:+,.0f} t={r["t"]:+.2f} ex5={r["ex_top5"]:+.3f}', flush=True)

    for cid, fld in (('A1', 'rv_profile'), ('A2', 'spread_over_r')):
        tr_book = b0[b0.split == 'TRAIN']
        cut = float(tr_book[fld].median())
        keep = s[s[fld] <= cut]                  # both passing fields are LOWER in the tail
        bk = book_ranked(keep, 12, 4)
        bk = bk[bk.split.isin(SPLITS)]
        for sp in SPLITS:
            r = week_row(bk, sp, f'{cid} {fld} <= {cut:.3f}')
            r['null_green_p95'] = green_null(bk, sp, rng)
            r['ex_top5'] = ex_top5(bk, sp)
            out.append(r)
            print(f'  {cid} {fld:14s} {sp:5s} n={r["n"]:5d} /wk={r["per_wk"]:5.1f} '
                  f'gross={r["gross"]:+.3f} net={r["net"]:+.3f} green={r["green"]:5.1f} '
                  f'(null p95 {r["null_green_p95"]:5.1f}) ${r["total"]:+,.0f} t={r["t"]:+.2f} '
                  f'ex5={r["ex_top5"]:+.3f} MDE={r["mde"]:.3f}', flush=True)
        for lab, q in (('H1', bk[(bk.split == 'TRAIN') & (bk.day < '2025-07-01')]),
                       ('H2', bk[(bk.split == 'TRAIN') & (bk.day >= '2025-07-01')])):
            print(f'    {cid} {lab}: n={len(q)} net={q.net.mean():+.3f} ${q.pnl.sum():+,.0f}',
                  flush=True)
    pd.DataFrame(out).to_csv(f'{D11}/admit35.csv', index=False)
    return 0


if __name__ == '__main__':
    sys.exit(main())
