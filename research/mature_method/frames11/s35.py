#!/usr/bin/env python3
"""F35 — THE TAIL IS THE BOOK: is the tail IDENTIFIABLE at the entry bar?

PREREG.md §F35.  Eleven declared cells, one per causal entry-bar field, on the B2 booked book.

  python3 s35.py            # writes cells35.csv, s35.log to stdout

Reads only; TEST is never loaded (`FREEZE.md`).
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

from common6 import base_book, mde                       # noqa: E402
from common3 import clustered_t                          # noqa: E402

D11 = f'{ROOT}/research/mature_method/frames11'
SPLITS = ('TRAIN', 'VAL')
TAIL_FRAC = 0.05
NDRAW = 2000
SEED = 20260921
AVAIL_FLOOR = 0.80


def repro_gate(b):
    """G-B2, asserted before any number is read (frames10 §0b)."""
    for sp, n_want, pnl_want in (('TRAIN', 1622, -17346.0), ('VAL', 706, 893.0)):
        d = b[b.split == sp]
        n, pnl = len(d), float(d.pnl.sum())
        print(f'  G-B2 {sp:5s} n={n:5d} (want {n_want}) pnl=${pnl:+,.0f} (want ${pnl_want:+,.0f})',
              flush=True)
        assert n == n_want, f'G-B2 {sp}: n {n} != {n_want}'
        assert abs(pnl - pnl_want) < 1.0, f'G-B2 {sp}: pnl {pnl} != {pnl_want}'
    print('  G-B2 MATCH', flush=True)


def label_tail(b):
    """TAIL = the top TAIL_FRAC of trades by `net` WITHIN its split (rank-trimmed, as the clause)."""
    b = b.copy()
    b['is_tail'] = False
    for sp in SPLITS:
        d = b[b.split == sp]
        k = max(1, int(round(len(d) * TAIL_FRAC)))
        idx = d.net.sort_values(ascending=False).index[:k]
        b.loc[idx, 'is_tail'] = True
    return b


# ------------------------------------------------------------------------------- the statistics
def diff_t(d, field):
    """(tail mean, rest mean, difference, day-clustered t of the difference, n_tail, n_rest).

    The t is the clustered t of a per-trade contrast: x_i = field_i * (1/p if tail else -1/(1-p)),
    whose mean IS the tail-minus-rest difference and whose day-clustered SE is the programme's.
    """
    d = d[d[field].notna()]
    if len(d) < 10 or d.is_tail.sum() < 3:
        return (np.nan,) * 4 + (int(d.is_tail.sum()), int((~d.is_tail).sum()))
    p = float(d.is_tail.mean())
    if not (0 < p < 1):
        return (np.nan,) * 4 + (int(d.is_tail.sum()), int((~d.is_tail).sum()))
    x = np.where(d.is_tail, d[field] / p, -d[field] / (1 - p))
    q = pd.DataFrame({'day': d.day.values, 'net': x})
    t = clustered_t(q, 'net')
    mt = float(d.loc[d.is_tail, field].mean())
    mr = float(d.loc[~d.is_tail, field].mean())
    return mt, mr, mt - mr, t, int(d.is_tail.sum()), int((~d.is_tail).sum())


def null_p(d, field, rng):
    """p-value of |tail - rest| against a count-matched label null.

    The tail LABEL is reshuffled NDRAW times preserving the per-day count of tail trades, so the
    null keeps the day structure (a day with 3 monsters keeps 3 tail labels somewhere that day).
    """
    d = d[d[field].notna()]
    if len(d) < 10 or d.is_tail.sum() < 3:
        return np.nan, np.nan
    v = d[field].values.astype(float)
    days = d.day.values
    obs = abs(diff_t(d, field)[2])
    order = np.argsort(days, kind='mergesort')
    v_s, days_s = v[order], days[order]
    tail_s = d.is_tail.values[order]
    bounds = np.flatnonzero(np.r_[True, days_s[1:] != days_s[:-1]])
    bounds = np.r_[bounds, len(days_s)]
    ntail_day = [int(tail_s[bounds[i]:bounds[i + 1]].sum()) for i in range(len(bounds) - 1)]
    n = len(v_s)
    draws = np.empty(NDRAW)
    p = tail_s.mean()
    for k in range(NDRAW):
        lab = np.zeros(n, dtype=bool)
        for i, nt in enumerate(ntail_day):
            if nt:
                a, b = bounds[i], bounds[i + 1]
                lab[a + rng.choice(b - a, size=nt, replace=False)] = True
            # a day with no tail trade contributes none, exactly as observed
        if lab.sum() == 0 or lab.sum() == n:
            draws[k] = 0.0
            continue
        draws[k] = abs(v_s[lab].mean() - v_s[~lab].mean())
    return float((draws >= obs).mean()), float(np.percentile(draws, 95))


# ------------------------------------------------------------------------------- the fields
def build_fields(b):
    """The eleven declared fields, each named with its provenance. Returns (df, kinds)."""
    b = b.copy()
    b['spread_over_r'] = b.sp_pct / b.r_pct.clip(lower=0.05)
    b['advd'] = b.adv20 * b.price
    if 'asset_class' not in b.columns:
        import c7
        b['asset_class'] = b.symbol.map(c7.asset_class(b.symbol.unique()))
    b['is_wrapper'] = (b.asset_class == 'wrapper').astype(float)
    b['log_advd'] = np.log10(b.advd.clip(lower=1.0))
    b['log_price'] = np.log10(b.price.clip(lower=0.01))
    cont = {
        'C1': ('rv_profile', 'relative volume at the break (breaks2)'),
        'C2': ('dollar_frac', "the bar's dollar volume / ADV$ (book6)"),
        'C3': ('spread_over_r', 'imputed spread / R (book6 sp_pct, r_pct)'),
        'C4': ('dist_open_pct', '% above the 09:30 open (breaks2)'),
        'C5': ('gap_pct', 'session gap vs prev close (PIT panel)'),
        'C6': ('is_wrapper', 'wrapper = 1, common = 0 (orb_asset_class)'),
        'C7': ('log_price', 'log10 entry price (book6)'),
        'C8': ('entry_m', 'entry minute, ET minutes past midnight (book6)'),
        'C9': ('spy_r5_pct', "SPY's 5-min return into the bar (day_fields)"),
        'C10': ('r_pct', 'stop width as % of price (book6)'),
        'C11': ('log_advd', 'log10 ADV$ (book6 adv20 x price)'),
    }
    return b, cont


BANDS = {
    'C7': [(0, 30, '$20-30'), (30, 100, '$30-100'), (100, 1e9, '>=$100')],
    'C8': [(577, 630, '09:37-10:30'), (630, 690, '10:30-11:30'),
           (690, 780, '11:30-13:00'), (780, 842, '13:00-14:01')],
    'C11': [(0, 25e6, '<$25M'), (25e6, 150e6, '$25-150M'), (150e6, 1e18, '>=$150M')],
    'C6': None,
}


def main() -> int:
    rng = np.random.default_rng(SEED)
    print('F35 — is the tail identifiable at the entry bar?  (PREREG §F35, 11 cells)', flush=True)
    b, sig = base_book(verbose=True)
    b = b[b.split.isin(SPLITS)].copy()
    repro_gate(b)

    # the entry-bar fields that live on the break row, not on book6
    extra = ['rv_profile', 'dist_open_pct', 'cum_dollar', 'bar_vol', 'open_px']
    have = [c for c in extra if c in b.columns]
    print(f'  fields already on the booked rows: {have}', flush=True)

    # gap: from the point-in-time panel, known at 09:30
    import c9
    pan = c9.panel()[['day', 'symbol', 'gap_pct']]
    b = b.merge(pan, on=['day', 'symbol'], how='left')

    if 'dollar_frac' not in b.columns or b.dollar_frac.isna().all():
        b['dollar_frac'] = b.cum_dollar / (b.adv20 * b.price).replace(0, np.nan)

    b = label_tail(b)
    b, cont = build_fields(b)
    print(f'  tail labelled: TRAIN {int(b[(b.split=="TRAIN")].is_tail.sum())} of '
          f'{int((b.split=="TRAIN").sum())} | VAL {int(b[(b.split=="VAL")].is_tail.sum())} of '
          f'{int((b.split=="VAL").sum())}', flush=True)
    for sp in SPLITS:
        d = b[b.split == sp]
        print(f'  tail {sp}: mean net {d[d.is_tail].net.mean():+.3f} R over {int(d.is_tail.sum())} '
              f'| rest {d[~d.is_tail].net.mean():+.3f} R | book {d.net.mean():+.3f} R', flush=True)

    rows = []
    for cid, (fld, prov) in cont.items():
        avail = {sp: float(b.loc[b.split == sp, fld].notna().mean()) for sp in SPLITS}
        ok_avail = all(v >= AVAIL_FLOOR for v in avail.values())
        res = {}
        for sp in SPLITS:
            d = b[b.split == sp]
            mt, mr, df_, t, nt, nr = diff_t(d, fld)
            pv, p95 = null_p(d, fld, rng)
            res[sp] = dict(mt=mt, mr=mr, d=df_, t=t, nt=nt, nr=nr, p=pv, p95=p95)
        # TRAIN halves
        h = {}
        for lab, q in (('H1', b[(b.split == 'TRAIN') & (b.day < '2025-07-01')]),
                       ('H2', b[(b.split == 'TRAIN') & (b.day >= '2025-07-01')])):
            h[lab] = diff_t(q, fld)[2]
        tr, va = res['TRAIN'], res['VAL']
        same_sign = (tr['d'] == tr['d'] and va['d'] == va['d']
                     and np.sign(tr['d']) == np.sign(va['d']) and tr['d'] != 0)
        passes = bool(same_sign and abs(tr['t']) >= 2.0 and abs(va['t']) >= 1.0
                      and tr['p'] <= 0.05 and ok_avail)
        rows.append(dict(cell=cid, field=fld, provenance=prov,
                         avail_train=avail['TRAIN'], avail_val=avail['VAL'],
                         availability_ok=ok_avail,
                         tail_train=tr['mt'], rest_train=tr['mr'], diff_train=tr['d'],
                         t_train=tr['t'], p_null_train=tr['p'], null_p95_train=tr['p95'],
                         tail_val=va['mt'], rest_val=va['mr'], diff_val=va['d'],
                         t_val=va['t'], p_null_val=va['p'], null_p95_val=va['p95'],
                         diff_h1=h['H1'], diff_h2=h['H2'],
                         same_sign=same_sign, SEPARATES=passes))
        print(f'  {cid:4s} {fld:16s} TRAIN tail {tr["mt"]:+9.3f} rest {tr["mr"]:+9.3f} '
              f'diff {tr["d"]:+8.3f} t {tr["t"]:+5.2f} p {tr["p"]:.3f} | '
              f'VAL diff {va["d"]:+8.3f} t {va["t"]:+5.2f} p {va["p"]:.3f} | '
              f'H1 {h["H1"]:+7.3f} H2 {h["H2"]:+7.3f} | avail {avail["TRAIN"]*100:.0f}/'
              f'{avail["VAL"]*100:.0f}% | {"SEPARATES" if passes else "no"}', flush=True)

    # the banded reads (diagnostics; the tail's SHARE per level vs the same null)
    print('\n  BANDED DIAGNOSTICS — the tail\'s share of each level (expected 5.0 %):', flush=True)
    for cid, bands in BANDS.items():
        fld = cont[cid][0]
        raw = {'C7': 'price', 'C8': 'entry_m', 'C11': 'advd', 'C6': 'asset_class'}[cid]
        for sp in SPLITS:
            d = b[b.split == sp]
            if bands is None:
                lv = [(k, d[d[raw] == k]) for k in sorted(d[raw].dropna().unique())]
            else:
                lv = [(lab, d[(d[raw] >= lo) & (d[raw] < hi)]) for lo, hi, lab in bands]
            parts = ' '.join(f'{lab}: {100*g.is_tail.mean():4.1f}% (n={len(g)})' for lab, g in lv
                             if len(g))
            print(f'  {cid:4s} {sp:5s} {parts}', flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(f'{D11}/cells35.csv', index=False)
    n_sep = int(out.SEPARATES.sum())
    print(f'\n  11 cells scored, {n_sep} SEPARATE the tail on the pre-committed rule', flush=True)
    for sp in SPLITS:
        d = b[b.split == sp]
        print(f'  MDE {sp}: {mde(d):.3f} R on the book\'s own net', flush=True)
    # how much of the book the tail carries — the size of the question
    for sp in SPLITS:
        d = b[b.split == sp]
        print(f'  tail carries {sp}: ${d[d.is_tail].pnl.sum():+,.0f} of ${d.pnl.sum():+,.0f} | '
              f'ex-tail net {d[~d.is_tail].net.mean():+.3f} R', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
