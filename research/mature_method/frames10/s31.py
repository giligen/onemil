#!/usr/bin/env python3
"""frames10 / F31 — THE R-UNIT AUDIT.

`R = move% / stop%`.  Every headline object of this programme was measured in R, and a rule that
tightens the stop inflates R without moving a cent of price.  This script re-expresses the EIGHT
objects declared in `PREREG.md` §1.1 in **% of entry price** beside R, with each object's stop-width
distribution, and applies the pre-committed flip rule (§1.3).

No bars are walked: every object already carries a per-trade stop width (`r_pct`) or a realized
per-cent move.  Read-only on every store.  TEST is never loaded.

  python3 s31.py            -> cells31.csv, the printed tables
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
for _p in ('frames10', 'frames9', 'frames8', 'frames7', 'hod_frames6', 'hod_frames5',
           'hod_frames4', 'hod_frames3'):
    sys.path.insert(0, f'{ROOT}/research/mature_method/{_p}')
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_filter_stack')
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_break')

D10 = f'{ROOT}/research/mature_method/frames10'
D9 = f'{ROOT}/research/mature_method/frames9'
D8 = f'{ROOT}/research/mature_method/frames8'
D7 = f'{ROOT}/research/mature_method/frames7'
D6 = f'{ROOT}/research/mature_method/hod_frames6'
SPLITS = ('TRAIN', 'VAL')
ROWS = []


def q(v):
    """(median, p25, p75) of a stop-width distribution, in % of price."""
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if not len(v):
        return (np.nan,) * 3
    return tuple(float(x) for x in np.percentile(v, [50, 25, 75]))


def emit(obj, split, n, r_mean, pct_mean, stop, note='', r_alt=None):
    med, p25, p75 = q(stop)
    ROWS.append(dict(obj=obj, split=split, n=n, R=r_mean, pct=pct_mean, stop_med=med,
                     stop_p25=p25, stop_p75=p75, R_alt=r_alt, note=note))
    alt = '' if r_alt is None or not np.isfinite(r_alt) else f' (book R {r_alt:+.3f})'
    print(f'{obj:26s} {split:5s} n={n:6d}  R={r_mean:+.4f}{alt}  '
          f'%price={pct_mean:+.4f}%  stop%={med:.2f} [{p25:.2f}–{p75:.2f}]  {note}', flush=True)


# ------------------------------------------------------------------- O1/O2/O3/O4 — HOD's objects
def hod_objects():
    from common4 import load_breaks4, admit
    from common5 import sigset5, S, S2
    from common6 import base_book
    br = load_breaks4(verbose=False)
    S.build_impute(S2.load_pop())
    b, _ = base_book(br, verbose=False)
    ref = {'TRAIN': (1622, -17346.0), 'VAL': (706, 893.0)}
    for sp in SPLITS:
        w = S.week_stats(b, sp)
        assert w['n'] == ref[sp][0] and abs(w['total'] - ref[sp][1]) < 1.0, f'B2 repro FAIL {sp}'
    print('  G-B2 reproduced (1,622 / -$17,346 TRAIN; 706 / +$893 VAL)', flush=True)

    # O1 — the shipped book, gross and net
    b = b.copy()
    b['move_pct'] = b.rr * b.r_pct
    b['net_pct'] = b.net * b.r_pct
    for sp in SPLITS:
        d = b[b.split == sp]
        emit('O1 HOD B2 gross', sp, len(d), float(d.rr.mean()), float(d.move_pct.mean()), d.r_pct)
        emit('O1 HOD B2 net', sp, len(d), float(d.net.mean()), float(d.net_pct.mean()), d.r_pct)

    # O2 — the pond bound: a random pond name at HOD's clock (arm u of pass 9)
    sig9 = pd.read_csv(f'{D9}/sig9.csv', dtype={'day': str, 'symbol': str},
                       keep_default_na=False, na_values=[''],
                       usecols=['day', 'symbol', 'entry_m', 'r_pct', 'split'])
    U = pd.read_csv(f'{D9}/w9_u.csv', dtype={'day': str, 'symbol': str, 'ctrl': str, 'pond': str},
                    keep_default_na=False, na_values=[''])
    U = U.merge(sig9, on=['day', 'symbol', 'entry_m'], how='left')
    U = U[U.split.isin(SPLITS) & U.r_pct.notna()]
    for g in ('G3', 'X0'):
        U[f'mp_{g}'] = U[f'rr_{g}'] * U.r_pct
    for sp in SPLITS:
        d = U[U.split == sp]
        for g in ('G3', 'X0'):
            emit(f'O2 pond bound {g}', sp, len(d), float(d[f'rr_{g}'].mean()),
                 float(d[f'mp_{g}'].mean()), d.r_pct, note='arm u, pass 9')

    # O8 — the stop-width buckets of the SAME arm-u population (SUPP B, re-priced)
    bnd = [(0, 1.5), (1.5, 3), (3, 6), (6, 1e9)]
    for lo, hi in bnd:
        d = U[(U.r_pct >= lo) & (U.r_pct < hi)]
        if len(d) < 20:
            continue
        lab = f'O8 stop {lo:g}-{hi:g}%' if hi < 1e9 else 'O8 stop >=6%'
        emit(lab, 'TR+VA', len(d), float(d.rr_G3.mean()), float(d.mp_G3.mean()), d.r_pct,
             note='G3, arm u')

    # O3 / O4 — the paired selection margin and its wrapper split
    import f29
    d = f29.margin_frame()
    m = {}
    for g in ('G3', 'X0'):
        d[f'mp_{g}'] = d[f'm_{g}'] * d.r_pct
    for sp in SPLITS:
        s = d[d.split == sp]
        for g in ('G3', 'X0'):
            m[(sp, g)] = float(s[f'm_{g}'].mean())
            emit(f'O3 margin {g}', sp, len(s), m[(sp, g)], float(s[f'mp_{g}'].mean()), s.r_pct,
                 note='paired, sig - own controls')
    assert abs(m[('TRAIN', 'G3')] - 0.1829) < 0.002 and abs(m[('VAL', 'G3')] - 0.2452) < 0.002, \
        f'F29 margin repro FAIL {m}'
    print('  G-F29 margin reproduced (+0.183 / +0.245 G3)', flush=True)
    for lvl in ('wrapper', 'stock'):
        s = d[d.cls == lvl]
        for sp in SPLITS:
            ss = s[s.split == sp]
            emit(f'O4 margin {lvl}', sp, len(ss), float(ss.m_G3.mean()),
                 float(ss.mp_G3.mean()), ss.r_pct, note='G3')
    return b, d


# ------------------------------------------------------------------------ O5 — the live ORB book
def orb_object():
    o = pd.read_csv(f'{ROOT}/research/orb_gates2/book_G3_meas.csv',
                    dtype={'symbol': str, 'date': str}, keep_default_na=False, na_values=[''])
    o['split'] = np.where(o.date < '2026-01-01', 'TRAIN',
                          np.where(o.date < '2026-06-01', 'VAL', 'TEST'))
    for sp, n in (('TRAIN', 282), ('VAL', 177)):
        assert int((o.split == sp).sum()) == n, 'ORB repro FAIL'
    print('  G-ORB reproduced (282 / 177 picks)', flush=True)
    f = o[(o.entered.astype(str).isin(('1', 'True', 'true'))) & o.pnl_pct.notna() &
          (o.range_size_pct > 0)].copy()
    f['R'] = f.pnl_pct / f.range_size_pct
    for sp in SPLITS:
        d = f[f.split == sp]
        emit('O5 ORB book_G3', sp, len(d), float(d.R.mean()), float(d.pnl_pct.mean()),
             d.range_size_pct, note='fills only; stop% = range_size_pct')
    return f


# ------------------------------------------------------------------------- O6 — the live BF book
def bf_object():
    b = pd.read_csv(f'{ROOT}/research/bf_frequency/runs/P1.csv',
                    dtype={'symbol': str, 'date': str}, keep_default_na=False, na_values=[''])
    assert len(b) == 56 and abs(float(b.pnl.sum()) - 139113.67) < 0.01, 'BF repro FAIL'
    print('  G-BF reproduced (56 trades / $139,113.67)', flush=True)
    b['split'] = np.where(b.date < '2026-01-01', 'TRAIN',
                          np.where(b.date < '2026-06-01', 'VAL', 'TEST'))
    b['stop_pct'] = (b.entry_price - b.stop_loss) / b.entry_price * 100
    b['R_price'] = b.pnl_pct / b.stop_pct
    b['R_book'] = b.pnl / 2000.0
    for sp in SPLITS:
        d = b[b.split == sp]
        if not len(d):
            continue
        emit('O6 BF P1', sp, len(d), float(d.R_price.mean()), float(d.pnl_pct.mean()),
             d.stop_pct, note='R_price = pnl_pct/stop%', r_alt=float(d.R_book.mean()))
    return b


# ------------------------------------------------------------------------ O7 — the F23 mirror
def mirror_object():
    p = pd.read_csv(f'{D7}/p23.csv', dtype={'day': str, 'symbol': str, 'ctrl': str, 'clock': str},
                    keep_default_na=False, na_values=[''])
    bk = pd.read_csv(f'{D6}/book6.csv', dtype={'day': str, 'symbol': str},
                     usecols=['day', 'symbol', 'entry_m', 'split', 'r_pct'])
    p = p.merge(bk, on=['day', 'symbol', 'entry_m'], how='left')
    p = p[p.split.isin(SPLITS) & p.r_pct.notna() & p.rr.notna()]
    bf = pd.read_csv(f'{ROOT}/research/fuckup_audit/O_halt/PASSIVE/borrow_flags.csv',
                     dtype={'symbol': str})
    ok = set(bf[(bf.shortable.astype(str) == 'True') &
                (bf.easy_to_borrow.astype(str) == 'True')].symbol)
    p = p[p.ctrl.isin(ok) & (p.down_pct > -10.0) & (p.clock == 'sig')]
    p['mp'] = p.rr * p.r_pct
    for lab, d in (('O7 mirror S-base', p), ('O7 mirror S-nm3', p[p.rng_pct <= 3.0])):
        for sp in SPLITS:
            s = d[d.split == sp]
            emit(lab, sp, len(s), float(s.rr.mean()), float(s.mp.mean()), s.r_pct,
                 note='gross, short R')
    return p


def flips(df):
    """The pre-committed flip rule (PREREG §1.3): sign flips, then the two rank orders."""
    print('\n== THE FLIP LIST (pre-committed rule, PREREG §1.3) ==', flush=True)
    out = []
    for r in df.itertuples():
        if not (np.isfinite(r.R) and np.isfinite(r.pct)):
            continue
        if np.sign(r.R) != np.sign(r.pct) and abs(r.R) > 1e-9 and abs(r.pct) > 1e-9:
            out.append(f'(a) SIGN  {r.obj} / {r.split}: R {r.R:+.4f} vs {r.pct:+.4f} % of price')
    for split in SPLITS:
        for tag, objs in (('books', ('O1 HOD B2 gross', 'O5 ORB book_G3', 'O6 BF P1')),
                          ('margins', ('O3 margin G3', 'O4 margin wrapper'))):
            d = df[(df.split == split) & df.obj.isin(objs)]
            if len(d) < 2:
                continue
            by_r = list(d.sort_values('R', ascending=False).obj)
            by_p = list(d.sort_values('pct', ascending=False).obj)
            v = 'SAME' if by_r == by_p else 'FLIPPED'
            print(f'  rank {tag:8s} {split:5s}: by R {by_r}  |  by %price {by_p}  -> {v}')
            if by_r != by_p:
                out.append(f'(b) RANK  {tag} {split}: {by_r} -> {by_p}')
    if out:
        for o in out:
            print('  ' + o, flush=True)
    else:
        print('  none', flush=True)
    return out


def main():
    print('== frames10 / F31 — THE R-UNIT AUDIT ==', flush=True)
    print('== reproduction gates ==', flush=True)
    hod_objects()
    orb_object()
    bf_object()
    mirror_object()
    df = pd.DataFrame(ROWS)
    df.to_csv(f'{D10}/cells31.csv', index=False)
    fl = flips(df)
    print(f'\ncells -> {D10}/cells31.csv ({len(df)} rows); flips {len(fl)}', flush=True)


if __name__ == '__main__':
    main()
