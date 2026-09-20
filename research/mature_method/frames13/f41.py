#!/usr/bin/env python3
"""F41 — THE OVERNIGHT CONTROL AS THE DENOMINATOR FOR THE THREE LIVE BOOKS.

  python3 f41.py bfwalk   # the ONE re-walk this frame needs: BF's arms WITH the shipped partial
  python3 f41.py score    # the 8 cells, the three readings, the ramp-band reading

No new walk for ORB: pass 7's `frames7/p24.csv` already walked ORB's arms under ORB's OWN exit
spec (`c7.walk_orb`: static lock +1.75 R -> +0.5 R, flat 15:45), at ORB's own clock and on ORB's
own universe.  BF's pass-7 walk used `partial=False`; the SHIPPED P1 book carries the 50 % @ +2 R
profit partial, so BF's arms are re-walked here with it — same bars, same pools, same seed.

Declared in `frames13/PREREG.md` §2 BEFORE any cell was read.  Read-only on every store.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/research/mature_method/frames7')
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_filter_stack')
from c7 import (D7, SPLITS, arrays, clustered_t, half_of, idx_of_minute, load_bars,   # noqa: E402
                mde, split_of, walk_bf, BF_FLAT_M)

D13 = f'{ROOT}/research/mature_method/frames13'
STATE = f'{D13}/f41_state.json'
OUT_BF = f'{D13}/p41_bf.csv'
SEED = 20260920
N_LATER = 10
TOL = 5
ORB_BOOK = f'{ROOT}/research/orb_gates2/book_G3_meas.csv'
BF_BOOK = f'{ROOT}/research/bf_frequency/runs/P1.csv'

# The book headlines F41 is asked to re-read, from the books' own reports (not recomputed here).
HEADLINE = {
    ('orb', 'TRAIN'): 0.2161, ('orb', 'VAL'): 0.4695,          # frames7 §1.2, walker-internal
    ('bf', 'TRAIN'): 0.6914, ('bf', 'VAL'): 0.8163,
}
BOOK_R = {                                                      # frames10 F31 §1.1, book-internal
    ('orb', 'TRAIN'): 0.2348, ('orb', 'VAL'): 0.1887,
    ('bf', 'TRAIN'): 0.9117, ('bf', 'VAL'): 0.5555,
}


def repro():
    """The reproduction gates, asserted BEFORE any cell is read (RUNBOOK step 1)."""
    o = pd.read_csv(ORB_BOOK, dtype={'symbol': str, 'date': str},
                    keep_default_na=False, na_values=[''])
    o['split'] = o.date.map(split_of)
    for sp, n in (('TRAIN', 282), ('VAL', 177)):
        got = int((o.split == sp).sum())
        assert got == n, f'ORB repro FAIL {sp}: {got} != {n}'
    b = pd.read_csv(BF_BOOK, dtype={'symbol': str, 'date': str},
                    keep_default_na=False, na_values=[''])
    tot = float(b.pnl.sum())
    assert len(b) == 56 and abs(tot - 139113.67) < 0.01, f'BF repro FAIL: {len(b)} / {tot}'
    print(f'  G-ORB  book_G3_meas 282 TRAIN / 177 VAL picks — MATCH', flush=True)
    print(f'  G-BF   runs/P1.csv 56 trades / ${tot:,.2f} — MATCH to the cent', flush=True)
    return o, b


# ------------------------------------------------------------------------------ the BF re-walk
def bfwalk():
    """BF's four arms re-walked WITH the shipped 50 % @ +2 R partial.  Resumable per session."""
    repro()
    bk = pd.read_csv(f'{D7}/book_bf.csv', dtype={'symbol': str, 'day': str},
                     keep_default_na=False, na_values=[''])
    pb = pd.read_csv(f'{D7}/poolb_bf.csv', dtype={'symbol': str, 'day': str, 'ctrl': str},
                     keep_default_na=False, na_values=[''])
    pu = pd.read_csv(f'{D7}/poolu_bf.csv', dtype={'symbol': str, 'day': str, 'ctrl': str},
                     keep_default_na=False, na_values=[''])
    done = set(json.load(open(STATE))['done']) if os.path.exists(STATE) else set()
    rng = np.random.default_rng(SEED)
    days = sorted(set(bk.day))
    print(f'BF re-walk WITH the partial: {len(days)} sessions, {len(done)} already done',
          flush=True)
    for day in days:
        if day in done:
            continue
        tr = bk[bk.day == day]
        p_b, p_u = pb[pb.day == day], pu[pu.day == day]
        syms = set(tr.symbol.astype(str)) | set(p_b.ctrl.astype(str)) | set(p_u.ctrl.astype(str))
        bars = load_bars(day, sorted(syms))
        arr = {}
        for s, gg in bars.items():
            a = arrays(gg)
            if a is not None:
                arr[s] = a
        rows = []
        for r in tr.itertuples():
            A = arr.get(r.symbol)
            if A is None:
                continue
            o, h, l, c, v, m = A
            em = int(r.entry_m) + 1            # the obtainable fill: the NEXT bar's open (rail 1b)
            e0 = idx_of_minute(m, em)
            if e0 < 0 or m[e0] > em + TOL:
                continue
            key = ('bf', day, r.symbol, em, r.split)
            for pp in (False, True):
                rr, why, xm = walk_bf(o, h, l, c, m, e0, float(r.r_pct), partial=pp)
                if rr == rr:
                    rows.append(key + ('sig', r.symbol, em, rr, why, int(pp)))
            cand = m[(m > em) & (m <= BF_FLAT_M - 30)]
            if len(cand):
                for mm in rng.choice(cand, size=min(N_LATER, len(cand)), replace=False):
                    k = idx_of_minute(m, mm)
                    for pp in (False, True):
                        rr, why, xm = walk_bf(o, h, l, c, m, k, float(r.r_pct), partial=pp)
                        if rr == rr:
                            rows.append(key + ('a2', r.symbol, int(mm), rr, why, int(pp)))
            for tag, pool in (('b', p_b), ('u', p_u)):
                for cs in pool[pool.symbol == r.symbol].ctrl.astype(str).unique():
                    B = arr.get(cs)
                    if B is None:
                        continue
                    bo, bh, bl, bc, bv, bm = B
                    k = idx_of_minute(bm, em)
                    if k < 0 or bm[k] > em + TOL:
                        continue
                    for pp in (False, True):
                        rr, why, xm = walk_bf(bo, bh, bl, bc, bm, k, float(r.r_pct), partial=pp)
                        if rr == rr:
                            rows.append(key + (tag, cs, em, rr, why, int(pp)))
        if rows:
            pd.DataFrame(rows, columns=['book', 'day', 'symbol', 'entry_m', 'split', 'arm',
                                        'ctrl', 'ctrl_m', 'rr', 'why', 'partial']).to_csv(
                OUT_BF, mode='a', header=not os.path.exists(OUT_BF), index=False)
        done.add(day)
        json.dump({'done': sorted(done)}, open(STATE, 'w'))
        print(f'  {day}: {len(rows)} rows', flush=True)
    print('BF re-walk done', flush=True)


# ------------------------------------------------------------------------------ the scoring
def paired(sig, ctrl, keys=('day', 'symbol', 'entry_m')):
    """Per booked trade: the signal's R minus the MEAN of its own controls.  Day-clustered."""
    g = ctrl.groupby(list(keys)).rr.mean()
    s = sig.set_index(list(keys)).rr
    j = s.to_frame('sig').join(g.to_frame('ctrl'), how='inner').dropna()
    if not len(j):
        return np.nan, np.nan, np.nan, 0
    d = (j.sig - j.ctrl).values
    days = j.index.get_level_values('day').values
    return float(d.mean()), clustered_t(d, days), mde(d), len(d)


def score():
    print('== F41 reproduction gates ==', flush=True)
    repro()

    p = pd.read_csv(f'{D7}/p24.csv', dtype={'symbol': str, 'ctrl': str, 'day': str},
                    keep_default_na=False, na_values=[''])
    orb = p[p.book == 'orb'].copy()
    bf0 = p[p.book == 'bf'].copy()                      # pass 7's BF, partial OFF
    bfn = pd.read_csv(OUT_BF, dtype={'symbol': str, 'ctrl': str, 'day': str},
                      keep_default_na=False, na_values=[''])
    bf1 = bfn[bfn.partial == 1].copy()                  # the SHIPPED P1 exit spec
    bf_off = bfn[bfn.partial == 0].copy()               # the re-walk's own partial-OFF control

    # the re-walk must reproduce pass 7's partial-OFF arms — the parity gate on this frame
    for arm in ('sig', 'b', 'u'):
        a = bf0[bf0.arm == arm].rr.mean()
        b = bf_off[bf_off.arm == arm].rr.mean()
        print(f'  G-BFWALK arm {arm}: pass-7 {a:+.4f} vs this re-walk partial-OFF {b:+.4f} '
              f'(d {b - a:+.5f})', flush=True)

    rows = []
    for name, fr in (('orb', orb), ('bf', bf1)):
        for sp in SPLITS:
            f = fr[fr.split == sp]
            sig = f[f.arm == 'sig']
            for arm, tag in (('b', 'matched non-signal'), ('u', 'universe bound')):
                c = f[f.arm == arm]
                C = float(c.rr.mean())
                H = float(sig.rr.mean())
                d, tc, md, n = paired(sig, c)
                rows.append(dict(book=name, split=sp, arm=arm, what=tag,
                                 n_sig=len(sig), n_ctrl=len(c),
                                 control_R=C, headline_R=H, share=C / H if H else np.nan,
                                 paired=d, tc=tc, mde=md, n_pairs=n))
    t = pd.DataFrame(rows)
    t.to_csv(f'{D13}/cells41.csv', index=False)
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 30)
    print('\n== THE 8 SCORED CELLS — the CONTROL FIRST, as the frame pre-commits ==')
    print(t[['book', 'split', 'arm', 'n_sig', 'n_ctrl', 'control_R', 'headline_R', 'share',
             'paired', 'tc', 'mde']].to_string(index=False, float_format=lambda v: f'{v:+.4f}'))

    print('\n== THE THREE READINGS (PREREG §2.3: |C| < 0.25H both splits = R-ZERO; '
          'C >= 0.25H on either = R-POS; C < 0 both = R-NEG) ==')
    for name in ('orb', 'bf'):
        for arm in ('b', 'u'):
            s = t[(t.book == name) & (t.arm == arm)].set_index('split')
            C = s.control_R
            H = s.headline_R
            if (C < 0).all():
                rd = 'R-NEG'
            elif ((C / H) >= 0.25).any():
                rd = 'R-POS'
            elif (C.abs() < 0.25 * H).all():
                rd = 'R-ZERO'
            else:
                rd = 'MIXED'
            print(f'  {name.upper():4s} arm {arm} ({"matched non-signal" if arm == "b" else "universe bound"}): '
                  f'control {C["TRAIN"]:+.4f} / {C["VAL"]:+.4f} R  vs headline '
                  f'{H["TRAIN"]:+.4f} / {H["VAL"]:+.4f}  -> share '
                  f'{C["TRAIN"] / H["TRAIN"] * 100:.0f} % / {C["VAL"] / H["VAL"] * 100:.0f} %  '
                  f'=> **{rd}**')

    print('\n== BOTH HALVES (gross R, walker-internal) ==')
    for name, fr in (('orb', orb), ('bf', bf1)):
        f = fr.copy()
        f['half'] = [half_of(d) if split_of(d) == 'TRAIN' else 'VAL' for d in f.day]
        g = f.pivot_table(index='arm', columns='half', values='rr', aggfunc='mean')
        print(f'\n-- {name.upper()} --')
        print(g.to_string(float_format=lambda v: f'{v:+.4f}'))

    print('\n== BF: WHAT THE SHIPPED PARTIAL DOES TO SIGNAL AND CONTROL (PREREG P41.2) ==')
    for sp in SPLITS:
        for arm in ('sig', 'b', 'u'):
            a = float(bf_off[(bf_off.split == sp) & (bf_off.arm == arm)].rr.mean())
            b = float(bf1[(bf1.split == sp) & (bf1.arm == arm)].rr.mean())
            print(f'  {sp:5s} arm {arm:3s}: partial OFF {a:+.4f} -> ON {b:+.4f}  ({b - a:+.4f})')

    print('\n== THE RAMP-BAND READING (PREREG §2.4 — reported, NOT applied) ==')
    for name in ('orb', 'bf'):
        for sp in SPLITS:
            C = float(t[(t.book == name) & (t.split == sp) & (t.arm == 'b')].control_R.iloc[0])
            print(f'  {name.upper():4s} {sp}: book R {BOOK_R[(name, sp)]:+.4f} | exit-geometry '
                  f'control {C:+.4f} | difference {BOOK_R[(name, sp)] - C:+.4f}')


if __name__ == '__main__':
    {'bfwalk': bfwalk, 'score': score}[sys.argv[1]]()
