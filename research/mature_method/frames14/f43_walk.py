#!/usr/bin/env python3
"""F43 stage 1 — THE GEOMETRY FLOOR WALK.

Prices SEVEN exit geometries with **NO admission rule** on pass 6's 288,174 detector-free arm-d
controls (`hod_frames6/pd6.csv`), stop width a pure function of price (s in {2, 3, 4} %), so every
cell is a property of the UNIVERSE and the clock and never of a signal.

  X1  +2R bracket, flat 15:55          — HOD-break's geometry (already on disk as frames11/w34.csv;
                                         recomputed here and ASSERTED identical, which is the gate)
  X2  ORB static lock, flat 15:45      — arm +1.75R -> stop +0.5R forever, no target
  X3  BF R-trail, no partial, flat 15:45
  X4  BF R-trail + 50 % @ +2R partial, stop -> breakeven, flat 15:45
  X5  bare stop, no target, flat 15:55
  (X6 hold-to-next-open and X7 MOC need no bar walk — they are daily-panel joins, done in f43.py)

The bar walks are VECTORISED rewrites of `hod_frames6.common6.walk_from` and
`frames7.c7.walk_orb / walk_bf`; `--gate` asserts they reproduce those python loops exactly before
any row is written.

Resumable per session. Every store READ-ONLY. TEST never loaded (day < 2026-06-01).

  python3 f43_walk.py --gate     # parity gates only
  python3 f43_walk.py            # gates, then append to w43.csv (state in w43_state.json)
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
for _p in ('frames14', 'frames11', 'hod_frames6', 'hod_frames5', 'hod_frames4', 'hod_frames3',
           'hod_frames2', 'frames7'):
    sys.path.insert(0, f'{ROOT}/research/mature_method/{_p}')
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_filter_stack')

from common6 import walk_from, bars_arrays, EOD_M, SLIP          # noqa: E402
from c7 import walk_orb, walk_bf                                 # noqa: E402
from pass2 import load_bars                                      # noqa: E402

D6 = f'{ROOT}/research/mature_method/hod_frames6'
D11 = f'{ROOT}/research/mature_method/frames11'
D14 = f'{ROOT}/research/mature_method/frames14'
OUT = f'{D14}/w43.csv'
ST = f'{D14}/w43_state.json'
TEST_FROM = '2026-06-01'
STOPS = (0.02, 0.03, 0.04)
ORB_FLAT_M, BF_FLAT_M, HOD_FLAT_M = 945, 945, 955

GEOS = ('x1', 'x2', 'x3', 'x4', 'x5')
HDR = ['day', 'ctrl', 'ctrl_entry_m', 'entry'] + [
    f'{g}_{k}_{int(s * 100)}' for s in STOPS for g in GEOS for k in ('rr', 'why', 'xm')]


# --------------------------------------------------------------------- the vectorised geometries
def _first(mask, s0):
    """Index of the first True at or after s0, or -1."""
    if not mask.any():
        return -1
    return s0 + int(np.argmax(mask))


def v_x1(o, h, l, c, m, e, stop):
    """+2R bracket, flat 15:55 — walk_from's convention, returning the exit minute too."""
    n = len(o)
    E = float(o[e]); R = E - stop
    if not (R > 0):
        return np.nan, '', -1
    tgt = E + 2.0 * R
    s0 = e + 1
    if s0 >= n:
        return (float(c[-1]) - E) / R, 'eod', int(m[-1])
    eod = m[s0:] >= EOD_M
    hs = l[s0:] <= stop
    ht = c[s0:] >= tgt
    any_ = eod | hs | ht
    if not any_.any():
        return (float(c[-1]) - E) / R, 'eod', int(m[-1])
    j = int(np.argmax(any_)); k = s0 + j
    if eod[j]:
        px, why = float(o[k]), 'eod'
    elif hs[j]:
        px, why = float(min(stop, o[k]) * (1.0 - SLIP)), 'stop'
    else:
        px, why = float(tgt), 'target'
    return (px - E) / R, why, int(m[k])


def v_x5(o, h, l, c, m, e, stop, flat_m=HOD_FLAT_M):
    """Bare stop, NO target, flat at `flat_m`."""
    n = len(o)
    E = float(o[e]); R = E - stop
    if not (R > 0):
        return np.nan, '', -1
    s0 = e + 1
    if s0 >= n:
        return (float(c[-1]) - E) / R, 'eod', int(m[-1])
    fl = m[s0:] >= flat_m
    hs = l[s0:] <= stop
    any_ = fl | hs
    if not any_.any():
        return (float(c[-1]) - E) / R, 'eod', int(m[-1])
    j = int(np.argmax(any_)); k = s0 + j
    if fl[j]:
        return (float(o[k]) - E) / R, 'flat', int(m[k])
    px = float(min(stop, o[k]) * (1.0 - SLIP))
    return (px - E) / R, 'stop', int(m[k])


def v_x2(o, h, l, c, m, e, r_pct, lock_arm_r=1.75, lock_stop_r=0.5, flat_m=ORB_FLAT_M):
    """ORB static lock. Vectorised twin of c7.walk_orb (same per-bar priority: flat, stop, arm)."""
    n = len(o)
    if e < 0 or e + 1 >= n:
        return np.nan, '', -1
    E = float(o[e]); R = E * (r_pct / 100.0)
    if not (R > 0) or not (E > 0):
        return np.nan, '', -1
    s0 = e + 1
    stop0 = E - R
    k_flat = _first(m[s0:] >= flat_m, s0)
    lim = k_flat if k_flat >= 0 else n
    k_s0 = _first(l[s0:lim] <= stop0, s0)
    k_arm = _first(h[s0:lim] >= E + lock_arm_r * R, s0)
    if k_s0 >= 0 and (k_arm < 0 or k_s0 <= k_arm):
        # the stop fires at or before the arming bar — c7 checks the stop BEFORE arming
        px = min(stop0, float(o[k_s0])) * (1.0 - SLIP)
        return (px - E) / R, 'stop', int(m[k_s0])
    if k_arm < 0:
        if k_flat >= 0:
            return (float(o[k_flat]) - E) / R, 'flat', int(m[k_flat])
        return (float(c[-1]) - E) / R, 'eod', int(m[-1])
    stop1 = E + lock_stop_r * R
    k_s1 = _first(l[k_arm + 1:lim] <= stop1, k_arm + 1)
    if k_s1 >= 0:
        px = min(stop1, float(o[k_s1])) * (1.0 - SLIP)
        return (px - E) / R, 'lock', int(m[k_s1])
    if k_flat >= 0:
        return (float(o[k_flat]) - E) / R, 'flat', int(m[k_flat])
    return (float(c[-1]) - E) / R, 'eod', int(m[-1])


def v_bf(o, h, l, c, m, e, r_pct, activate_at_r=2.0, trail_r=1.0, partial=False,
         partial_r=2.0, partial_frac=0.5, flat_m=BF_FLAT_M):
    """BF R-trail (+/- the shipped partial). Vectorised twin of c7.walk_bf.

    At bar k the stop in force is a function of `hi_k` = max(E, h[e+1 .. k-1]) ONLY — c7 updates
    `hi` AFTER the stop check — so the whole stop path is a shifted cummax and the exit bar is one
    argmax. The partial fires at the first bar whose high reaches E + partial_r x R, which for the
    shipped parameters is the same bar the trail arms on.
    """
    n = len(o)
    if e < 0 or e + 1 >= n:
        return np.nan, '', -1
    E = float(o[e]); R = E * (r_pct / 100.0)
    if not (R > 0) or not (E > 0):
        return np.nan, '', -1
    s0 = e + 1
    k_flat = _first(m[s0:] >= flat_m, s0)
    lim = k_flat if k_flat >= 0 else n
    if lim <= s0:
        if k_flat >= 0:
            return (float(o[k_flat]) - E) / R, 'flat', int(m[k_flat])
        return (float(c[-1]) - E) / R, 'eod', int(m[-1])
    hh = h[s0:lim]
    hi = np.empty(len(hh))
    hi[0] = E
    if len(hh) > 1:
        hi[1:] = np.maximum(E, np.maximum.accumulate(hh[:-1]))
    armed = (hi - E) / R >= activate_at_r
    stop = np.where(armed, np.maximum(E - R, hi - trail_r * R), E - R)
    if partial:
        k_p = _first(hh >= E + partial_r * R, s0)          # absolute index of the partial bar
        if k_p >= 0:
            j = k_p - s0
            after = np.arange(len(hh)) > j                 # the floor is live from the NEXT bar
            stop = np.where(after, np.maximum(stop, E), stop)
    k_s = _first(l[s0:lim] <= stop, s0)
    prr, frac, ptag = 0.0, 1.0, ''
    if partial:
        k_p = _first(hh >= E + partial_r * R, s0)
        if k_p >= 0 and (k_s < 0 or k_p < k_s):
            prr = partial_frac * (float(c[k_p]) - E) / R
            frac = 1.0 - partial_frac
            ptag = 'pp+'
    if k_s >= 0:
        px = min(float(stop[k_s - s0]), float(o[k_s])) * (1.0 - SLIP)
        tag = 'trail_stop' if (hi[k_s - s0] - E) / R >= activate_at_r else 'stop'
        return prr + frac * (px - E) / R, ptag + tag, int(m[k_s])
    if k_flat >= 0:
        return prr + frac * (float(o[k_flat]) - E) / R, ptag + 'flat', int(m[k_flat])
    return prr + frac * (float(c[-1]) - E) / R, ptag + 'eod', int(m[-1])


# ----------------------------------------------------------------------------------- the gates
def gates(nsamp=250):
    """G-X1: the vectorised X1 must reproduce frames11/w34.csv exactly (which itself reproduced
    book6.rr). G-X2/3/4: the vectorised ORB and BF twins must reproduce c7's python loops."""
    w = pd.read_csv(f'{D11}/w34.csv', dtype={'day': str, 'ctrl': str}).dropna(subset=['rr_2'])
    g = w.sample(min(nsamp, len(w)), random_state=43)
    worst = {k: 0.0 for k in ('x1', 'x2', 'x3', 'x4')}
    nchk = 0
    for day, gg in g.groupby('day'):
        bars = load_bars(day, sorted(gg.ctrl.unique()))
        for r in gg.itertuples():
            a = bars.get(r.ctrl)
            if a is None:
                continue
            arr = bars_arrays(a)
            if arr is None:
                continue
            o, h, l, c, v, mm = arr
            idx = np.flatnonzero(mm == int(r.ctrl_entry_m))
            if not len(idx):
                continue
            e = int(idx[0])
            if e + 1 >= len(o):
                continue
            E = float(o[e])
            for s in STOPS:
                rr1, _, _ = v_x1(o, h, l, c, mm, e, E * (1.0 - s))
                ref = float(getattr(r, f'rr_{int(s * 100)}'))
                if rr1 == rr1 and ref == ref:
                    worst['x1'] = max(worst['x1'], abs(rr1 - ref))
                rp = s * 100.0
                for tag, vf, pf in (('x2', v_x2(o, h, l, c, mm, e, rp), walk_orb(o, h, l, c, mm, e, rp)),
                                    ('x3', v_bf(o, h, l, c, mm, e, rp), walk_bf(o, h, l, c, mm, e, rp)),
                                    ('x4', v_bf(o, h, l, c, mm, e, rp, partial=True),
                                     walk_bf(o, h, l, c, mm, e, rp, partial=True))):
                    if vf[0] == vf[0] and pf[0] == pf[0]:
                        worst[tag] = max(worst[tag], abs(vf[0] - pf[0]))
                        assert vf[1] == pf[1] and vf[2] == pf[2], \
                            f'{tag} tag/minute mismatch {vf} vs {pf} on {r.ctrl} {day} s={s}'
            nchk += 1
    print(f'  GATES on {nchk} control keys x {len(STOPS)} stops:', flush=True)
    for k, v_ in worst.items():
        print(f'    {k}: max |diff| = {v_:.3e}', flush=True)
    assert worst['x1'] < 1e-12, 'vectorised X1 does not reproduce w34.csv'
    for k in ('x2', 'x3', 'x4'):
        assert worst[k] < 1e-12, f'vectorised {k} does not reproduce c7'
    print('  ALL GATES PASS', flush=True)


def main(argv):
    gates()
    if '--gate' in argv:
        return 0
    d = pd.read_csv(f'{D6}/pd6.csv', dtype={'day': str, 'symbol': str, 'ctrl': str})
    d = d[d.day < TEST_FROM]
    print(f'  arm-d controls: {len(d):,} over {d.day.nunique()} sessions (TEST cut off)', flush=True)
    done = set()
    if os.path.exists(ST):
        done = set(json.load(open(ST))['days'])
    if not os.path.exists(OUT):
        with open(OUT, 'w') as f:
            f.write(','.join(HDR) + '\n')
    days = sorted(set(d.day) - done)
    print(f'  {len(days)} sessions to walk ({len(done)} already done)', flush=True)
    nrow = 0
    for i, day in enumerate(days):
        gg = d[d.day == day]
        bars = load_bars(day, sorted(gg.ctrl.unique()))
        arrs = {}
        for s_, a in bars.items():
            x = bars_arrays(a)
            if x is not None:
                arrs[s_] = x
        rows = []
        for r in gg.itertuples():
            a = arrs.get(r.ctrl)
            if a is None:
                continue
            o, h, l, c, v, mm = a
            idx = np.flatnonzero(mm == int(r.ctrl_entry_m))
            if not len(idx):
                continue
            e = int(idx[0])
            if e + 1 >= len(o) or int(mm[e]) >= EOD_M:
                continue
            E = float(o[e])
            row = [day, r.ctrl, int(r.ctrl_entry_m), E]
            for s in STOPS:
                rp = s * 100.0
                for g in (v_x1(o, h, l, c, mm, e, E * (1.0 - s)),
                          v_x2(o, h, l, c, mm, e, rp),
                          v_bf(o, h, l, c, mm, e, rp),
                          v_bf(o, h, l, c, mm, e, rp, partial=True),
                          v_x5(o, h, l, c, mm, e, E * (1.0 - s))):
                    row += [g[0], g[1], g[2]]
            rows.append(row)
        if rows:
            pd.DataFrame(rows, columns=HDR).to_csv(OUT, mode='a', header=False, index=False)
            nrow += len(rows)
        done.add(day)
        json.dump({'days': sorted(done)}, open(ST, 'w'))
        if i % 20 == 0 or i == len(days) - 1:
            print(f'  {i + 1}/{len(days)} {day} rows so far {nrow:,}', flush=True)
    print(f'  DONE — {nrow:,} rows written', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
