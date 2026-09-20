#!/usr/bin/env python3
"""hod_frames6 — shared loaders + THE PLACEBO SIMULATOR.  ONE definition for F20, F19, F21.

The population object is unchanged from passes 4 and 5 (`common4.load_breaks4`), so the `B2`
reproduction gate holds byte-for-byte.  What is NEW here is `walk_from()` — a second, independently
written exit walk that prices a bracket entered at an ARBITRARY minute.  It is the object the whole
placebo rests on, so it is required to reproduce the booked trades' own `rr` exactly (R3 in
PREREG §0); `assert_parity()` does that and raises.
"""
import os, sys
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT); sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_frames4')
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_frames5')
from common5 import (S, S2, SPLITS, RISK, load_breaks5, sigset5, book_portfolio,  # noqa: E402,F401
                     attach_instrument, Sheet, HDR, SEP)
from common4 import (load_breaks4, book_ranked, book_oracle, clustered_t, halves,  # noqa: E402,F401
                     admit, daily_ctx, BR_COLS, RD)

D6 = f'{ROOT}/research/mature_method/hod_frames6'
OPEN_M, EOD_M, LAST_M = 570, 955, 930
SLIP = 0.001                      # the shipped stop slippage, identical to walk2.vwalk
FIRST_ENTRY_M, LAST_ENTRY_M = 577, 841     # 09:37 .. 14:01 — the book's own entry window


# --------------------------------------------------------------------------- the placebo simulator
def walk_from(o, h, l, c, m, e, stop):
    """Price ONE bracket entered at the open of bar index `e`, stop at `stop`, target at +2R.

    Written from the prose spec of `hod_frames2/walk2.vwalk` (priority EOD -> stop -> target from
    bar e+1; stop fills at min(stop, that bar's open) minus one slip; target fills AT the target;
    EOD fills at that bar's open).  Returns (exit_idx, exit_px, why, rr).
    """
    n = len(o)
    E = float(o[e])
    R = E - stop
    if not (R > 0):
        return -1, np.nan, '', np.nan
    tgt = E + 2.0 * R
    s0 = e + 1
    if s0 >= n:
        return n - 1, float(c[-1]), 'eod', (float(c[-1]) - E) / R
    eod = m[s0:] >= EOD_M
    hs = l[s0:] <= stop
    ht = c[s0:] >= tgt
    any_ = eod | hs | ht
    if not any_.any():
        return n - 1, float(c[-1]), 'eod', (float(c[-1]) - E) / R
    j = int(np.argmax(any_)); k = s0 + j
    if eod[j]:
        px, why = float(o[k]), 'eod'
    elif hs[j]:
        px, why = float(min(stop, o[k]) * (1.0 - SLIP)), 'stop'
    else:
        px, why = float(tgt), 'target'
    return k, px, why, (px - E) / R


def bars_arrays(gg):
    """(o,h,l,c,v,m) numpy arrays for the RTH window walk2 used: 09:30 <= m < 16:00."""
    rth = gg[(gg.m >= OPEN_M) & (gg.m < 960)]
    if len(rth) < 10:
        return None
    o, h, l, c, v = (rth[k].values.astype(float) for k in ('o', 'h', 'l', 'c', 'v'))
    return o, h, l, c, v, rth.m.values.astype(int)


def new_hod_mask(h):
    """True where bar i made a NEW high of day (h[i] >= running max of h[0..i-1]); bar 0 is True."""
    prev = np.maximum.accumulate(h)
    out = np.empty(len(h), dtype=bool)
    out[0] = True
    out[1:] = h[1:] >= prev[:-1]
    return out


# --------------------------------------------------------------------------- the reference book
def base_book(br=None, verbose=True):
    """`B2` — the reference book every cell in six passes is measured against."""
    if br is None:
        br = load_breaks4(verbose=verbose)
    S.build_impute(S2.load_pop())            # the cost model MUST be built before attach_cost
    s = sigset5(admit(br, pd.Series(True, index=br.index)))
    return book_ranked(s, 12, 4), s


def repro_line(b, tag=''):
    for sp in SPLITS:
        w = S.week_stats(b, sp)
        print(f'  R {tag}{sp:5s} n={w["n"]:5d} /wk={w["per_wk"]:5.1f} gross={w["gross"]:+.3f} '
              f'net={w["net"]:+.3f} green={w["green"]:5.1f} total=${w["total"]:+,.0f}', flush=True)


def mde(d, col='net', power=0.8):
    """Two-sided 80 %-power minimum detectable effect on the mean of `col`, day-clustered n."""
    if len(d) < 5:
        return np.nan
    sd = float(d[col].std(ddof=1))
    nd = d.day.nunique()
    per = len(d) / max(nd, 1)
    eff_n = len(d) / max(1.0, 1.0 + (per - 1) * 0.1)      # modest intra-day ICC allowance
    return 2.8 * sd / np.sqrt(eff_n)
