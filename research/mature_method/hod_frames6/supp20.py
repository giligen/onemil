#!/usr/bin/env python3
"""hod_frames6 / F20 supplement — THE CAUSALITY TRACE the headline cannot be reported without.

Arms a and b disagree in SIGN.  Before either is interpreted, three things must be checked, and all
three are rails already standing in CLAUDE.md, not new cells:

  S1  CAUSALITY OF THE CONTROL (rail 2).  Arm a draws minutes from the whole session of a symbol-day
      whose MEMBERSHIP was established by a break that may not have happened yet.  A minute BEFORE
      the break is therefore selected with future information; a minute AFTER it is not.  If the
      break-minus-control difference lives only in the before-half, arm a is §2.3 in a new costume.
  S2  ARM c IS NOT A PLACEBO (rail 2, again).  "15 minutes before the break" is defined BY the break.
      Its number is reported as what it is — an oracle, not a control.
  S3  TAIL DEPENDENCE (rail 5) and TIME-OF-DAY.  The difference ex-top-1 %/5 % of the control's own R,
      and restricted to controls within +/-30 minutes of the break's own clock.
"""
import os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common6 import D6, SPLITS                                              # noqa: E402
from score20 import clustered_paired_t, KEY, RD                             # noqa: E402


def paired(bk, g, lab, split, extra=''):
    b = bk[bk.split == split]
    cm = g.groupby(KEY, sort=True).rr.agg(['mean', 'size']).reset_index()
    pr = b.merge(cm, on=KEY, how='inner')
    if len(pr) < 5:
        print(f'| {lab} | {split} | — | — | — | — | {extra} |', flush=True)
        return
    pr['diff'] = pr.rr - pr['mean']
    print(f'| {lab} | {split} | {len(pr)} | {pr.rr.mean():+.4f} | {pr["mean"].mean():+.4f} | '
          f'{pr["diff"].mean():+.4f} | {clustered_paired_t(pr, "diff"):+.2f} | {extra} |', flush=True)


def main():
    bk = pd.read_csv(f'{D6}/book6.csv', **RD)
    sp_of = dict(zip(zip(bk.day, bk.symbol, bk.entry_m), bk.split))
    a = pd.read_csv(f'{D6}/pa6.csv', **RD)
    a['split'] = [sp_of.get(t, '') for t in zip(a.day, a.symbol, a.entry_m)]
    a = a[a.split.isin(SPLITS)]
    b = pd.read_csv(f'{D6}/pb6.csv', **RD)
    b['split'] = [sp_of.get(t, '') for t in zip(b.day, b.symbol, b.entry_m)]
    b = b[b.split.isin(SPLITS)]

    print('== S1  CAUSALITY OF THE CONTROL — arm a split at the break minute ==', flush=True)
    print('  A control minute BEFORE the break is selected using the knowledge that the day would', flush=True)
    print('  later produce a booked break.  A control minute AFTER it is not: at the break bar the', flush=True)
    print('  engine already knows the day qualified.  Only the AFTER half is a causal comparison.', flush=True)
    print('\n| arm | split | n trades | break grossR | control grossR | paired Δ | day-clust t | note |',
          flush=True)
    print('|---|---|---|---|---|---|---|---|', flush=True)
    for sp in SPLITS:
        paired(bk, a[a.split == sp], 'a ALL minutes', sp, 'as scored (look-ahead in membership)')
    for sp in SPLITS:
        paired(bk, a[(a.split == sp) & (a.ctrl_entry_m < a.entry_m)], 'a BEFORE the break', sp,
               'NOT causal — membership uses the future')
    for sp in SPLITS:
        paired(bk, a[(a.split == sp) & (a.ctrl_entry_m > a.entry_m)], 'a AFTER the break', sp,
               '**CAUSAL** — day membership already established')
    for sp in SPLITS:
        g = a[(a.split == sp) & (a.ctrl_entry_m > a.entry_m) &
              (a.ctrl_entry_m <= a.entry_m + 30)]
        paired(bk, g, 'a AFTER, within 30 min', sp, 'causal + clock-matched')
    print('', flush=True)

    print('== S2  arm c is NOT a placebo ==', flush=True)
    c = pd.read_csv(f'{D6}/pc6.csv', **RD)
    c['split'] = [sp_of.get(t, '') for t in zip(c.day, c.symbol, c.entry_m)]
    c = c[c.split.isin(SPLITS)]
    print('  The mark "entry_m - 15" exists only because a break happened at entry_m.  Its +1.2 R is', flush=True)
    print('  the value of buying 15 minutes before a run you already know is coming — an ORACLE, and', flush=True)
    print('  it is reported as one.  Coverage is also below the PREREG 80 % rail, so it is a', flush=True)
    print('  DIAGNOSTIC and not a cell:', flush=True)
    for sp in SPLITS:
        d = c[c.split == sp]
        nb = int((bk.split == sp).sum())
        print(f'    {sp}: {len(d)}/{nb} = {len(d) / nb * 100:.1f} % coverage, mean gross R '
              f'{d.rr.mean():+.4f}, target-hit {(d.why == "target").mean() * 100:.1f} %', flush=True)
    print('', flush=True)

    print('== S3  TAIL DEPENDENCE and the time-of-day control (arm a AFTER, arm b) ==', flush=True)
    print('| arm | split | paired Δ all | ex-top-1 % ctrl | ex-top-5 % ctrl | median Δ | '
          'break WR | ctrl WR |', flush=True)
    print('|---|---|---|---|---|---|---|---|', flush=True)
    for lab, g in (('a AFTER', a[a.ctrl_entry_m > a.entry_m]), ('b', b)):
        for sp in SPLITS:
            gg = g[g.split == sp]
            bb = bk[bk.split == sp]
            out = []
            for q in (1.0, 0.99, 0.95):
                h = gg[gg.rr <= gg.rr.quantile(q)] if q < 1 else gg
                cm = h.groupby(KEY, sort=True).rr.mean().rename('m').reset_index()
                pr = bb.merge(cm, on=KEY, how='inner')
                out.append(float((pr.rr - pr.m).mean()))
            cm = gg.groupby(KEY, sort=True).rr.mean().rename('m').reset_index()
            pr = bb.merge(cm, on=KEY, how='inner')
            print(f'| {lab} | {sp} | {out[0]:+.4f} | {out[1]:+.4f} | {out[2]:+.4f} | '
                  f'{float((pr.rr - pr.m).median()):+.4f} | {(pr.rr > 0).mean() * 100:.1f} % | '
                  f'{(gg.rr > 0).mean() * 100:.1f} % |', flush=True)
    print('', flush=True)

    print('== S3b  the BREAK\'s own tail removed (CLAUDE.md rail 5, the version that matters) ==',
          flush=True)
    print('| arm | split | paired Δ all | ex-top-1 % of the BREAK | ex-top-5 % of the BREAK | n |',
          flush=True)
    print('|---|---|---|---|---|---|', flush=True)
    for lab, g in (('a AFTER (causal)', a[a.ctrl_entry_m > a.entry_m]), ('b', b)):
        for sp in SPLITS:
            gg = g[g.split == sp]; bb = bk[bk.split == sp]
            cm = gg.groupby(KEY, sort=True).rr.mean().rename('m').reset_index()
            pr = bb.merge(cm, on=KEY, how='inner'); pr['diff'] = pr.rr - pr.m
            o = [float(pr['diff'].mean())]
            rk = pr.rr.rank(method='first', ascending=False)   # RANK-based: `rr` has a point mass
            for q in (0.01, 0.05):                            # at exactly +2R, so quantiles tie
                o.append(float(pr[rk > q * len(pr)]['diff'].mean()))
            print(f'| {lab} | {sp} | {o[0]:+.4f} | {o[1]:+.4f} | {o[2]:+.4f} | {len(pr)} |',
                  flush=True)
    print('', flush=True)

    print('== S4  the decomposition, in one table (gross R per trade, matched design) ==', flush=True)
    print('| object | TRAIN | VAL | what it holds fixed |', flush=True)
    print('|---|---|---|---|', flush=True)
    d = pd.read_csv(f'{D6}/pd6.csv', **RD)
    d['split'] = [sp_of.get(t, '') for t in zip(d.day, d.symbol, d.entry_m)]
    d = d[d.split.isin(SPLITS)]
    rows = [('the BOOKED break', {s: bk[bk.split == s].rr.mean() for s in SPLITS},
             'nothing — the book itself'),
            ('same symbol-day, any non-break minute', {s: a[a.split == s].rr.mean() for s in SPLITS},
             'day + name (look-ahead in membership)'),
            ('same symbol-day, non-break minute AFTER the break',
             {s: a[(a.split == s) & (a.ctrl_entry_m > a.entry_m)].rr.mean() for s in SPLITS},
             'day + name, CAUSAL'),
            ('matched non-signal name, same minute', {s: b[b.split == s].rr.mean() for s in SPLITS},
             'clock + price + ADV + asset class'),
            ('matched non-signal name, random non-break minute',
             {s: d[d.split == s].rr.mean() for s in SPLITS}, 'nothing — the universe bound')]
    for lab, v, note in rows:
        print(f'| {lab} | {v["TRAIN"]:+.4f} | {v["VAL"]:+.4f} | {note} |', flush=True)


if __name__ == '__main__':
    main()
