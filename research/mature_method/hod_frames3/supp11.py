#!/usr/bin/env python3
"""hod_frames3 / F11 — SUPPLEMENTARY DIAGNOSTICS (S1..S6).  None of these is a declared cell and
none carries a decision; they exist to say WHAT the `consol_bars >= 20` admission actually does
once the four declared components have all failed to reproduce it."""
import sys
import numpy as np, pandas as pd

sys.path.insert(0, '/home/ec2-user/onemil/research/mature_method/hod_frames3')
from common3 import (S, S2, clustered_t, halves, sigset, load_breaks, admit)   # noqa: E402

SPLITS = S.SPLITS


def line(nm, b):
    for sp in SPLITS:
        w = S.week_stats(b, sp); d = b[b.split == sp]
        print(f'  {nm:<38s} {sp:5s} n {w["n"]:5d} /wk {w["per_wk"]:5.1f} gross {w["gross"]:+.3f} '
              f'net {w["net"]:+.3f} green {w["green"]:5.1f}% $ {w["total"]:+9,.0f} '
              f'tc {clustered_t(d):+.2f}')
    g, n, ok = halves(b)
    print(f'      halves H1 {g[0]:+.3f} | H2 {g[1]:+.3f} | VAL {g[2]:+.3f} -> same-signed+: {ok}')


def main():
    pop = S2.load_pop(); S.build_impute(pop)
    br = load_breaks(verbose=False)
    base_rows = admit(br, pd.Series(True, index=br.index))
    base = sigset(base_rows)
    ge20_rows = admit(br, br.consol_bars >= 20)
    ge20 = sigset(ge20_rows)

    print('== S1 — the SPY 09:35 gate ALONE on the base (the control C1 never had) ==')
    line('S1 base (= B2)', S.apply_book(base, 12, 4))
    line('S1 base x spy_r5>0', S.apply_book(base[base.spy_r5_pct > 0], 12, 4))
    line('S1 ge20 (R3)', S.apply_book(ge20, 12, 4))
    line('S1 C1 = ge20 x spy_r5>0', S.apply_book(ge20[ge20.spy_r5_pct > 0], 12, 4))
    clock = sigset(admit(br, br.break_m >= 590))
    line('S1 clock590 x spy_r5>0', S.apply_book(clock[clock.spy_r5_pct > 0], 12, 4))

    print('\n== S2 — what the ge20 ADMISSION changes: same break as the base, or a later one? ==')
    b0 = base_rows[['day', 'symbol', 'break_m']].rename(columns={'break_m': 'base_m'})
    g = ge20_rows.merge(b0, on=['day', 'symbol'], how='left')
    g['changed'] = g.break_m != g.base_m
    gs = sigset(g)
    for sp in SPLITS:
        d = gs[gs.split == sp]
        ch, un = d[d.changed], d[~d.changed]
        print(f'  {sp}: pre-book rows {len(d)} | SAME break as base {len(un)} ({len(un)/len(d):.0%}) '
              f'gross {un.rr.mean():+.3f} | LATER break {len(ch)} gross {ch.rr.mean():+.3f} | '
              f'median minutes later {float((ch.break_m - ch.base_m).median()):.0f}')
    bk = S.apply_book(gs, 12, 4)
    for sp in SPLITS:
        d = bk[bk.split == sp]
        ch, un = d[d.changed], d[~d.changed]
        print(f'  {sp} BOOKED: same {len(un)} gross {un.rr.mean():+.3f} $ {un.pnl.sum():+,.0f} | '
              f'later {len(ch)} gross {ch.rr.mean():+.3f} $ {ch.pnl.sum():+,.0f}')

    print('\n== S3 — C1 weekly concentration and the tail ==')
    c1 = S.apply_book(ge20[ge20.spy_r5_pct > 0], 12, 4)
    for sp in SPLITS:
        d = c1[c1.split == sp]
        w = d.groupby('wk').pnl.sum().sort_values()
        tot = float(w.sum())
        print(f'  {sp}: total ${tot:+,.0f} | best week ${w.iloc[-1]:+,.0f} '
              f'({w.iloc[-1]/tot*100 if tot else float("nan"):.0f}% of it) | '
              f'net {d.net.mean():+.3f} -> ex-top-5% {d.net[d.net <= d.net.quantile(0.95)].mean():+.3f}')

    print('\n== S4 — `consol_bars` as a CONTINUOUS ranker on the base first break (gross R) ==')
    for sp in SPLITS:
        d = base[base.split == sp]
        q = pd.qcut(d.consol_bars, 5, labels=False, duplicates='drop')
        print(f'  {sp}: ' + ' | '.join(
            f'Q{i+1} n {int((q==i).sum())} gross {d.rr[q==i].mean():+.3f}'
            for i in sorted(pd.unique(q.dropna()))))

    print('\n== S5 — the ge20 admission split by whether SPY was up (the day gate) ==')
    for sp in SPLITS:
        d = ge20[ge20.split == sp]
        up, dn = d[d.spy_r5_pct > 0], d[d.spy_r5_pct <= 0]
        print(f'  {sp}: SPY up n {len(up)} gross {up.rr.mean():+.3f} | SPY down n {len(dn)} '
              f'gross {dn.rr.mean():+.3f} | delta {up.rr.mean()-dn.rr.mean():+.3f}')
        d2 = base[base.split == sp]
        up2, dn2 = d2[d2.spy_r5_pct > 0], d2[d2.spy_r5_pct <= 0]
        print(f'        base: SPY up n {len(up2)} gross {up2.rr.mean():+.3f} | down n {len(dn2)} '
              f'gross {dn2.rr.mean():+.3f} | delta {up2.rr.mean()-dn2.rr.mean():+.3f}')

    print('\n== S6 — the four F11 components as CONTINUOUS rankers (base first break, gross R) ==')
    for f in ('hl_n20', 'lo_slope20', 'atr_ratio', 'break_m', 'price'):
        for sp in SPLITS:
            d = base[base[f].notna() & (base.split == sp)]
            if len(d) < 100:
                continue
            q = pd.qcut(d[f], 5, labels=False, duplicates='drop')
            print(f'  {f:<11s} {sp:5s} ' + ' | '.join(
                f'Q{i+1} {d.rr[q==i].mean():+.3f}' for i in sorted(pd.unique(q.dropna()))))
    print('\nDONE', flush=True)


if __name__ == '__main__':
    main()
