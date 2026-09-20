#!/usr/bin/env python3
"""frames9 / F28 stage 2 — SCORE the pond cells.

Per (pond, rung): the pond's own universe bound (arm u — a random pond name at the HOD clock under
the HOD bracket), the matched non-signal control (arm b), the detector's picks, the SELECTION
MARGIN (paired, day-clustered), and the absolute book under G3 (the bare stop) and X0 (the shipped
+2 R cap) with gross, the re-measured booked cost, net, green weeks against a count-matched null,
weekly $ at $100 risk, trades/wk, both TRAIN halves, VAL, ex-top-5 % and the 80 %-power MDE.

The walker parity gate (PREREG §0 d) runs first and raises.  TEST is never loaded.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c9                                                              # noqa: E402
import g8                                                              # noqa: E402
from common4 import book_ranked, clustered_t                           # noqa: E402
from common5 import S                                                  # noqa: E402

D8, D9 = c9.D8, c9.D9
RISK = 100.0
KEYS = ['day', 'symbol', 'entry_m']


def net_of(rr, why, sp_pct, r_pct):
    """PREREG cost: entry half-spread always, exit half-spread x the leg-weighted ratio."""
    half = 0.5 * np.asarray(sp_pct, float) / np.clip(np.asarray(r_pct, float), 0.05, None)
    ratio = np.array([g8.exit_ratio(str(w)) for w in why], dtype=float)
    return np.asarray(rr, float) - half - half * ratio


def rank_trim(v, frac=0.05):
    v = np.asarray(v, dtype=float)
    if len(v) < 20:
        return np.nan
    k = int(np.floor(len(v) * (1 - frac)))
    o = np.sort(v)[:k]
    return float(o.mean()) if len(o) else np.nan


def mkbook(s, g):
    b = s.copy()
    b['rr'] = b[f'rr_{g}'].astype(float)
    b['why'] = b[f'why_{g}'].astype(str)
    b['exit_m'] = b[f'xm_{g}'].astype(float)
    b = b[b.rr.notna() & (b.exit_m > 0)]
    b['exit_m'] = b.exit_m.astype(int)
    b['net'] = net_of(b.rr.values, b.why.values, b.sp_pct.values, b.r_pct.values)
    b['netb'] = b['net']
    return b


def report(name, b, rows, note=''):
    for sp in c9.SPLITS:
        d = b[b.split == sp]
        if len(d) < 3:
            continue
        w = S.week_stats(d, sp)
        nb = S.null_band(d, sp)
        ct = clustered_t(d)
        h1 = d[(d.half == 'H1')]
        h2 = d[(d.half == 'H2')]
        r = dict(cell=name, split=sp, n=w['n'], per_wk=w['per_wk'], gross=w['gross'],
                 cost=float((d.rr - d.net).mean()), net=w['net'], clust_t=ct, green=w['green'],
                 null_p95=nb[3], redstreak=w['redstreak'], worst=w['worst'], total=w['total'],
                 wk_dollar=w['wk_mean'], mdd=w['mdd'], ex5=rank_trim(d.net.values),
                 h1=float(h1.net.mean()) if len(h1) else np.nan,
                 h2=float(h2.net.mean()) if len(h2) else np.nan,
                 imputed=float(d.imputed.mean() * 100),
                 mde=2.80 * float(d.net.std(ddof=1) / np.sqrt(len(d))) if len(d) > 2 else np.nan,
                 note=note)
        rows.append(r)
        print(f'{name:22s} {sp:5s} n={w["n"]:5d} /wk={w["per_wk"]:5.1f} gross={w["gross"]:+.4f} '
              f'cost={r["cost"]:.4f} net={w["net"]:+.4f} t={ct:+5.2f} green={w["green"]:5.1f} '
              f'(null p95 {r["null_p95"]:5.1f}) wk$={w["wk_mean"]:+8.1f} tot=${w["total"]:+8.0f} '
              f'worst={w["worst"]:+7.0f} ex5={r["ex5"]:+.4f} H1={r["h1"]:+.4f} H2={r["h2"]:+.4f} '
              f'imp={r["imputed"]:.0f}% mde={r["mde"]:.3f}', flush=True)


def main():
    s = pd.read_csv(f'{D9}/sig9.csv', dtype={'day': str, 'symbol': str},
                    keep_default_na=False, na_values=[''])
    W = pd.read_csv(f'{D9}/w9_sig.csv', dtype={'day': str, 'symbol': str},
                    keep_default_na=False, na_values=[''])
    B = pd.read_csv(f'{D9}/w9_b.csv', dtype={'day': str, 'symbol': str, 'ctrl': str, 'pond': str},
                    keep_default_na=False, na_values=[''])
    U = pd.read_csv(f'{D9}/w9_u.csv', dtype={'day': str, 'symbol': str, 'ctrl': str, 'pond': str},
                    keep_default_na=False, na_values=[''])
    s = s[s.in_UNION].merge(W, on=KEYS, how='inner')
    print(f'walked pond signals {len(s)} | arm b {len(B)} | arm u {len(U)}', flush=True)

    # ---- PREREG §0 d — the walker parity gate against frames8 -------------------------------
    W8 = pd.read_csv(f'{D8}/w_sig.csv', dtype={'day': str, 'symbol': str},
                     keep_default_na=False, na_values=[''],
                     usecols=KEYS + ['rr_X0', 'rr_G3'])
    j = s.merge(W8, on=KEYS, how='inner', suffixes=('', '_8'))
    dmax = max(float(np.nanmax(np.abs(j.rr_X0 - j.rr_X0_8))),
               float(np.nanmax(np.abs(j.rr_G3 - j.rr_G3_8))))
    assert dmax <= 1e-9, f'PARITY FAIL vs frames8/w_sig.csv: max |delta| = {dmax:.3e}'
    print(f'  G-PARITY: {len(j)} signals shared with frames8/w_sig.csv, max |delta rr| = '
          f'{dmax:.2e} — MATCH', flush=True)

    # ---- C0 — the overlap diagnostic, reported FIRST ----------------------------------------
    o = pd.read_csv(c9.ORB_BOOK, dtype={'symbol': str, 'date': str},
                    keep_default_na=False, na_values=[''])
    obk = set(zip(o.date, o.symbol))
    bf = pd.read_csv(c9.BF_BOOK, dtype={'symbol': str, 'date': str},
                     keep_default_na=False, na_values=[''])
    bbk = set(zip(bf.date, bf.symbol))
    print('\n=== C0 — THE OVERLAP DIAGNOSTIC (does the HOD rule re-label the pond book\'s own '
          'picks?) ===', flush=True)
    for rung in c9.RUNGS:
        x = s[s.next_open >= rung]
        for pond, bk in (('ORB', obk), ('BF', bbk)):
            xx = x[x[f'in_{pond}']]
            ov = sum(1 for k in zip(xx.day, xx.symbol) if k in bk)
            print(f'  rung ${rung:>4.0f} {pond:3s}: {len(xx):5d} pond signals, {ov:3d} on a '
                  f'(day, symbol) the {pond} book itself takes = {100*ov/max(len(xx),1):.1f} % '
                  f'(the book has {len(bk)} picks in TRAIN+VAL+TEST)', flush=True)

    rows, mrows = [], []
    print('\n=== C1-C9 — THE POND CELLS ===', flush=True)
    for pond in ('ORB', 'BF', 'UNION'):
        for rung in c9.RUNGS:
            x = s[(s.next_open >= rung) & s[f'in_{pond}']]
            if len(x) < 10:
                continue
            # arm membership: UNION uses each signal's own pond pool
            kx = set(zip(x.day, x.symbol, x.entry_m))
            bsel = B[[k in kx for k in zip(B.day, B.symbol, B.entry_m)]]
            usel = U[[k in kx for k in zip(U.day, U.symbol, U.entry_m)]]
            if pond in ('ORB', 'BF'):
                bsel, usel = bsel[bsel.pond == pond], usel[usel.pond == pond]
            else:
                bsel = bsel.drop_duplicates(KEYS + ['ctrl'])
                usel = usel.drop_duplicates(KEYS + ['ctrl'])
            print(f'\n-- pond {pond}  rung ${rung:.0f}  signals {len(x)} '
                  f'(TRAIN {int((x.split=="TRAIN").sum())} / VAL {int((x.split=="VAL").sum())}) '
                  f'| imputed {x.imputed.mean()*100:.0f} % | arm b {len(bsel)} arm u {len(usel)} --',
                  flush=True)
            for g in c9.GE:
                cb = bsel.groupby(KEYS)[f'rr_{g}'].mean().rename('cb')
                cu = usel.groupby(KEYS)[f'rr_{g}'].mean().rename('cu')
                for sp in c9.SPLITS:
                    xs = x[(x.split == sp) & x[f'rr_{g}'].notna()]
                    if len(xs) < 5:
                        continue
                    pb = xs.merge(cb, on=KEYS, how='inner')
                    pu = xs.merge(cu, on=KEYS, how='inner')
                    cov_b = len(pb) / len(xs)
                    cov_u = len(pu) / len(xs)
                    db = (pb[f'rr_{g}'] - pb.cb)
                    du = (pu[f'rr_{g}'] - pu.cu)
                    h1 = pb[pb.half == 'H1']
                    h2 = pb[pb.half == 'H2']
                    m = dict(pond=pond, rung=rung, geom=g, split=sp, n=len(xs),
                             sig=float(xs[f'rr_{g}'].mean()),
                             arm_b=float(bsel[bsel.day.isin(set(xs.day))][f'rr_{g}'].mean()),
                             bound_u=float(usel[usel.day.isin(set(xs.day))][f'rr_{g}'].mean()),
                             margin=float(db.mean()), t_margin=clustered_t(pb.assign(net=db)),
                             margin_h1=float((h1[f'rr_{g}'] - h1.cb).mean()) if len(h1) else np.nan,
                             margin_h2=float((h2[f'rr_{g}'] - h2.cb).mean()) if len(h2) else np.nan,
                             vs_u=float(du.mean()), t_u=clustered_t(pu.assign(net=du)),
                             cov_b=cov_b, cov_u=cov_u,
                             mde=2.80 * float(db.std(ddof=1) / np.sqrt(len(db))))
                    mrows.append(m)
                    flag = '' if min(cov_b, cov_u) >= 0.80 else '  [AVAILABILITY < 80 % -> DIAGNOSTIC]'
                    print(f'   {g} {sp:5s} n={len(xs):4d} sig={m["sig"]:+.4f}  '
                          f'b={m["arm_b"]:+.4f}  POND BOUND u={m["bound_u"]:+.4f}  | '
                          f'MARGIN sig-b={m["margin"]:+.4f} t={m["t_margin"]:+5.2f} '
                          f'(H1 {m["margin_h1"]:+.4f} H2 {m["margin_h2"]:+.4f}) | '
                          f'sig-u={m["vs_u"]:+.4f} t={m["t_u"]:+5.2f} | cov b/u '
                          f'{100*cov_b:.0f}/{100*cov_u:.0f} % mde={m["mde"]:.3f}{flag}', flush=True)
                b = mkbook(x, g)
                if len(b) >= 10:
                    bk = book_ranked(b, 12, 4)
                    bk['pnl'] = bk.net * RISK
                    report(f'{pond}/${rung:.0f}/{g}', bk, rows)

    # ---- C10 — HOD's own baseline at the same rungs (no pond restriction) -------------------
    print('\n=== C10 — HOD\'s OWN baseline at the same rungs (no pond restriction) ===', flush=True)
    W8f = pd.read_csv(f'{D8}/w_sig.csv', dtype={'day': str, 'symbol': str},
                      keep_default_na=False, na_values=[''])
    sall = pd.read_csv(f'{D9}/sig9.csv', dtype={'day': str, 'symbol': str},
                       keep_default_na=False, na_values=[''])
    base = sall.merge(W8f, on=KEYS, how='inner')
    for g in c9.GE:
        b = mkbook(base[base.next_open >= 20.0], g)
        bk = book_ranked(b, 12, 4)
        bk['pnl'] = bk.net * RISK
        report(f'HOD/$20/{g}', bk, rows, note='baseline')

    pd.DataFrame(rows).to_csv(f'{D9}/cells28_books.csv', index=False)
    pd.DataFrame(mrows).to_csv(f'{D9}/cells28_margin.csv', index=False)
    print('\nS9 DONE', flush=True)


if __name__ == '__main__':
    main()
