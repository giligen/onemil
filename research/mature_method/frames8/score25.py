#!/usr/bin/env python3
"""frames8 / F25 stage 2 — SCORE every declared geometry cell, then the decomposition.

Reads the stage-1 walk (`w_sig.csv`, `w_a.csv`, `w_b.csv`, `w_d.csv`) and prints, per PREREG §4:
gross, the RE-MEASURED booked cost for that cell's own exit mix, net, the exit mix, green weeks,
red streak, worst week, weekly $ at $100 risk, trades/wk, both TRAIN halves, VAL, day-clustered t,
the count-matched permutation null on green weeks, ex-top-5 % and the 80 %-power MDE.

Nothing is written outside `frames8/`.  TEST is filtered out everywhere.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
for p in ('frames8', 'frames7', 'hod_frames6', 'hod_frames5', 'hod_frames4'):
    sys.path.insert(0, f'{ROOT}/research/mature_method/{p}')
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_filter_stack')
import g8                                                              # noqa: E402
from common6 import base_book  # noqa: E402,F401
from common5 import S                                                  # noqa: E402
from common4 import book_ranked, clustered_t                           # noqa: E402

D8 = f'{ROOT}/research/mature_method/frames8'
RISK = 100.0
GE = list(g8.GEOMS) + ['G1a', 'G1b', 'G2v', 'G2pl']
SPL = ('TRAIN', 'VAL')


def net_of(rr, why, sp_pct, r_pct):
    """PREREG §4 cost: entry half-spread always, exit half-spread x the leg-weighted ratio."""
    half = 0.5 * sp_pct / np.clip(r_pct, 0.05, None)
    ratio = np.array([g8.exit_ratio(str(w)) for w in why], dtype=float)
    return rr - half - half * ratio


def mkbook(sig, g):
    b = sig.copy()
    b['rr'] = b[f'rr_{g}'].astype(float)
    b['why'] = b[f'why_{g}'].astype(str)
    b['exit_m'] = b[f'xm_{g}'].astype(float)
    b = b[b.rr.notna()]
    b['net'] = net_of(b.rr.values, b.why.values, b.sp_pct.values, b.r_pct.values)
    b['netb'] = b['net']
    b['pnl'] = b.net * RISK
    return b


def exmix(b):
    w = b.why.astype(str)
    fam = np.where(w.str.contains('target'), 'target',
          np.where(w.str.contains('lock'), 'lock',
          np.where(w.str.contains('trail_stop'), 'trail',
          np.where(w.str.contains('stop'), 'stop', 'eod'))))
    v = pd.Series(fam).value_counts(normalize=True) * 100
    return {k: float(v.get(k, 0.0)) for k in ('target', 'lock', 'trail', 'stop', 'eod')}


def rank_trim(v, frac=0.05):
    """Rank-based trim of the top `frac` (quantile trimming ties on the target point mass)."""
    v = np.asarray(v, dtype=float)
    if len(v) < 20:
        return np.nan
    k = int(np.floor(len(v) * (1 - frac)))
    o = np.sort(v)[:k]
    return float(o.mean()) if len(o) else np.nan


def report(name, b, rows, note=''):
    for sp in SPL:
        d = b[b.split == sp]
        if not len(d):
            continue
        w = S.week_stats(d, sp)
        mx = exmix(d)
        nb = S.null_band(d, sp)
        ct = clustered_t(d)
        halves = {}
        for hn, dd in (('H1', d[d.day < '2025-07-01']), ('H2', d[d.day >= '2025-07-01'])):
            halves[hn] = float(dd.net.mean()) if len(dd) else np.nan
        r = dict(cell=name, split=sp, n=w['n'], per_wk=w['per_wk'], gross=w['gross'],
                 cost=float((d.rr - d.net).mean()), net=w['net'], clust_t=ct,
                 green=w['green'], null_p95=nb[3] if isinstance(nb, tuple) else np.nan,
                 redstreak=w['redstreak'], worst=w['worst'], total=w['total'],
                 wk_dollar=w['wk_mean'], mdd=w['mdd'],
                 ex5=rank_trim(d.net.values), h1=halves['H1'], h2=halves['H2'],
                 mde=2.80 * float(d.net.std(ddof=1) / np.sqrt(len(d))) if len(d) > 2 else np.nan,
                 **{f'x_{k}': v for k, v in mx.items()}, note=note)
        rows.append(r)
        print(f'{name:9s} {sp:5s} n={w["n"]:5d} /wk={w["per_wk"]:5.1f} gross={w["gross"]:+.4f} '
              f'cost={r["cost"]:.4f} net={w["net"]:+.4f} t={ct:+5.2f} green={w["green"]:5.1f} '
              f'(null p95 {r["null_p95"]:5.1f}) wk$={w["wk_mean"]:+8.1f} tot=${w["total"]:+9.0f} '
              f'worst={w["worst"]:+8.0f} rs={w["redstreak"]:2d} ex5={r["ex5"]:+.4f} '
              f'H1={halves["H1"]:+.4f} H2={halves["H2"]:+.4f} mde={r["mde"]:.3f} | '
              f'tgt{mx["target"]:4.0f} lock{mx["lock"]:4.0f} trl{mx["trail"]:4.0f} '
              f'stp{mx["stop"]:4.0f} eod{mx["eod"]:4.0f}', flush=True)


def main():
    W = pd.read_csv(f'{D8}/w_sig.csv', dtype={'day': str, 'symbol': str},
                    keep_default_na=False, na_values=[''])
    from f25_walk import load_sig
    sig, bk = load_sig()
    sig = sig.merge(W, on=['day', 'symbol', 'entry_m'], how='inner', suffixes=('', '_w'))
    sig = sig[sig.split.isin(SPL)]
    print(f'walked signals in TRAIN+VAL: {len(sig)}  booked {int(sig.booked.sum())}', flush=True)
    bkd = sig[sig.booked]

    rows = []
    print('\n=== F25 §1.3  THE BOOKED BOOK UNDER EACH GEOMETRY  (B2 set held fixed) ===', flush=True)
    for g in GE:
        if f'rr_{g}' not in sig.columns:
            continue
        b = mkbook(bkd, g)
        if not len(b):
            continue
        report(g, b, rows)

    print('\n=== F25 RB  the RE-BOOKED honesty check (run_book on each geometry\'s own exit_m) ===',
          flush=True)
    for g in ('X0', 'G1', 'G2', 'G3'):
        s = mkbook(sig, g)
        s = s[s.exit_m > 0]
        s['exit_m'] = s.exit_m.astype(int)
        rb = book_ranked(s, 12, 4)
        rb['pnl'] = rb.net * RISK
        report('RB_' + g, rb, rows, note='re-booked')

    pd.DataFrame(rows).to_csv(f'{D8}/cells25.csv', index=False)

    # ------------------------------------------------------------------ the decomposition
    print('\n=== F25 §1.4  THE DECOMPOSITION UNDER EACH GEOMETRY (gross R) ===', flush=True)
    keys = ['day', 'symbol', 'entry_m']
    A = pd.read_csv(f'{D8}/w_a.csv', dtype={'day': str, 'symbol': str},
                    keep_default_na=False, na_values=[''])
    B = pd.read_csv(f'{D8}/w_b.csv', dtype={'day': str, 'symbol': str, 'ctrl': str},
                    keep_default_na=False, na_values=[''])
    Dd = pd.read_csv(f'{D8}/w_d.csv', dtype={'day': str, 'symbol': str, 'ctrl': str},
                     keep_default_na=False, na_values=[''])
    bset = set(zip(bkd.day, bkd.symbol, bkd.entry_m))
    sp_of = dict(zip(zip(bkd.day, bkd.symbol, bkd.entry_m), bkd.split))

    def tag(df):
        k = list(zip(df.day, df.symbol, df.entry_m))
        df = df[[x in bset for x in k]].copy()
        df['split'] = [sp_of[x] for x in zip(df.day, df.symbol, df.entry_m)]
        df['half'] = np.where(df.day < '2025-07-01', 'H1', 'H2')
        return df

    A, B, Dd = tag(A), tag(B), tag(Dd)
    print(f'coverage: a\' {A.groupby("split").size().to_dict()}  '
          f'b {B.groupby("split").size().to_dict()}  d {Dd.groupby("split").size().to_dict()}',
          flush=True)
    drows = []
    for g in g8.GEOMS:
        sg = bkd.copy()
        sg['rr'] = sg[f'rr_{g}']
        for sp in SPL:
            s_ = sg[(sg.split == sp) & sg.rr.notna()]
            a_ = A[A.split == sp]
            b_ = B[B.split == sp]
            d_ = Dd[Dd.split == sp]
            # paired: booked minus the mean of its own controls, day-clustered
            pa = s_.merge(a_.groupby(keys)[f'rr_{g}'].mean().rename('ca'), on=keys, how='inner')
            pb = s_.merge(b_.groupby(keys)[f'rr_{g}'].mean().rename('cb'), on=keys, how='inner')
            pdd = s_.merge(d_.groupby(keys)[f'rr_{g}'].mean().rename('cd'), on=keys, how='inner')
            r = dict(geom=g, split=sp, sig=float(s_.rr.mean()),
                     a=float(a_[f'rr_{g}'].mean()), b=float(b_[f'rr_{g}'].mean()),
                     d=float(d_[f'rr_{g}'].mean()),
                     d_h1=float(d_[d_.half == 'H1'][f'rr_{g}'].mean()),
                     d_h2=float(d_[d_.half == 'H2'][f'rr_{g}'].mean()),
                     da=float((pa.rr - pa.ca).mean()),
                     ta=clustered_t(pa.assign(net=pa.rr - pa.ca)),
                     db=float((pb.rr - pb.cb).mean()),
                     tb=clustered_t(pb.assign(net=pb.rr - pb.cb)),
                     dd=float((pdd.rr - pdd.cd).mean()),
                     td=clustered_t(pdd.assign(net=pdd.rr - pdd.cd)),
                     mde_a=2.80 * float((pa.rr - pa.ca).std(ddof=1) / np.sqrt(len(pa))))
            drows.append(r)
            print(f'{g:5s} {sp:5s} sig={r["sig"]:+.4f}  a\'={r["a"]:+.4f}  b={r["b"]:+.4f}  '
                  f'd={r["d"]:+.4f} (H1 {r["d_h1"]:+.4f} H2 {r["d_h2"]:+.4f})  | '
                  f'sig-a\'={r["da"]:+.4f} t={r["ta"]:+5.2f}  sig-b={r["db"]:+.4f} t={r["tb"]:+5.2f}'
                  f'  sig-d={r["dd"]:+.4f} t={r["td"]:+5.2f}  mde_a={r["mde_a"]:.3f}', flush=True)
    pd.DataFrame(drows).to_csv(f'{D8}/decomp25.csv', index=False)
    print('SCORE25 DONE', flush=True)


if __name__ == '__main__':
    main()
