#!/usr/bin/env python3
"""frames10 / F32 — THE WRAPPER OBJECT, BOTH SIDES.  The scorer.

(L) the LONG: HOD's shipped detector restricted to WRAPPERS, bare-stop exit, the shipped slot rule,
    with the pre-committed ex-top-5 % kill BINDING.
(S) the SHORT: the non-moving wrapper cohort pass 9 isolated at -0.25 R, walked by `w32.py`, scored
    at three clocks and two range thresholds, with the two falsifiable mechanism tests.

Every cell, threshold, bar and kill is declared in `PREREG.md` §2 and is not re-decided here.
TEST is never loaded.

  python3 s32.py            -> cells32_long.csv, cells32_short.csv, the printed tables
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

import g8                                                              # noqa: E402
from common4 import load_breaks4, admit, book_ranked, clustered_t      # noqa: E402
from common5 import sigset5, S, S2                                     # noqa: E402

D10 = f'{ROOT}/research/mature_method/frames10'
D8 = f'{ROOT}/research/mature_method/frames8'
SPLITS = ('TRAIN', 'VAL')
RISK = 100.0
SHORT_MULT = 1.8            # F2 §3 / F23 — the short side pays 1.8x per R
BORROW = f'{ROOT}/research/fuckup_audit/O_halt/PASSIVE/borrow_flags.csv'
LROWS, SROWS = [], []


def rank_trim(v, frac=0.05):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 20:
        return np.nan
    k = int(np.floor(len(v) * (1 - frac)))
    o = np.sort(v)[:k]
    return float(o.mean()) if len(o) else np.nan


def imputed_spread(price, entry_m):
    """The programme's price-band x hour-band spread imputation, for rows with no measured NBBO."""
    pb = pd.cut(pd.Series(price), S.PB_EDGES, labels=S.PB_LAB)
    hb = pd.cut(pd.Series(entry_m), S.HB_EDGES, labels=S.HB_LAB)
    return np.array([S.IMPUTE.get((p, h), S.IMPUTE_GLOBAL) for p, h in zip(pb, hb)], dtype=float)


def report(rows, name, b, note=''):
    """The standard cell row: gross, booked cost, net, green vs its own count-matched null,
    weekly $ at $100/R, both TRAIN halves, VAL, ex-top-5 %, day-clustered t, MDE."""
    for sp in SPLITS:
        d = b[b.split == sp]
        if len(d) < 5:
            continue
        w = S.week_stats(d, sp)
        nb = S.null_band(d, sp)
        ct = clustered_t(d)
        h1, h2 = d[d.half == 'H1'], d[d.half == 'H2']
        r = dict(cell=name, split=sp, n=w['n'], per_wk=w['per_wk'], gross=w['gross'],
                 cost=float((d.rr - d.net).mean()), net=w['net'], clust_t=ct, green=w['green'],
                 null_p95=nb[3], worst=w['worst'], total=w['total'], wk_dollar=w['wk_mean'],
                 mdd=w['mdd'], ex5=rank_trim(d.net.values),
                 h1=float(h1.net.mean()) if len(h1) else np.nan,
                 h2=float(h2.net.mean()) if len(h2) else np.nan,
                 imputed=float(d.imputed.mean() * 100) if 'imputed' in d else np.nan,
                 mde=2.80 * float(d.net.std(ddof=1) / np.sqrt(len(d))) if len(d) > 2 else np.nan,
                 note=note)
        rows.append(r)
        print(f'{name:24s} {sp:5s} n={w["n"]:6d} /wk={w["per_wk"]:6.1f} gross={w["gross"]:+.4f} '
              f'cost={r["cost"]:.4f} net={w["net"]:+.4f} t={ct:+5.2f} green={w["green"]:5.1f} '
              f'(null p95 {r["null_p95"]:5.1f}) wk$={w["wk_mean"]:+9.1f} tot=${w["total"]:+9.0f} '
              f'ex5={r["ex5"]:+.4f} H1={r["h1"]:+.4f} H2={r["h2"]:+.4f} mde={r["mde"]:.3f} {note}',
              flush=True)


def bar_verdict(rows, name, tail_binding):
    """The pre-committed bar (PREREG §2.1 / §2.2), applied cell by cell."""
    d = [r for r in rows if r['cell'] == name]
    if len(d) < 2:
        return 'NO-DATA (a split is missing)'
    fails = []
    for r in d:
        if not (r['wk_dollar'] > 0):
            fails.append(f"{r['split']} weekly $ {r['wk_dollar']:+.0f}")
        if not (r['green'] >= 50):
            fails.append(f"{r['split']} green {r['green']:.1f} %")
        if not (r['per_wk'] >= 10):
            fails.append(f"{r['split']} {r['per_wk']:.1f} tr/wk")
        if not (r['clust_t'] >= 2):
            fails.append(f"{r['split']} clustered t {r['clust_t']:+.2f}")
        if tail_binding and not (r['ex5'] > 0):
            fails.append(f"{r['split']} ex-top-5 % {r['ex5']:+.3f}")
    tr = [r for r in d if r['split'] == 'TRAIN']
    if tr and not (np.sign(tr[0]['h1']) == np.sign(tr[0]['h2'])):
        fails.append(f"TRAIN halves {tr[0]['h1']:+.3f} / {tr[0]['h2']:+.3f} opposite-signed")
    return 'CLEARS THE BAR' if not fails else 'FAILS: ' + '; '.join(fails)


# ================================================================== (L) the LONG wrapper book
def long_book():
    print('\n===== F32 (L) — LONG admission `wrapper`, the shipped cascade =====', flush=True)
    br = load_breaks4(verbose=False)
    S.build_impute(S2.load_pop())
    from common6 import base_book
    b0, _ = base_book(br, verbose=False)
    ref = {'TRAIN': (1622, -17346.0), 'VAL': (706, 893.0)}
    for sp in SPLITS:
        w = S.week_stats(b0, sp)
        assert w['n'] == ref[sp][0] and abs(w['total'] - ref[sp][1]) < 1.0, f'B2 repro FAIL {sp}'
    print('  G-B2 reproduced (1,622 / -$17,346 TRAIN; 706 / +$893 VAL)', flush=True)

    sig = sigset5(admit(br, pd.Series(True, index=br.index)), min_price=20.0)
    sig = sig[sig.split.isin(SPLITS)].copy()
    W = pd.read_csv(f'{D8}/w_sig.csv', dtype={'day': str, 'symbol': str},
                    keep_default_na=False, na_values=[''],
                    usecols=['day', 'symbol', 'entry_m'] +
                            [f'{x}_{g}' for g in ('G3', 'X0') for x in ('rr', 'why', 'xm')])
    keys = ['day', 'symbol', 'entry_m']
    n0 = len(sig)
    sig = sig.merge(W, on=keys, how='inner')
    print(f'  admitted signals {n0} -> walked {len(sig)} ({len(sig) / n0 * 100:.1f} % availability)',
          flush=True)
    import c9
    cls = c9.asset_class(list(sig.symbol.unique()))
    sig['cls'] = sig.symbol.map(cls)
    sig['half'] = np.where(sig.day < '2025-07-01', 'H1', 'H2')
    sh = (sig.cls == 'wrapper').mean() * 100
    print(f'  wrapper share of the admitted set {sh:.1f} % ({int((sig.cls == "wrapper").sum())} '
          f'signals)', flush=True)

    for g in ('G3', 'X0'):
        s = sig.copy()
        s['rr'] = s[f'rr_{g}'].astype(float)
        s['why'] = s[f'why_{g}'].astype(str)
        s['exit_m'] = pd.to_numeric(s[f'xm_{g}'], errors='coerce')
        s = s[s.rr.notna() & (s.exit_m > 0)].copy()
        s['exit_m'] = s.exit_m.astype(int)
        half = 0.5 * s.sp_pct / s.r_pct.clip(lower=0.05)
        ratio = np.array([g8.exit_ratio(w) for w in s.why], dtype=float)
        s['net'] = s.rr - half - half * ratio
        s['netb'] = s['net']
        for lab, d in ((f'L wrapper {g}', s[s.cls == 'wrapper']),
                       (f'L stock {g}', s[s.cls == 'stock'])):
            bk = book_ranked(d, 12, 4)
            report(LROWS, lab, bk, note='')
        print(f'  -> {"L wrapper " + g:22s} {bar_verdict(LROWS, f"L wrapper {g}", True)}',
              flush=True)


# ================================================================== (S) the SHORT wrapper book
def short_book():
    print('\n===== F32 (S) — SHORT the non-moving wrapper =====', flush=True)
    p = pd.read_csv(f'{D10}/p32.csv', dtype={'day': str, 'symbol': str, 'anchor': str,
                                             'why': str}, keep_default_na=False, na_values=[''])
    n_raw = len(p)
    p['split'] = np.where(p.day < '2026-01-01', 'TRAIN',
                          np.where(p.day < '2026-06-01', 'VAL', 'TEST'))
    p = p[p.split.isin(SPLITS)].copy()
    p['half'] = np.where(p.day < '2025-07-01', 'H1', 'H2')
    p['wk'] = pd.to_datetime(p.day).dt.to_period('W-FRI').astype(str)
    p['entry_m'] = p.e_m.astype(int)
    p['exit_m'] = pd.to_numeric(p.xm, errors='coerce').fillna(955).astype(int)

    nofill = float((p.why == 'nofill').mean()) * 100
    badstop = float((p.why == 'badstop').mean()) * 100
    sho = float((p.down_pct <= -10.0).mean()) * 100
    bf = pd.read_csv(BORROW, dtype={'symbol': str})
    ok = set(bf[(bf.shortable.astype(str) == 'True') &
                (bf.easy_to_borrow.astype(str) == 'True')].symbol)
    seen = set(bf.symbol)
    known = float(p.symbol.isin(seen).mean()) * 100
    borrowable = float(p[p.symbol.isin(seen)].symbol.isin(ok).mean()) * 100
    print(f'  rows walked {n_raw:,} | TRAIN+VAL {len(p):,} | no-fill {nofill:.1f} % | '
          f'stop<=0 {badstop:.1f} % | Reg SHO 201 {sho:.2f} % | borrow flags known {known:.1f} % | '
          f'BORROWABLE {borrowable:.1f} %', flush=True)
    print(f'  BORROW, the rail that decides this frame: {len([s for s in p.symbol.unique() if s in ok])}'
          f' of {p.symbol.nunique()} wrapper names are shortable+easy-to-borrow', flush=True)
    for lab, d in (('levered |L|>=2', p[p.lev.abs() >= 2]), ('inverse/1x', p[p.lev.abs() == 1]),
                   ('leverage unparsed', p[p.lev.isna()])):
        if len(d):
            print(f'    {lab:18s} rows {len(d):7d} borrowable {d.symbol.isin(ok).mean() * 100:5.1f} %'
                  f' | names {d.symbol.nunique():4d} borrowable names '
                  f'{len([s for s in d.symbol.unique() if s in ok]):4d}', flush=True)
    for sp in SPLITS:
        d = p[p.split == sp]
        print(f'    {sp:5s} rows {len(d):7d} borrowable {d.symbol.isin(ok).mean() * 100:5.1f} %',
              flush=True)
    print(f'  underlying move available on {float(p.u_move.notna().mean()) * 100:.1f} % of rows '
          f'(the availability rail floor is 80 %)', flush=True)
    print(f'  leverage parsed on {float(p.lev.notna().mean()) * 100:.1f} % of rows', flush=True)

    S.build_impute(S2.load_pop())
    p['sp_pct'] = imputed_spread(p.entry.values, p.entry_m.values)
    p['imputed'] = True
    half = 0.5 * p.sp_pct / p.r_pct.clip(lower=0.05)
    ratio = np.array([g8.exit_ratio(w) for w in p.why.astype(str)], dtype=float)
    p['net'] = p.rr - SHORT_MULT * (half + half * ratio)
    p['netb'] = p['net']

    # the declared screens
    q = p[(p.why != 'nofill') & (p.why != 'badstop') & p.rr.notna() & (p.r_pct >= 1.0) &
          (p.down_pct > -10.0) & p.symbol.isin(ok)].copy()
    print(f'  after fill + r_min + Reg SHO + borrow screens: {len(q):,} rows', flush=True)

    # the DECLARED sensitivity arm (PREREG §2.2 last cell): the same six cells with the borrow
    # screen OFF.  NOT EXECUTABLE — kept so the mechanism tests of §2.3 have power, and labelled
    # that way in every row.
    qd = p[(p.why != 'nofill') & (p.why != 'badstop') & p.rr.notna() & (p.r_pct >= 1.0) &
           (p.down_pct > -10.0)].copy()
    print(f'  DIAGNOSTIC arm (no borrow screen, NOT EXECUTABLE): {len(qd):,} rows', flush=True)

    best = None
    for tag, src, executable in (('S', q, True), ('S*', qd, False)):
        for T in (630, 660, 720):
            for X in (2.0, 3.0):
                d = src[(src.clock == T) & (src.rng_pct <= X) &
                        (src.u_move.abs() <= 1.0)].copy()
                lab = f'{tag} {T // 60:02d}:{T % 60:02d} rng<={X:g}%'
                if len(d) < 20:
                    print(f'{lab:24s} n={len(d)} — under the floor, not scored', flush=True)
                    continue
                d['pnl'] = d.net * RISK
                report(SROWS, lab, d, note='executable' if executable else 'DIAGNOSTIC no-borrow')
                v = bar_verdict(SROWS, lab, False)
                print(f'  -> {lab:22s} {v}', flush=True)
                if not executable:
                    sc = [r for r in SROWS if r['cell'] == lab]
                    score = min(r['net'] for r in sc) if len(sc) == 2 else -9
                    if best is None or score > best[0]:
                        best = (score, T, X, d)
    return p, qd, best


def mechanism(q, best):
    """The two pre-registered falsifiers (PREREG §2.3). Printed whatever they read.

    Measured on the WHOLE diagnostic population at range <= 3 % (all three clocks pooled), not on
    the single best cell: a mechanism test needs power, and the per-cell populations are 100-200
    rows. Stated here rather than chosen afterwards.
    """
    print('\n===== F32 — THE MECHANISM TESTS =====', flush=True)
    base = q[q.rng_pct <= 3.0].copy()
    flat = base[base.u_move.abs() <= 1.0]
    print(f'  population: the DIAGNOSTIC arm (no borrow screen — the executable population is far '
          f'too small to test a mechanism on), range <= 3 %, all three clocks: {len(base):,} rows, '
          f'of which underlying-flat {len(flat):,}', flush=True)

    out = {}
    print('\n  (i) MONOTONICITY IN LEVERAGE — decay (wrapper open->15:55 %) and the short gross R')
    print(f'  {"|L|":>4s} {"n days":>7s} {"decay %":>9s} {"VAL decay":>10s} '
          f'{"n trades":>9s} {"gross R":>9s} {"VAL gross":>10s} {"all-days decay %":>17s}')
    dec, gr, decv, grv = {}, {}, {}, {}
    for L in (1.0, 2.0, 3.0):
        d = flat[flat.lev.abs() == L]
        if len(d) < 20:
            print(f'  {L:4.0f} n={len(d)} — under the floor', flush=True)
            continue
        u = d.drop_duplicates(['day', 'symbol'])
        uv = u[u.split == 'VAL']
        dv = d[d.split == 'VAL']
        alld = q[q.lev.abs() == L].drop_duplicates(['day', 'symbol'])
        dec[L] = float(u.day_ret.mean())
        decv[L] = float(uv.day_ret.mean()) if len(uv) else np.nan
        gr[L] = float(d.rr.mean())
        grv[L] = float(dv.rr.mean()) if len(dv) else np.nan
        print(f'  {L:4.0f} {len(u):7d} {dec[L]:+9.4f} {decv[L]:+10.4f} {len(d):9d} '
              f'{gr[L]:+9.4f} {grv[L]:+10.4f} {float(alld.day_ret.mean()):+17.4f}', flush=True)
    print(f'\n  (i-secondary) by the variance-drag coefficient k = L(L-1)/2')
    for k in sorted(set(flat.kdrag.dropna())):
        d = flat[flat.kdrag == k]
        if len(d) < 20:
            continue
        u = d.drop_duplicates(['day', 'symbol'])
        print(f'  k={k:4.1f} n_days={len(u):6d} decay={float(u.day_ret.mean()):+8.4f} % '
              f'n={len(d):6d} gross={float(d.rr.mean()):+8.4f} R '
              f'(lev {sorted(set(d.lev.dropna()))})', flush=True)

    def monotone(m):
        ks = sorted(m)
        vals = [m[k] for k in ks]
        return len(ks) >= 3 and all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))
    # DECAY is a return: "larger decay" = MORE negative, so the test is on -decay rising.
    dec_ok = (monotone({k: -v for k, v in dec.items()}) and
              monotone({k: -v for k, v in decv.items() if np.isfinite(v)}))
    gr_ok = monotone(gr) and monotone({k: v for k, v in grv.items() if np.isfinite(v)})
    out['(i) leverage monotone'] = bool(dec_ok and gr_ok)
    print(f'\n  (i) VERDICT: decay monotone {dec_ok} | short gross monotone {gr_ok} -> '
          f'{"PASS" if out["(i) leverage monotone"] else "FAIL"}', flush=True)

    print('\n  (ii) MAXIMUM WHEN THE UNDERLYING IS FLAT — short gross R by |underlying move|')
    print(f'  {"bucket":>12s} {"n":>7s} {"gross R":>9s} {"TRAIN":>9s} {"VAL":>9s} {"net R":>9s} '
          f'{"decay %":>9s}')
    buckets, vals = [('<=1 %', 0, 1), ('1-3 %', 1, 3), ('>=3 %', 3, 1e9)], {}
    for lab, lo, hi in buckets:
        d = base[(base.u_move.abs() >= lo) & (base.u_move.abs() < hi)]
        if len(d) < 20:
            continue
        vals[lab] = float(d.rr.mean())
        tr, v = d[d.split == 'TRAIN'], d[d.split == 'VAL']
        u = d.drop_duplicates(['day', 'symbol'])
        print(f'  {lab:>12s} {len(d):7d} {vals[lab]:+9.4f} '
              f'{float(tr.rr.mean()) if len(tr) else np.nan:+9.4f} '
              f'{float(v.rr.mean()) if len(v) else np.nan:+9.4f} {float(d.net.mean()):+9.4f} '
              f'{float(u.day_ret.mean()):+9.4f}', flush=True)
    order = [vals.get(l) for l, _, _ in buckets if l in vals]
    out['(ii) flat is the max'] = bool(len(order) == 3 and order[0] > order[1] > order[2])
    print(f'\n  (ii) VERDICT: {"PASS" if out["(ii) flat is the max"] else "FAIL"} '
          f'(required: gross falls monotonically as |underlying move| rises)', flush=True)
    return out


def realism(best):
    """Declared secondary: the best (S) cell under the shipped slot rule (12/day, 4 concurrent)."""
    if best is None:
        return
    _, T, X, d = best
    print('\n===== F32 (S) — the realism cut: run_book(12/day, 4 concurrent) =====', flush=True)
    bk = book_ranked(d, 12, 4)
    report(SROWS, f'S* booked {T // 60:02d}:{T % 60:02d} rng<={X:g}%', bk,
           note='12/day, 4 conc, DIAGNOSTIC no-borrow')


def main():
    long_book()
    p, q, best = short_book()
    m = mechanism(q, best)
    realism(best)
    pd.DataFrame(LROWS).to_csv(f'{D10}/cells32_long.csv', index=False)
    pd.DataFrame(SROWS).to_csv(f'{D10}/cells32_short.csv', index=False)
    print(f'\ncells -> cells32_long.csv ({len(LROWS)}) / cells32_short.csv ({len(SROWS)})',
          flush=True)
    print(f'mechanism: {m}', flush=True)


if __name__ == '__main__':
    main()
