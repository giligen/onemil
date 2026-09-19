#!/usr/bin/env python3
"""hod_frames3 / F11 — identify `consol_bars >= 20`, the programme's only unexplained survivor.

Cells exactly as declared in PREREG.md (commit b4b8171, before any cell was scored).
TEST is sealed.  Read-only on every DB.  One process.
"""
import os, sys
import numpy as np, pandas as pd

sys.path.insert(0, '/home/ec2-user/onemil/research/mature_method/hod_frames3')
from common3 import (ROOT, D, RD, S, S2, clustered_t, halves, sigset, load_breaks,  # noqa: E402
                     admit)

SPLITS = S.SPLITS
HDR = ('| cell                               | split | n     | /wk   | grossR | cost   | netR   |'
       '  t    | tc    | green | rs | worst $  | total $   | MDD $    | ex5    | imp |')
SEP = '|' + '|'.join(['-' * 6] * 16) + '|'
CELLS, BOOKS = [], {}


def show(name, b, note=''):
    BOOKS[name] = b
    out = [HDR, SEP] if not CELLS else []
    for sp in SPLITS:
        w = S.week_stats(b, sp); d = b[b.split == sp]
        cost = float((d.rr - d.net).mean()) if len(d) else np.nan
        out.append(f'| {name:<34s} | {sp:5s} | {w["n"]:5d} | {w["per_wk"]:5.1f} | {w["gross"]:+.3f} | '
                   f'{cost:+.3f} | {w["net"]:+.3f} | {w["t"]:+5.2f} | {clustered_t(d):+5.2f} | '
                   f'{w["green"]:5.1f} | {w["redstreak"]:2d} | {w["worst"]:8.0f} | {w["total"]:9.0f} | '
                   f'{w["mdd"]:8.0f} | {w["ex5"]:+.3f} | {w["imp"]:3.0f} |')
    print('\n'.join(out), flush=True)
    g, n, ok = halves(b)
    for sp in SPLITS:
        w = S.week_stats(b, sp); d = b[b.split == sp]
        w.update(cell=name, split=sp, tc=clustered_t(d),
                 cost=float((d.rr - d.net).mean()) if len(d) else np.nan,
                 mde=2.80 * float(d.net.std(ddof=1) / np.sqrt(len(d))) if len(d) > 2 else np.nan,
                 big_range=float((d.day_range_pct >= 10).mean() * 100) if len(d) else np.nan,
                 med_px=float(d.price.median()) if len(d) else np.nan,
                 med_spr=float((d.sp_pct / d.r_pct).median()) if len(d) else np.nan,
                 h1=g[0], h2=g[1], vgross=g[2], half_ok=ok, note=note)
        CELLS.append(w)
    print(f'    halves H1 {g[0]:+.3f} (n {n[0]}) | H2 {g[1]:+.3f} (n {n[1]}) | VAL {g[2]:+.3f} '
          f'(n {n[2]}) -> same-signed POSITIVE in all three: {ok}', flush=True)
    return b


def cluster_delta_t(y, g, d):
    """Cluster-robust (clusters = trading days) t of the difference in means, via OLS y = a + b.d."""
    X = np.column_stack([np.ones(len(y)), d.astype(float)])
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    e = y - X @ beta
    meat = np.zeros((2, 2))
    df = pd.DataFrame(X * e[:, None]); df['g'] = g
    for _, sub in df.groupby('g'):
        s = sub[[0, 1]].values.sum(axis=0)
        meat += np.outer(s, s)
    V = XtX_inv @ meat @ XtX_inv
    return float(beta[1]), float(beta[1] / np.sqrt(V[1, 1])) if V[1, 1] > 0 else np.nan


def sep_test(pop, mask, label, col='rr'):
    """kept vs rejected inside one population: the F11 difference test."""
    print(f'\n-- SEPARATION: {label}  (column = {col}) --')
    print('| split | n kept | n rej | kept | rej | delta | iid t | CLUSTERED t | halves kept |')
    print('|---|---|---|---|---|---|---|---|---|')
    for sp in SPLITS:
        d = pop[pop.split == sp]
        mk = mask.reindex(d.index).fillna(False).values
        k, r = d[mk], d[~mk]
        if len(k) < 5 or len(r) < 5:
            continue
        dl = float(k[col].mean() - r[col].mean())
        se = np.sqrt(k[col].var(ddof=1) / len(k) + r[col].var(ddof=1) / len(r))
        _, tc = cluster_delta_t(d[col].values.astype(float), d.day.values, mk)
        h = ''
        if sp == 'TRAIN':
            h = (f'H1 {k[k.day<"2025-07-01"][col].mean():+.3f} / '
                 f'H2 {k[k.day>="2025-07-01"][col].mean():+.3f}')
        print(f'| {sp} | {len(k)} | {len(r)} | {k[col].mean():+.3f} | {r[col].mean():+.3f} | '
              f'{dl:+.3f} | {dl/se:+.2f} | {tc:+.2f} | {h} |')


def main():
    print('== hod_frames3 / F11 — identify `consol_bars >= 20` ==')
    print(f'   splits scored: {SPLITS}   (TEST sealed)\n', flush=True)

    # ---- STEP 0: reproduction -------------------------------------------------------------
    pop = S2.load_pop(); S.build_impute(pop)
    b2 = S.apply_book(S2.sig_set(pop, **S2.BASES['B2']), 12, 4)
    b2['day_range_pct'] = np.nan
    print('== R1/R2 REPRODUCTION (ref B2: 1,622/30.6/-0.039/-0.107/32.1%/-17,346 | '
          '706/30.7/+0.083/+0.013/43.5%/+893) ==')
    show('R1/R2 B2 shipped (pop.csv)', b2)

    br = load_breaks()
    print(f'  feat3 coverage: hl_n20 {br.hl_n20.notna().mean():.1%} | lo_slope20 '
          f'{br.lo_slope20.notna().mean():.1%} | atr_ratio {br.atr_ratio.notna().mean():.1%} | '
          f'consol_bars {br.consol_bars.notna().mean():.1%}', flush=True)

    base = sigset(admit(br, pd.Series(True, index=br.index)))
    show('R2b B2 rebuilt (this pass)', S.apply_book(base, 12, 4))

    ge20 = sigset(admit(br, br.consol_bars >= 20))
    print('\n== R3 REPRODUCTION — hod_fresh `consol_bars >= 20` rung x n5 stop '
          '(ref 1,411/26.6/+0.028/-0.041/34.0%/-5,749 | 714/31.0/+0.050/-0.023/47.8%/-1,629) ==')
    bg = show('R3 consol_bars>=20 (ge20)', S.apply_book(ge20, 12, 4))
    print('\n== R4 REPRODUCTION — C1 = ge20 x spy_r5>0 '
          '(ref 731/13.8/+0.100/+0.033/43.4%/+2,391 | 368/16.0/+0.123/+0.050/47.8%/+1,827) ==')
    c1 = show('R4 C1 = ge20 x spy_r5>0', S.apply_book(ge20[ge20.spy_r5_pct > 0], 12, 4))

    # ---- availability audit on the new fields ----------------------------------------------
    print('\n== AVAILABILITY AUDIT (B2 pre-book) — coverage, missingness on winners vs losers ==')
    for c in ('hl_n20', 'lo_slope20', 'atr_ratio', 'atr_now', 'consol_bars', 'touch_n'):
        v = pd.to_numeric(base[c], errors='coerce'); win = base.rr > 0
        mw, ml = float(v[win].isna().mean()), float(v[~win].isna().mean())
        flag = '  <-- DROP (outcome-dependent missingness)' if abs(mw - ml) > 0.05 else ''
        print(f'  {c:<12s} cov {v.notna().mean():6.1%}  miss win {mw:6.1%} loss {ml:6.1%}{flag}')
    print('\n   structural clock floors (min break_m of each admission, TRAIN):')
    for lab, m in (('ge20', br.consol_bars >= 20), ('hl_n20 defined', br.hl_n20.notna()),
                   ('atr_ratio defined', br.atr_ratio.notna())):
        d = br[m.fillna(False).values]
        print(f'     {lab:<20s} min break_m {int(d.break_m.min())} '
              f'(= {570 + (int(d.break_m.min()) - 570)} -> '
              f'{int(d.break_m.min())//60:02d}:{int(d.break_m.min())%60:02d} ET)')
    bb0 = S.apply_book(base, 12, 4)
    bb0t = bb0[bb0.split == 'TRAIN']
    bgt0 = bg[bg.split == 'TRAIN']
    print('\n   what ge20 SELECTS vs the base (TRAIN book rows): '
          f'price med {bgt0.price.median():.2f} vs {bb0t.price.median():.2f} | '
          f'spread/R med {(bgt0.sp_pct/bgt0.r_pct).median():.4f} vs '
          f'{(bb0t.sp_pct/bb0t.r_pct).median():.4f} | '
          f'break_m med {bgt0.break_m.median():.0f} vs {bb0t.break_m.median():.0f}')

    # ---- the declared d1/d2 thresholds, printed BEFORE the cells ---------------------------
    bgt = bg[bg.split == 'TRAIN']
    P20 = float(bgt.price.median())
    S20 = float((bgt.sp_pct / bgt.r_pct).median())
    print(f'\n   PREREG §2 thresholds from R3 TRAIN: P20 = {P20:.4f}  S20 = {S20:.6f}')

    # ================= THE CELLS ============================================================
    print('\n\n================ F11 — THE DECOMPOSITION (10 cells) ================')
    cells = {
        'F11-a1 hl_n20>=12': admit(br, br.hl_n20 >= 12),
        'F11-a2 hl_n20>=16': admit(br, br.hl_n20 >= 16),
        'F11-a3 lo_slope20>0': admit(br, br.lo_slope20 > 0),
        'F11-b1 atr_ratio<=0.8': admit(br, br.atr_ratio <= 0.8),
        'F11-b2 atr_ratio<=0.6': admit(br, br.atr_ratio <= 0.6),
        'F11-c1 break_m>=590 [clock]': admit(br, br.break_m >= 590),
        'F11-c2 break_m>=600 [clock]': admit(br, br.break_m >= 600),
    }
    for nm, rows in cells.items():
        show(nm, S.apply_book(sigset(rows), 12, 4))
    # d1/d2 are FIRST-BREAK FILTERS on the base admission (scan rule named in PREREG)
    show(f'F11-d1 price>={P20:.1f} [filter]', S.apply_book(base[base.price >= P20], 12, 4))
    show(f'F11-d2 spread/R<={S20:.4f} [filter]',
         S.apply_book(base[(base.sp_pct / base.r_pct) <= S20], 12, 4))

    # ---- the difference test ---------------------------------------------------------------
    c1pop = sigset(admit(br, br.break_m >= 590))
    sep_test(c1pop, c1pop.consol_bars >= 20,
             'inside the clock admission (break_m>=590): consol_bars>=20 kept vs rejected')
    sep_test(c1pop[c1pop.spy_r5_pct > 0], c1pop[c1pop.spy_r5_pct > 0].consol_bars >= 20,
             'the same, on SPY-up days only (C1 s own day gate)')
    sep_test(base, base.break_m >= 590,
             'inside the BASE first-break population: break_m>=590 kept vs rejected')

    # ---- the conditional cross -------------------------------------------------------------
    cf = pd.DataFrame(CELLS)
    ab = [c for c in cells if c.startswith(('F11-a', 'F11-b'))]
    elig = []
    for nm in ab:
        t_ = cf[(cf.cell == nm) & (cf.split == 'TRAIN')].iloc[0]
        v_ = cf[(cf.cell == nm) & (cf.split == 'VAL')].iloc[0]
        if t_.half_ok and min(t_.per_wk, v_.per_wk) >= 10:
            elig.append((min(t_.h1, t_.h2, t_.vgross), nm))
    print(f'\n-- F11-x1 conditional cross: eligible (a)/(b) cells = {sorted(elig, reverse=True)}')
    if elig:
        best = sorted(elig, reverse=True)[0][1]
        bb = cells[best]
        show(f'F11-x1 {best} x spy_r5>0', S.apply_book(sigset(bb[bb.spy_r5_pct > 0]), 12, 4))
    else:
        print('   NOT SCORED — no (a)/(b) cell is same-signed-positive on gross in H1/H2/VAL at '
              '>=10 trades/week.  Reported as declared.')

    # ---- nulls, bars, MDE ------------------------------------------------------------------
    print('\n\n== count-matched permutation null on green weeks (2,000 draws) ==')
    print('| cell | split | observed green % | null mean | [p5, p95] | outside? |')
    print('|---|---|---|---|---|---|')
    nulls = []
    for nm, bb in BOOKS.items():
        for sp in SPLITS:
            obs, mu_, p5, p95 = S.null_band(bb, sp)
            o = ('ABOVE' if obs == obs and obs > p95 else
                 ('below' if obs == obs and obs < p5 else 'inside'))
            print(f'| {nm} | {sp} | {obs:.1f} | {mu_:.1f} | [{p5:.1f}, {p95:.1f}] | {o} |')
            nulls.append(dict(cell=nm, split=sp, obs=obs, mu=mu_, p5=p5, p95=p95, outside=o))
    pd.DataFrame(nulls).to_csv(f'{D}/nulls11.csv', index=False)
    cf = pd.DataFrame(CELLS); cf.to_csv(f'{D}/cells11.csv', index=False)

    print('\n== BOTH BARS ==')
    g1 = cf[(cf.split == 'TRAIN') & (cf.net > 0) & (cf.t >= 2.0) & (cf.tc >= 2.0) & (cf.per_wk >= 10)]
    print(f'G1 (TRAIN net>0, iid t>=2, CLUSTERED t>=2, >=10/wk): {len(g1)} of '
          f'{cf[cf.split=="TRAIN"].cell.nunique()} -> {list(g1.cell)}')
    ship = []
    for nm in cf.cell.unique():
        t_ = cf[(cf.cell == nm) & (cf.split == 'TRAIN')].iloc[0]
        v_ = cf[(cf.cell == nm) & (cf.split == 'VAL')].iloc[0]
        if (t_.total > 0 and v_.total > 0 and t_.green >= 50 and v_.green >= 50 and
                min(t_.per_wk, v_.per_wk) >= 10 and t_.tc >= 2.0 and t_.half_ok):
            ship.append(nm)
    print(f'LIVE-EXPLORATION BAR: {len(ship)} -> {ship}')
    print('\nMDE (80% power, per trade, net):')
    for nm, bb in BOOKS.items():
        row = ' '.join(f'{sp} {2.80*bb[bb.split==sp].net.std(ddof=1)/np.sqrt(max(len(bb[bb.split==sp]),1)):.3f}'
                       for sp in SPLITS if len(bb[bb.split == sp]) > 2)
        print(f'  {nm:36s} {row}')

    print('\n== PREREG VERDICT RULE ==')
    fam = {}
    for nm in cf.cell.unique():
        t_ = cf[(cf.cell == nm) & (cf.split == 'TRAIN')].iloc[0]
        v_ = cf[(cf.cell == nm) & (cf.split == 'VAL')].iloc[0]
        fam[nm] = bool(t_.half_ok and min(t_.per_wk, v_.per_wk) >= 10)
    ab_ok = [n for n in fam if n.startswith(('F11-a', 'F11-b')) and fam[n]]
    cd_ok = [n for n in fam if n.startswith(('F11-c', 'F11-d')) and fam[n]]
    print(f'  (a)/(b) mechanism cells same-signed-positive at >=10/wk: {ab_ok}')
    print(f'  (c)/(d) clock/liquidity cells same-signed-positive at >=10/wk: {cd_ok}')
    if cd_ok and not ab_ok:
        print('  -> C1 IS RETIRED as a clock/liquidity artefact (PREREG §2 rule).')
    elif ab_ok and not cd_ok:
        print('  -> C1 IS A MECHANISM (PREREG §2 rule); the frame extends it.')
    else:
        print('  -> UNRESOLVED by the pre-committed rule; the difference test is the evidence.')
    print('\nDONE', flush=True)


if __name__ == '__main__':
    main()
