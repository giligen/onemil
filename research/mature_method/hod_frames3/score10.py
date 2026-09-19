#!/usr/bin/env python3
"""hod_frames3 / F10 — the quarantined footprint field, with the downstream gate re-specified.

`dollar_frac` = cumulative $ volume 09:30 -> the break bar, as % of the symbol's 20-prior-session
ADV$.  Pass 2 EXCLUDED it before scoring; PREREG §2 withdraws the exclusion and keeps the
already-there / arrived-after split as a REPORTED DIAGNOSTIC.  The only selector is era-consistency.

Cells exactly as declared in PREREG.md (commit b4b8171).  TEST sealed.  Read-only.  One process.
"""
import os, sys
import numpy as np, pandas as pd

sys.path.insert(0, '/home/ec2-user/onemil/research/mature_method/hod_frames3')
from common3 import (ROOT, D, S, S2, clustered_t, halves, sigset, load_breaks, admit)  # noqa: E402

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
                 h1=g[0], h2=g[1], vgross=g[2], half_ok=ok, note=note)
        CELLS.append(w)
    print(f'    halves H1 {g[0]:+.3f} (n {n[0]}) | H2 {g[1]:+.3f} (n {n[1]}) | VAL {g[2]:+.3f} '
          f'(n {n[2]}) -> same-signed POSITIVE in all three: {ok}', flush=True)
    return b


def diagnostic(pre, thr, lab):
    """PREREG §2: the already-there / arrived-after split — REPORTED, never an exclusion."""
    for sp in SPLITS:
        d = pre[(pre.split == sp) & pre.dollar_frac.notna()]
        top, rest = d[d.dollar_frac >= thr], d[d.dollar_frac < thr]
        if len(top) < 20 or len(rest) < 20:
            continue
        dp = float((top.rng_day >= 10).mean() - (rest.rng_day >= 10).mean()) * 100
        ds = float(top.rng_sig.mean() - rest.rng_sig.mean())
        da = float(top.rng_after.mean() - rest.rng_after.mean())
        med = top.rng_sig.median()
        a, b_ = top[top.rng_sig >= med], top[top.rng_sig < med]
        print(f'| {lab} | {sp} | {len(top)} | {(top.rng_day>=10).mean():.1%} | '
              f'{(rest.rng_day>=10).mean():.1%} | {dp:+.1f}pp | {ds:+.2f}pp | {da:+.2f}pp | '
              f'{top.rr.mean():+.3f} | {rest.rr.mean():+.3f} | already-wide {a.rr.mean():+.3f} '
              f'(WR {(a.rr>0).mean():.0%}) / arrived-after {b_.rr.mean():+.3f} '
              f'(WR {(b_.rr>0).mean():.0%}) |')


def main():
    print('== hod_frames3 / F10 — the footprint field `dollar_frac`, gate re-specified ==')
    print(f'   splits scored: {SPLITS}   (TEST sealed)\n', flush=True)

    pop = S2.load_pop(); S.build_impute(pop)
    b2 = S.apply_book(S2.sig_set(pop, **S2.BASES['B2']), 12, 4)
    print('== R1/R2 REPRODUCTION (ref -17,346 | +893) ==')
    show('R1/R2 B2 shipped (pop.csv)', b2)

    br = load_breaks()
    base = sigset(admit(br, pd.Series(True, index=br.index)))
    show('R2b B2 rebuilt (this pass)', S.apply_book(base, 12, 4))

    tr = base[base.split == 'TRAIN']
    print('\n== AVAILABILITY AUDIT — `dollar_frac` on the B2 pre-book set ==')
    for sp in SPLITS:
        d = base[base.split == sp]; v = d.dollar_frac; win = d.rr > 0
        print(f'  {sp}: cov {v.notna().mean():.1%}  miss win {v[win].isna().mean():.1%} '
              f'loss {v[~win].isna().mean():.1%}  (>5pp gap drops the field)')
    qs = {p: float(tr.dollar_frac.quantile(p / 100.0)) for p in (50, 60, 70, 80, 90)}
    print(f'\n   TRAIN percentiles of dollar_frac (% of ADV$): '
          + ' | '.join(f'p{p} {v:.1f}' for p, v in qs.items()))
    print(f'   imputed-cost share on the TRAIN pre-book set: {tr.imputed.mean():.1%}')

    print('\n== THE RE-SPECIFIED DOWNSTREAM DIAGNOSTIC (reported, NOT an exclusion) ==')
    print('| rung | split | n top | P(EOD>=10%) top | rest | dP | d rng_sig | d rng_after | '
          'gross top | gross rest | §2.3 split inside the rung |')
    print('|' + '|'.join(['---'] * 11) + '|')
    for p, v in qs.items():
        diagnostic(base, v, f'dollar_frac>=p{p}')

    print('\n\n================ F10 — THE LADDER (5 cells) ================')
    for p, v in qs.items():
        show(f'F10 dollar_frac>=p{p} ({v:.1f}%)', S.apply_book(base[base.dollar_frac >= v], 12, 4))

    cf = pd.DataFrame(CELLS)
    elig = []
    for p, v in qs.items():
        nm = f'F10 dollar_frac>=p{p} ({v:.1f}%)'
        t_ = cf[(cf.cell == nm) & (cf.split == 'TRAIN')].iloc[0]
        vv = cf[(cf.cell == nm) & (cf.split == 'VAL')].iloc[0]
        ok = bool(t_.half_ok) and min(t_.per_wk, vv.per_wk) >= 10
        print(f'   {nm}: halves same-signed+ {bool(t_.half_ok)}  /wk {t_.per_wk:.1f}/{vv.per_wk:.1f} '
              f'-> eligible {ok}')
        if ok:
            elig.append((min(t_.h1, t_.h2, t_.vgross), p, v, nm))
    print(f'\n== PREREG SELECTOR (era-consistency only): eligible rungs {[e[3] for e in elig]}')
    if not elig:
        print('   NO rung is era-consistent -> F10 is DEAD; F10-6/7/8 are NOT scored, as declared.')
    else:
        _, p, v, nm = sorted(elig, reverse=True)[0]
        print(f'   SELECTED RUNG: {nm}')
        sel = base[base.dollar_frac >= v]
        show(f'F10-6 p{p} x spy_r5>0', S.apply_book(sel[sel.spy_r5_pct > 0], 12, 4))
        show(f'F10-7 p{p} x consol_bars>=20', S.apply_book(sel[sel.consol_bars >= 20], 12, 4))
        show(f'F10-8 p{p} x C1', S.apply_book(sel[(sel.consol_bars >= 20) & (sel.spy_r5_pct > 0)],
                                              12, 4))
        miss = sel[sel.spread_mean.isna()][['day', 'symbol', 'entry_m']].drop_duplicates()
        miss.to_csv(f'{D}/nbbo3_todo.csv', index=False)
        print(f'\n   NBBO FETCH LIST: {len(miss)} of {len(sel)} rows of the selected rung have no '
              f'measured quote -> {D}/nbbo3_todo.csv  (run fetch_nbbo3.py, then re-run with the '
              f'measured cost)')

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
    pd.DataFrame(nulls).to_csv(f'{D}/nulls10.csv', index=False)
    cf = pd.DataFrame(CELLS); cf.to_csv(f'{D}/cells10.csv', index=False)

    print('\n== BOTH BARS ==')
    g1 = cf[(cf.split == 'TRAIN') & (cf.net > 0) & (cf.t >= 2.0) & (cf.tc >= 2.0) & (cf.per_wk >= 10)]
    print(f'G1: {len(g1)} of {cf[cf.split=="TRAIN"].cell.nunique()} -> {list(g1.cell)}')
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
    print('\nDONE', flush=True)


if __name__ == '__main__':
    main()
