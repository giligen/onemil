#!/usr/bin/env python3
"""hod_frames4 / F13 — RANK, DON'T RACE: the slot rule.

Cells exactly as declared in PREREG.md §1 (commit 92db20a, before any cell was scored).
TEST sealed.  Read-only.  One process.
"""
import sys
import numpy as np, pandas as pd

sys.path.insert(0, '/home/ec2-user/onemil/research/mature_method/hod_frames4')
from common4 import (ROOT, D4, S, SPLITS, RISK, clustered_t, halves,      # noqa: E402
                     book_ranked, book_oracle)

HDR = ('| cell                                  | split | n     | /wk   | grossR | cost   | netR   |'
       '  t    | tc    | green | rs | worst $  | total $   | MDD $    | ex5    |')
SEP = '|' + '|'.join(['-' * 6] * 15) + '|'
CELLS, BOOKS = [], {}


def show(name, b, note='', first=[True]):
    BOOKS[name] = b
    if first[0]:
        print(HDR); print(SEP); first[0] = False
    for sp in SPLITS:
        w = S.week_stats(b, sp); d = b[b.split == sp]
        cost = float((d.rr - d.net).mean()) if len(d) else np.nan
        print(f'| {name:<37s} | {sp:5s} | {w["n"]:5d} | {w["per_wk"]:5.1f} | {w["gross"]:+.3f} | '
              f'{cost:+.3f} | {w["net"]:+.3f} | {w["t"]:+5.2f} | {clustered_t(d):+5.2f} | '
              f'{w["green"]:5.1f} | {w["redstreak"]:2d} | {w["worst"]:8.0f} | {w["total"]:9.0f} | '
              f'{w["mdd"]:8.0f} | {w["ex5"]:+.3f} |', flush=True)
    g, n, ok = halves(b)
    for sp in SPLITS:
        w = S.week_stats(b, sp); d = b[b.split == sp]
        w.update(cell=name, split=sp, tc=clustered_t(d), note=note,
                 cost=float((d.rr - d.net).mean()) if len(d) else np.nan,
                 mde=2.80 * float(d.net.std(ddof=1) / np.sqrt(len(d))) if len(d) > 2 else np.nan,
                 h1=g[0], h2=g[1], vgross=g[2], half_ok=ok)
        CELLS.append(w)
    print(f'    halves H1 {g[0]:+.3f} (n {n[0]}) | H2 {g[1]:+.3f} (n {n[1]}) | VAL {g[2]:+.3f} '
          f'(n {n[2]}) -> same-signed POSITIVE in all three: {ok}', flush=True)
    return b


def main():
    s = pd.read_csv(f'{D4}/sig4.csv', dtype={'symbol': str, 'day': str, 'wk': str, 'split': str,
                                             'why': str}, keep_default_na=False, na_values=[''])
    print(f'== hod_frames4 / F13 — the slot rule ==  pre-book signals {len(s)}, days '
          f'{s.day.nunique()}\n   splits scored: {SPLITS}   (TEST sealed)\n', flush=True)

    # ---------- controls -------------------------------------------------------------------
    c0 = show('F13-C0 first-come, symbol tie [=B2]', book_ranked(s, 12, 4))
    print('\n-- F13-C1: 200 seeded RANDOM tie-breaks within the minute (the alphabet s own noise) --')
    band = {}
    for sp in SPLITS:
        band[sp] = dict(green=[], total=[], gross=[])
    for seed in range(200):
        rg = np.random.default_rng(1000 + seed)
        b = book_ranked(s, 12, 4, rng=rg)
        for sp in SPLITS:
            w = S.week_stats(b, sp)
            band[sp]['green'].append(w['green']); band[sp]['total'].append(w['total'])
            band[sp]['gross'].append(w['gross'])
    print('| split | green % mean [p5,p95] | total $ mean [p5,p95] | gross mean [p5,p95] | C0 |')
    print('|---|---|---|---|---|')
    for sp in SPLITS:
        w0 = S.week_stats(c0, sp)
        g_, t_, r_ = (np.array(band[sp][k]) for k in ('green', 'total', 'gross'))
        print(f'| {sp} | {g_.mean():.1f} [{np.percentile(g_,5):.1f}, {np.percentile(g_,95):.1f}] '
              f'| {t_.mean():+.0f} [{np.percentile(t_,5):+.0f}, {np.percentile(t_,95):+.0f}] '
              f'| {r_.mean():+.4f} [{np.percentile(r_,5):+.4f}, {np.percentile(r_,95):+.4f}] '
              f'| {w0["green"]:.1f} / {w0["total"]:+.0f} / {w0["gross"]:+.4f} |', flush=True)
    RTB = band

    # ---------- the oracle ceiling ----------------------------------------------------------
    print('\n\n================ F13 ORACLE CEILING (bounds, NOT strategies) ================',
          flush=True)
    show('F13-O1 oracle top-4/day', book_oracle(s, 4, None))
    show('F13-O2 oracle top-8/day', book_oracle(s, 8, None))
    show('F13-O3 oracle top-12/day', book_oracle(s, 12, None))
    show('F13-O4 oracle top-12/day, 4 conc', book_oracle(s, 12, 4))

    # ---------- causal rankings -------------------------------------------------------------
    print('\n\n================ F13 CAUSAL RANKINGS (within-minute) ================', flush=True)
    s = s.copy()
    s['dist_own'] = s.dist_open_pct / s.med_rng.replace(0, np.nan)
    s['spr_r'] = s.sp_pct / s.r_pct.clip(lower=0.05)
    tr = s[s.split == 'TRAIN']
    zc = {}
    for f, sgn in (('rv_profile', 1), ('dollar_frac', 1), ('dist_own', 1), ('spr_r', -1)):
        mu, sd = float(tr[f].mean()), float(tr[f].std(ddof=1))
        zc[f] = (mu, sd, sgn)
        print(f'   z-param (TRAIN) {f:<12s} mean {mu:+.4f} sd {sd:.4f} sign {sgn:+d}')
    s['comp_z'] = sum(sgn * ((s[f] - mu) / sd).fillna(0.0) for f, (mu, sd, sgn) in zc.items())

    rank_cells = {
        'F13-r1 rank rv_profile desc': (s.rv_profile, True),
        'F13-r2 rank dollar_frac desc': (s.dollar_frac, True),
        'F13-r3 rank dist_open/med_rng desc': (s.dist_own, True),
        'F13-r4 rank spread/R asc [cheapest]': (s.spr_r, False),
        'F13-r5 rank composite z desc': (s.comp_z, True),
    }
    for nm, (sc, desc) in rank_cells.items():
        show(nm, book_ranked(s, 12, 4, score=sc, descending=desc))

    # F13-r6 — declared void if degenerate; the degeneracy is MEASURED, not asserted
    deg = s.groupby(['day', 'entry_m']).spy_r5_pct.nunique().max()
    print(f'\n-- F13-r6 SPY-state x entry-minute: within a (day, minute) the score takes '
          f'{int(deg)} distinct value(s) -> **VOID by the availability rail** (it cannot order '
          f'simultaneous candidates).  Declared and reported, not dropped silently.', flush=True)

    # ---------- how much does the ORDER even change? ----------------------------------------
    print('\n-- churn of the booked set vs first-come (share of C0 trades replaced) --')
    for nm in rank_cells:
        b = BOOKS[nm]
        for sp in SPLITS:
            a = set(c0[c0.split == sp].index); z = set(b[b.split == sp].index)
            print(f'   {nm:<38s} {sp:5s} kept {len(a & z):4d} of {len(a):4d} '
                  f'({len(a & z)/max(len(a),1):.1%}), swapped {len(z - a):4d}')

    # ---------- slot count and reserve ------------------------------------------------------
    print('\n\n================ F13 SLOT COUNT / RESERVE ================', flush=True)
    cf = pd.DataFrame(CELLS)
    best, bestv = None, -1e18
    for nm in rank_cells:
        t_ = cf[(cf.cell == nm) & (cf.split == 'TRAIN')].iloc[0]
        v_ = cf[(cf.cell == nm) & (cf.split == 'VAL')].iloc[0]
        v = t_.total + v_.total
        if v > bestv:
            best, bestv = nm, v
    print(f'   best causal ranker by TRAIN+VAL dollars = {best} ({bestv:+.0f})', flush=True)
    sc, desc = rank_cells[best]
    show(f'F13-n8  {best} @ 8 conc', book_ranked(s, 12, 8, score=sc, descending=desc))
    show(f'F13-n12 {best} @ 12 conc', book_ranked(s, 12, 12, score=sc, descending=desc))
    show('F13-rs1 reserve 1 of 4 until 10:30', book_ranked(s, 12, 4, reserve=1, reserve_until=630))
    show('F13-rs2 reserve 2 of 4 until 10:30', book_ranked(s, 12, 4, reserve=2, reserve_until=630))

    # ---------- nulls, bars, MDE ------------------------------------------------------------
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
    pd.DataFrame(nulls).to_csv(f'{D4}/nulls13.csv', index=False)
    cf = pd.DataFrame(CELLS); cf.to_csv(f'{D4}/cells13.csv', index=False)

    print('\n== F13 PRE-COMMITTED SELECTOR (ranking beats the race?) ==')
    print('| ranking | split | green vs C0 | $ vs C0 | outside the random tie-break band? | halves |')
    print('|---|---|---|---|---|---|')
    winners = []
    for nm in list(rank_cells) + [f'F13-n8  {best} @ 8 conc', f'F13-n12 {best} @ 12 conc',
                                  'F13-rs1 reserve 1 of 4 until 10:30',
                                  'F13-rs2 reserve 2 of 4 until 10:30']:
        okall = True
        for sp in SPLITS:
            r = cf[(cf.cell == nm) & (cf.split == sp)].iloc[0]
            w0 = S.week_stats(c0, sp)
            g_, t_ = np.array(RTB[sp]['green']), np.array(RTB[sp]['total'])
            out = ('ABOVE' if r.total > np.percentile(t_, 95) else
                   ('below' if r.total < np.percentile(t_, 5) else 'inside'))
            ok = (r.green > w0['green'] and r.total > w0['total'] and out == 'ABOVE')
            okall = okall and ok
            print(f'| {nm} | {sp} | {r.green:.1f} vs {w0["green"]:.1f} | {r.total:+.0f} vs '
                  f'{w0["total"]:+.0f} | $ {out} | {r.half_ok} |')
        if okall and cf[(cf.cell == nm) & (cf.split == 'TRAIN')].iloc[0].half_ok:
            winners.append(nm)
    print(f'\n  -> cells beating first-come on BOTH splits, outside the tie-break band, halves '
          f'same-signed positive: {winners if winners else "NONE"}')

    print('\n== BOTH BARS ==')
    g1 = cf[(cf.split == 'TRAIN') & (cf.net > 0) & (cf.t >= 2.0) & (cf.tc >= 2.0) &
            (cf.per_wk >= 10) & (~cf.cell.str.startswith('F13-O'))]
    print(f'G1 (TRAIN net>0, iid t>=2, CLUSTERED t>=2, >=10/wk; oracles excluded): {len(g1)} '
          f'-> {list(g1.cell)}')
    ship = []
    for nm in cf.cell.unique():
        if nm.startswith('F13-O'):
            continue
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
        print(f'  {nm:40s} {row}')
    print('\nDONE', flush=True)


if __name__ == '__main__':
    main()
