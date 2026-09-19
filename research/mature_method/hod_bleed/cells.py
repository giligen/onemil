#!/usr/bin/env python3
"""hod_bleed PART 2 — the declared cells of PREREG.md + ADDENDUM.md.  Nothing here is run before
both are committed.  TEST is sealed: no TEST number unless FREEZE.md exists AND --test.
"""
import json, os, sys
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT); sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_break')
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_filter_stack')
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_fresh')
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_bleed')
import score as S             # noqa: E402
import score2 as S2           # noqa: E402
import core as C              # noqa: E402
from walk3 import load_pop4   # noqa: E402

D = f'{ROOT}/research/mature_method/hod_bleed'
OUT = open(f'{D}/part2.txt', 'w')
ROWS, NULLS = [], []
CACHE = {}
H1 = ('2025-01-01', '2025-06-30')
H2 = ('2025-07-01', '2025-12-31')


def p(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); OUT.write(s + '\n'); OUT.flush()


def clustered_t(d, col):
    """CRVE by DAY: se = sqrt(sum_d (sum_{i in d} (x_i - xbar))^2) / n."""
    x = d[col].values
    if len(x) < 3:
        return np.nan
    xb = x.mean()
    g = pd.Series(x - xb).groupby(d.day.values).sum().values
    se = np.sqrt((g ** 2).sum()) / len(x)
    return xb / se if se > 0 else np.nan


def mde(x):
    """smallest per-trade effect detectable at 80% power, 5% two-sided."""
    return 2.802 * np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 2 else np.nan


def recost(x):
    """net / netb from the simulated rr and the exit-side cost ratio of THIS exit."""
    half = 0.5 * x.sp_pct / x.r_pct.clip(lower=0.05)
    hb = 0.5 * x.sp_band / x.r_pct.clip(lower=0.05)
    x['net'] = x.rr - half - half * x.ratio
    x['netb'] = x.rr - hb - hb * x.ratio
    return x


def apply_sim(sg, fn):
    """Run one declared exit over a signal frame; returns it with rr/why/ratio/exit_m/net set."""
    x = sg[sg.sid.notna()].copy()
    rr, wy, rt, em = [], [], [], []
    for sid in x.sid.astype(int):
        Q = CACHE[sid]
        k, v, w, r = fn(Q)
        rr.append(v); wy.append(w); rt.append(r); em.append(int(Q['m'][k]))
    x['rr'] = rr; x['why'] = wy; x['ratio'] = rt; x['exit_m'] = em
    return recost(x)


def book(s):
    if not len(s):
        return s.assign(pnl=pd.Series(dtype=float))
    from trading.hod_break import run_book
    rows = [(r.day, int(r.entry_m), int(r.exit_m), r.symbol, r.Index) for r in s.itertuples()]
    b = s.loc[[t[4] for t in run_book(rows, 12, 4)]].copy()
    b['pnl'] = b.net * S.RISK
    return b


def emit(name, base, fixed, ship, rebook, note=''):
    """One cell row: the fixed-cohort trade-off + the re-booked week shape."""
    rec = dict(cell=name, base=base, note=note)
    for sp in ('TRAIN', 'H1', 'H2', 'VAL'):
        if sp == 'H1':
            f = fixed[(fixed.day >= H1[0]) & (fixed.day <= H1[1])]
            sh = ship[(ship.day >= H1[0]) & (ship.day <= H1[1])]
        elif sp == 'H2':
            f = fixed[(fixed.day >= H2[0]) & (fixed.day <= H2[1])]
            sh = ship[(ship.day >= H2[0]) & (ship.day <= H2[1])]
        else:
            f = fixed[fixed.split == sp]; sh = ship[ship.split == sp]
        d = f[['day', 'net', 'rr']].copy()
        d['net0'] = sh.net.values; d['rr0'] = sh.rr.values
        d['dn'] = d.net - d.net0
        lost = d[d.rr < d.rr0 - 1e-12]; cut = d[d.rr > d.rr0 + 1e-12]
        rec[f'{sp}_n'] = len(d)
        rec[f'{sp}_dnet'] = float(d.dn.mean()) if len(d) else np.nan
        rec[f'{sp}_net'] = float(d.net.mean()) if len(d) else np.nan
        rec[f'{sp}_gross'] = float(d.rr.mean()) if len(d) else np.nan
        rec[f'{sp}_cost'] = float((d.rr - d.net).mean()) if len(d) else np.nan
        rec[f'{sp}_t'] = float(d.dn.mean() / (d.dn.std(ddof=1) / np.sqrt(len(d)))) \
            if len(d) > 2 and d.dn.std(ddof=1) > 0 else np.nan
        rec[f'{sp}_tc'] = float(clustered_t(d, 'dn')) if len(d) > 2 else np.nan
        rec[f'{sp}_lost_n'] = len(lost); rec[f'{sp}_lost_R'] = float((lost.rr0 - lost.rr).sum())
        rec[f'{sp}_cut_n'] = len(cut); rec[f'{sp}_cut_R'] = float((cut.rr - cut.rr0).sum())
        rec[f'{sp}_mde'] = float(mde(d.dn.values)) if len(d) > 2 else np.nan
        if sp in ('TRAIN', 'VAL'):
            w = S.week_stats(rebook, sp)
            for k in ('n', 'per_wk', 'gross', 'net', 'green', 'redstreak', 'worst', 'total', 'mdd'):
                rec[f'{sp}_bk_{k}'] = w[k]
    ROWS.append(rec)
    p(f'{name:30s} {base:3s} | TRAIN dNet {rec["TRAIN_dnet"]:+.4f} (H1 {rec["H1_dnet"]:+.4f} '
      f'H2 {rec["H2_dnet"]:+.4f}) t {rec["TRAIN_t"]:+5.2f} tc {rec["TRAIN_tc"]:+5.2f} | '
      f'VAL dNet {rec["VAL_dnet"]:+.4f} tc {rec["VAL_tc"]:+5.2f} | '
      f'lost {rec["TRAIN_lost_n"]:4d}/-{rec["TRAIN_lost_R"]:6.1f}R cut {rec["TRAIN_cut_n"]:4d}/'
      f'+{rec["TRAIN_cut_R"]:6.1f}R | book TRAIN {rec["TRAIN_bk_n"]:5.0f} '
      f'{rec["TRAIN_bk_per_wk"]:5.1f}/wk net {rec["TRAIN_bk_net"]:+.3f} grn '
      f'{rec["TRAIN_bk_green"]:5.1f}% ${rec["TRAIN_bk_total"]:+8,.0f} | VAL {rec["VAL_bk_n"]:5.0f} '
      f'net {rec["VAL_bk_net"]:+.3f} grn {rec["VAL_bk_green"]:5.1f}% ${rec["VAL_bk_total"]:+8,.0f} '
      f'{note}')
    return rec


def main():
    ADD = json.load(open(f'{D}/addendum.json'))
    E1 = [tuple(x) for x in ADD['E1']]
    ASTAR, SSTAR = ADD['a_star'], ADD['s_star']
    pop = S2.load_pop(); S.build_impute(pop)
    keys = pd.read_csv(f'{D}/keys.csv', **C.RD)
    global CACHE
    P = C.Paths()
    for sid in P.sig.index:
        if P.has(int(sid)):
            CACHE[int(sid)] = C.prep(P.get(int(sid)))
    p(f'paths {len(CACHE)} signals')

    def attach(sg):
        k = keys.copy(); sg = sg.copy()
        sg['_st'] = sg.stop.round(6); k['_st'] = k.stop.round(6)
        return sg.merge(k[['day', 'symbol', 'entry_m', '_st', 'sid']],
                        on=['day', 'symbol', 'entry_m', '_st'], how='left').reset_index(drop=True)

    SG = {}
    for nm in ('B0', 'B2'):
        sg = S2.sig_set(pop, **S2.BASES[nm])
        SG[nm] = attach(sg[sg.split.isin(('TRAIN', 'VAL'))])
    del pop
    import score4 as S4
    p4 = load_pop4()
    c1 = S4.sig_set4(p4, rung='ge20', stop='n5')
    c1 = c1[c1.split.isin(('TRAIN', 'VAL'))]
    c1 = c1[(c1.spy_r5_pct > 0).fillna(False)]
    SG['C1'] = attach(c1)
    del p4, c1

    SHIP, SHIPBK = {}, {}
    p('=' * 190)
    p('REFERENCE ROWS — the shipped exit, re-simulated (the parity gate is part1.txt)')
    p('=' * 190)
    for nm in ('B0', 'B2', 'C1'):
        s = apply_sim(SG[nm], C.sim_base)
        b = book(s)
        SHIP[nm] = s; SHIPBK[nm] = b
        for sp in ('TRAIN', 'VAL'):
            w = S.week_stats(b, sp)
            p(f'  {nm} shipped {sp:5s} n{w["n"]:5d} {w["per_wk"]:5.1f}/wk gross {w["gross"]:+.3f} '
              f'net {w["net"]:+.3f} cost/R {w["gross"] - w["net"]:.4f} green {w["green"]:5.1f}% '
              f'rs{w["redstreak"]:2d} worst ${w["worst"]:+,.0f} ${w["total"]:+,.0f}')

    def run(name, nm, fn, note=''):
        """fixed-cohort = the shipped book's own trades under the new exit; re-booked = run_book."""
        idx = SHIPBK[nm].index
        fixed = apply_sim(SG[nm].loc[idx], fn)
        shipc = SHIP[nm].loc[idx]
        reb = book(apply_sim(SG[nm], fn))
        return emit(name, nm, fixed, shipc, reb, note)

    # ---------------------------------------------------------------- E1
    p('\n' + '=' * 190)
    p(f'E1 — THE RATCHET.  {len(E1)} cells, the top-{len(E1)} (a, s) pairs by TRAIN dR '
      f'(ADDENDUM.md, committed before this ran)')
    p('=' * 190)
    for a, s in E1:
        run(f'E1 a={a:.1f} s={s:+.1f}', 'B0', lambda Q, a=a, s=s: C.sim_ratchet(Q, a, s))

    # ---------------------------------------------------------------- E2
    p('\n' + '=' * 190)
    p('E2 — THE TIME-CONDITIONED RATCHET (arm only if a is reached within N minutes)')
    p('=' * 190)
    for (a, s) in E1[:2]:
        for N in (6, 10):
            run(f'E2 a={a:.1f} s={s:+.1f} N={N}', 'B0',
                lambda Q, a=a, s=s, N=N: C.sim_ratchet(Q, a, s, n_max=N))

    # ---------------------------------------------------------------- E3
    p('\n' + '=' * 190)
    p('E3 — RETRACE FROM PEAK (armed at a*, exit at the next open when close <= MFE - d)')
    p('=' * 190)
    for d_ in (0.3, 0.4, 0.5):
        run(f'E3 a={ASTAR:.1f} d={d_:.1f}', 'B0', lambda Q, d_=d_: C.sim_peak(Q, ASTAR, d_))
    run(f'E3 a=0.3 d=0.4', 'B0', lambda Q: C.sim_peak(Q, 0.3, 0.4))

    # ---------------------------------------------------------------- E4
    p('\n' + '=' * 190)
    p(f'E4 — THE OWNER\'S SIGNALS as exits on a trade armed at a={ASTAR:.1f}')
    p('=' * 190)
    TRIG = [('E4a vol 2x + down close', 'vol'), ('E4b vol fade 3 bars', 'fade'),
            ('E4c close < session VWAP', 'vwap'), ('E4d MACD hist < 0', 'hist'),
            ('E4e MACD signal cross', 'cross'), ('E4f any 2 of 3', 'two3')]
    for nm_, key in TRIG:
        run(nm_, 'B0', lambda Q, key=key: C.sim_trigger(Q, key, ASTAR))

    # ---------------------------------------------------------------- E5
    df = pd.DataFrame(ROWS)
    e4 = df[df.cell.str.startswith('E4')].sort_values(['TRAIN_dnet', 'VAL_dnet'], ascending=False)
    best_t = dict(TRIG)[e4.iloc[0].cell]
    p('\n' + '=' * 190)
    p(f'E5 — THE COMBINATION: ratchet a={ASTAR:.1f} s={SSTAR:+.1f} + the best E4 trigger '
      f'({e4.iloc[0].cell})')
    p('=' * 190)
    for nm in ('B0', 'B2'):
        run(f'E5 ratchet+{best_t}', nm, lambda Q: C.sim_combo(Q, ASTAR, SSTAR, best_t))

    # ---------------------------------------------------------------- E6
    p('\n' + '=' * 190)
    p('E6 — THE PARTIAL, STOP UNMOVED.  50% off at p R; the runner keeps the ORIGINAL stop')
    p('=' * 190)
    for pp in (0.3, 0.4, 0.5):
        run(f'E6 partial 50% @ {pp:.1f}R', 'B0', lambda Q, pp=pp: C.sim_partial(Q, pp))

    # ---------------------------------------------------------------- E7
    df = pd.DataFrame(ROWS)
    cand = df[df.cell.str.startswith(('E1', 'E2', 'E3', 'E4', 'E5', 'E6'))]
    best = cand.sort_values(['TRAIN_dnet', 'VAL_dnet'], ascending=False).iloc[0]
    p('\n' + '=' * 190)
    p(f'E7 — the best exit of E1-E6 ({best.cell}) on hod_fresh C1 '
      f'(consol_bars>=20 x last-5-bar stop x spy_r5>0)')
    p('=' * 190)
    FN = {}
    for a, s in E1:
        FN[f'E1 a={a:.1f} s={s:+.1f}'] = (lambda a=a, s=s: (lambda Q: C.sim_ratchet(Q, a, s)))()
    for (a, s) in E1[:2]:
        for N in (6, 10):
            FN[f'E2 a={a:.1f} s={s:+.1f} N={N}'] = \
                (lambda a=a, s=s, N=N: (lambda Q: C.sim_ratchet(Q, a, s, n_max=N)))()
    for d_ in (0.3, 0.4, 0.5):
        FN[f'E3 a={ASTAR:.1f} d={d_:.1f}'] = (lambda d_=d_: (lambda Q: C.sim_peak(Q, ASTAR, d_)))()
    FN['E3 a=0.3 d=0.4'] = lambda Q: C.sim_peak(Q, 0.3, 0.4)
    for nm_, key in TRIG:
        FN[nm_] = (lambda key=key: (lambda Q: C.sim_trigger(Q, key, ASTAR)))()
    FN[f'E5 ratchet+{best_t}'] = lambda Q: C.sim_combo(Q, ASTAR, SSTAR, best_t)
    for pp in (0.3, 0.4, 0.5):
        FN[f'E6 partial 50% @ {pp:.1f}R'] = (lambda pp=pp: (lambda Q: C.sim_partial(Q, pp)))()
    run(f'E7 {best.cell} on C1', 'C1', FN[best.cell], 'the best exit x the only net-positive admission')

    # ---------------------------------------------------------------- nulls on the top cells
    p('\n' + '=' * 190)
    p('COUNT-MATCHED PERMUTATION NULL (2,000 draws, per-week pick count fixed) — top cells')
    p('=' * 190)
    df = pd.DataFrame(ROWS)
    top = df[df.cell.str.startswith(('E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7'))] \
        .sort_values('TRAIN_dnet', ascending=False).head(6)
    for r in top.itertuples():
        fn = FN.get(r.cell) or (FN[best.cell] if r.cell.startswith('E7') else None)
        nm = r.base
        if fn is None:
            continue
        reb = book(apply_sim(SG[nm], fn))
        for sp in ('TRAIN', 'VAL'):
            o, mu, lo, hi = S.null_band(reb, sp)
            tag = 'ABOVE' if o > hi else ('below' if o < lo else 'inside')
            p(f'  {r.cell:30s} {nm} {sp:5s} observed green {o:5.1f}%  null mean {mu:5.1f}% '
              f'[{lo:.1f}, {hi:.1f}] -> {tag}')
            NULLS.append(dict(cell=r.cell, base=nm, split=sp, obs=o, mu=mu, p5=lo, p95=hi))
    for nm in ('B0', 'B2', 'C1'):
        for sp in ('TRAIN', 'VAL'):
            o, mu, lo, hi = S.null_band(SHIPBK[nm], sp)
            p(f'  {nm + " shipped":30s} {nm} {sp:5s} observed green {o:5.1f}%  null mean {mu:5.1f}% '
              f'[{lo:.1f}, {hi:.1f}]')

    pd.DataFrame(ROWS).to_csv(f'{D}/cells.csv', index=False)
    pd.DataFrame(NULLS).to_csv(f'{D}/nulls.csv', index=False)

    # ---------------------------------------------------------------- the ship bar
    p('\n' + '=' * 190)
    p('THE SHIP BAR (PREREG §6): TRAIN dNet >= +0.10 R with both halves same-signed, VAL dNet > 0, '
      'TRAIN day-clustered t >= 2.0, green weeks not worse on both splits')
    p('=' * 190)
    d = pd.DataFrame(ROWS)
    d = d[~d.cell.str.contains('shipped')]
    g0 = {nm: {sp: S.week_stats(SHIPBK[nm], sp)['green'] for sp in ('TRAIN', 'VAL')}
          for nm in ('B0', 'B2', 'C1')}
    ok = []
    for r in d.itertuples():
        b1 = r.TRAIN_dnet >= 0.10
        b2 = (r.H1_dnet > 0) == (r.H2_dnet > 0)
        b3 = r.VAL_dnet > 0
        b4 = (r.TRAIN_tc >= 2.0)
        b5 = (r.TRAIN_bk_green >= g0[r.base]['TRAIN'] - 1e-9) and \
             (r.VAL_bk_green >= g0[r.base]['VAL'] - 1e-9)
        if b1 and b2 and b3 and b4 and b5:
            ok.append(r.cell)
    p(f'  cells passing all four: {len(ok)} of {len(d)}   {ok}')
    bb = d.sort_values('TRAIN_dnet', ascending=False).iloc[0]
    p(f'  best TRAIN dNet: {bb.cell} {bb.base}  {bb.TRAIN_dnet:+.4f} '
      f'(H1 {bb.H1_dnet:+.4f} / H2 {bb.H2_dnet:+.4f} / VAL {bb.VAL_dnet:+.4f}), '
      f'clustered t {bb.TRAIN_tc:+.2f} / {bb.VAL_tc:+.2f}, MDE {bb.TRAIN_mde:.4f} / {bb.VAL_mde:.4f}')
    p(f'  the book under it: TRAIN net {bb.TRAIN_bk_net:+.4f} (shipped '
      f'{S.week_stats(SHIPBK[bb.base], "TRAIN")["net"]:+.4f}), VAL net {bb.VAL_bk_net:+.4f} '
      f'(shipped {S.week_stats(SHIPBK[bb.base], "VAL")["net"]:+.4f})')
    OUT.close()


if __name__ == '__main__':
    main()
