#!/usr/bin/env python3
"""hod_losers PART 1 — the anatomy of the HOD-break book's losses.

Descriptive only. No cells, no gates, no selection. Reuses the EXACT machinery of
`hod_break/score.py` (cost model, run_book, week stats) and `hod_filter_stack/score2.py`
(base populations B0 / B2) so every number here is the same book the two reports scored.

Read-only on every DB/CSV. One process.
"""
import os, sys
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT); sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_break')
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_filter_stack')
import score as S            # noqa: E402
import score2 as S2          # noqa: E402

D = f'{ROOT}/research/mature_method/hod_losers'
OUT = open(f'{D}/part1.txt', 'w')


def p(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); OUT.write(s + '\n'); OUT.flush()


def lorenz(vals, shares=(0.05, 0.10, 0.20)):
    """Share of the total NEGATIVE mass carried by the worst q of units, and of the
    total POSITIVE mass carried by the best q. Units are the groupby keys."""
    v = np.sort(np.asarray(vals, dtype=float))
    n = len(v)
    neg_tot = -v[v < 0].sum()
    pos_tot = v[v > 0].sum()
    out = {}
    for q in shares:
        k = max(1, int(round(q * n)))
        out[f'worst{int(q*100)}'] = (-v[:k].sum() / neg_tot * 100) if neg_tot > 0 else np.nan
        out[f'best{int(q*100)}'] = (v[-k:].sum() / pos_tot * 100) if pos_tot > 0 else np.nan
    out['n'] = n
    out['neg_units'] = int((v < 0).sum())
    out['tot'] = float(v.sum())
    return out


def main():
    p('=' * 100)
    p('PART 1 — THE ANATOMY.  B0 = shipped population, B2 = corrected base (hod_filter_stack §2)')
    p('=' * 100)

    pop = S2.load_pop()
    S.build_impute(pop.rename(columns={}))   # imputation cells fitted on the same rows score2 uses
    books = {}
    for name, kw in (('B0', S2.BASES['B0']), ('B2', S2.BASES['B2'])):
        sg = S2.sig_set(pop, **kw)
        bk = S.apply_book(sg, 12, 4)
        books[name] = (sg, bk)
        p(f'\n[{name}] pre-book signals {len(sg)}   booked {len(bk)}')
        for sp in S.SPLITS:
            d = bk[bk.split == sp]
            w = S.week_stats(bk, sp)
            p(f'   {sp:5s} n {w["n"]:5d}  /wk {w["per_wk"]:5.1f}  gross {w["gross"]:+.3f}  '
              f'net {w["net"]:+.3f}  green {w["green"]:.1f}%  total ${w["total"]:+,.0f}  '
              f'worst ${w["worst"]:+,.0f}  rs {w["redstreak"]}')

    # ---------------------------------------------------------------- 1. CONCENTRATION
    p('\n' + '=' * 100)
    p('1. CONCENTRATION — Lorenz of net $ (at $100 risk) by DAY / WEEK / SYMBOL / HOUR')
    p('=' * 100)
    p(f'{"base":4s} {"split":5s} {"unit":7s} {"n":>5s} {"neg":>5s} | '
      f'{"worst5%":>8s} {"worst10%":>8s} {"worst20%":>8s} | {"best5%":>7s} {"best10%":>8s} {"best20%":>8s} | {"total$":>9s}')
    for name in ('B0', 'B2'):
        bk = books[name][1]
        for sp in S.SPLITS:
            d = bk[bk.split == sp]
            for unit, key in (('DAY', 'day'), ('WEEK', 'wk'), ('SYMBOL', 'symbol')):
                g = d.groupby(key).pnl.sum()
                L = lorenz(g.values)
                p(f'{name:4s} {sp:5s} {unit:7s} {L["n"]:5d} {L["neg_units"]:5d} | '
                  f'{L["worst5"]:7.1f}% {L["worst10"]:7.1f}% {L["worst20"]:7.1f}% | '
                  f'{L["best5"]:6.1f}% {L["best10"]:7.1f}% {L["best20"]:7.1f}% | {L["tot"]:+9,.0f}')

    p('\n-- per-trade Lorenz (the trade-level view of the same thing) --')
    for name in ('B0', 'B2'):
        bk = books[name][1]
        for sp in S.SPLITS:
            d = bk[bk.split == sp]
            L = lorenz(d.pnl.values)
            p(f'{name} {sp:5s} TRADE   n {L["n"]:5d} neg {L["neg_units"]:5d} | worst5% {L["worst5"]:5.1f}% '
              f'worst10% {L["worst10"]:5.1f}% worst20% {L["worst20"]:5.1f}% | best5% {L["best5"]:5.1f}% '
              f'best10% {L["best10"]:5.1f}% best20% {L["best20"]:5.1f}%')

    p('\n-- by HOUR of entry (net $ total, and mean R) --')
    HB = [569, 585, 600, 660, 720, 840]
    HL = ['0930-0945', '0945-1000', '1000-1100', '1100-1200', '1200-1400']
    for name in ('B0', 'B2'):
        bk = books[name][1].copy()
        bk['hb'] = pd.cut(bk.entry_m, HB, labels=HL)
        for sp in S.SPLITS:
            d = bk[bk.split == sp]
            g = d.groupby('hb', observed=True).agg(n=('pnl', 'size'), tot=('pnl', 'sum'),
                                                   gross=('rr', 'mean'), net=('net', 'mean'))
            p(f'  [{name} {sp}] ' + '  '.join(
                f'{i}: n{int(r.n)} ${r.tot:+,.0f} g{r.gross:+.2f}' for i, r in g.iterrows()))

    # ---------------------------------------------------------------- 2. LOSING DAYS
    p('\n' + '=' * 100)
    p('2. LOSING DAYS — worst-20% vs best-20% of TRADED days')
    p('=' * 100)
    dayctx = pd.read_csv(f'{ROOT}/research/mature_method/hod_filter_stack/day_ctx.csv',
                         dtype={'day': str}, keep_default_na=False, na_values=[''])
    p('day_ctx columns: ' + ', '.join(dayctx.columns))
    blite_n = None
    for name in ('B0',):
        sg, bk = books[name]
        for sp in S.SPLITS:
            d = bk[bk.split == sp]
            g = d.groupby('day').agg(pnl=('pnl', 'sum'), n=('pnl', 'size'),
                                     wins=('rr', lambda x: (x > 0).sum()),
                                     grossR=('rr', 'mean'))
            g['stopfrac'] = d.groupby('day').why.apply(lambda x: (x == 'stop').mean())
            g['med_hold'] = d.groupby('day').apply(lambda x: (x.exit_m - x.entry_m).median(),
                                                   include_groups=False)
            g = g.sort_values('pnl')
            k = max(1, int(round(0.20 * len(g))))
            worst, best = g.iloc[:k], g.iloc[-k:]
            mid = g.iloc[k:-k]
            gg = g.join(dayctx.set_index('day'), how='left')
            wc, bc = gg.loc[worst.index], gg.loc[best.index]
            p(f'\n[{name} {sp}] traded days {len(g)}  worst20% k={k}')
            p(f'   {"":22s} {"WORST20%":>12s} {"MID":>12s} {"BEST20%":>12s}')
            rows = [('net $ total', worst.pnl.sum(), mid.pnl.sum(), best.pnl.sum()),
                    ('net $ / day', worst.pnl.mean(), mid.pnl.mean(), best.pnl.mean()),
                    ('trades / day', worst.n.mean(), mid.n.mean(), best.n.mean()),
                    ('gross R / trade', worst.grossR.mean(), mid.grossR.mean(), best.grossR.mean()),
                    ('stop fraction', worst.stopfrac.mean(), mid.stopfrac.mean(), best.stopfrac.mean()),
                    ('median hold (min)', worst.med_hold.median(), mid.med_hold.median(), best.med_hold.median())]
            for lbl, a, b, c in rows:
                p(f'   {lbl:22s} {a:12,.2f} {b:12,.2f} {c:12,.2f}')
            for col in dayctx.columns:
                if col == 'day' or not np.issubdtype(gg[col].dtype, np.number):
                    continue
                p(f'   {col:22s} {wc[col].median():12.4f} {gg.loc[mid.index, col].median():12.4f} '
                  f'{bc[col].median():12.4f}')

            # within-day correlation of outcomes: mean pairwise agreement of win/loss
            def agree(x):
                w = (x > 0).values.astype(float)
                if len(w) < 2:
                    return np.nan
                m = w.mean()
                # intraclass: variance of the day mean vs binomial expectation
                return m
            dd = d.copy(); dd['win'] = (dd.rr > 0).astype(float)
            per = dd.groupby('day').win.agg(['mean', 'size'])
            per = per[per['size'] >= 3]
            pbar = dd.win.mean()
            obs_var = float(np.average((per['mean'] - pbar) ** 2, weights=per['size']))
            exp_var = float(np.average(pbar * (1 - pbar) / per['size'], weights=per['size']))
            rho = (obs_var - exp_var) / max(pbar * (1 - pbar), 1e-9)
            p(f'   within-day win-rate ICC (rho)  {rho:+.4f}   '
              f'(0 = trades independent; >0 = outcomes cluster by day)   days>=3tr {len(per)}')
            # same on worst / best days only
            for lbl, idx in (('worst20%', worst.index), ('best20%', best.index)):
                sub = dd[dd.day.isin(idx)]
                pp = sub.groupby('day').win.agg(['mean', 'size'])
                pp = pp[pp['size'] >= 3]
                if len(pp) < 3:
                    continue
                pb = sub.win.mean()
                ov = float(np.average((pp['mean'] - pb) ** 2, weights=pp['size']))
                ev = float(np.average(pb * (1 - pb) / pp['size'], weights=pp['size']))
                p(f'      {lbl:9s} WR {pb:.3f}  rho {(ov - ev) / max(pb * (1 - pb), 1e-9):+.4f}')

            # do the failures cluster in one 30-min window on losing days?
            for lbl, idx in (('worst20%', worst.index), ('best20%', best.index)):
                sub = d[d.day.isin(idx)].copy()
                sub['w30'] = (sub.entry_m // 30) * 30
                per_day_top = sub.groupby(['day', 'w30']).pnl.sum().groupby('day').min()
                tot = sub.groupby('day').pnl.sum()
                share = (per_day_top / tot.replace(0, np.nan)).clip(-5, 5)
                # concentration of the day's loss in its worst 30-min bucket
                nb = sub.groupby('day').w30.nunique()
                p(f'      {lbl:9s} distinct 30-min entry buckets/day median {nb.median():.1f}; '
                  f'worst-bucket share of day pnl median {share.median():.2f}')
            # exit clustering: do stops on losing days land in the same window?
            for lbl, idx in (('worst20%', worst.index), ('best20%', best.index)):
                sub = d[d.day.isin(idx) & (d.why == 'stop')].copy()
                if not len(sub):
                    continue
                sub['x30'] = (sub.exit_m // 30) * 30
                cnt = sub.groupby(['day', 'x30']).size().groupby('day').max()
                tot = sub.groupby('day').size()
                p(f'      {lbl:9s} stops/day {tot.mean():.1f}; largest single 30-min EXIT bucket holds '
                  f'{(cnt / tot).mean() * 100:.0f}% of the day\'s stops')

    # ---------------------------------------------------------------- 3. LOSING WEEKS
    p('\n' + '=' * 100)
    p('3. LOSING WEEKS — shape and red-streak structure')
    p('=' * 100)
    for name in ('B0', 'B2'):
        bk = books[name][1]
        for sp in S.SPLITS:
            d = bk[bk.split == sp]
            wk = S.ALL_WEEKS[sp]
            w = d.groupby('wk').pnl.sum().reindex(wk).fillna(0.0)
            n = d.groupby('wk').size().reindex(wk).fillna(0)
            sgn = (w > 0).astype(int).values
            # runs
            runs, cur = [], 1
            for i in range(1, len(sgn)):
                if sgn[i] == sgn[i - 1]:
                    cur += 1
                else:
                    runs.append((sgn[i - 1], cur)); cur = 1
            runs.append((sgn[-1], cur))
            red = [c for s_, c in runs if s_ == 0]
            grn = [c for s_, c in runs if s_ == 1]
            # runs test: expected runs under independence
            n1, n0 = int(sgn.sum()), int((1 - sgn).sum())
            nr = len(runs)
            mu = 2 * n1 * n0 / (n1 + n0) + 1 if (n1 and n0) else np.nan
            sd = np.sqrt(2 * n1 * n0 * (2 * n1 * n0 - n1 - n0) /
                         ((n1 + n0) ** 2 * (n1 + n0 - 1))) if (n1 > 1 and n0 > 1) else np.nan
            z = (nr - mu) / sd if sd == sd and sd > 0 else np.nan
            lo = lorenz(w.values)
            p(f'\n[{name} {sp}] weeks {len(w)} green {n1} red {n0}; runs {nr} vs E[{mu:.1f}] z={z:+.2f} '
              f'({"runs<E => streaky/regime" if z < -1 else "runs~E => noise/alternating"})')
            p(f'   red runs {sorted(red, reverse=True)}   green runs {sorted(grn, reverse=True)}')
            p(f'   worst5% of weeks carry {lo["worst5"]:.1f}% of all red-week $, worst10% {lo["worst10"]:.1f}%, '
              f'worst20% {lo["worst20"]:.1f}%; best20% carry {lo["best20"]:.1f}% of green-week $')
            # autocorrelation of weekly pnl and of the sign
            if len(w) > 4:
                a1 = float(pd.Series(w.values).autocorr(1))
                s1 = float(pd.Series(sgn.astype(float)).autocorr(1))
                p(f'   weekly $ lag-1 autocorr {a1:+.3f}; week-SIGN lag-1 autocorr {s1:+.3f}')

    # ---------------------------------------------------------------- 5/6 cheap parts
    p('\n' + '=' * 100)
    p('5/6 (bar-free part) — the LEVEL and the STOCK CONTEXT, on fields already causal in pop.csv')
    p('=' * 100)
    for name in ('B0', 'B2'):
        sg = books[name][0].copy()
        sg['dist_atr'] = sg.dist_open_pct / sg.atr14_pct.replace(0, np.nan)
        sg['near_round'] = (np.minimum(sg.level % 1.0, 1.0 - (sg.level % 1.0)) <= 0.05).astype(int)
        sg['at_20d_high'] = (sg.dist_20d_high_pct >= 0).astype(int)
        sg['rebreak'] = (sg.n_break > 0).astype(int)
        sg['hod_is_open'] = (sg.hod_age_bars >= 900).astype(int)
        for f, lbl in (('rebreak', 'FIRST break of day (0) vs RE-break (1)'),
                       ('near_round', 'level within 5c of a round dollar'),
                       ('at_20d_high', 'level at/above the 20-day high'),
                       ('is_wrapper', 'leveraged/inverse wrapper'),
                       ('above_vwap', 'level above session VWAP')):
            if f not in sg.columns:
                continue
            p(f'\n  [{name}] {lbl}')
            for sp in S.SPLITS:
                d = sg[sg.split == sp]
                g = d.groupby(f).agg(n=('rr', 'size'), gross=('rr', 'mean'), net=('net', 'mean'),
                                     wr=('rr', lambda x: (x > 0).mean() * 100))
                p(f'     {sp:5s} ' + '   '.join(
                    f'{int(i)}: n{int(r.n)} gross{r.gross:+.3f} net{r.net:+.3f} wr{r.wr:.0f}%'
                    for i, r in g.iterrows()))
        for f in ('dist_atr', 'hod_age_bars', 'dist_20d_high_pct', 'n_break'):
            p(f'\n  [{name}] {f} — TRAIN-cut quintiles, gross R both splits')
            tr = sg[sg.split == 'TRAIN']
            v = pd.to_numeric(tr[f], errors='coerce')
            q = np.unique(v.quantile([.2, .4, .6, .8]).dropna().values)
            if len(q) < 2:
                p('     degenerate'); continue
            for sp in S.SPLITS:
                d = sg[sg.split == sp]
                b = np.digitize(pd.to_numeric(d[f], errors='coerce').fillna(-1e18), q)
                g = pd.DataFrame(dict(b=b, rr=d.rr.values, net=d.net.values)).groupby('b').agg(
                    n=('rr', 'size'), gross=('rr', 'mean'), net=('net', 'mean'))
                p(f'     {sp:5s} edges {np.round(q, 3)} | ' + '  '.join(
                    f'Q{int(i)} n{int(r.n)} {r.gross:+.3f}' for i, r in g.iterrows()))

    OUT.close()


if __name__ == '__main__':
    main()
