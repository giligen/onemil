#!/usr/bin/env python3
"""hod_frames3 / F12 — ADAPTIVE vs FROZEN: refit the admission on a rolling window, week by week.

Procedure EXACTLY as declared in PREREG.md §2 (commit b4b8171, before any OOS week was booked):
the refit sees only the L complete weeks strictly before the week it trades (asserted in code), it
refits ONLY the hour band (contiguous entry-minute decile range, width >= 4) plus ONE ranked feature
cut from the declared short list {rv_profile, dollar_frac, consol_bars, spy_r5_pct}, and the next
week is booked strictly out of sample with the shipped 12/4 book at $100 risk.

TEST is sealed: no OOS week is dated after 2026-05-31.  Read-only.  One process.
"""
import os, sys
import numpy as np, pandas as pd

sys.path.insert(0, '/home/ec2-user/onemil/research/mature_method/hod_frames3')
from common3 import (ROOT, D, S, S2, clustered_t, sigset, load_breaks, admit)   # noqa: E402

FEATS = ('rv_profile', 'dollar_frac', 'consol_bars', 'spy_r5_pct')
ARMS = (20, 26, 34)
MIN_TRAINW = 30
KEEP_FRAC = 0.40


def band_candidates(trw):
    """Contiguous entry-minute decile ranges of width >= 4, plus the full range."""
    q = [float(trw.entry_m.quantile(k / 10.0)) for k in range(11)]
    out = [(None, None)]
    for a in range(0, 7):
        for b in range(a + 3, 10):
            lo, hi = q[a], q[b + 1]
            if hi > lo:
                out.append((lo, hi))
    return out


def band_mask(d, lo, hi):
    if lo is None:
        return pd.Series(True, index=d.index)
    return (d.entry_m >= lo) & (d.entry_m <= hi)


def cut_candidates(trw):
    """{>=p33, >=p67} per declared feature (+ spy_r5>0), plus the null 'no cut'."""
    out = [(None, None)]
    for f in FEATS:
        v = trw[f].dropna()
        if len(v) < MIN_TRAINW:
            continue
        for p in (33, 67):
            out.append((f, float(v.quantile(p / 100.0))))
        if f == 'spy_r5_pct':
            out.append((f, 0.0))
    return out


def cut_mask(d, f, thr):
    if f is None:
        return pd.Series(True, index=d.index)
    return d[f] >= thr


def refit(trw):
    """The declared refit: best band by mean net R, then best admissible cut."""
    best_b, best_v = (None, None), -9e9
    for (lo, hi) in band_candidates(trw):
        m = band_mask(trw, lo, hi)
        if int(m.sum()) < MIN_TRAINW:
            continue
        v = float(trw.net[m.values].mean())
        if v > best_v:
            best_b, best_v = (lo, hi), v
    tb = trw[band_mask(trw, *best_b).values]
    need = KEEP_FRAC * len(tb)
    best_c, best_cv = (None, None), -9e9
    for (f, thr) in cut_candidates(tb):
        m = cut_mask(tb, f, thr)
        if int(m.fillna(False).sum()) < max(need, MIN_TRAINW):
            continue
        v = float(tb.net[m.fillna(False).values].mean())
        if v > best_cv:
            best_c, best_cv = (f, thr), v
    return best_b, best_c


def week_pnl(sig):
    b = S.apply_book(sig, 12, 4)
    if not len(b):
        return 0.0, 0, b
    return float(b.pnl.sum()), len(b), b


def stats(weeks, pnl, ntr, label, days=None, nets=None):
    w = pd.Series(pnl, index=weeks)
    streak = mx = 0
    for v in (w < 0).values:
        streak = streak + 1 if v else 0
        mx = max(mx, streak)
    cum = w.cumsum()
    d = dict(cell=label, weeks=len(w), green=float((w > 0).mean() * 100), redstreak=mx,
             worst=float(w.min()), total=float(w.sum()), mdd=float((cum - cum.cummax()).min()),
             per_wk=float(np.sum(ntr) / len(w)), n=int(np.sum(ntr)))
    if nets is not None and len(nets):
        nn = pd.DataFrame(dict(net=nets[0], day=nets[1]))
        d['t'] = float(nn.net.mean() / (nn.net.std(ddof=1) / np.sqrt(len(nn)))) if len(nn) > 2 else np.nan
        d['tc'] = clustered_t(nn)
    return d


def main():
    print('== hod_frames3 / F12 — ADAPTIVE vs FROZEN ==\n', flush=True)
    pop = S2.load_pop(); S.build_impute(pop)
    b2 = S.apply_book(S2.sig_set(pop, **S2.BASES['B2']), 12, 4)
    for sp in S.SPLITS:
        w = S.week_stats(b2, sp)
        print(f'   R1/R2 B2 {sp}: n {w["n"]} /wk {w["per_wk"]:.1f} gross {w["gross"]:+.3f} '
              f'net {w["net"]:+.3f} green {w["green"]:.1f}% total ${w["total"]:,.0f}')

    br = load_breaks()
    base = sigset(admit(br, pd.Series(True, index=br.index))).copy()
    base = base[base.day <= '2026-05-31']
    weeks, seen = [], set()                                          # EVERY market week, once
    for w in list(S.ALL_WEEKS['TRAIN']) + list(S.ALL_WEEKS['VAL']):
        if w not in seen:
            seen.add(w); weeks.append(w)
    print(f'\npre-book signals {len(base)} over {len(weeks)} market weeks '
          f'[{base.day.min()} .. {base.day.max()}]   (TEST sealed)')

    rows, detail = [], []
    for L in ARMS:
        oos = weeks[L:]
        log, booked = [], []
        for k, w in enumerate(oos, start=L):
            trw = base[base.wk.isin(weeks[k - L:k])]
            cur = base[base.wk == w]
            if len(trw) < MIN_TRAINW or not len(cur):
                log.append(dict(wk=w, band=(None, None), cut=(None, None), pnl=0.0, n=0,
                                fpnl=0.0, fn=0))
                continue
            assert trw.day.max() < cur.day.min(), f'LEAKAGE at {w}'
            bnd, cut = refit(trw)
            m = band_mask(cur, *bnd) & cut_mask(cur, *cut).fillna(False)
            p, n, bk = week_pnl(cur[m.values])
            fp, fn, _ = week_pnl(cur)
            if len(bk):
                booked.append(bk[['day', 'net']])
            log.append(dict(wk=w, band=bnd, cut=cut, pnl=p, n=n, fpnl=fp, fn=fn))
        lg = pd.DataFrame(log)
        bkall = pd.concat(booked) if booked else pd.DataFrame(columns=['day', 'net'])
        churn = float(np.mean([lg.band.iloc[i] != lg.band.iloc[i - 1] or
                               lg.cut.iloc[i] != lg.cut.iloc[i - 1]
                               for i in range(1, len(lg))]) * 100)
        chg = np.array([False] + [lg.band.iloc[i] != lg.band.iloc[i - 1] or
                                  lg.cut.iloc[i] != lg.cut.iloc[i - 1] for i in range(1, len(lg))])
        rows.append(dict(**stats(lg.wk, lg.pnl.values, lg.n.values, f'F12-r{L} REFIT'),
                         churn=churn,
                         net=float(bkall.net.mean()) if len(bkall) else np.nan,
                         tc=clustered_t(bkall) if len(bkall) > 3 else np.nan,
                         t=float(bkall.net.mean() / (bkall.net.std(ddof=1) /
                                                     np.sqrt(len(bkall)))) if len(bkall) > 3 else np.nan,
                         after_change=float(lg.pnl.values[chg].mean()) if chg.any() else np.nan,
                         after_same=float(lg.pnl.values[~chg].mean()) if (~chg).any() else np.nan))
        fz = S.apply_book(base[base.wk.isin(oos)], 12, 4)
        rows.append(dict(**stats(lg.wk, lg.fpnl.values, lg.fn.values, f'F12-f FROZEN (L={L} weeks)'),
                         churn=0.0, after_change=np.nan, after_same=np.nan,
                         net=float(fz.net.mean()), tc=clustered_t(fz),
                         t=float(fz.net.mean() / (fz.net.std(ddof=1) / np.sqrt(len(fz))))))
        lg['L'] = L
        detail.append(lg)
        # ---- the halves rail (OOS weeks inside TRAIN vs inside VAL) ------------------------
        lg['yr'] = lg.wk.str[:4]
        h2 = lg[lg.wk < '2026-01-01']; va = lg[lg.wk >= '2026-01-01']
        print(f'\n-- L={L}: {len(oos)} OOS weeks [{oos[0]} .. {oos[-1]}] | churn {churn:.0f}% --')
        for lab, x in (('OOS-in-TRAIN (H2-2025)', h2), ('OOS-in-VAL', va)):
            if not len(x):
                continue
            print(f'   {lab:<24s} refit ${x.pnl.sum():+9,.0f} green {(x.pnl>0).mean()*100:5.1f}% | '
                  f'frozen ${x.fpnl.sum():+9,.0f} green {(x.fpnl>0).mean()*100:5.1f}% | '
                  f'beats frozen: {x.pnl.sum() > x.fpnl.sum() and (x.pnl>0).mean() >= (x.fpnl>0).mean()}')
        # ---- the SIZING-refit control ------------------------------------------------------
        if L in (26, 34):
            slog = []
            for k, w in enumerate(oos, start=L):
                trw = base[base.wk.isin(weeks[k - L:k])]
                cur = base[base.wk == w]
                if len(trw) < MIN_TRAINW or not len(cur):
                    slog.append(dict(wk=w, pnl=0.0, n=0)); continue
                q1, q2 = trw.entry_m.quantile(1 / 3.), trw.entry_m.quantile(2 / 3.)
                terc = pd.cut(trw.entry_m, [-np.inf, q1, q2, np.inf], labels=[0, 1, 2])
                mr = trw.groupby(terc, observed=True).net.mean().sort_values()
                mult = {int(mr.index[-1]): 1.5, int(mr.index[0]): 0.5}
                for t in (0, 1, 2):
                    mult.setdefault(t, 1.0)
                b = S.apply_book(cur, 12, 4)
                if not len(b):
                    slog.append(dict(wk=w, pnl=0.0, n=0)); continue
                ct = pd.cut(b.entry_m, [-np.inf, q1, q2, np.inf], labels=[0, 1, 2]).astype(int)
                slog.append(dict(wk=w, pnl=float((b.net * 100.0 * ct.map(mult)).sum()), n=len(b)))
            sl = pd.DataFrame(slog)
            rows.append(dict(**stats(sl.wk, sl.pnl.values, sl.n.values, f'F12-s{L} SIZING refit'),
                             churn=np.nan, after_change=np.nan, after_same=np.nan,
                             net=np.nan, tc=np.nan, t=np.nan))

    tab = pd.DataFrame(rows)
    tab.to_csv(f'{D}/cells12.csv', index=False)
    pd.concat(detail).to_csv(f'{D}/f12_weeks.csv', index=False)
    print('\n\n== F12 — the OOS books ==')
    print('| cell | OOS weeks | trades | /wk | green % | red streak | worst $ | total $ | MDD $ | '
          'churn % | $ after a change | $ after no change |')
    print('|' + '|'.join(['---'] * 12) + '|')
    for r in rows:
        print(f'| {r["cell"]} | {r["weeks"]} | {r["n"]} | {r["per_wk"]:.1f} | {r["green"]:.1f} | '
              f'{r["redstreak"]} | {r["worst"]:,.0f} | {r["total"]:+,.0f} | {r["mdd"]:,.0f} | '
              f'{r["churn"] if r["churn"]==r["churn"] else float("nan"):.0f} | '
              f'{r["after_change"]:+,.0f} | {r["after_same"]:+,.0f} |')

    print('\n== PREREG READING RULE ==')
    win = []
    for L in ARMS:
        r = tab[tab.cell == f'F12-r{L} REFIT'].iloc[0]
        f = tab[tab.cell == f'F12-f FROZEN (L={L} weeks)'].iloc[0]
        lg = pd.concat(detail); lg = lg[lg.L == L]
        h2 = lg[lg.wk < '2026-01-01']; va = lg[lg.wk >= '2026-01-01']
        both = all(len(x) and x.pnl.sum() > x.fpnl.sum() and
                   (x.pnl > 0).mean() >= (x.fpnl > 0).mean() for x in (h2, va))
        print(f'  L={L}: beats frozen on $ AND green in BOTH OOS halves: {both} | '
              f'>=10 tr/wk: {r.per_wk >= 10} | clustered t {r.tc:+.2f} (>=2: {r.tc >= 2}) | '
              f'refit total ${r.total:+,.0f} vs frozen ${f.total:+,.0f}')
        if both and r.per_wk >= 10 and r.tc >= 2:
            win.append(L)
    print(f'\n  arms satisfying the rule: {win}')
    print('  -> READING (i) REGIME' if win else '  -> READING (ii) NOISE')
    print('\nDONE', flush=True)


if __name__ == '__main__':
    main()
