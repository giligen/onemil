#!/usr/bin/env python3
"""PIT survivorship re-run for F4-LO and F3-LO-12-1 on a DELISTING-INCLUSIVE Nasdaq tape.

REPORT_F2_A1.md named this the blocker: the Alpaca panel is 100% survivors (3,806 of 3,807 scored
symbols still quoted in 2026) and `delisted_names.parquet` has zero overlap with `universe.parquet`.
This script answers the only question that matters -- HOW MUCH of the long-only point estimate is the
missing cohort -- by running ONE code path over ONE tape twice:

    (a) PIT      : every symbol the XNAS.ITCH daily tape shows trading that month, delisted included
    (b) SURVIVOR : the same tape restricted to symbols Alpaca still lists today

The difference is the survivorship bias, measured rather than argued.

The tape is UNADJUSTED, so a split detector is built and VALIDATED against the Alpaca panel, where
the true cumulative factor is known (close_adj/close_raw). A detector that cannot reproduce the truth
on the survivors may not be trusted on the delisted names, and the validation share is reported.

Window 2018-05 -> 2023-12 (the tape's coverage, truncated at the VAL end -- TEST stays sealed).
"""
import json
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

D = '/home/ec2-user/onemil/research/multiday/data'
OUT = '/home/ec2-user/onemil/research/multiday/out_pit'
XNAS = ['/home/ec2-user/onemil/research/fuckup_audit/N_databento/N3/xnas_daily.parquet',
        '/home/ec2-user/onemil/research/fuckup_audit/R_daily/xnas_daily_2024H1.parquet']
START, END = '2018-05-01', '2023-12-31'
MIN_RAW_CLOSE, MIN_ADV = 5.0, 1_000_000.0
SPLITS = {'TRAIN': ('2018-05-01', '2021-12-31'), 'VAL': ('2022-01-01', '2023-12-31')}
SELL_FEE_BPS, IMPACT_COEF_BPS, BOOK_USD = 0.4, 10.0, 66_000.0


def log(m):
    print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


def clean_symbol(s):
    """ITCH suffix classes (+ warrant, = unit, ^ preferred, ...) and dotted classes are not our universe."""
    return s.isalpha() and 1 <= len(s) <= 5


# ------------------------------------------------------------------ tape


def load_tape():
    parts = []
    for f in XNAS:
        df = pq.read_table(f, columns=['bar_date', 'symbol', 'close', 'volume']).to_pandas()
        df = df[(df['bar_date'] >= START) & (df['bar_date'] <= END)]
        parts.append(df)
    t = pd.concat(parts, ignore_index=True)
    del parts
    t['symbol'] = t['symbol'].astype(str)
    t = t[t['symbol'].map(clean_symbol)]
    t = t[(t['close'] > 0) & np.isfinite(t['close'])]
    t = t.drop_duplicates(['symbol', 'bar_date'], keep='last')
    log(f'tape rows {len(t):,} | symbols {t["symbol"].nunique():,} '
        f'| {t["bar_date"].min()} -> {t["bar_date"].max()}')
    return t


def densify(t):
    syms = sorted(t['symbol'].unique())
    dates = sorted(t['bar_date'].unique())
    sidx = {s: i for i, s in enumerate(syms)}
    didx = {d: i for i, d in enumerate(dates)}
    C = np.full((len(syms), len(dates)), np.nan, dtype=np.float32)
    V = np.full((len(syms), len(dates)), np.nan, dtype=np.float32)
    si = t['symbol'].map(sidx).to_numpy(dtype=np.int32)
    di = t['bar_date'].map(didx).to_numpy(dtype=np.int32)
    C[si, di] = t['close'].to_numpy(dtype=np.float32)
    V[si, di] = t['volume'].to_numpy(dtype=np.float32)
    log(f'dense {C.shape[0]} symbols x {C.shape[1]} sessions')
    return syms, pd.to_datetime(dates), C, V


# ------------------------------------------------------------------ splits


# Real US split ratios only.  The first attempt used every p/q for p,q <= 40, which is so dense that
# any price jump rounds to "a round factor" and the detector degenerated (37.7% validation).
# Real US split ratios, restricted to >= 1.5:1 and their reciprocals.  Two earlier attempts failed
# here: p/q for p,q <= 40 is so dense that any jump "rounds to a split", and an 18% price threshold
# makes every big up-day a candidate.  Forward splits below 3:2 are NOT detected -- disclosed, and
# they move a price by < 50% so they corrupt a 12-month formation return far less than a 1:10 reverse.
# Real US split ratios, 2:1 and above plus their reciprocals.  Three earlier attempts failed here and
# the validation named each one: p/q for p,q <= 40 is dense enough that any jump "rounds to a split"
# (37.7%); an 18% price threshold makes every big up-day a candidate (0.0% precision); and keeping
# 3:2 / 5:3 puts the ratio set right on top of the ordinary +-50% biotech day, which produced every
# single false positive in the third run (8.6% precision, all FPs at 1.5 / 1.667 / 0.667 / 0.6).
# 3:2 and 5:3 splits are therefore NOT detected -- disclosed; they distort a 12-month formation
# return by 50-67%, where the reverse splits that dominate the delisting cohort distort it by 10-25x.
SPLIT_RATIOS = np.array(sorted(set(
    [2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0,
     40.0, 50.0, 60.0, 75.0, 100.0, 150.0, 200.0]
    + [1 / x for x in [2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 20.0,
                       25.0, 30.0, 40.0, 50.0, 60.0, 75.0, 100.0, 150.0, 200.0]])))


def detect_splits(C, V, thresh=0.90, vol_tol=2.5, dvol_lo=0.2, dvol_hi=5.0):
    """Volume-confirmed split detector -> cumulative factor F with C_adj = C * F.

    A split on session t shows as a price jump `r = C[t]/C[t-1]` far from 1 whose implied share
    factor `f = 1/r` is a near-round ratio AND is confirmed by the share volume stepping by ~f
    (10-session medians either side).  Nothing else is touched: this deliberately does NOT try to
    remove dividends -- the tape is used for the $5/ADV gates (raw, correct as-is) and for returns
    over 5- and 21-session windows, where the dividend drop is a few bps.
    """
    n_s, n_d = C.shape
    F = np.ones((n_s, n_d), dtype=np.float32)
    n_found = 0
    for i in range(n_s):
        c = C[i]
        fin = np.nonzero(np.isfinite(c))[0]
        if len(fin) < 30:
            continue
        cuts = []
        for k in range(1, len(fin)):
            t0, t1 = fin[k - 1], fin[k]
            r = c[t1] / c[t0]
            if abs(np.log(r)) < np.log(1 + thresh):
                continue
            f = 1.0 / r
            vpre = np.nanmedian(V[i, max(0, t1 - 10):t1])
            vpost = np.nanmedian(V[i, t1:t1 + 10])
            if not (np.isfinite(vpre) and np.isfinite(vpost) and vpre > 0 and vpost > 0):
                continue
            f_vol = vpost / vpre
            # The stock also MOVES on its split day (TSLA 5:1 2020-08-31 implies 4.31x from price
            # alone), so the price-implied factor cannot be matched tightly.  Pick the ratio that
            # fits the price AND the share-volume step together, then confirm on dollar volume.
            lp, lv = np.log(f), np.log(max(f_vol, 1e-9))
            ls = np.log(SPLIT_RATIOS)
            near = SPLIT_RATIOS[np.argmin(np.abs(ls - lp) + 0.5 * np.abs(ls - lv))]
            tol = 0.20
            if abs(np.log(near / f)) > np.log(1 + tol):
                continue
            if not (near / vol_tol <= f_vol <= near * vol_tol):
                continue
            dpre = np.nanmedian((C[i] * V[i])[max(0, t1 - 10):t1])
            dpost = np.nanmedian((C[i] * V[i])[t1:t1 + 10])
            if not (np.isfinite(dpre) and np.isfinite(dpost) and dpre > 0):
                continue
            if not (dvol_lo <= dpost / dpre <= dvol_hi):
                continue
            cuts.append((t1, near))
        if not cuts:
            continue
        n_found += len(cuts)
        # price BEFORE a forward split must be divided by the factor -> multiply by 1/f cumulatively
        acc = 1.0
        prev = n_d
        for t1, f in reversed(cuts):
            F[i, t1:prev] = acc
            acc = acc / f
            prev = t1
        F[i, :prev] = acc
    log(f'split detector: {n_found} events on {C.shape[0]} symbols')
    return F


def validate_detector(syms, dates, C, V, F):
    """Precision/recall of the split detector against Alpaca's own corporate-action factor.

    On Alpaca, `daily_factor[t] = (adj[t]/adj[t-1]) / (raw[t]/raw[t-1])` is ~1 on an ordinary day,
    ~1.005 on a dividend ex-day, and equals the SPLIT ratio on a split day.  So the truth set is
    every session whose daily factor is outside [0.87, 1.15], with its factor; the detector's cut set
    is compared to it day by day on the symbols present in both panels.  (The first attempt compared
    cumulative endpoint factors against a ratio set dense enough to absorb dividends, and measured
    37.7% -- a broken test on top of a broken detector.)
    """
    z = np.load(f'{D}/panel_f3f4.npz', allow_pickle=True)
    asym = {s: i for i, s in enumerate(list(z['symbols']))}
    aidx = {d: i for i, d in enumerate(pd.to_datetime(list(z['sessions'])))}
    ca, cr = z['close_adj'], z['close_raw']
    tp = fp = fn = 0
    n_sym = 0
    misses, falses = [], []
    for i, s in enumerate(syms):
        j = asym.get(s)
        if j is None:
            continue
        cols = [aidx.get(d) for d in dates]
        have = np.array([c is not None for c in cols])
        cidx = np.array([c if c is not None else 0 for c in cols])
        with np.errstate(invalid='ignore', divide='ignore'):
            ra = ca[j, cidx][1:] / ca[j, cidx][:-1]
            rr = cr[j, cidx][1:] / cr[j, cidx][:-1]
            fac = ra / rr
        valid = have[1:] & have[:-1] & np.isfinite(fac)
        if valid.sum() < 200:
            continue
        n_sym += 1
        truth = {int(t + 1): float(fac[t]) for t in np.nonzero(valid & ((fac > 1.15) | (fac < 0.87)))[0]}
        det = {}
        prev = F[i, 1:] / F[i, :-1]
        for t in np.nonzero(np.isfinite(prev) & (np.abs(np.log(np.maximum(prev, 1e-9))) > 1e-4))[0]:
            det[int(t + 1)] = float(prev[t])
        for t, f in truth.items():
            hit = [u for u in det if abs(u - t) <= 2]
            if hit and abs(det[hit[0]] / f - 1) < 0.05:
                tp += 1
            else:
                fn += 1
                if len(misses) < 12:
                    misses.append((s, str(dates[t].date()), round(f, 3),
                                   round(det[hit[0]], 3) if hit else None))
        for t, f in det.items():
            if not [u for u in truth if abs(u - t) <= 2]:
                fp += 1
                if len(falses) < 12:
                    falses.append((s, str(dates[t].date()), round(f, 3)))
    prec = 100.0 * tp / max(1, tp + fp)
    rec = 100.0 * tp / max(1, tp + fn)
    log(f'detector validation on {n_sym} dual-listed symbols: TP {tp} FP {fp} FN {fn} '
        f'-> precision {prec:.1f}% recall {rec:.1f}%')
    return dict(n_symbols=n_sym, tp=tp, fp=fp, fn=fn, precision=prec, recall=rec,
                missed=misses, false_positives=falses)


# ------------------------------------------------------------------ the two cells


def run_cells(syms, dates, C, V, F, universe_mask, tag, terminal_haircut=0.0,
              jump_screen=False, cells=('F4-LO', 'F3-LO-12-1')):
    """F4-LO (weekly industry-adj reversal) and F3-LO-12-1 on this tape/universe.

    `universe_mask` is a boolean per symbol.  `terminal_haircut`: a position whose symbol stops
    trading mid-hold is marked to its last close and then charged this fraction (0.0 = the main
    run's frozen-position convention; 1.0 = the name goes to zero).
    """
    Cadj = C * F
    n_s, n_d = C.shape
    prev, cur = Cadj[:, :-1], Cadj[:, 1:]
    with np.errstate(invalid='ignore', divide='ignore'):
        rr = cur / prev - 1.0
    ret = np.zeros_like(Cadj)
    ret[:, 1:] = np.nan_to_num(rr, nan=0.0, posinf=0.0, neginf=0.0)
    del rr, prev, cur
    fin = np.isfinite(Cadj)
    last_fin = np.where(fin.any(axis=1), n_d - 1 - np.argmax(fin[:, ::-1], axis=1), -1)
    # >90% single-session move: on an UNADJUSTED tape this is a split far more often than a return.
    big = np.abs(ret) > 0.9
    bigcum = np.cumsum(big.astype(np.int32), axis=1)

    def has_jump(si, t0, t1):
        """any >90% session move in (t0, t1]"""
        return bigcum[si, t1] - bigcum[si, max(t0, 0)] > 0

    dv = np.nan_to_num(C * V, nan=0.0)
    okv = np.isfinite(C * V).astype(np.float32)
    adv = np.full_like(C, np.nan)
    for s in range(0, n_s, 2000):
        e = min(s + 2000, n_s)
        cs = np.cumsum(dv[s:e].astype(np.float64), axis=1)
        co = np.cumsum(okv[s:e].astype(np.float64), axis=1)
        num, den = cs.copy(), co.copy()
        num[:, 20:] = cs[:, 20:] - cs[:, :-20]
        den[:, 20:] = co[:, 20:] - co[:, :-20]
        with np.errstate(invalid='ignore', divide='ignore'):
            adv[s:e] = np.where(den >= 10, num / np.maximum(den, 1), np.nan)
    del dv, okv

    # SIC from the Alpaca universe; delisted names have none -> F4's industry residual cannot be
    # built for them, so F4's PIT arm measures the bias only on names with a SIC.  Reported.
    uni = pd.read_parquet(f'{D}/universe.parquet', columns=['symbol', 'sic2'])
    sicmap = dict(zip(uni['symbol'], uni['sic2'].astype(str)))
    sic = np.array([sicmap.get(s, '') for s in syms], dtype=object)

    mdf = pd.DataFrame({'d': dates})
    me = mdf.groupby(mdf['d'].dt.to_period('M')).tail(1).index.to_numpy()
    we = mdf.groupby(mdf['d'].dt.to_period('W')).tail(1).index.to_numpy()

    def elig(t):
        return (universe_mask & np.isfinite(C[:, t]) & (C[:, t] >= MIN_RAW_CLOSE)
                & np.isfinite(adv[:, t]) & (adv[:, t] >= MIN_ADV) & np.isfinite(Cadj[:, t]))

    def portfolio(trades, anchors):
        num = np.zeros(n_d); den = np.zeros(n_d)
        for si, a, b, cost in trades:
            idx = np.arange(a + 1, b + 1)
            r = ret[si, idx].astype(np.float64).copy()
            r[0] -= cost / 2e4
            r[-1] -= cost / 2e4
            if terminal_haircut > 0 and last_fin[si] < b:
                k = int(np.searchsorted(idx, last_fin[si], side='right'))
                if 0 <= k < len(r):
                    r[k] -= terminal_haircut
            np.add.at(num, idx, r)
            np.add.at(den, idx, 1.0)
        with np.errstate(invalid='ignore', divide='ignore'):
            return pd.Series(np.where(den > 0, num / np.maximum(den, 1e-9), 0.0), index=dates)

    def cost_of(si, a, npos):
        order = BOOK_USD / max(npos, 1)
        with np.errstate(invalid='ignore', divide='ignore'):
            frac = np.minimum(order / np.maximum(0.01 * adv[si, a], 1.0), 1.0)
        return 2 * IMPACT_COEF_BPS * frac + SELL_FEE_BPS

    res = {}
    n_screened = [0]

    # ---- F4-LO
    book, bench = [], []
    for j in range(2, len(we) - 1):
        t = we[j]
        if t - 5 < 0:
            continue
        ok = elig(t) & np.isfinite(Cadj[:, t - 5]) & np.array([s not in ('', 'nan', 'None')
                                                               for s in sic])
        idx = np.nonzero(ok)[0]
        if len(idx) < 100:
            continue
        with np.errstate(invalid='ignore', divide='ignore'):
            r1w = Cadj[idx, t] / Cadj[idx, t - 5] - 1.0
        good = np.isfinite(r1w)
        idx, r1w = idx[good], r1w[good]
        df = pd.DataFrame({'si': idx, 'r': r1w.astype(np.float64), 'sic': sic[idx]})
        g = df.groupby('sic')['r']
        df['n'], df['s'] = g.transform('size'), g.transform('sum')
        df = df[df['n'] >= 5].copy()
        if len(df) < 100:
            continue
        df['resid'] = df['r'] - (df['s'] - df['r']) / (df['n'] - 1)
        q = int(np.ceil(len(df) / 10))
        df = df.sort_values('resid')
        a, b = we[j] + 1, we[j + 1] + 1
        if b >= n_d:
            continue
        lo, rest = df.iloc[:q], df.iloc[q:]
        for sub, sink, np_ in ((lo, book, len(lo)), (rest, bench, len(rest))):
            for si in sub['si'].to_numpy():
                if not (np.isfinite(Cadj[si, a]) and np.isfinite(Cadj[si, b])):
                    continue
                if jump_screen and (has_jump(si, t - 5, t) or has_jump(si, a, b)):
                    n_screened[0] += 1
                    continue
                sink.append((int(si), int(a), int(b), cost_of(si, a, np_)))
    res['F4-LO'] = (portfolio(book, we), portfolio([(s, a, b, 0.0) for s, a, b, _ in bench], we),
                    len(book))
    log(f'  {tag}: F4 book {len(book)} trades, {n_screened[0]} screened for a >90% session move')
    if 'F3-LO-12-1' not in cells:
        me = me[:0]

    # ---- F3-LO-12-1
    book, bench = [], []
    for k in range(13, max(0, len(me) - 1)):
        t = me[k]
        ok = elig(t) & np.isfinite(Cadj[:, me[k - 12]]) & np.isfinite(Cadj[:, me[k - 1]])
        idx = np.nonzero(ok)[0]
        if len(idx) < 100:
            continue
        with np.errstate(invalid='ignore', divide='ignore'):
            sig = Cadj[idx, me[k - 1]] / Cadj[idx, me[k - 12]] - 1.0
        good = np.isfinite(sig)
        idx, sig = idx[good], sig[good]
        o = np.argsort(sig)
        q = int(np.ceil(len(idx) / 10))
        a, b = me[k] + 1, me[k + 1] + 1
        if b >= n_d:
            continue
        top, rest = idx[o[-q:]], idx[o[:-q]]
        for arr, sink, np_ in ((top, book, len(top)), (rest, bench, len(rest))):
            for si in arr:
                if np.isfinite(Cadj[si, a]) and np.isfinite(Cadj[si, b]):
                    sink.append((int(si), int(a), int(b), cost_of(si, a, np_)))
    if book:
        res['F3-LO-12-1'] = (portfolio(book, me),
                             portfolio([(s, a, b, 0.0) for s, a, b, _ in bench], me), len(book))

    out = {}
    for name, (bk, bn, ntr) in res.items():
        if name not in cells:
            continue
        exc = bk - bn
        m = (1 + exc).resample('ME').prod() - 1
        for split, (s0, s1) in SPLITS.items():
            mm = m[(m.index >= s0) & (m.index <= s1)]
            mm = mm[mm != 0]
            if len(mm) < 6:
                continue
            mu, sd = mm.mean(), mm.std(ddof=1)
            out[f'{name}|{split}'] = dict(
                tag=tag, n_trades=ntr, n_months=len(mm), mean_bps=float(mu * 1e4),
                t=float(mu / (sd / np.sqrt(len(mm)))) if sd > 0 else None,
                pct_pos=float(100 * (mm > 0).mean()))
    return out


def main():
    import os
    os.makedirs(OUT, exist_ok=True)
    t = load_tape()
    syms, dates, C, V = densify(t)
    del t
    F = detect_splits(C, V)
    val = validate_detector(syms, dates, C, V, F)

    z = np.load(f'{D}/panel_f3f4.npz', allow_pickle=True)
    alive = set(z['symbols'])
    surv = np.array([s in alive for s in syms])
    log(f'universe: PIT {len(syms):,} symbols | still listed on Alpaca {surv.sum():,} '
        f'({100*surv.mean():.1f}%) | delisted {int((~surv).sum()):,}')

    results = {}
    # F3-LO-12-1 needs a CLEAN 12-month formation return, which needs a trustworthy split
    # adjustment, which this tape cannot give (see the validation above).  It is run only as the
    # disclosed unreliable arm.  F4-LO's windows are 5 sessions, so it needs no split adjustment at
    # all: the raw tape plus a >90%-session-move screen (applied IDENTICALLY to both arms, so it
    # cannot manufacture a survivorship difference) is enough.
    ONES = np.ones_like(F)
    for tag, mask, hc in (('PIT', np.ones(len(syms), dtype=bool), 0.0),
                          ('PIT_haircut100', np.ones(len(syms), dtype=bool), 1.0),
                          ('SURVIVOR', surv, 0.0)):
        log(f'running {tag} (F4, raw tape + jump screen) ...')
        results.update({f'{k}|{tag}': v for k, v in
                        run_cells(syms, dates, C, V, ONES, mask, tag, hc, jump_screen=True,
                                  cells=('F4-LO',)).items()})
    for tag, mask in (('PIT_unreliable', np.ones(len(syms), dtype=bool)), ('SURVIVOR_unreliable', surv)):
        log(f'running {tag} (F3 12-1, detector-adjusted -- NOT trustworthy) ...')
        results.update({f'{k}|{tag}': v for k, v in
                        run_cells(syms, dates, C, V, F, mask, tag, 0.0,
                                  cells=('F3-LO-12-1',)).items()})

    json.dump(dict(validation=val, n_symbols=len(syms), n_survivors=int(surv.sum()),
                   cells=results), open(f'{OUT}/pit.json', 'w'), indent=1, default=float)
    for k, v in sorted(results.items()):
        print(f'{k:42s} n={v["n_trades"]:6d} mo={v["n_months"]:3d} '
              f'{v["mean_bps"]:+9.1f} bps  t {v["t"] if v["t"] is None else round(v["t"],2)}  '
              f'%+ {v["pct_pos"]:.1f}', flush=True)


if __name__ == '__main__':
    sys.exit(main())
