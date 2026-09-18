#!/usr/bin/env python3
"""INDEPENDENT reimplementation of research cells F4-LO and F3-LO-12-1.

Written from a prose specification only (adversarial check, CLAUDE.md
"No research claim ships without an independent check"). No other
implementation of these cells was read.

Outputs -> research/multiday/out_indep_f4f3/
"""
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
OUT = os.path.join(BASE, "out_indep_f4f3")
os.makedirs(OUT, exist_ok=True)

YEARS = list(range(2016, 2024))  # 2024+ is explicitly out of scope
PRICE_FLOOR = 5.00
ADV_FLOOR = 1_000_000.0
BOOK_USD = 66_000.0
MIN_XSEC = 100
ADV_WIN = 20
ADV_MIN_OBS = 10

SPLITS = {
    "TRAIN": ("2016-01-01", "2021-12-31"),
    "VAL": ("2022-01-01", "2023-12-31"),
}

T0 = time.time()


def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------- panel build
def load_panels():
    uni = pd.read_parquet(os.path.join(DATA, "universe.parquet"),
                          columns=["symbol", "kind", "sic2"])
    common = uni[uni["kind"] == "common"]
    syms = sorted(set(common["symbol"]) | {"SPY"})
    sym_set = set(syms)
    sic2 = {r.symbol: (r.sic2 if isinstance(r.sic2, str) and r.sic2 != "" else None)
            for r in common.itertuples()}
    log(f"universe: {len(common)} common symbols (+SPY); "
        f"{sum(v is not None for v in sic2.values())} with sic2")

    def daynum(s):
        return pd.to_datetime(pd.Series(s).values).values.astype("datetime64[D]").astype(np.int64)

    # Session calendar = dates on which SPY has a bar in the ADJUSTED panel
    cal_days = []
    for y in YEARS:
        d = pd.read_parquet(f"{DATA}/prices_by_year/all/year={y}.parquet",
                            columns=["symbol", "date"],
                            filters=[("symbol", "==", "SPY")])
        cal_days.append(daynum(d["date"]))
    cal_int = np.unique(np.concatenate(cal_days))
    cal = cal_int.astype("datetime64[D]")
    log(f"session calendar: {len(cal)} sessions {cal[0]} .. {cal[-1]}")

    symbols = [s for s in syms if s != "SPY"]  # traded universe = common only
    sym_pos = {s: i for i, s in enumerate(symbols)}
    T, N = len(cal), len(symbols)

    A = np.full((T, N), np.nan)
    R = np.full((T, N), np.nan)
    DV = np.full((T, N), np.nan)

    def idx_of(df):
        col = df["symbol"]
        if isinstance(col.dtype, pd.CategoricalDtype):
            lut = np.array([sym_pos.get(c, -1) for c in col.cat.categories], dtype=np.int64)
            si = lut[col.cat.codes.values]
        else:
            si = np.array([sym_pos.get(c, -1) for c in col.values], dtype=np.int64)
        dv = daynum(df["date"])
        di = np.searchsorted(cal_int, dv)
        di = np.clip(di, 0, T - 1)
        ok = (si >= 0) & (cal_int[di] == dv)
        return di[ok], si[ok], ok

    for y in YEARS:
        a = pd.read_parquet(f"{DATA}/prices_by_year/all/year={y}.parquet",
                            columns=["symbol", "date", "close"])
        di, si, ok = idx_of(a)
        A[di, si] = a["close"].values[ok]
        na = len(di)
        del a
        r = pd.read_parquet(f"{DATA}/prices_by_year/raw/year={y}.parquet",
                            columns=["symbol", "date", "close", "vwap", "volume"])
        di, si, ok = idx_of(r)
        R[di, si] = r["close"].values[ok]
        DV[di, si] = (r["vwap"].values[ok].astype(np.float64) *
                      r["volume"].values[ok].astype(np.float64))
        log(f"  {y}: adj {na:,} cells, raw {len(di):,} cells")
        del r

    log(f"panel matrices built: {T} sessions x {N} symbols")
    return cal, symbols, None, sic2, A, R, DV


def rolling_adv(DV):
    """Mean of dollar volume over the 20 sessions ending at t, >=10 obs."""
    valid = np.isfinite(DV)
    vals = np.where(valid, DV, 0.0)
    cs = np.vstack([np.zeros((1, DV.shape[1])), np.cumsum(vals, axis=0)])
    cc = np.vstack([np.zeros((1, DV.shape[1])), np.cumsum(valid, axis=0)])
    T = DV.shape[0]
    lo = np.maximum(np.arange(1, T + 1) - ADV_WIN, 0)
    s = cs[1:] - cs[lo]
    c = cc[1:] - cc[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        adv = np.where(c >= ADV_MIN_OBS, s / np.maximum(c, 1), np.nan)
    return adv


# ------------------------------------------------------------------ portfolio
def portfolio_series(trades, ret, T, with_cost=True):
    """Equal-weighted daily portfolio return series.

    trades: dict of arrays si (symbol idx), e (entry session idx),
            x (exit session idx), cost (bps round trip).
    A position earns daily returns on sessions e+1 .. x.
    Half the round-trip cost hits the entry day, half the exit day, expressed
    as the MEAN half-cost over the trades entering / exiting that day (each
    trade is 1/n of the leg's notional).
    """
    contrib = np.zeros(T)
    nopen = np.zeros(T)
    ec_sum = np.zeros(T); ec_n = np.zeros(T)
    xc_sum = np.zeros(T); xc_n = np.zeros(T)
    si, e, x, cost = trades["si"], trades["e"], trades["x"], trades["cost"]
    for k in range(len(si)):
        a, b = e[k] + 1, x[k] + 1
        if b > a:
            contrib[a:b] += ret[a:b, si[k]]
            nopen[a:b] += 1
        if with_cost:
            ec_sum[e[k]] += cost[k] / 2.0; ec_n[e[k]] += 1
            xc_sum[x[k]] += cost[k] / 2.0; xc_n[x[k]] += 1
    out = np.zeros(T)
    m = nopen > 0
    out[m] = contrib[m] / nopen[m]
    if with_cost:
        m = ec_n > 0
        out[m] -= (ec_sum[m] / ec_n[m]) / 1e4
        m = xc_n > 0
        out[m] -= (xc_sum[m] / xc_n[m]) / 1e4
    return out


def monthly(series, cal):
    s = pd.Series(series, index=pd.to_datetime(pd.Index(cal)))
    return (1.0 + s).groupby(s.index.to_period("M")).prod() - 1.0


def stats(book_m, bench_m, lo, hi):
    ex = (book_m - bench_m).dropna()
    per_lo, per_hi = pd.Period(lo, "M"), pd.Period(hi, "M")
    ex = ex[(ex.index >= per_lo) & (ex.index <= per_hi)]
    ex = ex[ex != 0.0]
    n = len(ex)
    if n < 2:
        return dict(n_months=n, mean_bps=float("nan"), t=float("nan"), pct_pos=float("nan"))
    mean = ex.mean()
    t = mean / (ex.std(ddof=1) / math.sqrt(n))
    return dict(n_months=n, mean_bps=round(mean * 1e4, 2), t=round(float(t), 3),
                pct_pos=round(float((ex > 0).mean() * 100), 1))


# ---------------------------------------------------------------- cell engine
def make_trades(cohorts, A, adv, sym_ix, T):
    """cohorts: list of (decision_idx, book_sym_idx_array, bench_sym_idx_array,
    entry_idx, exit_idx). Returns book/bench trade dicts with costs."""
    out = {}
    for leg in ("book", "bench"):
        si, e, x, gross, cost = [], [], [], [], []
        for (d, bk, bn, ei, xi) in cohorts:
            cand = bk if leg == "book" else bn
            pe, px = A[ei, cand], A[xi, cand]
            ok = np.isfinite(pe) & np.isfinite(px)
            cand = cand[ok]
            if len(cand) == 0:
                continue
            g = A[xi, cand] / A[ei, cand] - 1.0
            order_usd = BOOK_USD / len(cand)
            a = adv[ei, cand]
            frac = np.where(np.isfinite(a) & (a > 0),
                            np.minimum(order_usd / (0.01 * np.where(a > 0, a, 1.0)), 1.0),
                            1.0)
            c = 2 * 10 * frac + 0.4
            si.append(cand); e.append(np.full(len(cand), ei))
            x.append(np.full(len(cand), xi)); gross.append(g); cost.append(c)
        out[leg] = dict(si=np.concatenate(si), e=np.concatenate(e),
                        x=np.concatenate(x), gross=np.concatenate(gross),
                        cost=np.concatenate(cost))
    return out


def subset(tr, mask):
    return {k: v[mask] for k, v in tr.items()}


def in_split(tr, lo, hi, cal):
    d_e = pd.to_datetime(pd.Index(cal))[tr["e"]]
    d_x = pd.to_datetime(pd.Index(cal))[tr["x"]]
    return (d_e >= lo) & (d_x <= hi) & (d_e >= lo) & (d_x <= hi)


# ------------------------------------------------------------------- cells
def cell_f4(cal, symbols, sic2, A, R, adv, elig, ret):
    T, N = A.shape
    sic_arr = np.array([sic2.get(s) or "" for s in symbols])
    have_sic = sic_arr != ""
    dts = pd.to_datetime(pd.Index(cal))
    iso = dts.isocalendar()
    key = pd.Series(list(zip(iso.year.values, iso.week.values)))
    # last index per ISO week, in calendar order
    last_idx = []
    for i in range(T):
        if i == T - 1 or key.iloc[i + 1] != key.iloc[i]:
            last_idx.append(i)
    W = np.array(last_idx)
    log(f"F4: {len(W)} week-end sessions")

    cohorts = []
    skipped = 0
    maxfeat = {}
    for j in range(len(W) - 1):
        w, wn = W[j], W[j + 1]
        if w - 5 < 0 or wn + 1 >= T or w + 1 >= T:
            continue
        ok = elig[w] & np.isfinite(A[w]) & np.isfinite(A[w - 5]) & have_sic
        idx = np.flatnonzero(ok)
        if len(idx) == 0:
            continue
        r1w = A[w, idx] / A[w - 5, idx] - 1.0
        g = sic_arr[idx]
        df = pd.DataFrame({"i": idx, "r": r1w, "g": g})
        grp = df.groupby("g")["r"]
        cnt = grp.transform("size")
        ssum = grp.transform("sum")
        df = df[cnt >= 5]
        if len(df) < MIN_XSEC:
            skipped += 1
            continue
        cnt = cnt[df.index]; ssum = ssum[df.index]
        ind_mean = (ssum - df["r"]) / (cnt - 1)
        df["resid"] = df["r"] - ind_mean
        n = len(df)
        k = int(math.ceil(n / 10.0))
        order = np.argsort(df["resid"].values, kind="stable")
        bk = df["i"].values[order[:k]]
        bn = df["i"].values[order[k:]]
        cohorts.append((w, bk, bn, w + 1, wn + 1))
        mx = np.nanmax(ret[w - 20:w + 1, bk], axis=0) if w - 20 >= 0 else np.full(len(bk), np.nan)
        for s_i, v in zip(bk, mx):
            maxfeat[(w, s_i)] = (v, R[w, s_i])
    log(f"F4: {len(cohorts)} weekly cohorts, {skipped} weeks skipped (<{MIN_XSEC})")
    return cohorts, maxfeat


def cell_f3(cal, A, adv, elig):
    T, N = A.shape
    dts = pd.to_datetime(pd.Index(cal))
    per = dts.to_period("M")
    last_idx = [i for i in range(T) if i == T - 1 or per[i + 1] != per[i]]
    M = np.array(last_idx)
    log(f"F3: {len(M)} month-end sessions")
    cohorts = []
    skipped = 0
    for k in range(12, len(M) - 1):
        m, mn = M[k], M[k + 1]
        m1, m12 = M[k - 1], M[k - 12]
        if m + 1 >= T or mn + 1 >= T:
            continue
        ok = elig[m] & np.isfinite(A[m1]) & np.isfinite(A[m12])
        idx = np.flatnonzero(ok)
        if len(idx) < MIN_XSEC:
            skipped += 1
            continue
        sig = A[m1, idx] / A[m12, idx] - 1.0
        n = len(idx)
        kk = int(math.ceil(n / 10.0))
        order = np.argsort(-sig, kind="stable")
        bk = idx[order[:kk]]
        bn = idx[order[kk:]]
        cohorts.append((m, bk, bn, m + 1, mn + 1))
    log(f"F3: {len(cohorts)} monthly cohorts, {skipped} skipped")
    return cohorts


# ---------------------------------------------------------------------- main
def report_cell(name, cohorts, cal, symbols, A, adv, ret, extra=None):
    T = A.shape[0]
    tr = make_trades(cohorts, A, adv, None, T)
    book, bench = tr["book"], tr["bench"]
    dts = pd.to_datetime(pd.Index(cal))
    res = {}
    rows = []
    for split, (lo, hi) in SPLITS.items():
        lo_ts, hi_ts = pd.Timestamp(lo), pd.Timestamp(hi)
        bm = (dts[book["e"]] >= lo_ts) & (dts[book["x"]] <= hi_ts)
        nm = (dts[bench["e"]] >= lo_ts) & (dts[bench["x"]] <= hi_ts)
        b = subset(book, bm); nb = subset(bench, nm)
        bench_m = monthly(portfolio_series(nb, ret, T, with_cost=False), cal)

        def run(tsub):
            bkm = monthly(portfolio_series(tsub, ret, T, with_cost=True), cal)
            return stats(bkm, bench_m, lo, hi)

        full = run(b)
        full["n_trades"] = int(len(b["si"]))
        full["mean_cost_bps"] = round(float(np.mean(b["cost"])), 3) if len(b["si"]) else float("nan")
        # ex-top-1% / ex-top-5% by gross
        for pct, tag in ((0.01, "ex_top1"), (0.05, "ex_top5")):
            n = len(b["gross"])
            drop = int(math.ceil(pct * n))
            keep = np.argsort(b["gross"], kind="stable")[: n - drop]
            m = np.zeros(n, dtype=bool); m[keep] = True
            st = run(subset(b, m))
            full[tag] = {"n_trades": int(m.sum()), "mean_bps": st["mean_bps"], "t": st["t"]}
        res[split] = full
        # per-trade dump rows
        rows.append(pd.DataFrame({
            "symbol": [symbols[i] for i in b["si"]],
            "entry_date": dts[b["e"]].date,
            "exit_date": dts[b["x"]].date,
            "gross": b["gross"], "cost_bps": b["cost"],
            "split": split,
        }))
        if extra is not None:
            res[split].update(extra(b, split))
    pd.concat(rows, ignore_index=True).to_parquet(f"{OUT}/{name}_trades.parquet", index=False)
    return res


def main():
    cal, symbols, sym_ix, sic2, A, R, DV = load_panels()
    adv = rolling_adv(DV)
    del DV
    elig = (np.isfinite(R) & (R >= PRICE_FLOOR) & np.isfinite(adv) &
            (adv >= ADV_FLOOR) & np.isfinite(A))
    log(f"eligibility: mean {elig.sum(axis=1).mean():.0f} symbols/session")

    prev = np.vstack([np.full((1, A.shape[1]), np.nan), A[:-1]])
    with np.errstate(invalid="ignore", divide="ignore"):
        ret = A / prev - 1.0
    ret[~np.isfinite(ret)] = 0.0

    out = {}
    f4_cohorts, maxfeat = cell_f4(cal, symbols, sic2, A, R, adv, elig, ret)

    def f4_extra(b, split):
        mx = np.array([maxfeat[(e - 1, s)][0] for e, s in zip(b["e"], b["si"])])
        rc = np.array([maxfeat[(e - 1, s)][1] for e, s in zip(b["e"], b["si"])])
        net = b["gross"] - b["cost"] / 1e4
        d = {}
        try:
            q = pd.qcut(pd.Series(mx), 5, labels=False, duplicates="drop")
            d["max_quintiles_net_bps"] = {
                f"Q{int(k)+1}": round(float(net[(q == k).values].mean() * 1e4), 2)
                for k in sorted(pd.Series(q).dropna().unique())}
            d["max_quintile_edges"] = [round(float(v), 4) for v in
                                       np.nanpercentile(mx, [0, 20, 40, 60, 80, 100])]
        except Exception as ex:
            d["max_quintiles_net_bps"] = f"error: {ex}"
        tot = float(net.sum())
        sub = float(net[rc < 10.0].sum())
        d["sub10_share_of_net_pnl_pct"] = round(sub / tot * 100, 1) if tot != 0 else None
        d["sub10_n_trades"] = int((rc < 10.0).sum())
        return d

    out["F4-LO"] = report_cell("F4-LO", f4_cohorts, cal, symbols, A, adv, ret, extra=f4_extra)
    log("F4-LO done")

    f3_cohorts = cell_f3(cal, A, adv, elig)
    out["F3-LO-12-1"] = report_cell("F3-LO-12-1", f3_cohorts, cal, symbols, A, adv, ret)
    log("F3-LO-12-1 done")

    with open(f"{OUT}/indep.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
