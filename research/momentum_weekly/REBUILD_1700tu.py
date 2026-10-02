"""
REBUILD_1700tu.py -- independent rebuild of the guarded sleeve and the half-size gate
(spec: REBUILD_1700tu_spec.md). Extends the conventions of REBUILD_1700_sleeve.py with the two
RECON fixes (word-boundary name exclusions; ALL 20 names reset to target weight every Monday).
Memory-safe load: pyarrow row-group batches, symbol pre-filter on a necessary condition
(some row with close>=10 and dollar volume>=200M) before any frame is built, float32 prices.
Run: bash scripts/research_run.sh -m 2500M python3 research/momentum_weekly/REBUILD_1700tu.py
"""
import re, time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

D = "research/momentum_weekly/"
PANEL = D + "panel_2016_2026.parquet"
ASSETS = D + "1700c_assets.csv"
EXCL_RE = re.compile(r"\bETFs?\b|\bETNs?\b|\bFUNDs?\b|\bTRUSTs?\b|\bWARRANTS?\b|\bUNITS?\b|\bPREFERRED\b|\bRIGHTS?\b",
                     re.IGNORECASE)
ZZZT = re.compile(r"^Z[A-Z]ZZT$")
START, END = pd.Timestamp("2017-01-02"), pd.Timestamp("2026-09-28")
TOPN, CAP0 = 20, 50000.0


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load():
    """Two streaming passes over the parquet; returns dict symbol -> (dates, o,h,l,c,v)."""
    a = pd.read_csv(ASSETS, dtype=str)
    a.columns = [c.strip().lower() for c in a.columns]
    bad = set(a.loc[a["name"].fillna("").str.contains(EXCL_RE), "symbol"].dropna())
    log(f"name-excluded symbols: {len(bad):,}")
    pf = pq.ParquetFile(PANEL)
    ok = set()
    for b in pf.iter_batches(batch_size=2_000_000, columns=["symbol", "close", "volume"]):
        df = b.to_pandas()
        m = (df["close"] >= 10) & (df["close"] * df["volume"] >= 200e6)
        ok.update(df.loc[m, "symbol"].unique())
    log(f"pass 1: {len(ok):,} symbols ever pass price/dollar-volume")
    ok = {s for s in ok if s not in bad and not ZZZT.match(s)} | {"SPY"}
    parts = []
    for b in pf.iter_batches(batch_size=2_000_000):
        df = b.to_pandas()
        df = df[df["symbol"].isin(ok)]
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = df[c].astype("float32")
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    del parts
    df = df.sort_values(["symbol", "bar_date"], kind="stable").reset_index(drop=True)
    log(f"pass 2: {len(df):,} rows, {df['symbol'].nunique():,} symbols")
    out = {}
    sym = df["symbol"].values
    cuts = np.flatnonzero(sym[1:] != sym[:-1]) + 1
    bounds = np.r_[0, cuts, len(df)]
    dts = df["bar_date"].values
    arr = {c: df[c].values for c in ["open", "high", "low", "close", "volume"]}
    for i in range(len(bounds) - 1):
        s, e = bounds[i], bounds[i + 1]
        out[sym[s]] = (dts[s:e], *[arr[c][s:e].astype("float64") for c in ["open", "high", "low", "close", "volume"]])
    return out


def rebal_dates(cal):
    res = []
    for m in pd.date_range(START, END, freq="W-MON"):
        p = cal.searchsorted(m, side="left")
        if p < len(cal):
            res.append(cal[p])
    return sorted(set(res))


def features(data, cal, ts, tpos):
    """Per rebal: arrays of (symbol, signal, guard_bad, spread_proxy) for names passing the plain universe."""
    nR = len(ts)
    rows = [[] for _ in range(nR)]
    for sym, (d, o, h, l, c, v) in data.items():
        if sym == "SPY" or len(d) < 274:
            continue
        pos = np.searchsorted(d, ts)
        have = (pos < len(d)) & (d[np.minimum(pos, len(d) - 1)] == ts)
        r = np.zeros(len(c)); r[1:] = c[1:] / c[:-1] - 1
        gapd = np.zeros(len(c)); gapd[1:] = np.diff(d).astype("timedelta64[D]").astype(float)
        badday = ((r > 2.0) | (r < -0.75) | (gapd > 10)).astype(np.int64); badday[0] = 0
        cb = np.cumsum(badday)
        c1 = np.r_[0, np.cumsum(r)]; c2 = np.r_[0, np.cumsum(r * r)]
        dv = np.r_[0, np.cumsum(c * v)]
        for k in np.flatnonzero(have & (pos >= 272)):
            p = pos[k]
            if c[p] < 10 or p + 1 < 273:
                continue
            adv = (dv[p + 1] - dv[p - 19]) / 20
            if adv < 200e6:
                continue
            n = 252
            s1 = c1[p + 1] - c1[p + 1 - n]; s2 = c2[p + 1] - c2[p + 1 - n]
            var = (s2 - s1 * s1 / n) / (n - 1)
            if var <= 0:
                continue
            sig = (c[p - 21] / c[p - 252] - 1) / np.sqrt(var)
            if not np.isfinite(sig):
                continue
            bad = (cb[p] - cb[p - 272]) > 0
            sp = (h[p] - l[p]) / c[p] * 0.1 if c[p] > 0 else 0.0
            rows[k].append((sym, sig, bad, sp))
    return rows


def px(data, sym, date, use_open):
    d, o, h, l, c, v = data[sym]
    p = np.searchsorted(d, np.datetime64(date), side="right") - 1
    if p < 0:
        return np.nan
    if d[p] == np.datetime64(date) and use_open:
        return o[p]
    return c[p]


def gate_series():
    """Half flag by date: percentile of VIX/VIX3M among prior 251 ratios (<20% -> half)."""
    def rd(f):
        x = pd.read_csv(D + f); x["DATE"] = pd.to_datetime(x["DATE"], format="%m/%d/%Y")
        return x.set_index("DATE")["CLOSE"]
    r = (rd("REBUILD_1700tu_VIX.csv") / rd("REBUILD_1700tu_VIX3M.csv")).dropna()
    v = r.values; pct = np.full(len(v), np.nan)
    for i in range(len(v)):
        w = v[max(0, i - 251):i]
        if len(w) >= 126:
            pct[i] = ((w < v[i]).sum() + 0.5 * (w == v[i]).sum()) / len(w)
    return pd.Series(pct, index=r.index)


def simulate(data, cal, rebs, ts, picks, halfflag, name):
    """Equal-weight reset every Monday; weight 1/20 (or 1/40 when halfflag). Returns weekly eq, daily eq."""
    cash, hold = CAP0, {}  # hold: sym -> dollar value drifted to this Monday open
    wk, daily = [], []
    for i, reb in enumerate(rebs[:-1]):
        nxt = rebs[i + 1]
        tgt_syms = picks[i]
        if tgt_syms is None:
            wk.append((reb, cash + sum(hold.values()))); continue
        E = cash + sum(hold.values())
        w = 1.0 / (2 * TOPN if halfflag[i] else TOPN)
        cost = 0.0
        for s in set(hold) | set(tgt_syms):
            delta = abs((w * E if s in tgt_syms else 0.0) - hold.get(s, 0.0))
            sp = spmap[(i, s)] if (i, s) in spmap else 0.0
            cost += delta * min(0.0005 + 0.5 * sp, 0.0020)
        cash = E - cost - w * E * len(tgt_syms)
        wk.append((reb, E - cost))
        qty, new = {}, {}
        for s in tgt_syms:
            p0 = px(data, s, reb, True)
            qty[s] = (w * E) / p0
        days = cal[(cal >= reb) & (cal < nxt)]
        for dday in days:
            val = cash + sum(qty[s] * px(data, s, dday, False) for s in qty)
            daily.append((dday, val))
        hold = {s: qty[s] * px(data, s, nxt, True) for s in qty}
    return wk, daily


def stats(wk, daily):
    eq = pd.Series([v for _, v in wk], index=[d for d, _ in wk])
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1
    de = pd.Series([v for _, v in daily], index=[d for d, _ in daily])
    dd = (de / de.cummax() - 1).min()
    by = {}
    for y in sorted(set(eq.index.year)):
        s = eq[eq.index.year == y]
        nxt_start = eq[eq.index.year == y + 1]
        end = nxt_start.iloc[0] if len(nxt_start) else None
        prev = eq[eq.index.year == y - 1]
        start = s.iloc[0]
        by[y] = ((end if end is not None else eq.iloc[-1]) / start - 1) * 100
    ye = de.groupby(de.index.year).last()
    by = {}
    for y in ye.index:
        base = ye.get(y - 1, de.iloc[0])
        by[y] = (ye[y] / base - 1) * 100
    return eq, cagr, dd, by, de


def main():
    global spmap
    data = load()
    cal = pd.DatetimeIndex(data["SPY"][0])
    rebs = rebal_dates(cal)
    ts = np.array([cal[cal.searchsorted(r) - 1] for r in rebs], dtype="datetime64[ns]")
    log(f"{len(rebs)} rebalances")
    rows = features(data, cal, ts, None)
    log("features done")
    spmap, plain, guarded, removed = {}, [], [], []
    for i, rr in enumerate(rows):
        if len(rr) < TOPN:
            plain.append(None); guarded.append(None); continue
        for s, sg, b, sp in rr:
            spmap[(i, s)] = sp
        top = sorted(rr, key=lambda x: -x[1])[:TOPN]
        gtop = sorted([x for x in rr if not x[2]], key=lambda x: -x[1])[:TOPN]
        plain.append([x[0] for x in top]); guarded.append([x[0] for x in gtop])
        removed += [(rebs[i], x[0]) for x in top if x[2]]
    pct = gate_series()
    half = []
    for i in range(len(rebs)):
        p = pct.loc[:pd.Timestamp(ts[i])].iloc[-1] if pd.Timestamp(ts[i]) >= pct.index[0] else np.nan
        half.append(bool(p < 0.20) if np.isfinite(p) else False)
    res = {}
    for nm, pk, hf in [("plain", plain, [False] * len(rebs)), ("guarded", guarded, [False] * len(rebs)),
                       ("gated", guarded, half)]:
        wk, dly = simulate(data, cal, rebs, ts, pk, hf, nm)
        eq, cagr, dd, by, de = stats(wk, dly)
        res[nm] = (eq, cagr, dd, by)
        de.to_csv(D + f'REBUILD_1700tu_daily_{nm}.csv')
        log(f"{nm}: CAGR {cagr*100:.2f}% maxDD(daily) {dd*100:.1f}% end ${eq.iloc[-1]:,.0f}")
    valid = [i for i in range(len(rebs) - 1) if guarded[i] is not None]
    nh = sum(half[i] for i in valid)
    spells = sum(1 for k, i in enumerate(valid) if half[i] and (k == 0 or not half[valid[k - 1]]))
    log(f"half weeks {nh} of {len(valid)} valid; spells {spells}")
    rm = pd.DataFrame(removed, columns=["week", "sym"])
    rm.to_csv(D + "REBUILD_1700tu_removed.csv", index=False)
    log(f"guard-removed name-weeks {len(rm)}: {rm['sym'].value_counts().to_dict()}")
    for dt in ["2021-02-08", "2025-12-29", "2026-06-29"]:
        i = rebs.index(pd.Timestamp(dt)) if pd.Timestamp(dt) in rebs else None
        log(f"holdings {dt} guarded: {sorted(guarded[i]) if i is not None else 'n/a'}")
        log(f"holdings {dt} plain  : {sorted(plain[i]) if i is not None else 'n/a'}")
    for dt in ["2021-08-30", "2024-01-08", "2024-12-02"]:
        i = rebs.index(pd.Timestamp(dt)) if pd.Timestamp(dt) in rebs else None
        log(f"gate {dt}: half={half[i] if i is not None else 'n/a'} pct={pct.loc[:pd.Timestamp(ts[i])].iloc[-1]:.3f}"
            if i is not None else f"gate {dt}: n/a")
    ge, ga = res["guarded"][0], res["gated"][0]
    pd.DataFrame({"date": ge.index.date, "guarded_equity": ge.values, "gated_equity": ga.reindex(ge.index).values,
                  "half_flag": [int(half[rebs.index(d)]) for d in ge.index]}).to_csv(D + "REBUILD_1700tu_weekly.csv", index=False)
    by = pd.DataFrame({k: v[3] for k, v in res.items()}); by.index.name = "year"
    by.round(2).to_csv(D + "REBUILD_1700tu_by_year.csv")
    log("by year:\n" + by.round(1).to_string())
    log("DONE")


if __name__ == "__main__":
    main()
