"""
INDEPENDENT REBUILD of cells 1,649-1,651 (index overnight premium / conditional overnight / turn-of-month).
Built solely from research/index_overnight/PREREG_1649.md (incl. Amendment 1). Did NOT read cell_1649.py,
RESULT_1649_build.md, nights_1649.csv or tom_1651.csv before writing this file.

Data: research/lev_flow/data/minute/{SPY,QQQ,IWM}.parquet (Alpaca adjustment=all, SIP, extended hours),
      research/lev_flow/data/daily/{SPY,QQQ,IWM}.parquet. No fetching. TEST (>=2024-01-01) never touched.

Cost model interpretation: PREREG says "0.5 bp per auction leg plus 0.5 bp half-spread (1 bp per round trip is
generous)". Two auction legs per trade (buy MOC/sell MOO, or buy MOC/sell MOC) at 0.5bp all-in each => 1bp round
trip, matching the stated parenthetical exactly. Applied as buy_eff=price*(1+0.00005), sell_eff=price*(1-0.00005).
"""
import numpy as np
import pandas as pd
from pathlib import Path

ETFS = ["SPY", "QQQ", "IWM"]
DATA_MIN = Path("research/lev_flow/data/minute")
DATA_DAY = Path("research/lev_flow/data/daily")
OUT = Path("research/index_overnight")
COST_LEG = 0.00005  # 0.5 bp per auction leg
TEST_START = pd.Timestamp("2024-01-01").date()

# ---------- loaders ----------

def load_daily(sym):
    df = pd.read_parquet(DATA_DAY / f"{sym}.parquet")
    df["date"] = df["t"].dt.tz_convert("America/New_York").dt.date
    df = df.sort_values("date").reset_index(drop=True)
    return df[["date", "o", "c", "v"]]


def load_minute(sym):
    df = pd.read_parquet(DATA_MIN / f"{sym}.parquet")
    df["et"] = df["t"].dt.tz_convert("America/New_York")
    df["date"] = df["et"].dt.date
    df["tod"] = df["et"].dt.hour * 60 + df["et"].dt.minute
    return df[["date", "tod", "v"]]


def detect_early_closes(mdf):
    """Empirically detect NYSE 1pm early closes from the bar data itself (no pandas_market_calendars
    installed): the closing auction is a huge single-minute volume spike at the close. On a normal day
    that spike sits at tod=960 (16:00) and tod=780 (13:00) is an ordinary mid-session minute
    (v960 >> v780). On an early close the auction spike moves to tod=780 and 16:00 is post-close chatter
    (v780 >> v960). Verified directly on 2018-12-24 (known half day: v780=4.23M, v960=1,900) vs
    2018-12-26 (normal: v780=489K, v960=9.17M) before adopting this rule."""
    v780 = mdf[mdf["tod"] == 780].groupby("date")["v"].sum()
    v960 = mdf[mdf["tod"] == 960].groupby("date")["v"].sum()
    j = pd.concat([v780.rename("v780"), v960.rename("v960")], axis=1).fillna(0)
    # scale-free (ratio) test so it works for IWM/QQQ too, not just SPY's higher absolute volume
    return set(j[(j["v780"] > 3 * j["v960"]) & (j["v780"] > 1e5)].index)


# ---------- stats helpers ----------

def cluster_mean_t(values, clusters):
    """Cluster-robust t-stat for a simple mean (CR1 sandwich on an intercept-only regression)."""
    v = np.asarray(values, dtype=float)
    c = np.asarray(clusters)
    n = len(v)
    mean = v.mean()
    resid = v - mean
    csum = pd.Series(resid).groupby(c).sum().values
    g = len(csum)
    var = (csum ** 2).sum() / (n ** 2)
    if g > 1:
        var *= g / (g - 1)
    se = np.sqrt(var) if var > 0 else np.nan
    t = mean / se if se and se > 0 else np.nan
    return mean, se, t, g, n


def cluster_ols_diff(y, d, clusters):
    """Cluster-robust OLS of y on [1, d]; returns coef/se/t on d (the group-difference / 'excess')."""
    y = np.asarray(y, dtype=float)
    d = np.asarray(d, dtype=float)
    X = np.column_stack([np.ones(len(y)), d])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    uniq = np.unique(clusters)
    meat = np.zeros((2, 2))
    for cl in uniq:
        idx = clusters == cl
        Xg, ug = X[idx], resid[idx]
        score = Xg.T @ ug
        meat += np.outer(score, score)
    g, n, k = len(uniq), len(y), 2
    corr = (g / (g - 1)) * ((n - 1) / (n - k)) if g > 1 else 1.0
    cov = XtX_inv @ meat @ XtX_inv * corr
    se = np.sqrt(np.diag(cov))
    t = beta / se
    return beta[1], se[1], t[1]


def net_ret(buy_px, sell_px):
    buy_eff = buy_px * (1 + COST_LEG)
    sell_eff = sell_px * (1 - COST_LEG)
    return sell_eff / buy_eff - 1.0


# ---------- load everything ----------
daily = {s: load_daily(s) for s in ETFS}
minute = {s: load_minute(s) for s in ETFS}
early_flags = {s: detect_early_closes(minute[s]) for s in ETFS}
# require agreement in >=2 of 3 ETFs (NYSE-wide event) to be robust to a single-series artifact
from collections import Counter
cnt = Counter()
for s in ETFS:
    cnt.update(early_flags[s])
EARLY_CLOSES = sorted(d for d, c in cnt.items() if c >= 2)
print(f"Detected {len(EARLY_CLOSES)} early-close dates (>=2/3 ETFs agree): {EARLY_CLOSES}")

# ================= CELL 1,649: unconditional overnight =================
rows = []
for sym in ETFS:
    d = daily[sym]
    d = d[~d["date"].isin(EARLY_CLOSES)].reset_index(drop=True)
    for i in range(len(d) - 1):
        di, dj = d.loc[i], d.loc[i + 1]
        if dj["date"] >= TEST_START:
            continue
        if di["date"].year < 2016 or di["date"].year > 2023:
            continue
        gross = dj["o"] / di["c"] - 1.0
        net = net_ret(di["c"], dj["o"])
        rows.append(dict(etf=sym, date=di["date"], next_date=dj["date"], year=di["date"].year,
                          gross_bps=gross * 1e4, net_bps=net * 1e4))
nights_1649 = pd.DataFrame(rows).sort_values(["date", "etf"]).reset_index(drop=True)
nights_1649.to_csv(OUT / "nights_1649_rebuild.csv", index=False)
print(f"1649: {len(nights_1649)} ETF-nights, {nights_1649['date'].nunique()} unique nights")

# 24h buy-and-hold gross return (close-to-close), same aligned pairs, for Sharpe comparison
bh_rows = []
for sym in ETFS:
    d = daily[sym]
    d = d[~d["date"].isin(EARLY_CLOSES)].reset_index(drop=True)
    for i in range(len(d) - 1):
        di, dj = d.loc[i], d.loc[i + 1]
        if dj["date"] >= TEST_START or di["date"].year < 2016 or di["date"].year > 2023:
            continue
        bh_rows.append(dict(etf=sym, date=di["date"], bh_bps=(dj["c"] / di["c"] - 1) * 1e4))
bh = pd.DataFrame(bh_rows)

pooled_mean, pooled_se, pooled_t, g, n = cluster_mean_t(nights_1649["net_bps"], nights_1649["date"])
print(f"1649 POOLED 2016-2023: n={n} nights(ETF-obs) clusters={g} mean={pooled_mean:.3f}bps t={pooled_t:.2f}")

per_etf_1649 = {}
for sym in ETFS:
    sub = nights_1649[nights_1649.etf == sym]
    m, se, t, g_, n_ = cluster_mean_t(sub["net_bps"], sub["date"])
    on_sharpe = sub["net_bps"].mean() / sub["net_bps"].std(ddof=1) * np.sqrt(252)
    bh_sub = bh[bh.etf == sym]
    bh_sharpe = bh_sub["bh_bps"].mean() / bh_sub["bh_bps"].std(ddof=1) * np.sqrt(252)
    ex_worst = sub.sort_values("net_bps").iloc[int(np.ceil(len(sub) * 0.05)):]["net_bps"].mean()
    cum = (1 + sub.sort_values("date")["net_bps"] / 1e4).cumprod()
    dd = (cum / cum.cummax() - 1).min()
    worst_night_bps = sub["net_bps"].min()
    worst_night_dollar = worst_night_bps / 1e4 * 20000
    per_etf_1649[sym] = dict(n=n_, mean=m, t=t, on_sharpe=on_sharpe, bh_sharpe=bh_sharpe,
                              ex_worst5=ex_worst, maxdd=dd, worst_bps=worst_night_bps, worst_dollar=worst_night_dollar)
    print(f"  {sym}: n={n_} mean={m:.3f}bps t={t:.2f} ON_Sharpe={on_sharpe:.2f} BH_Sharpe={bh_sharpe:.2f} "
          f"exW5={ex_worst:.3f} maxDD={dd*100:.2f}% worst=${worst_night_dollar:.0f}")

per_year_1649 = nights_1649.groupby(["year", "etf"])["net_bps"].mean().unstack()
print("1649 per-year per-ETF mean net bps:\n", per_year_1649.round(2))

# pooled equal-weighted sleeve (avg of 3 ETFs per calendar night) for drawdown/worst-night-$
sleeve = nights_1649.groupby("date")["net_bps"].mean().sort_index()
sleeve_cum = (1 + sleeve / 1e4).cumprod()
sleeve_dd = (sleeve_cum / sleeve_cum.cummax() - 1).min()
sleeve_worst_bps = sleeve.min()
sleeve_worst_dollar = sleeve_worst_bps / 1e4 * 20000
sleeve_ann = (sleeve_cum.iloc[-1] ** (252 / len(sleeve)) - 1)
ex_worst5_pooled = sleeve.sort_values().iloc[int(np.ceil(len(sleeve) * 0.05)):].mean()
n_years_pos = (per_year_1649.mean(axis=1) > 0).sum()
n_etf_pos = (per_etf_1649[s]["mean"] > 0 for s in ETFS)
print(f"1649 SLEEVE(equal-wt 3ETF): maxDD={sleeve_dd*100:.2f}% worst_night=${sleeve_worst_dollar:.0f} "
      f"annret={sleeve_ann*100:.2f}% ex_worst5={ex_worst5_pooled:.3f}bps years_pos={n_years_pos}/8")

# ================= CELL 1,650: conditional overnight =================
st_rows = []
for sym in ETFS:
    m = minute[sym]
    m = m[~m["date"].isin(EARLY_CLOSES)]
    sess = m[(m["tod"] >= 570) & (m["tod"] <= 960)].groupby("date")["v"].sum().rename("sess_v")
    last30 = m[(m["tod"] >= 930) & (m["tod"] <= 960)].groupby("date")["v"].sum().rename("last30_v")
    st = pd.concat([sess, last30], axis=1).fillna(0).sort_index()
    st["s_t"] = st["last30_v"] / st["sess_v"]
    st["trail_med"] = st["s_t"].shift(1).rolling(60, min_periods=60).median()
    st["heavy"] = st["s_t"] >= 1.25 * st["trail_med"]
    st["etf"] = sym
    st = st.reset_index().rename(columns={"index": "date"})
    st_rows.append(st)
st_all = pd.concat(st_rows, ignore_index=True).dropna(subset=["trail_med"])

# join to next-night net_bps from the 1649 night table (same night definition, no early closes, no TEST)
nxt = nights_1649.rename(columns={"date": "date"})[["etf", "date", "net_bps", "year"]]
cond = st_all.merge(nxt, on=["etf", "date"], how="inner")
cond["half"] = np.where(cond["year"] <= 2019, "TRAIN", np.where(cond["year"] <= 2023, "VAL", "TEST"))
cond = cond[cond["half"] != "TEST"]

for half in ["TRAIN", "VAL"]:
    sub = cond[cond.half == half]
    heavy = sub[sub.heavy]
    comp = sub[~sub.heavy]
    hm, hse, ht, hg, hn = cluster_mean_t(heavy["net_bps"], heavy["date"])
    exw5 = heavy.sort_values("net_bps").iloc[int(np.ceil(len(heavy) * 0.05)):]["net_bps"].mean()
    excess, exc_se, exc_t = cluster_ols_diff(sub["net_bps"], sub["heavy"].astype(float), sub["date"])
    print(f"1650 {half}: heavy n={hn} clusters={hg} mean={hm:.3f}bps t={ht:.2f} exW5={exw5:.3f} "
          f"| excess_over_complement={excess:.3f}bps t={exc_t:.2f} | complement_n={len(comp)}")
    per_etf_h = heavy.groupby("etf")["net_bps"].mean()
    print(f"   per-ETF heavy mean: {per_etf_h.round(3).to_dict()}")
    # tercile table (cut within this half, pooled across ETFs, on ALL valid nights not just heavy)
    terc = pd.qcut(sub["s_t"], 3, labels=["T1_low", "T2_mid", "T3_high"])
    tt = sub.groupby(terc, observed=False)["net_bps"].mean()
    print(f"   tercile table (next-night net bps): {tt.round(3).to_dict()}")

# ================= CELL 1,651: turn-of-month =================
tom_rows = []
for sym in ETFS:
    d = daily[sym].copy()
    d["ym"] = d["date"].apply(lambda x: (x.year, x.month))
    last_session = d.groupby("ym")["date"].max()
    first3 = d.groupby("ym")["date"].apply(lambda s: sorted(s)[:3])
    months_sorted = sorted(last_session.index)
    for idx in range(len(months_sorted) - 1):
        ym = months_sorted[idx]
        nym = months_sorted[idx + 1]
        entry_date = last_session[ym]
        nxt3 = first3[nym]
        if len(nxt3) < 3:
            continue
        exit_date = nxt3[2]
        if entry_date.year < 2016 or entry_date.year > 2023:
            continue
        if exit_date >= TEST_START:
            continue
        if entry_date in EARLY_CLOSES or exit_date in EARLY_CLOSES:
            continue
        c_entry = d.loc[d.date == entry_date, "c"].iloc[0]
        c_exit = d.loc[d.date == exit_date, "c"].iloc[0]
        gross = c_exit / c_entry - 1
        net = net_ret(c_entry, c_exit)
        tom_rows.append(dict(etf=sym, entry_date=entry_date, exit_date=exit_date, event_month=ym,
                              year=entry_date.year, gross_bps=gross * 1e4, net_bps=net * 1e4))
tom_1651 = pd.DataFrame(tom_rows).sort_values(["entry_date", "etf"]).reset_index(drop=True)
tom_1651.to_csv(OUT / "tom_1651_rebuild.csv", index=False)

pm, pse, pt, pg, pn = cluster_mean_t(tom_1651["net_bps"], tom_1651["event_month"])
print(f"1651 POOLED 2016-2023: n={pn} clusters(months)={pg} mean={pm:.3f}bps t={pt:.2f}")
ex_top5_1651 = tom_1651.sort_values("net_bps").iloc[:-int(np.ceil(len(tom_1651) * 0.05))]["net_bps"].mean()
ex_worst5_1651 = tom_1651.sort_values("net_bps").iloc[int(np.ceil(len(tom_1651) * 0.05)):]["net_bps"].mean()
no2020 = tom_1651[tom_1651.year != 2020]
m_no2020, _, t_no2020, _, n_no2020 = cluster_mean_t(no2020["net_bps"], no2020["event_month"])
per_year_1651 = tom_1651.groupby(["year", "etf"])["net_bps"].mean().unstack()
years_pos = (per_year_1651.mean(axis=1) > 0).sum()
etfs_pos = (tom_1651.groupby("etf")["net_bps"].mean() > 0).sum()
print(f"1651 ex_top5%={ex_top5_1651:.3f}bps ex_worst5%={ex_worst5_1651:.3f}bps "
      f"no_2020: n={n_no2020} mean={m_no2020:.3f}bps t={t_no2020:.2f}")
print(f"1651 years_positive(pooled avg)={years_pos}/8 etfs_positive={etfs_pos}/3")
print("1651 per-year per-ETF:\n", per_year_1651.round(2))
print("1651 per-ETF pooled:\n", tom_1651.groupby("etf")["net_bps"].agg(["count", "mean"]).round(3))
largest_event = tom_1651.loc[tom_1651["net_bps"].idxmin()]
print(f"1651 worst single event: {largest_event.to_dict()}")

# refuter #1: first three events
first3_events = tom_1651[tom_1651.etf == "SPY"].sort_values("entry_date").head(3)
print("1651 refuter#1 first three SPY events:\n", first3_events[["entry_date", "exit_date", "event_month"]])

# refuter #2: MOC (daily close) vs 16:00 minute bar, mean abs diff bps on TOM event days
diffs = []
for sym in ETFS:
    m = minute[sym]
    close_bar = m[m["tod"] == 960][["date"]].copy()
    # get the 16:00 bar's close price -> need 'c' column; reload minute close prices for those dates
    mm = pd.read_parquet(DATA_MIN / f"{sym}.parquet")
    mm["et"] = mm["t"].dt.tz_convert("America/New_York")
    mm["date"] = mm["et"].dt.date
    mm["tod"] = mm["et"].dt.hour * 60 + mm["et"].dt.minute
    bar16 = mm[mm["tod"] == 960][["date", "c"]].rename(columns={"c": "close_bar_c"})
    ev_dates = pd.concat([tom_1651.loc[tom_1651.etf == sym, "entry_date"],
                           tom_1651.loc[tom_1651.etf == sym, "exit_date"]]).unique()
    dsub = daily[sym][daily[sym].date.isin(ev_dates)][["date", "c"]].rename(columns={"c": "daily_c"})
    j = dsub.merge(bar16, on="date", how="inner")
    j["diff_bps"] = (j["daily_c"] / j["close_bar_c"] - 1).abs() * 1e4
    diffs.append(j["diff_bps"])
    print(f"  {sym}: n_matched={len(j)} mean|diff|={j['diff_bps'].mean():.3f}bps max={j['diff_bps'].max():.3f}bps")
all_diffs = pd.concat(diffs)
print(f"1651 refuter#2 ALL: mean|diff|={all_diffs.mean():.3f}bps n={len(all_diffs)}")

# refuter #4: share of TOM nights that are also 1650 heavy-close nights
tom_nights = set()
for _, r in tom_1651.iterrows():
    tom_nights.add((r["etf"], r["entry_date"]))
heavy_set = set(zip(cond.loc[cond.heavy, "etf"], cond.loc[cond.heavy, "date"]))
overlap = len(tom_nights & heavy_set)
print(f"1651 refuter#4 TOM entry-nights that are ALSO 1650-heavy: {overlap}/{len(tom_nights)}")

print("DONE")
