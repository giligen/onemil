"""Cells 1,714-1,716: market intraday momentum on SPY/QQQ (PREREG_1714.md, frozen 2026-10-29).

First half-hour return -> last half-hour return.  TEST (2023-01 onward) is NEVER loaded: the 1-min parquet is
cut at 2022-12-31 immediately after reading.

Usage:
  python3 cell_1714.py --split train --symbol SPY      # computes + FREEZES TRAIN thresholds (thresholds_<SYM>.json)
  python3 cell_1714.py --split val   --symbol SPY      # refuses to run without frozen thresholds
  python3 cell_1714.py --assemble                      # RESULT_1714.md from part files + caveats_1714.md

Definitions (ET): r1 = close(09:59)/open(09:30)-1 ; r12 = close(15:29)/close(09:59)-1 ;
rday = close(15:29)/open(09:30)-1 ; r13 = close(15:59)/open(15:31)-1 (TARGET).
Entry = open of the 15:31 bar (market order at 15:30:30), exit = close of the 15:59 bar (proxy for the MOC print).
Cost = --cost-bps per side (default 1) -> round trip 2*cost.  Sensitivity 3 bps per side.  R := 25 bps.
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "index_orb", "cache")
HARD_CUTOFF = pd.Timestamp("2023-01-01")      # TEST is sealed: nothing at/after this date is ever kept
TRAIN_END = pd.Timestamp("2019-12-31")
VAL_START = pd.Timestamp("2020-01-01")
R_BPS = 25.0
MIN_BARS = 380
NULL_DRAWS = 1000
CELLS = ["1714A", "1714B", "1714C", "1715P50", "1715P70", "1715P85"]
CELL_DESC = {
    "1714A": "sign(r1)", "1714B": "sign(r1+r12)", "1714C": "sign(r1) if |r1|>TRAIN median",
    "1715P50": "sign(rday) if |rday|>TRAIN P50", "1715P70": "sign(rday) if |rday|>TRAIN P70 (PRIMARY)",
    "1715P85": "sign(rday) if |rday|>TRAIN P85",
}


def log(msg):
    """Verbose progress, flushed."""
    print(msg, flush=True)


def load_bars(sym):
    """Read the 1-min parquet and drop everything at/after the TEST start immediately."""
    df = pd.read_parquet(os.path.join(CACHE, f"{sym}_1min.parquet"))
    ts = df["timestamp"].dt.tz_localize(None)
    df = df.loc[ts < HARD_CUTOFF].copy()
    df["date"] = df["timestamp"].dt.tz_localize(None).dt.normalize()
    df["tod"] = df["timestamp"].dt.strftime("%H:%M")
    df = df.drop_duplicates(subset=["timestamp"])
    assert df["date"].max() < HARD_CUTOFF, "TEST leak"
    return df


def build_days(sym):
    """One row per eligible day (r1, r12, rday, r13) + exclusion table; also daily closes for the vol proxy."""
    df = load_bars(sym)
    nbars = df.groupby("date").size()
    need = {"o0930": ("09:30", "open"), "c0959": ("09:59", "close"), "c1529": ("15:29", "close"),
            "o1531": ("15:31", "open"), "c1559": ("15:59", "close")}
    cols = {}
    for k, (tod, fld) in need.items():
        sub = df.loc[df["tod"] == tod]
        cols[k] = sub.set_index("date")[fld]
    px = pd.DataFrame(cols)
    px["nbars"] = nbars
    daily_close = df.groupby("date")["close"].last()
    allday = pd.DataFrame(index=nbars.index)
    allday = allday.join(px)
    reason = pd.Series("ok", index=allday.index)
    reason[allday["nbars"] < MIN_BARS] = "lt380_bars"
    miss = allday[list(need)].isna().any(axis=1) & (reason == "ok")
    reason[miss] = "missing_key_bar"
    allday["reason"] = reason
    ok = allday[allday["reason"] == "ok"].copy()
    out = pd.DataFrame(index=ok.index)
    out["r1"] = ok["c0959"] / ok["o0930"] - 1
    out["r12"] = ok["c1529"] / ok["c0959"] - 1
    out["rday"] = ok["c1529"] / ok["o0930"] - 1
    out["r13"] = ok["c1559"] / ok["o1531"] - 1
    return out, allday["reason"], daily_close


def split_mask(idx, split):
    """Boolean mask of the split on a DatetimeIndex."""
    if split == "train":
        return idx <= TRAIN_END
    if split == "val":
        return (idx >= VAL_START) & (idx < HARD_CUTOFF)
    raise ValueError(split)


def thresholds_from_train(days):
    """TRAIN-derived frozen thresholds: |r1| median, |rday| P50/P70/P85."""
    tr = days[split_mask(days.index, "train")]
    return {"abs_r1_median": float(np.median(tr["r1"].abs())),
            "abs_rday_p50": float(np.percentile(tr["rday"].abs(), 50)),
            "abs_rday_p70": float(np.percentile(tr["rday"].abs(), 70)),
            "abs_rday_p85": float(np.percentile(tr["rday"].abs(), 85)),
            "n_train_days": int(len(tr))}


def positions(days, th):
    """Position (+1/-1/0) per day for every cell."""
    s = np.sign
    return {
        "1714A": s(days["r1"]),
        "1714B": s(days["r1"] + days["r12"]),
        "1714C": s(days["r1"]) * (days["r1"].abs() > th["abs_r1_median"]),
        "1715P50": s(days["rday"]) * (days["rday"].abs() > th["abs_rday_p50"]),
        "1715P70": s(days["rday"]) * (days["rday"].abs() > th["abs_rday_p70"]),
        "1715P85": s(days["rday"]) * (days["rday"].abs() > th["abs_rday_p85"]),
    }


def tstat(x):
    """iid t of the mean."""
    x = np.asarray(x, float)
    if len(x) < 3 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x))))


def week_keys(idx):
    """ISO year-week label per date."""
    iso = idx.isocalendar()
    return (iso["year"].astype(int) * 100 + iso["week"].astype(int)).values


def block_boot_ci(net, keys, draws=2000, seed=7):
    """Block bootstrap 95% CI of the mean; blocks = unique keys, resampled with replacement."""
    rng = np.random.default_rng(seed)
    ukeys = np.unique(keys)
    groups = [net[keys == k] for k in ukeys]
    means = []
    for _ in range(draws):
        pick = rng.integers(0, len(groups), len(groups))
        means.append(np.concatenate([groups[i] for i in pick]).mean())
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def weekly_table(net_by_day, all_days_idx, trade_mask):
    """Weekly sums in R for every eligible week (empty weeks = 0) and the cadence stats."""
    wk_all = week_keys(all_days_idx)
    uw = np.unique(wk_all)
    pos = {w: i for i, w in enumerate(uw)}
    sums = np.zeros(len(uw))
    wk_day = np.array([pos[w] for w in wk_all])
    np.add.at(sums, wk_day, np.where(trade_mask, net_by_day, 0.0) / R_BPS)
    strong = np.where(sums >= 5)[0]
    gaps = np.diff(strong) if len(strong) >= 2 else np.array([])
    return {"weeks": len(uw), "p10": float(np.percentile(sums, 10)), "green": float((sums > 0).mean()),
            "strong_share": float((sums >= 5).mean()), "n_strong": int(len(strong)),
            "gap_med": float(np.median(gaps)) if len(gaps) else float("nan"),
            "gap_p90": float(np.percentile(gaps, 90)) if len(gaps) else float("nan"),
            "sums": sums, "wk_day": wk_day}


def green_null(gross_bps, trade_mask, wk_day, nweeks, cost_rt, longonly, observed_n, seed=11):
    """Count-matched random-direction null: share of green weeks over NULL_DRAWS draws.
    LS: random +-1 on the same trade days.  LO: random subset of the LS trade days of the same size, long."""
    rng = np.random.default_rng(seed)
    tidx = np.where(trade_mask)[0]
    n = len(tidx)
    shares = np.empty(NULL_DRAWS)
    for d in range(NULL_DRAWS):
        net = np.zeros(len(gross_bps))
        if longonly:
            pick = rng.choice(tidx, size=observed_n, replace=False)
            net[pick] = gross_bps[pick] - cost_rt
        else:
            sg = rng.choice([-1.0, 1.0], size=n)
            net[tidx] = sg * gross_bps[tidx] - cost_rt
        sums = np.zeros(nweeks)
        np.add.at(sums, wk_day, net / R_BPS)
        shares[d] = (sums > 0).mean()
    return shares


def cell_stats(days, pos, cost, longonly, ls_mask_for_null):
    """All pass-bar statistics for one cell (days = split slice, pos aligned)."""
    cost_rt = 2 * cost
    p = pos.values.copy()
    if longonly:
        p = np.where(p > 0, p, 0)
    traded = p != 0
    gross_all = days["r13"].values * 1e4
    gross = p * gross_all - 0  # signed gross bps (0 on non-trade days)
    net_day = np.where(traded, gross - cost_rt, 0.0)
    net = net_day[traded]
    n = int(traded.sum())
    out = {"n": n}
    if n < 5:
        out["ok"] = False
        return out
    wk = week_keys(days.index)
    yr = days.index.year.values[traded]
    mo = (days.index.year.values * 100 + days.index.month.values)[traded]
    k = int(math.ceil(0.05 * n))
    srt = np.sort(net)
    wt = weekly_table(net_day, days.index, traded)
    nweeks = wt["weeks"]
    shares = green_null(gross_all, ls_mask_for_null, wt["wk_day"], nweeks, cost_rt, longonly, n)
    out.update({
        "ok": True, "gross_mean": float(gross[traded].mean()), "mean": float(net.mean()), "sd": float(net.std(ddof=1)),
        "t": tstat(net), "ci_year": block_boot_ci(net, yr), "ci_month": block_boot_ci(net, mo),
        "ex5": float(srt[:-k].mean()), "cap50": float(np.minimum(net, 50).mean()),
        "tpw": n / nweeks, "mean_3bps": float((gross[traded] - 6.0).mean()),
        "mean_3x": float((3 * gross[traded] - 6.0).mean()),
        "green": wt["green"], "null_mean": float(shares.mean()), "null_p95": float(np.percentile(shares, 95)),
        "null_p": float((shares >= wt["green"]).mean()), "wt": wt, "net_day": net_day, "traded": traded,
        "win": float((net > 0).mean()),
    })
    return out


def fmt(x, d=2):
    """Format a float, nan-safe."""
    return "nan" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def mdtable(header, rows):
    """Markdown table."""
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def ols(days):
    """OLS r13 ~ 1 + r1 + r12 in bps/bps units with HC1 robust t."""
    y = days["r13"].values * 1e4
    X = np.column_stack([np.ones(len(y)), days["r1"].values * 1e4, days["r12"].values * 1e4])
    b = np.linalg.solve(X.T @ X, X.T @ y)
    e = y - X @ b
    xtxi = np.linalg.inv(X.T @ X)
    meat = (X * e[:, None] ** 2).T @ X
    V = xtxi @ meat @ xtxi * len(y) / (len(y) - 3)
    se = np.sqrt(np.diag(V))
    r2 = 1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return b, b / se, r2


def decile_table(days):
    """Decile of r1 -> mean r13 (bps), n, t."""
    q = pd.qcut(days["r1"], 10, labels=False, duplicates="drop")
    rows = []
    for d in sorted(q.unique()):
        m = days.loc[q == d]
        rows.append([int(d) + 1, f"{m['r1'].mean() * 1e4:.1f}", len(m), fmt(m["r13"].mean() * 1e4),
                     fmt(tstat(m["r13"] * 1e4))])
    mono = np.corrcoef([r[0] for r in rows], [float(r[3]) for r in rows])[0, 1]
    return rows, float(mono)


def run(split, sym, cost):
    """Compute one (split, symbol) part and write trades CSVs + part markdown."""
    log(f"[{sym}/{split}] building days (TEST cut at load)")
    days_all, reason, _ = build_days(sym)
    spy_days_all, _, spy_close = build_days("SPY") if sym != "SPY" else (days_all, reason, None)
    if sym == "SPY":
        spy_close = load_bars("SPY").groupby("date")["close"].last()
    th_path = os.path.join(HERE, f"thresholds_{sym}.json")
    th_now = thresholds_from_train(days_all)
    if split == "train":
        json.dump(th_now, open(th_path, "w"), indent=1)
        th = th_now
        log(f"[{sym}] FROZEN TRAIN thresholds -> {th_path}: {th}")
    else:
        if not os.path.exists(th_path):
            sys.exit(f"ERROR: {th_path} missing - run --split train first (thresholds must be frozen before VAL)")
        th = json.load(open(th_path))
        assert all(abs(th[k] - th_now[k]) < 1e-12 for k in th if k != "n_train_days"), "threshold drift"
    sm = split_mask(days_all.index, split)
    days = days_all[sm]
    rs = reason[split_mask(reason.index, split)]
    P = positions(days, th)
    ls_mask = np.ones(len(days), bool)
    lines = [f"## {sym} {split.upper()} (cost {cost} bp/side, round trip {2 * cost} bp)\n"]
    # ---- data check
    cnt = rs.value_counts()
    lines.append("### Data check\n")
    lines.append(mdtable(["calendar session days", "eligible", "excluded <380 bars", "excluded missing key bar",
                          "first", "last"],
                         [[len(rs), int((rs == "ok").sum()), int(cnt.get("lt380_bars", 0)),
                           int(cnt.get("missing_key_bar", 0)), str(days.index.min().date()),
                           str(days.index.max().date())]]))
    lines.append("\n(Exclusion criteria exactly as PREREG: <380 bars, or no 09:30/09:59/15:29/15:31/15:59 bar; "
                 "half-day sessions fall under <380 bars.)\n")
    lines.append(f"### Frozen TRAIN thresholds ({sym}, from {th['n_train_days']} TRAIN days)\n")
    lines.append(mdtable(["|r1| median", "|rday| P50", "|rday| P70", "|rday| P85"],
                         [[f"{th['abs_r1_median'] * 1e4:.2f} bp", f"{th['abs_rday_p50'] * 1e4:.2f} bp",
                           f"{th['abs_rday_p70'] * 1e4:.2f} bp", f"{th['abs_rday_p85'] * 1e4:.2f} bp"]]) + "\n")
    # ---- cells
    res = {}
    for c in CELLS:
        for lo in (False, True):
            res[(c, lo)] = cell_stats(days, P[c], cost, lo, ls_mask & (P[c].values != 0))
        # trades CSV (long-short positions)
        p = P[c]
        tr = days.loc[p != 0, ["r1", "r12", "rday"]].copy()
        tr["position"] = p[p != 0].astype(int)
        tr["r13_gross"] = days.loc[p != 0, "r13"]
        tr["net"] = tr["position"] * tr["r13_gross"] - 2 * cost * 1e-4
        tr["net_longonly"] = np.where(tr["position"] > 0, tr["net"], np.nan)
        tr.index.name = "date"
        tr.to_csv(os.path.join(HERE, f"trades_1714_{c}_{split}_{sym}.csv"), float_format="%.8f")
    lines.append("### Cell table (net bps per trade; R = 25 bp)\n")
    hdr = ["cell", "side", "n", "trades/wk", "gross", "net", "sd", "iid t", "yr-boot CI", "mo-boot CI", "ex-top-5%",
           "cap+50", "win%", "net@3bp/side", "3x net", "green wk", "null mean/P95 (p)"]
    rows = []
    for c in CELLS:
        for lo in (False, True):
            r = res[(c, lo)]
            side = "LO" if lo else "LS"
            if not r["ok"]:
                rows.append([c, side, r["n"]] + ["n/a"] * (len(hdr) - 3))
                continue
            rows.append([c, side, r["n"], fmt(r["tpw"]), fmt(r["gross_mean"]), fmt(r["mean"]), fmt(r["sd"], 1),
                         fmt(r["t"]), f"[{fmt(r['ci_year'][0])},{fmt(r['ci_year'][1])}]",
                         f"[{fmt(r['ci_month'][0])},{fmt(r['ci_month'][1])}]", fmt(r["ex5"]), fmt(r["cap50"]),
                         fmt(100 * r["win"], 1), fmt(r["mean_3bps"]), fmt(r["mean_3x"]),
                         fmt(100 * r["green"], 1) + "%",
                         f"{fmt(100 * r['null_mean'], 1)}/{fmt(100 * r['null_p95'], 1)} ({fmt(r['null_p'], 3)})"])
    lines.append(mdtable(hdr, rows) + "\n")
    lines.append("Cells: " + "; ".join(f"{c} = {CELL_DESC[c]}" for c in CELLS) + ". LS = long-short, LO = long-only. "
                 "yr-boot = year-block bootstrap (few blocks, crude); mo-boot = month-block.\n")
    # ---- gates (VAL only)
    if split == "val":
        partner = "QQQ" if sym == "SPY" else "SPY"
        pdays_all, _, _ = build_days(partner)
        pth = thresholds_from_train(pdays_all)
        pdays = pdays_all[split_mask(pdays_all.index, "val")]
        pP = positions(pdays, pth)
        trdays = days_all[split_mask(days_all.index, "train")]
        trP = positions(trdays, th)
        lines.append("### Pass bar on VAL (every item PASS/FAIL; sleeve = SPY, QQQ must agree)\n")
        glines = []
        for c in CELLS:
            for lo in (False, True):
                r = res[(c, lo)]
                side = "LO" if lo else "LS"
                if not r["ok"]:
                    glines.append([c, side, "n too small"] + [""] * 8)
                    continue
                tr_r = cell_stats(trdays, trP[c], cost, lo, trP[c].values != 0)
                pr = cell_stats(pdays, pP[c], cost, lo, pP[c].values != 0)
                g = []
                g.append(("mean>=+2", r["mean"], r["mean"] >= 2))
                g.append(("t>=2", r["t"], r["t"] >= 2))
                g.append(("ex5>0", r["ex5"], r["ex5"] > 0))
                g.append(("cap50>0", r["cap50"], r["cap50"] > 0))
                g.append(("tpw>=2", r["tpw"], r["tpw"] >= 2))
                g.append(("green>=55 & >null", r["green"] * 100,
                          r["green"] >= 0.55 and r["green"] > r["null_mean"]))
                trt = tr_r["t"] if tr_r["ok"] else float("nan")
                same = tr_r["ok"] and np.sign(tr_r["mean"]) == np.sign(r["mean"]) and r["mean"] > 0 and trt >= 1
                g.append(("TRAIN same sign t>=1", tr_r["mean"] if tr_r["ok"] else float("nan"), bool(same)))
                g.append((f"{partner} VAL mean>0", pr["mean"] if pr["ok"] else float("nan"),
                          bool(pr["ok"] and pr["mean"] > 0)))
                npass = sum(1 for x in g if x[2])
                if r["mean"] <= 0 or r["ex5"] <= 0:
                    verdict = "REJECT"
                elif npass == len(g):
                    verdict = "PASS"
                elif r["t"] < 2:
                    verdict = "UNDETERMINED"
                else:
                    verdict = "FAIL"
                glines.append([c, side, f"{npass}/{len(g)}", verdict] +
                              [f"{fmt(x[1])} {'PASS' if x[2] else 'FAIL'}" for x in g] +
                              [f"TRAIN t={fmt(trt)}" if False else ""])
        hdr2 = ["cell", "side", "pass", "verdict", "mean>=+2bp", "t>=2", "ex-top-5%>0", "cap+50>0", ">=2 trades/wk",
                "green>=55% & >null", "TRAIN same sign, t>=1 (value=TRAIN mean)", f"{partner} VAL mean>0"]
        glines = [r[:12] for r in glines]
        lines.append(mdtable(hdr2, glines) + "\n")
        lines.append("Green-week gate = observed >= 55% AND observed > null mean (null P95 and p shown in the cell "
                     "table). Verdict rule (PREREG): all 8 gates -> PASS; mean<=0 or ex-top-5%<=0 -> REJECT; "
                     "mean>0 with t<2 -> UNDETERMINED.\n")
    # ---- weekly
    lines.append("### Weekly table (weekly sum of net in R = 25 bp; empty weeks count 0)\n")
    wrows = []
    for c in CELLS:
        for lo in (False, True):
            r = res[(c, lo)]
            if not r["ok"]:
                continue
            w = r["wt"]
            wrows.append([c, "LO" if lo else "LS", w["weeks"], fmt(w["p10"]), fmt(100 * w["green"], 1) + "%",
                          fmt(100 * w["strong_share"], 1) + "%", w["n_strong"], fmt(w["gap_med"], 1),
                          fmt(w["gap_p90"], 1)])
    lines.append(mdtable(["cell", "side", "weeks", "P10 (R)", "green", ">= +5R share", "n strong", "median gap (wk)",
                          "P90 gap (wk)"], wrows) + "\n")
    lines.append("`scripts/cadence_bar.py` takes a trades CSV (date+pnl_R) with its own split/TEST guards and a "
                 "full multi-criterion spec; not invoked, the weekly table is computed here.\n")
    # ---- 1716 report-only
    lines.append("### 1,716 report-only\n")
    b, tb, r2 = ols(days)
    lines.append("OLS r13 (bps) on r1 (bps) and r12 (bps), HC1 t: " + mdtable(
        ["const", "b(r1)", "t", "b(r12)", "t", "R2", "n"],
        [[fmt(b[0]), fmt(b[1], 4), fmt(tb[1]), fmt(b[2], 4), fmt(tb[2]), fmt(r2, 5), len(days)]]) + "\n")
    drows, mono = decile_table(days)
    lines.append("Decile of r1 (own-split deciles) -> mean r13 (gross bps); corr(decile, mean r13) = "
                 f"{mono:.2f} (monotonic = +1):\n")
    lines.append(mdtable(["decile", "mean r1 bp", "n", "mean r13 bp", "t"], drows) + "\n")
    # per-year
    yrows = []
    for y in sorted(set(days.index.year)):
        ym = days.index.year == y
        row = [y, int(ym.sum()), fmt(days.loc[ym, "r13"].mean() * 1e4)]
        for c in ("1714A", "1715P70"):
            r = res[(c, False)]
            if r["ok"]:
                nd = r["net_day"][ym & r["traded"]]
                row += [len(nd), fmt(nd.mean()), fmt(tstat(nd))]
            else:
                row += ["n/a"] * 3
        yrows.append(row)
    lines.append("Per-year (net bps, LS): unconditional mean r13 beside 1714A and 1715P70:\n")
    lines.append(mdtable(["year", "days", "mean r13 (all days)", "1714A n", "net", "t", "1715P70 n", "net", "t"],
                         yrows) + "\n")
    # vol terciles (SPY realised vol, causal: 20 daily returns ending the previous close)
    dret = np.log(spy_close).diff()
    vol = (dret.rolling(20).std().shift(1) * math.sqrt(252) * 100)
    trv = vol[vol.index <= TRAIN_END].dropna()
    q1, q2 = np.percentile(trv, [100 / 3, 200 / 3])
    vrows = []
    for c in ("1714A", "1715P70"):
        r = res[(c, False)]
        if not r["ok"]:
            continue
        v = vol.reindex(days.index).values
        tag = np.where(v <= q1, "low", np.where(v <= q2, "mid", "high"))
        tag = np.where(np.isnan(v), "na", tag)
        for t in ("low", "mid", "high"):
            m = (tag == t) & r["traded"]
            nd = r["net_day"][m]
            vrows.append([c, t, int(m.sum()), fmt(nd.mean()) if m.sum() else "n/a", fmt(tstat(nd)) if m.sum() > 2 else "n/a"])
    lines.append(f"SPY 20-day realised vol terciles (TRAIN cut points {q1:.1f}% / {q2:.1f}% annualised), LS net bps:\n")
    lines.append(mdtable(["cell", "vol tercile", "n", "net", "t"], vrows) + "\n")
    path = os.path.join(HERE, f"part_{split}_{sym}.md")
    open(path, "w").write("\n".join(lines))
    log(f"[{sym}/{split}] wrote {path}")
    for c in CELLS:
        r = res[(c, False)]
        if r["ok"]:
            log(f"  {c} LS n={r['n']} net={r['mean']:.2f} t={r['t']:.2f} ex5={r['ex5']:.2f}")


def assemble():
    """Concatenate part files + caveats into RESULT_1714.md."""
    head = ["# RESULT 1,714-1,716 - market intraday momentum SPY/QQQ (PREREG_1714.md)\n",
            "Built by `cell_1714.py`. TEST (2023-01+) never loaded (hard cut at load). TRAIN thresholds were frozen "
            "(`thresholds_<SYM>.json`) by the TRAIN runs before any VAL run. Entry = open of the 15:31 bar, exit = "
            "close of the 15:59 bar (proxy for the MOC print, NOT the official closing auction price). Cost 1 bp/side "
            "(sensitivity 3 bp/side). 3x column = 3 x gross - 6 bp (3 bp/side), never traded.\n"]
    for sp in ("train", "val"):
        for sym in ("SPY", "QQQ"):
            p = os.path.join(HERE, f"part_{sp}_{sym}.md")
            if os.path.exists(p):
                head.append(open(p).read())
    cv = os.path.join(HERE, "caveats_1714.md")
    if os.path.exists(cv):
        head.append(open(cv).read())
    open(os.path.join(HERE, "RESULT_1714.md"), "w").write("\n".join(head))
    log("wrote RESULT_1714.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val"])
    ap.add_argument("--symbol", choices=["SPY", "QQQ"])
    ap.add_argument("--cost-bps", type=float, default=1.0)
    ap.add_argument("--assemble", action="store_true")
    a = ap.parse_args()
    if a.assemble:
        assemble()
    else:
        if not a.split or not a.symbol:
            sys.exit("need --split and --symbol")
        run(a.split, a.symbol, a.cost_bps)
