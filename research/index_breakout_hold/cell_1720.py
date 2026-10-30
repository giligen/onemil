"""Cell 1,720: "QLD Claude AI Scalper" replayed on QQQ 30-min bars, 2016-2022 (TRAIN + VAL only).

Frozen rules (PREREG_1720.md incl. Amendment 1): signal at 30-min bar close when close > max(high of prior 30
bars, excluding the current bar), volume >= 1.75 x SMA20(volume) and close > open; fill at next bar open; one
position at a time; exit at the open of the bar after the 16th held bar if QLD-proxy PnL at that bar's close
< +0.5 %, else at the open of the bar after the 60th held bar. QLD proxy = 2x QQQ per trading day (daily
reset) compounded over the trade.

TEST (>= 2023-01-01) is dropped immediately after loading and asserted absent from every scored frame.
Usage: python research/index_breakout_hold/cell_1720.py
"""
import math
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "index_orb", "cache", "QQQ_1min.parquet")
HARD_CUTOFF = pd.Timestamp("2023-01-01")
MIN_BARS = 380
LOOKBACK, VOL_N, VOL_MULT, H1, H2, THRESH, LEV = 30, 20, 1.75, 16, 60, 0.005, 2.0
COSTS_BP = [0.0, 1.0, 3.0]
PRIMARY_COST = 1.0
NULL_DRAWS = 1000
SPLITS = {"TRAIN": ("2016-01-01", "2019-12-31"), "VAL": ("2020-01-01", "2022-12-31")}


def log(msg):
    """Verbose progress, flushed."""
    print(msg, flush=True)


def load_bars30():
    """Load QQQ 1-min, drop TEST at once, filter days (<380 bars, not 13 half-hour bars), resample to 30-min OHLCV."""
    df = pd.read_parquet(CACHE)
    ts = df["timestamp"].dt.tz_localize(None)
    df = df.loc[ts < HARD_CUTOFF].copy()
    df["ts"] = df["timestamp"].dt.tz_localize(None)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("ts")
    assert df["ts"].max() < HARD_CUTOFF, "TEST leak"
    df["date"] = df["ts"].dt.normalize()
    mins = df["ts"].dt.hour * 60 + df["ts"].dt.minute
    df = df.loc[(mins >= 570) & (mins < 960)].copy()          # regular session 09:30-15:59
    n = df.groupby("date").size()
    ok_days = n[n >= MIN_BARS].index
    log(f"calendar days {len(n)}, kept (>= {MIN_BARS} 1-min bars) {len(ok_days)}, dropped {len(n) - len(ok_days)}")
    df = df.loc[df["date"].isin(ok_days)].copy()
    df["b"] = df["ts"].dt.floor("30min")
    g = df.groupby("b")
    b30 = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                        "close": g["close"].last(), "volume": g["volume"].sum()})
    b30["date"] = b30.index.normalize()
    cnt = b30.groupby("date").size()
    full = cnt[cnt == 13].index
    log(f"days with 13 half-hour bars {len(full)} of {len(cnt)} (dropped {len(cnt) - len(full)} incomplete)")
    b30 = b30.loc[b30["date"].isin(full)].copy()
    assert b30.index.max() < HARD_CUTOFF, "TEST leak in 30-min frame"
    return b30


def find_signals(b):
    """Boolean array: signal at bar close per the published entry rule (prior-30 high excludes the current bar)."""
    hh = b["high"].shift(1).rolling(LOOKBACK).max()
    vsma = b["volume"].rolling(VOL_N).mean()                 # includes the current bar (indicator convention)
    sig = (b["close"] > hh) & (b["volume"] >= VOL_MULT * vsma) & (b["close"] > b["open"])
    return sig.fillna(False).to_numpy()


class Path:
    """Holds numpy price arrays and computes QLD-proxy / QQQ returns of a trade from entry bar j."""

    def __init__(self, b):
        self.o = b["open"].to_numpy()
        self.c = b["close"].to_numpy()
        self.d = b["date"].to_numpy()
        self.n = len(b)

    def qld(self, j, last_bar, exit_open_bar=None):
        """Daily-reset 2x compounded return from open[j] through close[last_bar] (or to open[exit_open_bar])."""
        pts = [(self.d[j], self.o[j])]
        for k in range(j, last_bar + 1):
            pts.append((self.d[k], self.c[k]))
        if exit_open_bar is not None:
            pts.append((self.d[exit_open_bar], self.o[exit_open_bar]))
        days, last = [], []                                   # last price per date, in order
        for dt, p in pts:
            if days and days[-1] == dt:
                last[-1] = p
            else:
                days.append(dt)
                last.append(p)
        f, prev = 1.0, pts[0][1]
        for p in last:
            f *= 1.0 + LEV * (p / prev - 1.0)
            prev = p
        return f - 1.0

    def outcome(self, j):
        """Apply the exit rule for an entry at open of bar j: (exit_bar, hold_bars, qld_gross, qqq_gross) or None."""
        if j + H1 >= self.n:
            return None
        pnl16 = self.qld(j, j + H1 - 1)
        e = j + H1 if pnl16 < THRESH else j + H2
        if e >= self.n:
            return None
        return e, e - j, self.qld(j, e - 1, e), self.o[e] / self.o[j] - 1.0


def run_strategy(b, sig, path):
    """One position at a time; returns trade list as dicts (entry bar j, exit bar e, returns)."""
    trades, i = [], LOOKBACK + VOL_N
    while i < path.n - 1:
        if not sig[i]:
            i += 1
            continue
        j = i + 1
        out = path.outcome(j)
        if out is None:
            break
        e, h, q, q1 = out
        trades.append({"sig_bar": i, "j": j, "e": e, "hold": h, "qld_gross": q, "qqq_gross": q1,
                       "entry_ts": b.index[j], "exit_ts": b.index[e]})
        i = e                                  # flat at open of e; first allowed signal bar is e
    return trades


def tstat(x):
    """Mean / (sd/sqrt(n)); nan when n < 3."""
    x = np.asarray(x, float)
    if len(x) < 3 or x.std(ddof=1) == 0:
        return float("nan")
    return x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))


def split_of(ts):
    """TRAIN / VAL label for an entry timestamp (None outside)."""
    for k, (a, z) in SPLITS.items():
        if pd.Timestamp(a) <= ts < pd.Timestamp(z) + pd.Timedelta(days=1):
            return k
    return None


def null_n2(bars_idx, all_out, per_year, bar_drift, cost, rng):
    """N2: random entry bars, same trades/year, non-overlapping, same exit rule; returns draws of mean excess."""
    years = bars_idx.year.to_numpy()
    cand = {y: np.array([j for j in range(len(years)) if years[j] == y and all_out[j] is not None]) for y in per_year}
    res = np.empty(NULL_DRAWS)
    for dr in range(NULL_DRAWS):
        ex = []
        for y, k in per_year.items():
            chosen = []
            for ix in rng.permutation(len(cand[y])):
                j = cand[y][ix]
                e = all_out[j][0]
                if all(e <= cj or j >= ce for cj, ce in chosen):
                    chosen.append((j, e))
                    ex.append(all_out[j][2] - 2 * cost * 1e-4 - all_out[j][1] * bar_drift)
                    if len(chosen) == k:
                        break
        res[dr] = np.mean(ex)
    return res


def maxdd(eq):
    """Max drawdown (negative number) of an equity array."""
    eq = np.asarray(eq)
    return float((eq / np.maximum.accumulate(eq) - 1).min())


def equity_2022(b, trades, cost):
    """Bar-level (open-to-next-open) 2022 equity: strategy (2x while in position) vs QQQ B&H and QLD-proxy B&H."""
    o = b["open"].to_numpy()
    ret = np.append(o[1:] / o[:-1] - 1.0, 0.0)
    pos = np.zeros(len(b))
    for t in trades:
        pos[t["j"]:t["e"]] = 1.0
    mask = (b.index >= pd.Timestamp("2022-01-01")) & (b.index < pd.Timestamp("2023-01-01"))
    r, p = ret[mask], pos[mask]
    p_prev = np.concatenate([[0.0], p[:-1]])
    srt = LEV * r * p - cost * 1e-4 * np.abs(p - p_prev)
    eq_s, eq_q, eq_l = np.cumprod(1 + srt), np.cumprod(1 + r), np.cumprod(1 + LEV * r)
    return {"strat_dd": maxdd(eq_s), "qqq_dd": maxdd(eq_q), "qld_bh_dd": maxdd(eq_l),
            "strat_ret": float(eq_s[-1] - 1), "qqq_ret": float(eq_q[-1] - 1), "qld_bh_ret": float(eq_l[-1] - 1),
            "inmkt": float(p.mean())}


def summarize(tr, bar_drift, cost):
    """Per-subset statistics for one cost level; tr is a trades DataFrame."""
    net = tr["qld_gross"] - 2 * cost * 1e-4
    bh = tr["hold"] * bar_drift
    ex = net - bh
    n = len(tr)
    k = max(1, int(math.ceil(0.05 * n)))
    ex_top = ex.sort_values(ascending=False).iloc[k:].mean() if n > k else float("nan")
    return {"n": n, "gross_mean_bps": 1e4 * tr["qld_gross"].mean(), "net_mean_bps": 1e4 * net.mean(),
            "qqq_mean_bps": 1e4 * tr["qqq_gross"].mean(), "bh_mean_bps": 1e4 * bh.mean(),
            "excess_bps": 1e4 * ex.mean(), "t": tstat(ex), "ex_top5_bps": 1e4 * ex_top,
            "capped3_bps": 1e4 * (net.clip(upper=0.03) - bh).mean(),
            "n1_lit_bps": 1e4 * (net - tr["qqq_gross"]).mean(),
            "se_bps": 1e4 * ex.std(ddof=1) / math.sqrt(n) if n > 2 else float("nan"),
            "max": net.max() if n else float("nan"), "min": net.min() if n else float("nan"),
            "win": float((net > 0).mean()) if n else float("nan")}


def md(header, rows):
    """Markdown table from header list and list of row lists."""
    s = "| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n"
    for r in rows:
        s += "| " + " | ".join(str(x) for x in r) + " |\n"
    return s


def f(x, d=1):
    """Format float, nan-safe."""
    return "nan" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def main():
    """Run the cell and write trades CSVs + RESULT_1720.md."""
    b = load_bars30()
    log(f"30-min bars {len(b)}, {b.index.min()} -> {b.index.max()}")
    path = Path(b)
    sig = find_signals(b)
    log(f"raw signal bars {int(sig.sum())}")
    trades = run_strategy(b, sig, path)
    tr = pd.DataFrame(trades)
    tr["split"] = tr["entry_ts"].map(split_of)
    tr["year"] = tr["entry_ts"].dt.year
    assert tr["entry_ts"].max() < HARD_CUTOFF
    log(f"trades {len(tr)}: {tr['split'].value_counts().to_dict()}")
    o = b["open"].to_numpy()
    bar_ret = np.append(o[1:] / o[:-1] - 1.0, np.nan)
    drift, bars_split = {}, {}
    for k, (a, z) in SPLITS.items():
        m = (b.index >= pd.Timestamp(a)) & (b.index < pd.Timestamp(z) + pd.Timedelta(days=1))
        drift[k] = LEV * np.nanmean(bar_ret[m])
        bars_split[k] = int(m.sum())
        log(f"{k}: bars {bars_split[k]}, QLD-proxy mean bar return {1e4 * drift[k]:.3f} bp")
    log("precomputing outcome for every entry bar (N2)")
    all_out = [path.outcome(j) if j >= 1 else None for j in range(path.n)]
    rng = np.random.default_rng(1720)
    res = {}
    for sp in ["TRAIN", "VAL"]:
        t = tr[tr["split"] == sp].copy()
        assert t["entry_ts"].max() < HARD_CUTOFF
        t.assign(qld_net_1bp=t["qld_gross"] - 2e-4 * PRIMARY_COST).to_csv(
            os.path.join(HERE, f"trades_1720_{sp.lower()}.csv"), index=False)
        per_year = t.groupby("year").size().to_dict()
        s = {c: summarize(t, drift[sp], c) for c in COSTS_BP}
        yrs = {y: summarize(t[t["year"] == y], drift[sp], PRIMARY_COST) for y in sorted(per_year)}
        n2 = null_n2(b.index, all_out, per_year, drift[sp], PRIMARY_COST, rng)
        pct = float((n2 < s[PRIMARY_COST]["excess_bps"] * 1e-4).mean() * 100)
        comp = (float(np.prod(1 + t["qld_gross"]) - 1), float(np.prod(1 + t["qld_gross"] - 2e-4 * PRIMARY_COST) - 1),
                float(np.prod(1 + t["qqq_gross"]) - 1))
        res[sp] = dict(t=t, s=s, yrs=yrs, pct=pct, n2=n2, held=int(t["hold"].sum()), comp=comp)
        log(f"{sp}: n={len(t)} excess {s[PRIMARY_COST]['excess_bps']:.1f} bps t={s[PRIMARY_COST]['t']:.2f} N2 pct {pct:.1f}")
    eq = equity_2022(b, trades, PRIMARY_COST)
    log(f"2022: {eq}")

    lines = ["# RESULT 1,720 - QLD 'Claude AI Scalper' replay on QQQ 30-min bars (TRAIN 2016-19, VAL 2020-22; TEST sealed)\n",
             "Rule as amended (Amendment 1): close > prior-30-bar high, volume >= 1.75 x SMA20, close > open; "
             "16/60-bar exit, 0.5 % threshold at bar 16.\n"]
    for sp in ["TRAIN", "VAL"]:
        r = res[sp]
        s1 = r["s"][PRIMARY_COST]
        n = s1["n"]
        lines.append(f"\n## {sp}\n")
        lines.append(md(["trades", "in-market share", "held bars", "mean hold (bars)", "win %", "QLD-proxy gross comp.",
                         "QLD-proxy net comp. (1bp/side)", "QQQ 1x gross comp.", "max trade", "min trade"],
                        [[n, f(100 * r["held"] / bars_split[sp]) + " %", r["held"], f(r["t"]["hold"].mean()),
                          f(100 * s1["win"]), f(100 * r["comp"][0]) + " %", f(100 * r["comp"][1]) + " %",
                          f(100 * r["comp"][2]) + " %", f(100 * s1["max"]) + " %", f(100 * s1["min"]) + " %"]]))
        lines.append("\nPer-trade means (bps). EXCESS = QLD-proxy net trade return - exposure-matched B&H "
                     "(hold bars x split mean QLD-proxy bar return). 'N1 literal' = QLD net - QQQ 1x same bars "
                     "(leverage-confounded).\n")
        rows = []
        for c in COSTS_BP:
            x = r["s"][c]
            rows.append([f"{c:g} bp/side", f(x["gross_mean_bps"]), f(x["net_mean_bps"]), f(x["qqq_mean_bps"]),
                         f(x["bh_mean_bps"]), f(x["excess_bps"]), f(x["t"], 2), f(x["ex_top5_bps"]),
                         f(x["capped3_bps"]), f(x["n1_lit_bps"])])
        lines.append(md(["cost", "QLD gross", "QLD net", "QQQ 1x", "B&H matched", "EXCESS", "t (trade)",
                         "ex-top-5% excess", "winner-capped(+3%) excess", "N1 literal"], rows))
        rows = []
        for y, x in r["yrs"].items():
            rows.append([y, x["n"], f(x["net_mean_bps"]), f(x["bh_mean_bps"]), f(x["excess_bps"]), f(x["t"], 2),
                         "FLAG" if (not math.isnan(x["t"]) and abs(x["t"]) > 6) else ""])
        lines.append("\nPer year (1 bp/side):\n")
        lines.append(md(["year", "trades", "QLD net bps/trade", "B&H matched", "excess bps/trade", "t", "abs t > 6"], rows))
        pos_years = sum(1 for x in r["yrs"].values() if x["excess_bps"] > 0)
        lines.append(f"\nYears with excess > 0: {pos_years} of {len(r['yrs'])}.  "
                     f"N2 (1000 random-entry draws, same trades/year): mean {f(1e4 * r['n2'].mean())} bps, "
                     f"95th pct {f(1e4 * np.percentile(r['n2'], 95))} bps, realised percentile **{f(r['pct'])}**.  "
                     f"MDE at n={n} (2 x se) = {f(2 * s1['se_bps'])} bps/trade.\n")
        big = (r["t"]["qld_gross"] > 0.5).any() or (r["t"]["qld_gross"] < -0.30).any()
        lines.append(f"Flags: {'trade beyond +50 % / -30 %' if big else 'none (no trade > +50 % / < -30 %)'}.\n")
    lines.append("\n## 2022 (bear year inside VAL), bar-level open-to-open equity, 1 bp/side\n")
    lines.append(md(["", "return", "max DD"],
                    [["Strategy (QLD proxy)", f(100 * eq["strat_ret"]) + " %", f(100 * eq["strat_dd"]) + " %"],
                     ["QQQ B&H", f(100 * eq["qqq_ret"]) + " %", f(100 * eq["qqq_dd"]) + " %"],
                     ["QLD-proxy B&H (reference)", f(100 * eq["qld_bh_ret"]) + " %", f(100 * eq["qld_bh_dd"]) + " %"]]))
    lines.append(f"\nIn-market share 2022: {f(100 * eq['inmkt'])} %.\n")

    T, V = res["TRAIN"], res["VAL"]
    v1 = V["s"][PRIMARY_COST]
    tr_pos = sum(1 for x in T["yrs"].values() if x["excess_bps"] > 0)
    val_pos = sum(1 for x in V["yrs"].values() if x["excess_bps"] > 0)
    items = [
        ("1a Excess>0 in >=3 of 4 TRAIN years and in VAL", tr_pos >= 3 and v1["excess_bps"] > 0,
         f"TRAIN {tr_pos} of {len(T['yrs'])} yrs positive; VAL {f(v1['excess_bps'])} bps (VAL years positive {val_pos} of {len(V['yrs'])})"),
        ("1b VAL excess t >= 2", bool(v1["t"] >= 2), f"t = {f(v1['t'], 2)}"),
        ("2 VAL excess > N2 95th pct", V["pct"] > 95, f"percentile {f(V['pct'])}"),
        ("3 2022: DD <= QQQ DD and return > QQQ", eq["strat_dd"] >= eq["qqq_dd"] and eq["strat_ret"] > eq["qqq_ret"],
         f"DD {f(100 * eq['strat_dd'])} vs {f(100 * eq['qqq_dd'])} %; ret {f(100 * eq['strat_ret'])} vs {f(100 * eq['qqq_ret'])} %"),
        ("4 ex-top-5 % excess > 0 and winner-capped > 0", bool(v1["ex_top5_bps"] > 0 and v1["capped3_bps"] > 0),
         f"{f(v1['ex_top5_bps'])} / {f(v1['capped3_bps'])} bps"),
        ("5 fills obtainable (next-bar open)", True, "by construction"),
    ]
    allpass = all(i[1] for i in items)
    lines.append("\n## Pass bar (frozen) - VAL, after TRAIN same-signed\n")
    lines.append(md(["item", "pass", "evidence"], [[a, "PASS" if p else "FAIL", c] for a, p, c in items]))
    lines.append(f"\n## VERDICT: **{'PASS' if allpass else 'REJECT'}** (TEST remains sealed; no variants, no refit).\n")
    lines.append("""
## Caveats about this implementation
- **N1 definition.** The task text defines N1 as QQQ 1x return over the same bars with excess = strategy - N1; that
  compares a 2x instrument with a 1x one and is dominated by leverage. The pass-bar EXCESS here is the
  exposure-matched B&H (hold bars x split mean QLD-proxy per-bar return, bar = open to next open), the PREREG's
  own words. The literal N1 column is reported beside it.
- QLD proxy = 2x per-day price change (daily reset) compounded; no expense ratio, tracking error or financing.
  The 0.5 % threshold at bar 16 is evaluated on the QLD proxy. The 2022 equity curve is bar-level (open to
  next open, 2x per bar without per-day reset), an approximation for drawdown only.
- Volume SMA20 includes the current bar and runs across days. Half-hour volume has a strong time-of-day shape
  (09:30 and 15:30 bars carry far more volume), so even at 1.75x the filter is partly a time-of-day filter.
- Days with < 380 1-min bars (early closes, holes) and days without all 13 half-hour bars are removed from the
  series, so the 30-bar lookback and the 16/60-bar holds span them (bars counted, not time).
- Signal bar i enters at open of i+1; after an exit at the open of bar e the first signal considered is bar e's
  close. Trades are assigned to a split by entry date; trades whose exit would fall beyond the data are dropped.
- N2 draws non-overlapping entries uniformly within each year (same count per year, same exit rule on the QLD
  proxy); random entries are not restricted to signal-like bars (as specified).
- Costs: 1 bp/side charged on the QLD-proxy return (sensitivity 0 and 3 bp shown).
- t statistics are trade-level; trades are non-overlapping but serially correlated through regime, so the SE is
  likely understated.
""")
    with open(os.path.join(HERE, "RESULT_1720.md"), "w") as fh:
        fh.write("\n".join(lines))
    log("wrote RESULT_1720.md")


if __name__ == "__main__":
    main()
