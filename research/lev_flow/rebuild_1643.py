"""Independent rebuild of cells 1,643-1,645 (leveraged-ETF rebalancing flow into the close).

Implemented ONLY from the prose spec in research/lev_flow/PREREG_1643.md, without reading
cell_1643.py or RESULT_1643_build.md (per the independent-check protocol in CLAUDE.md).

Mechanism: r_t = underlying return from the prior official daily close to the 15:30 ET minute
bar's close (last price known at 15:31:00). Fires when |r_t| >= 1.0%. Direction = sign(r_t).
  1,643 LONG on up days: buy the 15:31 ET bar's open, exit MOC at the official close.
  1,645 SHORT on down days: short the 15:31 ET bar's open, cover MOC at the official close.
  1,644 report-only: same entry, held to the next official open (reversal reading).

Costs: 1 bp half-spread for SPY/QQQ/IWM/TLT, 2 bps for the sector ETFs (SMH/XLF/XLE/GDX/XBI),
0.5 bp MOC cost on every exit (close or, as an assumption for the report-only 1,644 reading,
the next open too -- noted as a caveat since the spec doesn't price that leg explicitly).

Splits: TRAIN 2016-01-01..2020-12-31, VAL 2021-01-01..2023-12-31. TEST (2024+) is SEALED --
this script never reads or computes anything past 2023-12-31.
"""
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
OUT_CSV = Path(__file__).parent / "events_1643_rebuild.csv"

UNDERLYINGS = ["SPY", "QQQ", "IWM", "XLF", "XLE", "GDX", "XBI", "TLT", "SMH"]
ONE_BP_SET = {"SPY", "QQQ", "IWM", "TLT"}
TWO_BP_SET = {"XLF", "XLE", "GDX", "XBI", "SMH"}
EXIT_MOC_BPS = 0.5  # official-close MOC cost, one-way
THRESH = 0.01  # |r_t| >= 1.0%
ET = "America/New_York"

TRAIN_START, TRAIN_END = dt.date(2016, 1, 1), dt.date(2020, 12, 31)
VAL_START, VAL_END = dt.date(2021, 1, 1), dt.date(2023, 12, 31)
DATA_CUTOFF = dt.date(2023, 12, 31)  # hard stop -- TEST (2024+) is sealed, never touched


def early_close_dates(years):
    """NYSE 1pm-close sessions, hardcoded rule (per task instructions, no calendar lib):
    day after Thanksgiving (4th Thursday of Nov + 1), July 3 if a weekday, Dec 24 if a weekday.
    """
    out = set()
    for y in years:
        nov1 = dt.date(y, 11, 1)
        thursdays = [
            nov1 + dt.timedelta(days=i)
            for i in range(30)
            if (nov1 + dt.timedelta(days=i)).month == 11
            and (nov1 + dt.timedelta(days=i)).weekday() == 3
        ]
        thanksgiving = thursdays[3]
        out.add(thanksgiving + dt.timedelta(days=1))
        for month, day in [(7, 3), (12, 24)]:
            cand = dt.date(y, month, day)
            if cand.weekday() < 5:
                out.add(cand)
    return out


EARLY_CLOSES = early_close_dates(range(2016, 2024))


def load_daily(symbol):
    df = pd.read_parquet(DATA_DIR / "daily" / f"{symbol}.parquet")
    df["date"] = df["t"].dt.tz_convert(ET).dt.date
    df = df.sort_values("date").reset_index(drop=True)
    df["prior_close"] = df["c"].shift(1)
    df["next_open"] = df["o"].shift(-1)
    return df[["date", "c", "prior_close", "next_open"]].rename(columns={"c": "official_close"})


def load_minute_key_bars(symbol):
    df = pd.read_parquet(DATA_DIR / "minute" / f"{symbol}.parquet")
    ts_et = df["t"].dt.tz_convert(ET)
    df = df.assign(date=ts_et.dt.date, time=ts_et.dt.time)
    df = df[(df["time"] >= dt.time(9, 30)) & (df["time"] <= dt.time(16, 0))]
    b1530 = df[df["time"] == dt.time(15, 30)][["date", "c"]].rename(columns={"c": "c_1530"})
    b1531 = df[df["time"] == dt.time(15, 31)][["date", "o"]].rename(columns={"o": "o_1531"})
    return b1530.merge(b1531, on="date", how="inner")


def build_events(symbol):
    daily = load_daily(symbol)
    minute = load_minute_key_bars(symbol)
    df = minute.merge(daily, on="date", how="inner")
    df = df[df["date"] <= DATA_CUTOFF]  # never touch TEST (2024+)
    df = df[~df["date"].isin(EARLY_CLOSES)]
    df["symbol"] = symbol
    df["r_t"] = df["c_1530"] / df["prior_close"] - 1.0
    fired = df[df["r_t"].abs() >= THRESH].copy()
    if fired.empty:
        return fired
    fired["direction"] = np.sign(fired["r_t"]).astype(int)
    half_spread = 0.0001 if symbol in ONE_BP_SET else 0.0002
    exit_cost = EXIT_MOC_BPS / 10000.0

    long_raw = fired["official_close"] / fired["o_1531"] - 1.0
    short_raw = fired["o_1531"] / fired["official_close"] - 1.0
    fired["raw_r"] = np.where(fired["direction"] > 0, long_raw, short_raw)
    fired["net_bps"] = (fired["raw_r"] - half_spread - exit_cost) * 10000
    fired["book"] = np.where(fired["direction"] > 0, 1643, 1645)

    next_long = fired["next_open"] / fired["o_1531"] - 1.0
    next_short = fired["o_1531"] / fired["next_open"] - 1.0
    next_raw = np.where(fired["direction"] > 0, next_long, next_short)
    fired["net_bps_1644"] = (next_raw - half_spread - exit_cost) * 10000

    fired = fired.rename(columns={"o_1531": "entry", "official_close": "exit"})
    return fired[
        ["date", "symbol", "r_t", "direction", "book", "entry", "exit", "next_open",
         "net_bps", "net_bps_1644"]
    ]


def add_split(df):
    df = df.copy()
    conds = [
        (df["date"] >= TRAIN_START) & (df["date"] <= TRAIN_END),
        (df["date"] >= VAL_START) & (df["date"] <= VAL_END),
    ]
    df["split"] = np.select(conds, ["TRAIN", "VAL"], default="OUT")
    return df


def clustered_t(values, days):
    """Day-clustered t-stat for a mean (CR1 cluster-robust SE), cluster = calendar day."""
    x = np.asarray(values, dtype=float)
    d = np.asarray(days)
    n = len(x)
    if n < 2:
        return x.mean() if n else np.nan, np.nan, np.nan, len(pd.unique(d))
    xbar = x.mean()
    u = x - xbar
    clusters = pd.unique(d)
    g = len(clusters)
    s = sum(u[d == c].sum() ** 2 for c in clusters)
    var = s / (n**2)
    se = np.sqrt(var)
    if g > 1 and se > 0:
        corr = (g / (g - 1)) * ((n - 1) / (n - 1))
        se_adj = se * np.sqrt(corr)
        t = xbar / se_adj
    else:
        se_adj, t = np.nan, np.nan
    return xbar, se_adj, t, g


def ex_top_pct(values, pct=0.05):
    x = np.sort(np.asarray(values, dtype=float))[::-1]
    k = int(np.ceil(len(x) * pct))
    return x[k:].mean() if k < len(x) else np.nan


def tercile_table(sub, value_col="net_bps"):
    if len(sub) < 3:
        return None
    try:
        q = pd.qcut(sub["r_t"].abs(), 3, labels=["T1_low", "T2_mid", "T3_high"], duplicates="drop")
    except ValueError:
        return None
    return sub.groupby(q, observed=True)[value_col].mean()


SPLIT_DAYS = {
    "TRAIN": (TRAIN_END - TRAIN_START).days + 1,
    "VAL": (VAL_END - VAL_START).days + 1,
}
MD_OUT = Path(__file__).parent / "REBUILD_1643.md"


def cell_split_stats(df, cell, split):
    value_col = "net_bps_1644" if cell == 1644 else "net_bps"
    if cell == 1644:
        sub = df[df.split == split]
    else:
        sub = df[(df.book == cell) & (df.split == split)]
    if sub.empty:
        return None
    xbar, se, t, g = clustered_t(sub[value_col], sub["date"])
    weeks = SPLIT_DAYS[split] / 7.0
    ex5 = ex_top_pct(sub[value_col], 0.05)
    terc = tercile_table(sub, value_col)
    per_symbol = sub.groupby("symbol")[value_col].mean()
    positives = per_symbol[per_symbol > 0].index.tolist()
    return dict(
        n=len(sub), days=g, events_per_week=len(sub) / weeks, mean=xbar, t=t,
        ex_top5=ex5, tercile=terc, n_pos=len(positives), positives=positives,
        per_symbol=per_symbol,
    )


def main():
    all_events = pd.concat([build_events(s) for s in UNDERLYINGS], ignore_index=True)
    all_events = add_split(all_events)
    all_events = all_events[all_events["split"].isin(["TRAIN", "VAL"])].reset_index(drop=True)
    all_events = all_events.sort_values(["date", "symbol"]).reset_index(drop=True)

    out = all_events[
        ["date", "symbol", "r_t", "direction", "book", "split", "entry", "exit",
         "net_bps", "net_bps_1644"]
    ]
    out.to_csv(OUT_CSV, index=False)
    print(f"wrote {len(out)} events to {OUT_CSV}")

    # sanity check: sign convention -- on 1643 (up-day, long) rows, raw long return must be
    # positive iff official close > entry (15:31 open). Pure arithmetic identity check.
    b1643 = all_events[all_events["book"] == 1643]
    raw_r_1643 = b1643["exit"] / b1643["entry"] - 1.0
    raw_sign_ok = (np.sign(b1643["exit"] - b1643["entry"]) == np.sign(raw_r_1643)).mean()
    print(f"sign convention (raw_r sign matches exit>entry) match rate: {raw_sign_ok:.4f}")

    lines = []
    lines.append("# REBUILD_1643 -- independent rebuild of cells 1,643-1,645")
    lines.append("")
    lines.append("Built from PREREG_1643.md alone (cell_1643.py / RESULT_1643_build.md not read")
    lines.append("until after this file and events_1643_rebuild.csv were written).")
    lines.append("TRAIN=2016-2020, VAL=2021-2023, TEST(2024+) sealed -- not computed.")
    lines.append("")
    lines.append(f"Sign convention check (1643 raw_r sign == sign(exit-entry)): {raw_sign_ok:.4f} (expect 1.0000)")
    lines.append(f"Early-close sessions excluded: {len(EARLY_CLOSES)} "
                 f"(Thanksgiving+1, Jul-3-if-weekday, Dec-24-if-weekday, 2016-2023)")
    lines.append("")
    lines.append("| cell | split | n | days | ev/wk | mean bps | clust t | ex-top5% bps | #pos/9 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")

    stats_store = {}
    for cell in [1643, 1645, 1644]:
        for split in ["TRAIN", "VAL"]:
            st = cell_split_stats(all_events, cell, split)
            stats_store[(cell, split)] = st
            if st is None:
                lines.append(f"| {cell} | {split} | 0 | - | - | - | - | - | - |")
                continue
            lines.append(
                f"| {cell} | {split} | {st['n']} | {st['days']} | {st['events_per_week']:.2f} | "
                f"{st['mean']:.2f} | {st['t']:.2f} | {st['ex_top5']:.2f} | {st['n_pos']}/9 |"
            )

    lines.append("")
    lines.append("## |r_t| tercile table (mean net bps, low/mid/high tercile among that cell's fired events)")
    lines.append("")
    lines.append("| cell | split | T1_low | T2_mid | T3_high | monotone? |")
    lines.append("|---|---|---|---|---|---|")
    for cell in [1643, 1645, 1644]:
        for split in ["TRAIN", "VAL"]:
            st = stats_store.get((cell, split))
            if st is None or st["tercile"] is None:
                lines.append(f"| {cell} | {split} | - | - | - | - |")
                continue
            terc = st["tercile"]
            vals = [terc.get(k, np.nan) for k in ["T1_low", "T2_mid", "T3_high"]]
            mono = "YES" if (vals[0] <= vals[1] <= vals[2] or vals[0] >= vals[1] >= vals[2]) else "no"
            mono_dir = "" if any(pd.isna(v) for v in vals) else (" (rising)" if vals[2] > vals[0] else " (falling)")
            lines.append(
                f"| {cell} | {split} | {vals[0]:.2f} | {vals[1]:.2f} | {vals[2]:.2f} | {mono}{mono_dir} |"
            )

    lines.append("")
    lines.append("## Per-underlying mean net bps (VAL)")
    lines.append("")
    lines.append("| symbol | 1643 VAL | 1645 VAL |")
    lines.append("|---|---|---|")
    for sym in UNDERLYINGS:
        v43 = stats_store[(1643, "VAL")]
        v45 = stats_store[(1645, "VAL")]
        a = v43["per_symbol"].get(sym, np.nan) if v43 else np.nan
        b = v45["per_symbol"].get(sym, np.nan) if v45 else np.nan
        lines.append(f"| {sym} | {a:.2f} | {b:.2f} |")

    lines.append("")
    lines.append("## Cost / assumption notes")
    lines.append("- Entry cost: 1bp half-spread SPY/QQQ/IWM/TLT, 2bp SMH/XLF/XLE/GDX/XBI (per spec).")
    lines.append("- Exit cost: 0.5bp MOC on the official-close leg (1643/1645), per spec.")
    lines.append("- Cell 1644 (report-only) applies the SAME entry cost + assumes another 0.5bp on the")
    lines.append("  next-open exit leg -- the spec does not price this leg explicitly; flagged as an")
    lines.append("  assumption, not a spec fact. 1644 is not scored against the pass bar.")
    lines.append("- FOMC-day exclusion line and the SPY-only / worst-day / MDE lines from the full PREREG")
    lines.append("  report spec were NOT computed here (out of scope for this rebuild's deliverable list).")

    MD_OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {MD_OUT} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
