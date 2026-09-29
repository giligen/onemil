#!/usr/bin/env python3
"""Cell 1,666: independent rebuild of cell 1,488 (the HOD no-withdrawal pyramid)
at the measured cost.

PREREG: research/hod_entry/PREREG_1666.md (FROZEN 2026-09-29). Built from the
PREREG's prose alone -- the first implementation (any file/script whose name
contains 1487, 1488 or 1662) was never opened, per the independence rule.

Rule (verbatim, PREREG_1487 S1488): base entry at 1/3 risk; at the end of
minute fill_min+15, if no dip below level-$0.01 occurred in bars
fill_min+1 .. fill_min+15 and the position is still open, add 2/3 at the
open of the bar at fill_min+16 (+ entry cost), move the WHOLE position's
stop to level-$0.01; target = 2R from the ORIGINAL fill for the whole
position; if the position stops or targets before minute 15, there is no
add (book = base leg at 1/3 size only). Paired base = the same fill at 1x
size under the standard rule (original stop, 2R target, same EOD bar),
same cost -- walked independently bar by bar, never read from an upstream
file.

Time convention (verified against bars_sip.db before coding this): fill_min
/ exit_m are fractional MINUTES-SINCE-MIDNIGHT in America/New_York local
time. floor(fill_min) is the integer minute of the fill bar itself; bars
fill_min+1 .. fill_min+15 are the 15 one-minute bars strictly after it.
Verified two ways: ESTA 2025-07-07 fill=46.35 at fill_min=727.277 ->
floor 727 -> 12:07 ET -> 2025-07-07T16:07:00+00:00 (EDT, UTC-4): that bar's
OPEN is exactly 46.35. Its EOD exit_m=955.0 = 15:55 ET exactly, matching
this PREREG's own EOD definition, and 2025-07-07T19:55:00+00:00 exists.

bars_sip.db is SPARSE (only minutes with a print have a row) -- every bar
lookup below tolerates missing minutes; the "next available bar at/after"
convention is used for the fixed-instant add and EOD lookups, since an
order cannot act on a minute with no trade.

Gap handling for stop/target fills (not stated verbatim in the PREREG, but
required by CLAUDE.md's obtainability rule: a fill is a price the market
offered, reachable by a resting order): if a bar gaps through a level, the
order fills at the bar's open (worse for a stop, better for a target),
never at the untouched level itself.
"""

import csv
import logging
import math
import sqlite3
import sys
import time
from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

REPO = "/home/ec2-user/onemil"
FILLS_CSV = f"{REPO}/research/hod_entry/fills_1658.csv"
CAUSAL_CSV = f"{REPO}/research/hod_entry/causal_arming_causal.csv"
BARS_DB = f"{REPO}/research/bf_zero/bars_sip.db"
OUT_PERFILL = f"{REPO}/research/hod_entry/1666_per_fill.csv"
OUT_RESULT = f"{REPO}/research/hod_entry/RESULT_1666.md"
LOG_FILE = f"{REPO}/research/hod_entry/1666_rebuild.log"

ENTRY_COST = 0.0007   # 7 bps, each entry leg
STOP_COST = 0.0006    # 6 bps, stop exits
TARGET_COST = 0.0     # 0 bps, target fills
EOD_COST = 0.0011     # 11 bps, EOD (MOC-ish) exit
BASE_FRAC = 1.0 / 3.0
ADD_FRAC = 2.0 / 3.0
WITHDRAW_PAD = 0.01
CHECK_END_OFFSET = 15   # bars fill_min+1 .. fill_min+15
ADD_OFFSET = 16          # add at the open of the bar at fill_min+16
EOD_MINUTE = 955         # 15:55 ET, verbatim from the PREREG

NY = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

logging.basicConfig(
    filename=LOG_FILE, filemode="w", level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("rebuild_1666")
console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.INFO)
log.addHandler(console)


# --------------------------------------------------------------------------
# Bar store access -- one connection, cached per (symbol, day), PK-scoped.
# --------------------------------------------------------------------------

def et_minute_of_day(iso_ts: str) -> int:
    dt = datetime.fromisoformat(iso_ts).astimezone(NY)
    return dt.hour * 60 + dt.minute


class BarStore:
    def __init__(self, path):
        self.conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        self.cache = {}
        self.hits = 0
        self.misses = 0

    def day_bars(self, symbol, day):
        key = (symbol, day)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        rows = self.conn.execute(
            "SELECT t, o, h, l, c FROM bars WHERE symbol = ? AND day = ? ORDER BY t",
            (symbol, day),
        ).fetchall()
        out = sorted((et_minute_of_day(t), o, h, l, c) for t, o, h, l, c in rows)
        self.cache[key] = out
        return out


# --------------------------------------------------------------------------
# Bar walk primitives
# --------------------------------------------------------------------------

def walk(bars, stop_price, target_price):
    """First bar in `bars` (ascending, already windowed) whose low touches
    stop_price or whose high touches target_price; stop checked first
    inside a bar. Gap-aware: fills at the worse-of(open, level) for a stop,
    better-of(open, level) for a target. Returns (price, type, minute) or
    (None, None, None) if nothing triggers in this window."""
    for minute, o, h, l, c in bars:
        if l <= stop_price:
            return min(o, stop_price), "stop", minute
        if h >= target_price:
            return max(o, target_price), "target", minute
    return None, None, None


def eod_bar(day_bars):
    candidates = [b for b in day_bars if b[0] <= EOD_MINUTE]
    return candidates[-1] if candidates else None


def simulate(day_bars, fill, stop, level, fill_min):
    """Returns a dict with base_R, pyr_R, withdrew, added, exit_type, or
    None if bar coverage is insufficient to resolve both books to an exit."""
    floor_min = int(math.floor(fill_min))
    target = fill + 2.0 * (fill - stop)
    withdraw_level = level - WITHDRAW_PAD
    r_denom = fill - stop
    if r_denom <= 0:
        return None

    fill_eff = fill * (1 + ENTRY_COST)
    post = [b for b in day_bars if floor_min < b[0] <= EOD_MINUTE]
    eod = eod_bar(day_bars)

    # ---------------- paired base: independent, standalone, 1x size -------
    bp, bt, _ = walk(post, stop, target)
    if bp is None:
        if eod is None:
            return None
        bp, bt = eod[4], "eod"
    base_cost = {"stop": STOP_COST, "target": TARGET_COST, "eod": EOD_COST}[bt]
    base_R = (bp * (1 - base_cost) - fill_eff) / r_denom

    # ---------------- pyramid ----------------------------------------------
    window = [b for b in post if b[0] <= floor_min + CHECK_END_OFFSET]
    withdrew = any(b[3] <= withdraw_level for b in window)
    ep, et, _ = walk(window, stop, target)

    if ep is not None:
        # stopped or targeted at/before minute 15 -> no add, single leg
        cost = {"stop": STOP_COST, "target": TARGET_COST}[et]
        pyr_R = (ep * (1 - cost) - fill_eff) / r_denom
        added = False
        exit_type = et
    else:
        added = not withdrew
        if added:
            add_candidates = [b for b in post if b[0] >= floor_min + ADD_OFFSET]
            if not add_candidates:
                return None
            add_bar = add_candidates[0]
            add_eff = add_bar[1] * (1 + ENTRY_COST)
            new_stop = withdraw_level
            cont = [b for b in post if b[0] >= add_bar[0]]
            xp, xt, _ = walk(cont, new_stop, target)
        else:
            add_eff = None
            cont = [b for b in post if b[0] > floor_min + CHECK_END_OFFSET]
            xp, xt, _ = walk(cont, stop, target)

        if xp is None:
            if eod is None:
                return None
            xp, xt = eod[4], "eod"
        cost = {"stop": STOP_COST, "target": TARGET_COST, "eod": EOD_COST}[xt]
        exit_eff = xp * (1 - cost)
        if added:
            pyr_R = (BASE_FRAC * (exit_eff - fill_eff) + ADD_FRAC * (exit_eff - add_eff)) / r_denom
        else:
            pyr_R = (exit_eff - fill_eff) / r_denom
        exit_type = xt

    return dict(base_R=base_R, pyr_R=pyr_R, delta_R=pyr_R - base_R,
                withdrew=withdrew, added=added, exit_type=exit_type)


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

def mde(sd, n):
    if n < 2 or sd == 0 or np.isnan(sd):
        return float("nan")
    return 2.8016 * sd / math.sqrt(n)  # two-sided a=0.05, 80% power


def iid_t(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return float("nan")
    sd = x.std(ddof=1)
    if sd == 0:
        return float("nan")
    return x.mean() / (sd / math.sqrt(n))


def day_clustered_t(x, days):
    df = pd.DataFrame({"x": x, "day": days})
    day_means = df.groupby("day")["x"].mean()
    n_days = len(day_means)
    if n_days < 2:
        return float("nan")
    sd = day_means.std(ddof=1)
    if sd == 0:
        return float("nan")
    return day_means.mean() / (sd / math.sqrt(n_days))


def ex_top5(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0:
        return float("nan")
    k = math.ceil(0.05 * n)
    trimmed = x[: n - k] if k < n else x
    return trimmed.mean() if len(trimmed) else float("nan")


def fills_per_week(days):
    iso = pd.to_datetime(pd.Series(days)).dt.isocalendar()
    weeks = set(zip(iso["year"], iso["week"]))
    return len(days) / max(len(weeks), 1)


def own_stats(df, col):
    x = df[col].values
    return dict(
        n=len(x), mean=float(np.mean(x)) if len(x) else float("nan"),
        iid_t=iid_t(x), day_t=day_clustered_t(x, df["day"].values),
        ex_top5=ex_top5(x), fpw=fills_per_week(df["day"].values),
        mde=mde(np.std(x, ddof=1) if len(x) > 1 else float("nan"), len(x)),
    )


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def main():
    t0 = time.time()
    log.info("Loading fills_1658.csv and causal_arming_causal.csv")
    fills = pd.read_csv(FILLS_CSV)
    causal = pd.read_csv(CAUSAL_CSV)
    log.info("fills_1658 rows=%d causal rows=%d", len(fills), len(causal))

    causal_fill = causal[causal["status"] == "fill"].copy()
    dup_causal = causal_fill.duplicated(subset=["day", "symbol"]).sum()
    dup_fills = fills.duplicated(subset=["date", "symbol"]).sum()
    log.info("causal fill-status rows=%d dup(day,symbol)=%d ; fills_1658 dup(date,symbol)=%d",
             len(causal_fill), dup_causal, dup_fills)
    if dup_causal:
        causal_fill = causal_fill.drop_duplicates(subset=["day", "symbol"], keep="first")
        log.warning("Dropped %d duplicate (day,symbol) rows from causal fill set", dup_causal)

    merged = fills.merge(
        causal_fill[["day", "symbol", "fill", "stop", "level", "fill_min"]],
        left_on=["date", "symbol"], right_on=["day", "symbol"], how="inner",
    )
    log.info("Joined population: %d rows (fills_1658=%d, causal fill=%d)",
              len(merged), len(fills), len(causal_fill))
    unmatched = len(fills) - len(merged)
    log.info("Unmatched fills_1658 rows (no causal fill row found): %d", unmatched)

    store = BarStore(BARS_DB)
    results = []
    no_coverage = 0
    n = len(merged)
    for i, row in enumerate(merged.itertuples(index=False)):
        if i and i % 1000 == 0:
            log.info("progress %d/%d (%.0fs elapsed, %d/%d bar cache hit)",
                      i, n, time.time() - t0, store.hits, store.hits + store.misses)
        day_bars = store.day_bars(row.symbol, row.date)
        if not day_bars:
            no_coverage += 1
            continue
        sim = simulate(day_bars, row.fill, row.stop, row.level, row.fill_min)
        if sim is None:
            no_coverage += 1
            continue
        results.append(dict(
            fill_id=row.fill_id, day=row.date, symbol=row.symbol, split=row.split,
            r_pct=row.r_pct, withdrew=sim["withdrew"], added=sim["added"],
            base_R=sim["base_R"], pyr_R=sim["pyr_R"], delta_R=sim["delta_R"],
            exit_type=sim["exit_type"],
        ))

    log.info("Simulated %d fills, %d had insufficient bar coverage (%.1f%%), %.0fs elapsed",
              len(results), no_coverage, 100.0 * no_coverage / n, time.time() - t0)

    out = pd.DataFrame(results)
    out.to_csv(OUT_PERFILL, index=False)
    log.info("Wrote %s (%d rows)", OUT_PERFILL, len(out))

    coverage_pct = 100.0 * len(out) / n
    write_result_md(merged, out, no_coverage, n, unmatched, dup_causal, dup_fills)
    log.info("Wrote %s. Total elapsed %.0fs", OUT_RESULT, time.time() - t0)


def fmt(v, nd=3):
    if v is None or (isinstance(v, float) and (math.isnan(v))):
        return "n/a"
    return f"{v:.{nd}f}"


def write_result_md(merged, out, no_coverage, n_joined, unmatched, dup_causal, dup_fills):
    halves = sorted(out["split"].unique())
    lines = []
    lines.append("# RESULT 1,666 -- independent rebuild of the no-withdrawal pyramid (cell 1,488) at measured cost")
    lines.append("")
    lines.append("Built from PREREG_1666.md prose alone; no 1487/1488/1662 file opened.")
    lines.append("")
    lines.append("## Coverage")
    lines.append(f"- Joined population (fills_1658 x causal fill-status, inner join on day+symbol): {n_joined} rows "
                  f"(fills_1658={len(merged)+0 if False else ''}".rstrip() + "")
    lines[-1] = f"- Joined population (fills_1658 x causal fill-status, inner join on day+symbol): {n_joined} rows"
    lines.append(f"- fills_1658 rows with no matching causal fill row: {unmatched}")
    lines.append(f"- Duplicate (day,symbol) keys dropped: causal={dup_causal}, fills_1658={dup_fills}")
    lines.append(f"- Simulated to a determinate exit (both books): {len(out)}/{n_joined} "
                 f"({100.0*len(out)/n_joined:.1f}%); excluded for insufficient bar coverage: {no_coverage} "
                 f"({100.0*no_coverage/n_joined:.1f}%)")
    lines.append("")

    def pop_view(df, label):
        rows = []
        for half in halves:
            d = df[df["split"] == half]
            s = own_stats(d, "pyr_R")
            rows.append(f"| {half} | {s['n']} | {fmt(s['mean'])} | {fmt(s['iid_t'],2)} | {fmt(s['day_t'],2)} | "
                        f"{fmt(s['ex_top5'])} | {fmt(s['fpw'],1)} | {fmt(s['mde'])} |")
        return rows

    lines.append("## Own-book (pyramid) R, by half")
    lines.append("Primary population: r_pct >= 1.5%. Unfloored (all r_pct) reported beside each half.")
    lines.append("")
    lines.append("| half | pop | n | mean R | iid t | day-clust t | ex-top5% | fills/wk | MDE |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    prim = out[out["r_pct"] >= 1.5]
    for half in halves:
        for label, df in ((">=1.5%", prim), ("unfloored", out)):
            d = df[df["split"] == half]
            s = own_stats(d, "pyr_R")
            lines.append(f"| {half} | {label} | {s['n']} | {fmt(s['mean'])} | {fmt(s['iid_t'],2)} | "
                         f"{fmt(s['day_t'],2)} | {fmt(s['ex_top5'])} | {fmt(s['fpw'],1)} | {fmt(s['mde'])} |")
    lines.append("")

    lines.append("## Paired base (1x, standard rule, same fills), by half")
    lines.append("| half | pop | n | mean R | iid t | day-clust t | ex-top5% |")
    lines.append("|---|---|---|---|---|---|---|")
    for half in halves:
        for label, df in ((">=1.5%", prim), ("unfloored", out)):
            d = df[df["split"] == half]
            x = d["base_R"].values
            lines.append(f"| {half} | {label} | {len(x)} | {fmt(np.mean(x) if len(x) else float('nan'))} | "
                         f"{fmt(iid_t(x),2)} | {fmt(day_clustered_t(x, d['day'].values),2)} | {fmt(ex_top5(x))} |")
    lines.append("")

    lines.append("## Paired delta_R (pyramid - base), by half -- the pass-bar quantity")
    lines.append("| half | pop | n | mean dR | iid t | day-clust t | ex-top5% dR | MDE |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for half in halves:
        for label, df in ((">=1.5%", prim), ("unfloored", out)):
            d = df[df["split"] == half]
            x = d["delta_R"].values
            lines.append(f"| {half} | {label} | {len(x)} | {fmt(np.mean(x) if len(x) else float('nan'))} | "
                         f"{fmt(iid_t(x),2)} | {fmt(day_clustered_t(x, d['day'].values),2)} | "
                         f"{fmt(ex_top5(x))} | {fmt(mde(np.std(x,ddof=1) if len(x)>1 else float('nan'), len(x)))} |")
    lines.append("")

    lines.append("## Add mechanics, exposure, worst day (primary population, r_pct >= 1.5%)")
    lines.append("| half | add share | worst day (sum pyr_R) |")
    lines.append("|---|---|---|")
    for half in halves:
        d = prim[prim["split"] == half]
        add_share = float(d["added"].mean()) if len(d) else float("nan")
        by_day = d.groupby("day")["pyr_R"].sum()
        worst = float(by_day.min()) if len(by_day) else float("nan")
        lines.append(f"| {half} | {fmt(add_share)} | {fmt(worst)} |")
    lines.append("")
    lines.append("Every add multiplies the live position from 1/3 to full size (3x the base leg); "
                 "no per-trade dollar notional is in this data, so exposure is reported as this fixed "
                 "multiplier, not a dollar figure.")
    lines.append("")

    # Pass bar verdict
    lines.append("## Pass-bar verdict")
    def half_dr(half, pop):
        d = pop[pop["split"] == half]
        x = d["delta_R"].values
        return dict(mean=np.mean(x) if len(x) else float("nan"), t=iid_t(x), dayt=day_clustered_t(x, d["day"].values),
                    ex5=ex_top5(x), fpw=fills_per_week(d["day"].values))

    bar_rows = []
    all_pass_1487 = True
    all_pass_1662 = True
    for half in halves:
        r = half_dr(half, prim)
        own = own_stats(prim[prim["split"] == half], "pyr_R")
        p1487 = (r["mean"] >= 0.05)
        val_extra = ""
        if half == "VAL":
            val_extra = f" VAL t={fmt(r['t'],2)} (>=2.5? {r['t']>=2.5}), VAL book mean={fmt(own['mean'])} (>=0.10? {own['mean']>=0.10})"
            p1487 = p1487 and (r["t"] >= 2.5) and (own["mean"] >= 0.10)
        p1662 = (r["mean"] >= 0.05) and (r["t"] >= 2.5) and (r["dayt"] >= 2.5) and (r["ex5"] > 0) and (r["fpw"] >= 3)
        all_pass_1487 = all_pass_1487 and p1487
        all_pass_1662 = all_pass_1662 and p1662
        bar_rows.append(f"- {half}: dR mean={fmt(r['mean'])}, iid t={fmt(r['t'],2)}, day-clust t={fmt(r['dayt'],2)}, "
                        f"ex-top5%={fmt(r['ex5'])}, fills/wk={fmt(r['fpw'],1)}.{val_extra} "
                        f"1487-bar {'PASS' if p1487 else 'FAIL'}; 1662-bar {'PASS' if p1662 else 'FAIL'}")
    lines.extend(bar_rows)
    lines.append("")
    lines.append(f"**Combined: PREREG_1487 bar {'PASSES' if all_pass_1487 else 'FAILS'} on both halves; "
                 f"PREREG_1662 synthesis bar {'PASSES' if all_pass_1662 else 'FAILS'} on both halves.**")
    lines.append("")
    lines.append("Row-by-row agreement against cell 1,662's per-fill file (withdrawal flag, sign of pyr_R, "
                 ">=95% required before this reaches the owner) is a third-party step outside this build's "
                 "scope -- this agent never opened that file, per the independence rule.")
    lines.append("")

    lines.append("## Adequacy note")
    val_prim = prim[prim["split"] == "VAL"]
    tr_prim = prim[prim["split"] == "TRAIN-H2"]
    val_mde = mde(np.std(val_prim["delta_R"].values, ddof=1) if len(val_prim) > 1 else float("nan"), len(val_prim))
    tr_mde = mde(np.std(tr_prim["delta_R"].values, ddof=1) if len(tr_prim) > 1 else float("nan"), len(tr_prim))
    lines.append(f"MDE on paired dR at n: VAL={fmt(val_mde)} R, TRAIN-H2={fmt(tr_mde)} R (two-sided, 80% power, "
                 f"observed SD). Compare against the +0.05 R pass-bar threshold: an MDE close to or above 0.05 R "
                 f"means a null here is underpowered, not evidence of no effect -- only a point estimate clearly "
                 f"above the MDE (or a pass) should be read as a finding.")
    lines.append("")
    lines.append(f"Programme count on the HOD line: >= 1,700 (this cell = 1,666; per PREREG_1666 pass bar note).")

    with open(OUT_RESULT, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
