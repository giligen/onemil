#!/usr/bin/env python3
"""Cell 1,680: does the 50% partial at +1R buy CONSISTENCY on ORB?

PREREG: research/orb_exit/PREREG_1680.md (FROZEN 2026-09-30).
Owner: "net is zero, but green weeks? green days? must be better, right?"

Input: research/orb_exit/1679_per_fill.csv (per-fill R for the base exit,
rule (d) = scale50_1R_plus_live, rule (e) = live_lock_ref, on L2=BT
2025-01..2026-09 n478 and L1=live n123).

R floor ("R must exceed the spread"): R_pct = R_unit / entry must be >=
0.5%; only 1 row in the input fails it (an L1 fill) -- dropped here, same
convention as cell 1,679's R_FLOOR_PCT.

Dollar conversion: L2 uses a FIXED $375 risk per fill (current ramp
stage). L1 uses the fill's own actual risk dollars = shares * R_unit
("actual shares" per PREREG), verified against the actual_dollar column
(actual_R * shares * R_unit == actual_dollar on every row).

Weeks: Monday-start buckets of the fill date (== ISO week grouping;
reuses scripts/cadence_bar.week_monday so results are directly comparable
to the cadence bar's own weekly series). Weeks with zero fills are kept
as flat (R=0) weeks in every range, same convention as cadence_bar.py.

Null (PREREG override of cadence_bar's own C4 null): shuffle the rule's
per-fill R values across the fixed week-slot structure (i.e. permute
which fill's R lands in which week, holding the number of fills per week
constant) 1000x; read the green-week share of each shuffle; report the
percentile of the ACTUAL green-week share within that null distribution.
"""
import csv
import logging
import random
import statistics
import sys
from collections import defaultdict
from datetime import timedelta

sys.path.insert(0, "scripts")
import cadence_bar as cb  # noqa: E402  (week_monday, percentile, compute_cycles, score_c1, FLAT_THRESH_R, max_drawdown_and_underwater)

IN_CSV = "research/orb_exit/1679_per_fill.csv"
OUT_DIR = "research/orb_exit"
R_FLOOR_PCT = 0.005
L2_RISK_DOLLARS = 375.0
RULE_COL = {"base": "actual_R", "d": "scale50_1R_plus_live", "e": "live_lock_ref"}
N_NULL = 1000
SEED = 0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(f"{OUT_DIR}/1680_consistency.log", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("1680")


# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------

def load_rows():
    rows = []
    n_blank = 0
    with open(IN_CSV, newline="") as f:
        for row in csv.DictReader(f):
            row["_date"] = cb.parse_date(row["date"])
            r_pct = float(row["R_pct"])
            row["_floor_ok"] = r_pct >= R_FLOOR_PCT
            row["_risk_dollars"] = float(row["shares"]) * float(row["R_unit"])
            blank_cols = [col for col in RULE_COL.values() if row[col].strip() == ""]
            if blank_cols:
                n_blank += 1
                logger.warning(
                    "blank rule value(s) %s, dropped: ledger=%s fill_id=%s",
                    blank_cols, row["ledger"], row["fill_id"],
                )
                continue
            for key, col in RULE_COL.items():
                row["_r_%s" % key] = float(row[col])
            rows.append(row)
    n_fail = [r for r in rows if not r["_floor_ok"]]
    for r in n_fail:
        logger.warning(
            "R floor fail (R_pct=%.4f < %.3f), dropped: ledger=%s fill_id=%s",
            float(r["R_pct"]), R_FLOOR_PCT, r["ledger"], r["fill_id"],
        )
    kept = [r for r in rows if r["_floor_ok"]]
    logger.info(
        "loaded %d rows, dropped %d blank-rule + %d R-floor fail, kept %d",
        len(rows) + n_blank, n_blank, len(n_fail), len(kept),
    )
    # sanity: actual_dollar == actual_R * risk_dollars
    bad = [r for r in kept if abs(r["_r_base"] * r["_risk_dollars"] - float(r["actual_dollar"])) > 0.05]
    if bad:
        logger.error("%d rows fail the actual_dollar = actual_R * shares * R_unit identity -- dollar conversion suspect", len(bad))
    return kept


def dollar_of(row, rulekey):
    r = row["_r_%s" % rulekey]
    if row["ledger"] == "L2":
        return r * L2_RISK_DOLLARS
    return r * row["_risk_dollars"]


# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------

def make_splits(rows):
    l2 = [r for r in rows if r["ledger"] == "L2"]
    l1 = [r for r in rows if r["ledger"] == "L1"]
    splits = {
        "L2_2025": [r for r in l2 if r["half"] == "2025"],
        "L2_2026": [r for r in l2 if r["half"] == "2026"],
        "L2_whole": l2,
        "L1_whole": l1,
    }
    for name, rs in splits.items():
        if rs:
            lo, hi = min(r["_date"] for r in rs), max(r["_date"] for r in rs)
            logger.info("split %-9s n=%4d  %s .. %s", name, len(rs), lo, hi)
    return splits


# --------------------------------------------------------------------------
# Reads
# --------------------------------------------------------------------------

def per_fill_read(rows, rulekey):
    rs = [r["_r_%s" % rulekey] for r in rows]
    n = len(rs)
    if n == 0:
        return dict(n=0)
    mean_r = statistics.fmean(rs)
    win_rate = sum(1 for x in rs if x > 0) / n
    sd = statistics.stdev(rs) if n > 1 else None
    srt = sorted(rs)
    k = max(1, round(n * 0.05))
    ex_top5 = srt[: n - k] if k < n else []
    ex_top5_mean = statistics.fmean(ex_top5) if ex_top5 else None
    return dict(n=n, mean_r=mean_r, win_rate=win_rate, sd=sd, ex_top5_mean=ex_top5_mean)


def daily_agg(rows, rulekey):
    by_day = defaultdict(lambda: [0.0, 0.0])
    for r in rows:
        d = r["_date"]
        by_day[d][0] += r["_r_%s" % rulekey]
        by_day[d][1] += dollar_of(r, rulekey)
    days = sorted(by_day)
    r_series = [by_day[d][0] for d in days]
    dollar_series = [by_day[d][1] for d in days]
    return days, r_series, dollar_series


def daily_read(rows, rulekey):
    days, r_series, dollar_series = daily_agg(rows, rulekey)
    if not days:
        return dict(n_days=0), days, r_series, dollar_series
    green = sum(1 for v in r_series if v > 0)
    red = sum(1 for v in r_series if v < 0)
    green_share = green / (green + red) if (green + red) else None
    p10 = cb.percentile(r_series, 10)
    worst = min(r_series)
    mdd_r, _ = cb.max_drawdown_and_underwater(r_series)
    mdd_d, _ = cb.max_drawdown_and_underwater(dollar_series)
    return dict(
        n_days=len(days), green_share=green_share, p10=p10, worst=worst,
        mdd_r=mdd_r, mdd_dollar=mdd_d,
    ), days, r_series, dollar_series


def weekly_agg(rows, rulekey, lo, hi):
    by_week = defaultdict(lambda: [0.0, 0.0])
    for r in rows:
        wk = cb.week_monday(r["_date"])
        by_week[wk][0] += r["_r_%s" % rulekey]
        by_week[wk][1] += dollar_of(r, rulekey)
    start, end = cb.week_monday(lo), cb.week_monday(hi)
    weeks, r_series, dollar_series = [], [], []
    wk = start
    while wk <= end:
        rr, dd = by_week.get(wk, [0.0, 0.0])
        weeks.append(wk)
        r_series.append(rr)
        dollar_series.append(dd)
        wk += timedelta(days=7)
    return weeks, r_series, dollar_series


def green_share_of(vals, flat=cb.FLAT_THRESH_R):
    g = sum(1 for v in vals if v >= flat)
    rd = sum(1 for v in vals if v <= -flat)
    return (g / (g + rd)) if (g + rd) else None


def weekly_read(rows, rulekey, lo, hi):
    weeks, r_series, dollar_series = weekly_agg(rows, rulekey, lo, hi)
    if not weeks:
        return dict(n_weeks=0), weeks, r_series, dollar_series
    green_share = green_share_of(r_series)
    mean_r = statistics.fmean(r_series)
    sd = statistics.stdev(r_series) if len(r_series) > 1 else None
    sharpe = (mean_r / sd) if sd else None
    p10 = cb.percentile(r_series, 10)
    worst = min(r_series)
    mdd_r, uw = cb.max_drawdown_and_underwater(r_series)
    mdd_d, _ = cb.max_drawdown_and_underwater(dollar_series)
    weekly_pairs = list(zip(weeks, r_series))
    cycles, strong_idx = cb.compute_cycles(weekly_pairs, strong_r=5.0)
    c1 = cb.score_c1(cycles, gap_median_thresh=3.0, gap_p90_thresh=6.0)
    return dict(
        n_weeks=len(weeks), green_share=green_share, mean_r=mean_r, sd=sd,
        sharpe=sharpe, p10=p10, worst=worst, mdd_r=mdd_r, mdd_dollar=mdd_d,
        underwater_max=uw, n_strong=len(strong_idx), n_cycles=len(cycles),
        gap_median=c1["median"], gap_p90=c1["p90"],
    ), weeks, r_series, dollar_series


def fnum(x, nd=3):
    """Format a possibly-None float, or pass through an int/str as-is."""
    if x is None:
        return "N/A"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def null_percentile_fills(rows, rulekey, weeks):
    """weeks: list of Monday dates spanning the range (incl. zero-fill weeks)."""
    rng = random.Random(SEED)
    week_pos = {w: i for i, w in enumerate(weeks)}
    slot_idx = [week_pos[cb.week_monday(r["_date"])] for r in rows]
    fill_rs = [r["_r_%s" % rulekey] for r in rows]
    actual_by_week = [0.0] * len(weeks)
    for idx, val in zip(slot_idx, fill_rs):
        actual_by_week[idx] += val
    actual_share = green_share_of(actual_by_week)
    null_shares = []
    for _ in range(N_NULL):
        shuffled = fill_rs[:]
        rng.shuffle(shuffled)
        by_week = [0.0] * len(weeks)
        for idx, val in zip(slot_idx, shuffled):
            by_week[idx] += val
        s = green_share_of(by_week)
        if s is not None:
            null_shares.append(s)
    null_shares.sort()
    if actual_share is None or not null_shares:
        return actual_share, None, null_shares
    pctl = sum(1 for s in null_shares if s <= actual_share) / len(null_shares) * 100.0
    return actual_share, pctl, null_shares


def compounding_read(rows_l2_whole, lo, hi):
    out = {}
    for rulekey in ("base", "d"):
        for risk in (375.0, 750.0):
            weeks, r_series, _ = weekly_agg(rows_l2_whole, rulekey, lo, hi)
            cap = 65000.0
            for r in r_series:
                pnl = r * risk
                cap *= (1.0 + pnl / cap)
            n = len(weeks)
            cagr_wk = (cap / 65000.0) ** (1.0 / n) - 1.0 if n else None
            out[(rulekey, risk)] = dict(final_capital=cap, n_weeks=n, weekly_geo_growth=cagr_wk)
    return out


# --------------------------------------------------------------------------
# Consistency clause table (d vs base, L2 2025 and 2026)
# --------------------------------------------------------------------------

def clause_table(read_d, read_base, gap_d, gap_base, null_pctl_d, pf_d, pf_base):
    clauses = {}
    # PREREG clause 1 is the PER-FILL mean R (Read 1), not the weekly-sum mean.
    clauses["mean_R_within_0.03"] = dict(
        base=pf_base["mean_r"], d=pf_d["mean_r"],
        delta=pf_d["mean_r"] - pf_base["mean_r"],
        ok=abs(pf_d["mean_r"] - pf_base["mean_r"]) <= 0.03,
    )
    gs_base = read_base["green_share"] or 0.0
    gs_d = read_d["green_share"] or 0.0
    clauses["green_week_share_ge_base_plus5pp"] = dict(
        base=gs_base, d=gs_d, delta=gs_d - gs_base, ok=(gs_d >= gs_base + 0.05),
    )
    p10_base = read_base["p10"]
    p10_d = read_d["p10"]
    clauses["weekly_P10_ge_base_plus_0.5R"] = dict(
        base=p10_base, d=p10_d, delta=(p10_d - p10_base) if (p10_base is not None and p10_d is not None) else None,
        ok=(p10_d is not None and p10_base is not None and p10_d >= p10_base + 0.5),
    )
    sh_base = read_base["sharpe"]
    sh_d = read_d["sharpe"]
    thresh = sh_base * 1.15 if sh_base is not None else None
    clauses["weekly_sharpe_ge_base_x1.15"] = dict(
        base=sh_base, d=sh_d, threshold=thresh,
        ok=(sh_d is not None and thresh is not None and sh_d >= thresh),
    )
    # strong-week median gap: not-worse by > 1 week; N/A if either side lacks >=2 strong weeks
    if gap_base is None or gap_d is None:
        clauses["gap_median_not_worse_1wk"] = dict(base=gap_base, d=gap_d, ok=False, note="N/A: <2 strong weeks on one side")
    else:
        clauses["gap_median_not_worse_1wk"] = dict(base=gap_base, d=gap_d, ok=(gap_d <= gap_base + 1))
    clauses["null_percentile_ge_95"] = dict(value=null_pctl_d, ok=(null_pctl_d is not None and null_pctl_d >= 95.0))
    overall = all(c.get("ok") for c in clauses.values())
    return clauses, overall


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    rows = load_rows()
    splits = make_splits(rows)

    weekly_csv_rows = []
    daily_csv_rows = []
    all_reads = {}

    for split_name, rs in splits.items():
        if not rs:
            logger.warning("split %s empty -- skipped", split_name)
            continue
        lo, hi = min(r["_date"] for r in rs), max(r["_date"] for r in rs)
        for rulekey in ("base", "d", "e"):
            pf = per_fill_read(rs, rulekey)
            dr, days, d_r, d_d = daily_read(rs, rulekey)
            wr, weeks, w_r, w_d = weekly_read(rs, rulekey, lo, hi)
            actual_share, pctl, _ = null_percentile_fills(rs, rulekey, weeks)
            gday = dr.get("green_share")
            logger.info(
                "%-9s %-4s n=%3d meanR=%6.3f win=%4.1f%% wkGreen=%s dayGreen=%s wkP10=%s sharpe=%s nullPctl=%s",
                split_name, rulekey, pf["n"], pf["mean_r"] or 0, (pf["win_rate"] or 0) * 100,
                None if wr.get("green_share") is None else round(wr["green_share"], 3),
                None if gday is None else round(gday, 3),
                None if wr.get("p10") is None else round(wr["p10"], 3),
                None if wr.get("sharpe") is None else round(wr["sharpe"], 3),
                None if pctl is None else round(pctl, 1),
            )
            all_reads[(split_name, rulekey)] = dict(per_fill=pf, daily=dr, weekly=wr, null_pctl=pctl)
            for w, rr, dd in zip(weeks, w_r, w_d):
                weekly_csv_rows.append([split_name, rulekey, w.isoformat(), f"{rr:.6f}", f"{dd:.2f}"])
            for d, rr, dd in zip(days, d_r, d_d):
                daily_csv_rows.append([split_name, rulekey, d.isoformat(), f"{rr:.6f}", f"{dd:.2f}"])

    with open(f"{OUT_DIR}/1680_weekly.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "rule", "week_monday", "R_sum", "dollar_sum"])
        w.writerows(weekly_csv_rows)
    with open(f"{OUT_DIR}/1680_daily.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "rule", "date", "R_sum", "dollar_sum"])
        w.writerows(daily_csv_rows)
    logger.info("wrote 1680_weekly.csv (%d rows), 1680_daily.csv (%d rows)", len(weekly_csv_rows), len(daily_csv_rows))

    # Clause table for L2 2025 and L2 2026
    clause_results = {}
    for yr_split in ("L2_2025", "L2_2026"):
        rd = all_reads[(yr_split, "d")]
        rb = all_reads[(yr_split, "base")]
        clauses, overall = clause_table(
            rd["weekly"], rb["weekly"], rd["weekly"].get("gap_median"), rb["weekly"].get("gap_median"),
            rd["null_pctl"], rd["per_fill"], rb["per_fill"],
        )
        clause_results[yr_split] = (clauses, overall)

    overall_pass = all(ov for _, ov in clause_results.values())

    # Compounding read on L2 whole
    l2_whole = splits["L2_whole"]
    lo_w, hi_w = min(r["_date"] for r in l2_whole), max(r["_date"] for r in l2_whole)
    comp = compounding_read(l2_whole, lo_w, hi_w)

    # ---- RESULT_1680.md ----
    lines = []
    lines.append("# RESULT — cell 1,680: does the 50% partial at +1R buy consistency on ORB?")
    lines.append("")
    lines.append(f"PREREG: research/orb_exit/PREREG_1680.md. R floor drops 1/601 fills (L1 EIDO_2026-06-09_195).")
    lines.append("")
    lines.append("## Consistency bar clause table -- (d) 50% @ +1R vs base, L2 by year")
    lines.append("")
    lines.append("| clause | 2025 base | 2025 (d) | 2025 pass | 2026 base | 2026 (d) | 2026 pass |")
    lines.append("|---|---|---|---|---|---|---|")
    c25, _ = clause_results["L2_2025"]
    c26, _ = clause_results["L2_2026"]
    for key in c25:
        b25, d25v, ok25 = c25[key].get("base"), c25[key].get("d") or c25[key].get("value"), c25[key]["ok"]
        b26, d26v, ok26 = c26[key].get("base"), c26[key].get("d") or c26[key].get("value"), c26[key]["ok"]
        fmt = lambda x: "N/A" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))
        lines.append(f"| {key} | {fmt(b25)} | {fmt(d25v)} | {'PASS' if ok25 else 'FAIL'} | {fmt(b26)} | {fmt(d26v)} | {'PASS' if ok26 else 'FAIL'} |")
    lines.append("")
    lines.append(f"**Overall verdict: {'PASS' if overall_pass else 'FAIL'}** (all clauses must pass in BOTH years).")
    lines.append("")
    lines.append("## Green-day share (day sum > 0 vs < 0, days with >= 1 fill)")
    lines.append("")
    lines.append("| split | rule | n_days | green_share | daily P10 | worst day | MDD (R) | MDD ($) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for split_name in ("L2_2025", "L2_2026", "L2_whole", "L1_whole"):
        for rulekey in ("base", "d", "e"):
            if (split_name, rulekey) not in all_reads:
                continue
            dr = all_reads[(split_name, rulekey)]["daily"]
            if dr.get("n_days", 0) == 0:
                continue
            lines.append(
                f"| {split_name} | {rulekey} | {dr['n_days']} | {dr['green_share']:.3f} | "
                f"{dr['p10']:.3f} | {dr['worst']:.3f} | {dr['mdd_r']:.2f} | {dr['mdd_dollar']:.0f} |"
            )
    lines.append("")
    lines.append("## Weekly reads (all splits, all rules)")
    lines.append("")
    lines.append("| split | rule | n_wk | green_share | null_pctl | mean_R | SD | Sharpe | P10 | worst | strong_wk | gap_med | gap_p90 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for split_name in ("L2_2025", "L2_2026", "L2_whole", "L1_whole"):
        for rulekey in ("base", "d", "e"):
            if (split_name, rulekey) not in all_reads:
                continue
            wr = all_reads[(split_name, rulekey)]["weekly"]
            pctl = all_reads[(split_name, rulekey)]["null_pctl"]
            if wr.get("n_weeks", 0) == 0:
                continue
            lines.append(
                f"| {split_name} | {rulekey} | {wr['n_weeks']} | "
                f"{fnum(wr['green_share'])} | {fnum(pctl, 1)} | {fnum(wr['mean_r'])} | "
                f"{fnum(wr['sd'])} | {fnum(wr['sharpe'])} | "
                f"{fnum(wr['p10'])} | {fnum(wr['worst'])} | {wr['n_strong']} | "
                f"{fnum(wr['gap_median'], 1)} | {fnum(wr['gap_p90'], 1)} |"
            )
    lines.append("")
    lines.append("## Compounding read -- L2 whole (2025-01..2026-09), $65K start, weekly geometric compounding")
    lines.append("Weekly $ = R_sum_week x fixed risk-per-fill ($375 current stage / $750 next rung); cap *= (1 + $/cap) each week.")
    lines.append("")
    lines.append("| rule | risk/fill | n_wk | final capital | weekly geo growth |")
    lines.append("|---|---|---|---|---|")
    for rulekey in ("base", "d"):
        for risk in (375.0, 750.0):
            c = comp[(rulekey, risk)]
            lines.append(f"| {rulekey} | ${risk:.0f} | {c['n_weeks']} | ${c['final_capital']:,.0f} | {c['weekly_geo_growth']*100:.3f}% |")
    lines.append("")
    lines.append("## L1 (live) stability -- damaged-execution period, forward reference only")
    lines.append("")
    for rulekey in ("base", "d", "e"):
        if ("L1_whole", rulekey) not in all_reads:
            continue
        wr = all_reads[("L1_whole", rulekey)]["weekly"]
        pf = all_reads[("L1_whole", rulekey)]["per_fill"]
        lines.append(
            f"- {rulekey}: n={pf['n']} fills, mean R={pf['mean_r']:.3f}, weekly green share="
            f"{fnum(wr['green_share'])}, n_weeks={wr['n_weeks']}"
        )
    lines.append("")
    lines.append("Caveats: L1 spans 2026-05-19..2026-09-28 only (the damaged-execution period, small n);")
    lines.append("strong-week gap (C1) requires >=2 weeks at R>=+5 -- most splits here never reach a single")
    lines.append("+5R week at this book's per-fill scale, so gap_median/gap_p90/the clause built on them read N/A.")

    with open(f"{OUT_DIR}/RESULT_1680.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("wrote RESULT_1680.md (%d lines)", len(lines))
    logger.info("OVERALL VERDICT: %s", "PASS" if overall_pass else "FAIL")


if __name__ == "__main__":
    main()
