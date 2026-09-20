#!/usr/bin/env python3
"""
Cadence bar scorer — docs/cadence_bar.md (owner, 2026-09-20).

Scores a book by WEEKS and the CYCLE between strong weeks, not by trades.
A book is evaluated as a renewal process: strong weeks (net >= +5R) are the
renewals; everything between two of them is the cycle. The book passes if
renewals come often enough (C1), regularly enough (C7), the bleed between
them is shallow enough (C2/C3), it isn't a coin flip (C4), it trades often
enough for the math to be possible (C5), and its tail is real (C6).

Usage:
    python scripts/cadence_bar.py --trades trades.csv --split TRAIN
    python scripts/cadence_bar.py --trades trades.csv --split VAL \\
        --tail-audit tail_audit.csv --book orb --slots 4 \\
        --fill-model "next-open capped" --r-dollars 375

Input trades CSV: one row per CLOSED trade.
    date     - exit date, YYYY-MM-DD (required)
    pnl_R    - P&L in R (risk multiples), OR
    pnl,risk - dollar P&L and dollar risk-per-trade (R = pnl / risk)
    symbol   - optional, only used to join --tail-audit rows for C6

--live/--book reads live fills from the project's `trades` table (cache.db,
persistence.database.TradeDatabase.get_strategy_trades_in_window) instead of
--trades. NOT YET WIRED (TODO): the trades-table row shape (exact pnl/risk
column names per strategy) needs its own short investigation before this can
be trusted numerically — plumbing it in blind risked silently mis-scoring a
live book. Passing --live today raises NotImplementedError with this note.
"""
import argparse
import csv
import logging
import random
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta

logger = logging.getLogger("cadence_bar")

# --- §0/§3: splits, fixed per docs/cadence_bar.md, never inferred from data.
SPLIT_RANGES = {
    "TRAIN": (date(2025, 1, 1), date(2025, 12, 31)),
    "VAL": (date(2026, 1, 1), date(2026, 5, 31)),
}
TEST_START = date(2026, 6, 1)

FLAT_THRESH_R = 0.5  # §0: a week with |P&L| < 0.5 R is FLAT.


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def parse_date(s):
    """Parse a YYYY-MM-DD string into a datetime.date."""
    return datetime.strptime(s.strip(), "%Y-%m-%d").date()


def load_trades_csv(path):
    """Load a trades CSV into a list of {'date', 'r', 'symbol'} dicts.

    Accepts either a `pnl_R` column, or `pnl` + `risk` columns (R computed
    as pnl / risk). Rows with zero/missing risk when only pnl+risk is
    available are skipped with a WARNING (can't compute R, not a silent
    drop of unrelated data).
    """
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        has_pnl_r = "pnl_R" in fields
        has_pnl_risk = "pnl" in fields and "risk" in fields
        if not has_pnl_r and not has_pnl_risk:
            raise ValueError(
                f"{path}: need a 'pnl_R' column, or both 'pnl' and 'risk'; "
                f"found columns {fields}"
            )
        for i, row in enumerate(reader):
            d = parse_date(row["date"])
            r = None
            if has_pnl_r and row.get("pnl_R", "") not in ("", None):
                r = float(row["pnl_R"])
            elif has_pnl_risk:
                risk = float(row["risk"])
                if risk == 0:
                    logger.warning(
                        "row %d (date=%s): risk=0, cannot compute R — skipped",
                        i, row["date"],
                    )
                    continue
                r = float(row["pnl"]) / risk
            else:
                logger.warning(
                    "row %d (date=%s): no pnl_R and no usable pnl/risk — skipped",
                    i, row["date"],
                )
                continue
            rows.append({"date": d, "r": r, "symbol": row.get("symbol")})
    return rows


def load_trades_live(book):
    """Load closed trades for `book` from cache.db (live tracking, §4).

    TODO: not wired. persistence.database.TradeDatabase has
    get_strategy_trades_in_window(strategy, since_date, symbols=None),
    which is the right seam, but the exact pnl/risk column names on the
    `trades` row (and whether they're already R or need derivation per
    strategy) need a short dedicated investigation before scoring a live
    book on them. Wire this up, don't guess the column mapping.
    """
    raise NotImplementedError(
        "cadence_bar.py --live is not wired yet. The `trades` table is "
        "reachable via persistence.database.TradeDatabase."
        "get_strategy_trades_in_window(strategy, since_date) — see the "
        "module docstring TODO. Use --trades CSV in the meantime."
    )


def filter_split(trades, split, include_test):
    """Filter trades to the split's fixed date range (§0)."""
    if split == "ALL":
        lo, hi = SPLIT_RANGES["TRAIN"][0], SPLIT_RANGES["VAL"][1]
    else:
        lo, hi = SPLIT_RANGES[split]
    out = []
    for t in trades:
        if t["date"] < lo or t["date"] > hi:
            continue
        if t["date"] >= TEST_START and not include_test:
            logger.warning(
                "trade on %s is >= TEST_START (%s) — excluded (pass "
                "--include-test to override)", t["date"], TEST_START,
            )
            continue
        out.append(t)
    return out, lo, hi


# --------------------------------------------------------------------------
# Weekly series (§0 "Week" + §1)
# --------------------------------------------------------------------------

def week_monday(d):
    """Monday of the Mon-Fri session week containing date d."""
    return d - timedelta(days=d.weekday())


def build_weekly_series(trades, lo, hi):
    """Sum trade R into Mon-Fri weeks, spanning every week in [lo, hi] —
    including weeks with zero trades (they are a flat week, still part of
    the cadence walk). Returns a list of (monday_date, r_sum) tuples,
    sorted by date.
    """
    by_week = defaultdict(float)
    for t in trades:
        by_week[week_monday(t["date"])] += t["r"]
    start, end = week_monday(lo), week_monday(hi)
    weeks = []
    wk = start
    while wk <= end:
        weeks.append((wk, by_week.get(wk, 0.0)))
        wk += timedelta(days=7)
    return weeks


def classify_week(r, flat_thresh=FLAT_THRESH_R):
    """green / red / flat per §0 (flat = |P&L| < 0.5 R)."""
    if abs(r) < flat_thresh:
        return "flat"
    return "green" if r > 0 else "red"


# --------------------------------------------------------------------------
# Stats helpers
# --------------------------------------------------------------------------

def percentile(values, p):
    """Linear-interpolated percentile (numpy 'linear' convention), no numpy
    dependency. Returns None for an empty input.
    """
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (p / 100.0)
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] * (c - k) + s[c] * (k - f)


def max_drawdown_and_underwater(weekly_r):
    """Max drawdown (in R, as a positive number) and the longest run of
    consecutive weeks strictly under the running high-water mark, on the
    cumulative equity curve built from weekly_r.
    """
    peak = 0.0
    cum = 0.0
    mdd = 0.0
    cur_uw = 0
    max_uw = 0
    for r in weekly_r:
        cum += r
        if cum > peak:
            peak = cum
            cur_uw = 0
        else:
            cur_uw += 1
            max_uw = max(max_uw, cur_uw)
        mdd = max(mdd, peak - cum)
    return mdd, max_uw


# --------------------------------------------------------------------------
# §1 definitions: strong week / gap / cycle / bleed / cycle net
# --------------------------------------------------------------------------

def compute_cycles(weekly, strong_r):
    """Find strong weeks (net >= strong_r) and build the cycles between
    consecutive ones. Returns (cycles, strong_idx) where each cycle is a
    dict with gap (int weeks), bleed (R), cycle_net (R), start/end indices.
    N strong weeks give N-1 cycles/gaps (need >= 2 strong weeks for any).
    """
    strong_idx = [i for i, (_, r) in enumerate(weekly) if r >= strong_r]
    cycles = []
    for a, b in zip(strong_idx, strong_idx[1:]):
        gap = b - a
        bleed = sum(r for _, r in weekly[a + 1:b])
        cycle_net = bleed + weekly[b][1]
        cycles.append({
            "gap": gap, "bleed": bleed, "cycle_net": cycle_net,
            "start": a, "end": b,
        })
    return cycles, strong_idx


# --------------------------------------------------------------------------
# §2 criteria C1-C7
# --------------------------------------------------------------------------

def score_c1(cycles, gap_median_thresh, gap_p90_thresh):
    gaps = [c["gap"] for c in cycles]
    if not gaps:
        return {"pass": False, "median": None, "p90": None, "gaps": []}
    med = percentile(gaps, 50)
    p90 = percentile(gaps, 90)
    ok = med <= gap_median_thresh and p90 <= gap_p90_thresh
    return {"pass": ok, "median": med, "p90": p90, "gaps": gaps}


def score_c2(cycles, bleed_p90_thresh, cycle_pos_thresh):
    if not cycles:
        return {"pass": False, "bleed_p90": None, "net_pos_frac": None}
    bleeds = [c["bleed"] for c in cycles]
    bleed_p90 = percentile(bleeds, 90)
    net_pos_frac = sum(1 for c in cycles if c["cycle_net"] > 0) / len(cycles)
    ok = bleed_p90 >= bleed_p90_thresh and net_pos_frac >= cycle_pos_thresh
    return {"pass": ok, "bleed_p90": bleed_p90, "net_pos_frac": net_pos_frac}


def score_c3(weekly, p10_thresh, min_thresh, mdd_thresh, underwater_thresh):
    weekly_r = [r for _, r in weekly]
    if not weekly_r:
        return {"pass": False, "p10": None, "min": None, "mdd": None, "underwater": None}
    p10 = percentile(weekly_r, 10)
    wmin = min(weekly_r)
    mdd, uw = max_drawdown_and_underwater(weekly_r)
    ok = (p10 >= p10_thresh and wmin >= min_thresh
          and mdd <= mdd_thresh and uw <= underwater_thresh)
    return {"pass": ok, "p10": p10, "min": wmin, "mdd": mdd, "underwater": uw}


def _green_share(weekly):
    """green / (green + red), flat weeks excluded. None if no non-flat weeks."""
    g = sum(1 for _, r in weekly if classify_week(r) == "green")
    rd = sum(1 for _, r in weekly if classify_week(r) == "red")
    if g + rd == 0:
        return None
    return g / (g + rd)


def score_c4(weekly, trades_in_split, green_thresh, green_margin, n_null=1000, rng=None):
    """C4: green share must clear both an absolute bar and a count-matched
    null (mean green share of `n_null` sign-shuffled weekly series, built
    by flipping each trade's R sign 50/50, keeping trade dates fixed, and
    re-bucketing into weeks).
    """
    rng = rng or random.Random(0)
    real = _green_share(weekly)
    if real is None:
        return {"pass": False, "green": None, "null": None}

    lo = min(w for w, _ in weekly)
    hi = max(w for w, _ in weekly)
    weeks_index = []
    wk = lo
    while wk <= hi:
        weeks_index.append(wk)
        wk += timedelta(days=7)

    null_shares = []
    for _ in range(n_null):
        by_week = defaultdict(float)
        for t in trades_in_split:
            sign = 1.0 if rng.random() < 0.5 else -1.0
            by_week[week_monday(t["date"])] += sign * t["r"]
        shuffled_weekly = [(w, by_week.get(w, 0.0)) for w in weeks_index]
        s = _green_share(shuffled_weekly)
        if s is not None:
            null_shares.append(s)
    null_mean = sum(null_shares) / len(null_shares) if null_shares else None

    ok = (real >= green_thresh
          and null_mean is not None and real >= null_mean + green_margin)
    return {"pass": ok, "green": real, "null": null_mean}


def score_c5(n_trades, n_weeks, fills_wk_thresh):
    if n_weeks == 0:
        return {"pass": False, "fills_wk": None}
    rate = n_trades / n_weeks
    return {"pass": rate >= fills_wk_thresh, "fills_wk": rate}


def score_c6(trades_in_split, tail_audit_path):
    """Tail realism: every trade >= +3R must pass the obtainability audit,
    joined by (date, symbol) where both sides carry a symbol, else by date
    alone. Returns None (not m/n) when no --tail-audit was supplied — the
    caller prints "C6 not audited" per docs/cadence_bar.md §5.
    """
    if tail_audit_path is None:
        return None
    audit = {}
    with open(tail_audit_path, newline="") as f:
        for row in csv.DictReader(f):
            key = (parse_date(row["date"]), row.get("symbol") or None)
            audit[key] = int(row["obtainable"])

    tail_trades = [t for t in trades_in_split if t["r"] >= 3.0]
    n = len(tail_trades)
    m = 0
    failures = []
    for t in tail_trades:
        key = (t["date"], t.get("symbol") or None)
        ok = audit.get(key)
        if ok is None and t.get("symbol") is None:
            # No symbol on either side to disambiguate: match by date only.
            date_matches = [v for (d, _), v in audit.items() if d == t["date"]]
            ok = date_matches[0] if len(date_matches) == 1 else None
        if ok is None:
            logger.warning(
                "C6: no tail-audit row found for %s R=%.2f — treated as FAIL",
                key, t["r"],
            )
            ok = 0
        if ok:
            m += 1
        else:
            failures.append(key)
    return {"pass": m == n, "m": m, "n": n, "failures": failures}


def score_c7(cycles, weekly_r, min_cycles, gap_p90_bootstrap_thresh,
             strong_r, block_size=4, n_boot=1000, rng=None):
    """Power: >= min_cycles per split, and a 4-week block bootstrap of the
    weekly series puts the 75% upper bound of the bootstrap P90-gap
    distribution <= gap_p90_bootstrap_thresh weeks.
    """
    rng = rng or random.Random(0)
    n_cycles = len(cycles)
    n = len(weekly_r)
    p90_gaps = []
    if n > 0:
        for _ in range(n_boot):
            resampled = []
            while len(resampled) < n:
                start = rng.randrange(0, n)
                for k in range(block_size):
                    resampled.append(weekly_r[(start + k) % n])
                    if len(resampled) >= n:
                        break
            boot_weekly = [(None, r) for r in resampled]
            boot_cycles, _ = compute_cycles(boot_weekly, strong_r)
            gaps = [c["gap"] for c in boot_cycles]
            if gaps:
                p90_gaps.append(percentile(gaps, 90))
    boot_ub = percentile(p90_gaps, 75) if p90_gaps else None
    ok = (n_cycles >= min_cycles
          and boot_ub is not None and boot_ub <= gap_p90_bootstrap_thresh)
    return {"pass": ok, "cycles": n_cycles, "bootstrap_p90_ub": boot_ub}


# --------------------------------------------------------------------------
# §5 output block
# --------------------------------------------------------------------------

def _fmt(x, nd=2):
    return "N/A" if x is None else f"{x:.{nd}f}"


def render_report(*, book, split, slots, fill_model, r_dollars,
                   c1, c2, c3, c4, c5, c6, c7, weekly, trades_in_split):
    """Render the exact §5 report block."""
    pf = lambda ok: "pass" if ok else "fail"
    weekly_r = [r for _, r in weekly]
    top5 = sorted(weekly_r, reverse=True)[:max(1, len(weekly_r) // 20 or 1)]
    ex_top5 = sum(weekly_r) - sum(top5)
    top5_share = (sum(top5) / sum(weekly_r) * 100.0) if sum(weekly_r) else 0.0
    capped = sum(min(r, 5.0) for r in weekly_r)

    lines = []
    lines.append(
        f"CADENCE BAR  ({book}, {split}, live config: {slots} slots, "
        f"{fill_model}, R = ${r_dollars})"
    )
    lines.append(
        f"C1 gap       median {_fmt(c1['median'], 1)} wk  "
        f"P90 {_fmt(c1['p90'], 1)} wk        [{pf(c1['pass'])}]   "
        f"gaps: {c1['gaps']}"
    )
    lines.append(
        f"C2 bleed     P90 {_fmt(c2['bleed_p90'])} R     "
        f"cycles net>0 {_fmt((c2['net_pos_frac'] or 0) * 100, 0)}% "
        f"[{pf(c2['pass'])}]"
    )
    lines.append(
        f"C3 reds      P10 {_fmt(c3['p10'])} R  min {_fmt(c3['min'])} R  "
        f"MDD {_fmt(c3['mdd'])} R   under-water max "
        f"{c3['underwater'] if c3['underwater'] is not None else 'N/A'} wk   "
        f"[{pf(c3['pass'])}]"
    )
    lines.append(
        f"C4 green     {_fmt((c4['green'] or 0) * 100, 0)}%  "
        f"null {_fmt((c4['null'] or 0) * 100, 0)}%                   "
        f"[{pf(c4['pass'])}]"
    )
    lines.append(
        f"C5 fills/wk  {_fmt(c5['fills_wk'], 2)}                             "
        f"[{pf(c5['pass'])}]"
    )
    if c6 is None:
        lines.append("C6 tail      C6 not audited")
    else:
        lines.append(
            f"C6 tail      {c6['m']} of {c6['n']} ≥ 3R trades obtainable  "
            f"[{pf(c6['pass'])}]" + (f"   (failures: {c6['failures']})" if c6['failures'] else "")
        )
    lines.append(
        f"C7 power     cycles {c7['cycles']}   "
        f"bootstrap P90-gap 75% UB {_fmt(c7['bootstrap_p90_ub'], 1)} wk    "
        f"[{pf(c7['pass'])}]"
    )
    lines.append(
        f"diagnostics  ex-top-5% {_fmt(ex_top5)} R   capped {_fmt(capped)} R   "
        f"top-5 share {_fmt(top5_share, 1)}%   "
        f"weekly P&L histogram: {sorted(round(r, 1) for r in weekly_r)}"
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Cadence bar scorer (docs/cadence_bar.md) — scores a "
        "book by weeks and the cycle between strong weeks, not by trades."
    )
    p.add_argument("--trades", help="Trades CSV (date + pnl_R, or date + pnl + risk)")
    p.add_argument("--split", choices=["TRAIN", "VAL", "ALL"], default="ALL")
    p.add_argument("--include-test", action="store_true",
                    help="Allow trades on/after 2026-06-01 (sealed TEST split)")
    p.add_argument("--tail-audit", help="Optional CSV: date,symbol,obtainable for C6")
    p.add_argument("--live", action="store_true",
                    help="Read live fills from cache.db instead of --trades (TODO, not wired)")
    p.add_argument("--book", default="unknown", help="Book name, for --live and the report header")
    p.add_argument("--slots", default="N/A", help="Live slot count, for the report header")
    p.add_argument("--fill-model", default="N/A", help="Live fill model description, for the report header")
    p.add_argument("--r-dollars", default="N/A", help="Dollar value of 1R at current ramp stage")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for C4 null and C7 bootstrap")

    # §2 thresholds, all overridable.
    p.add_argument("--strong-r", type=float, default=5.0)
    p.add_argument("--gap-median", type=float, default=3.0)
    p.add_argument("--gap-p90", type=float, default=6.0)
    p.add_argument("--bleed-p90", type=float, default=-4.0)
    p.add_argument("--cycle-pos", type=float, default=0.75)
    p.add_argument("--week-p10", type=float, default=-2.0)
    p.add_argument("--week-min", type=float, default=-4.0)
    p.add_argument("--mdd", type=float, default=8.0)
    p.add_argument("--underwater", type=int, default=6)
    p.add_argument("--green", type=float, default=0.55)
    p.add_argument("--green-margin", type=float, default=0.10)
    p.add_argument("--fills-wk", type=float, default=3.0)
    p.add_argument("--min-cycles", type=int, default=10)
    return p


def run(args):
    """Run the full C1-C7 scoring pipeline and return the report string."""
    if args.live:
        trades = load_trades_live(args.book)
    else:
        if not args.trades:
            raise SystemExit("--trades CSV required (or --live --book NAME)")
        trades = load_trades_csv(args.trades)

    trades_in_split, lo, hi = filter_split(trades, args.split, args.include_test)
    weekly = build_weekly_series(trades_in_split, lo, hi)
    weekly_r = [r for _, r in weekly]
    cycles, strong_idx = compute_cycles(weekly, args.strong_r)
    rng = random.Random(args.seed)

    c1 = score_c1(cycles, args.gap_median, args.gap_p90)
    c2 = score_c2(cycles, args.bleed_p90, args.cycle_pos)
    c3 = score_c3(weekly, args.week_p10, args.week_min, args.mdd, args.underwater)
    c4 = score_c4(weekly, trades_in_split, args.green, args.green_margin, rng=rng)
    c5 = score_c5(len(trades_in_split), len(weekly), args.fills_wk)
    c6 = score_c6(trades_in_split, args.tail_audit)
    c7 = score_c7(cycles, weekly_r, args.min_cycles, args.gap_p90, args.strong_r, rng=rng)

    return render_report(
        book=args.book, split=args.split, slots=args.slots,
        fill_model=args.fill_model, r_dollars=args.r_dollars,
        c1=c1, c2=c2, c3=c3, c4=c4, c5=c5, c6=c6, c7=c7,
        weekly=weekly, trades_in_split=trades_in_split,
    )


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    print(run(args))


if __name__ == "__main__":
    main()
