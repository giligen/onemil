#!/usr/bin/env python3
"""
Independent rebuild of PREREG_1550 (research/overnight_high/PREREG_1550.md) FROM PROSE ONLY.
Built by a fresh agent that did not read cell_1550.py / build_panel.py / test_cell_1550.py /
RESULT_1550.md / cell_1550_nights.csv. Cross-checks against the frozen run happen elsewhere
(row-level Jaccard + net-bps diff); this script only implements the PREREG's own words.

RULE (PREREG "Rule" section, unchanged from the disclosed note, nothing re-tuned):
  Signal at the 15:59 close of day t: close_t >= highest CLOSE of the prior 252 sessions,
  volume_t >= 1.5 x ADV20, close >= $5, 20-day dollar volume >= $10M, no test tickers
  (^Z[A-Z]ZZT$, ^ZZ). Rank the day's qualifying names by volume ratio (volume_t / ADV20),
  take the top N (N=10 -> cell 1550, N=25 -> cell 1551). Buy at day-t MOC, sell at day t+1
  MOO. Return = open_{t+1} / close_t - 1. Both ADV20 and the 20-day dollar-volume gate are
  computed through t-1 (shift-then-roll) per the PREREG's own look-ahead refuter ("the volume
  ratio uses ADV through t-1"); the 252-session high window is likewise shifted so it never
  sees day t's own close.

SAMPLES:
  EXTENSION: Alpaca daily bars 2019-01-02..2024-06-28, research/overnight_high/alpaca_daily_2019_2024H1.parquet
             (already fetched upstream; PIT-listing union already applied there per the fetch task).
  PANEL:     Databento EQUS.SUMMARY daily, 2024-07-01..2026-09-04, from
             data/research/databento/equs_daily_2024H2.parquet + equs_daily_2025_2026.parquet
             (both already carry a resolved `symbol` column -- the instrument/symbol map is not
             needed to join them). Splits per PREREG: TRAIN < 2026-01-01, VAL 2026-01..05,
             TEST >= 2026-06-01 (disclosed as spent).

DOCUMENTED ASSUMPTION (data-limitation, reported not hidden): a literal "prior 252 SESSIONS"
high requires 252 trading days of history before the signal day. Both underlying files start
at their sample's first date (2019-01-02 for EXTENSION, 2024-07-01 for PANEL), so the first
~12-13 months of each sample cannot carry a fully-populated 252-session high and are excluded
(min_periods=252, no shortcut min_periods per the disclosed prior run's 60). This shrinks PANEL
TRAIN in particular -- reported explicitly below rather than loosened to preserve n.
"""
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = "/home/ec2-user/onemil"
os.chdir(ROOT)
sys.path.insert(0, ROOT)
from scripts.cadence_bar import build_weekly_series, compute_cycles, percentile  # noqa: E402

OUT = "research/overnight_high"
TEST_TICKER_RE = re.compile(r"^Z[A-Z]ZZT$|^ZZ")
HIGH_WINDOW, HIGH_MINP = 252, 252
ADV_WINDOW, ADV_MINP = 20, 20
VOL_MULT = 1.5
MIN_CLOSE = 5.0
MIN_DVOL20 = 1e7
NS = (10, 25)
COST_BPS = {"2bp": 0.0002, "5bp": 0.0005, "10bp": 0.0010}
PRIMARY_COST = "5bp"
NULL_DRAWS = 1000
NULL_SEED = 1550
CAP_RET = 0.05          # winner-cap
PRICE_SCALE_FLAG = 0.30  # +/-30% flags a night for the price-scale refuter
RISK_PER_NAME = 3000.0   # cadence-bar $ per name -> R = pnl / risk


def log(*a):
    print(*a, flush=True)


def load_extension():
    d = pd.read_parquet(f"{OUT}/alpaca_daily_2019_2024H1.parquet")
    d["bar_date"] = d["bar_date"].astype(str).str[:10]
    return d[["symbol", "bar_date", "open", "high", "low", "close", "volume"]]


def load_panel():
    cols = ["bar_date", "symbol", "open", "high", "low", "close", "volume"]
    a = pd.read_parquet("data/research/databento/equs_daily_2024H2.parquet", columns=cols)
    b = pd.read_parquet("data/research/databento/equs_daily_2025_2026.parquet", columns=cols)
    d = pd.concat([a, b], ignore_index=True)
    d["bar_date"] = d["bar_date"].astype(str).str[:10]
    return d


def build_features(d):
    """Causal per-symbol features. Every rolling stat is shift(1)'d before the roll so day t's
    own bar never leaks into its own gate (PREREG look-ahead refuter)."""
    d = d[d["symbol"].notna()].copy()
    d["symbol"] = d["symbol"].astype(str).str.strip()
    d = d[d["symbol"] != ""]
    before = d["symbol"].nunique()
    d = d[~d["symbol"].str.match(TEST_TICKER_RE)]
    log(f"  test-ticker exclusion: {before} -> {d['symbol'].nunique()} symbols")
    for k in ("open", "high", "low", "close", "volume"):
        d[k] = pd.to_numeric(d[k], errors="coerce")
    d = d.dropna(subset=["open", "high", "low", "close", "volume"])
    d = d[(d["open"] > 0) & (d["close"] > 0) & (d["volume"] >= 0)]
    d = d.sort_values(["symbol", "bar_date"]).reset_index(drop=True)
    g = d.groupby("symbol", sort=False)
    d["adv20"] = g["volume"].transform(lambda s: s.shift(1).rolling(ADV_WINDOW, min_periods=ADV_MINP).mean())
    dv = (d["close"] * d["volume"])
    d["dvol20"] = dv.groupby(d["symbol"]).transform(lambda s: s.shift(1).rolling(ADV_WINDOW, min_periods=ADV_MINP).mean())
    d["high252"] = g["close"].transform(lambda s: s.shift(1).rolling(HIGH_WINDOW, min_periods=HIGH_MINP).max())
    d["vol_ratio"] = d["volume"] / d["adv20"]
    d["open_next"] = g["open"].shift(-1)
    d["bar_date_next"] = g["bar_date"].shift(-1)
    d["ret_on_next"] = d["open_next"] / d["close"] - 1.0
    return d


def eligible_universe(d):
    """Base universe: close>=$5, 20d $vol>=$10M, has a next-session return, test tickers
    already dropped in build_features. This is the pool for the placebo and the null draw --
    it does NOT require the 252d-high/volume-shock signal."""
    e = d[(d["close"] >= MIN_CLOSE) & (d["dvol20"] >= MIN_DVOL20) & d["ret_on_next"].notna()
          & d["adv20"].notna() & (d["adv20"] > 0)].copy()
    return e


def rule_hits(e):
    """The H rule on top of the eligible universe: new 252-session close high on >=1.5x ADV20."""
    r = e[e["high252"].notna() & (e["close"] >= e["high252"]) & (e["vol_ratio"] >= VOL_MULT)].copy()
    r["score"] = r["vol_ratio"]
    return r


def top_n_per_day(r, n):
    return (r.sort_values(["bar_date", "score"], ascending=[True, False])
             .groupby("bar_date", group_keys=False).head(n))


def day_clustered_t(x, col="ret_net"):
    """Cluster by night (bar_date): one mean per day, then a t-stat across day-means."""
    day_means = x.groupby("bar_date")[col].mean()
    n_days = len(day_means)
    if n_days < 2:
        return float("nan"), n_days
    se = day_means.std(ddof=1) / np.sqrt(n_days)
    if se == 0:
        return float("nan"), n_days
    return day_means.mean() / se, n_days


def stats_block(x, label):
    """One row of the required battery for a (sample, cell) slice of top-N picks."""
    out = {"label": label, "n_nights": len(x), "n_days": x["bar_date"].nunique()}
    if len(x) == 0:
        return out
    gross = x["ret_on_next"]
    out["gross_bps"] = gross.mean() * 1e4
    for tag, c in COST_BPS.items():
        x[f"ret_net_{tag}"] = gross - c
    x["ret_net"] = x[f"ret_net_{PRIMARY_COST}"]
    out["net_bps_5bp"] = x["ret_net"].mean() * 1e4
    t, n_days = day_clustered_t(x, "ret_net")
    out["t_day_clustered"] = t
    naive_t = x["ret_net"].mean() / (x["ret_net"].std(ddof=1) / np.sqrt(len(x)))
    out["t_naive"] = naive_t
    lo5, hi95 = x["ret_net"].quantile(0.05), x["ret_net"].quantile(0.95)
    ex5 = x[(x["ret_net"] >= lo5) & (x["ret_net"] <= hi95)]
    lo1, hi99 = x["ret_net"].quantile(0.01), x["ret_net"].quantile(0.99)
    ex1 = x[(x["ret_net"] >= lo1) & (x["ret_net"] <= hi99)]
    out["ex_top5pct_bps"] = ex5["ret_net"].mean() * 1e4 if len(ex5) else float("nan")
    out["ex_top1pct_bps"] = ex1["ret_net"].mean() * 1e4 if len(ex1) else float("nan")
    out["winner_capped_bps"] = (x["ret_on_next"].clip(upper=CAP_RET) - COST_BPS[PRIMARY_COST]).mean() * 1e4
    out["flagged_gt30pct"] = int((x["ret_on_next"].abs() > PRICE_SCALE_FLAG).sum())
    out["flagged_gt30pct_share"] = out["flagged_gt30pct"] / len(x)
    return out


def placebo_and_null(e, r_topn, rng_seed):
    """Universe placebo (mean ret_on_next of the WHOLE eligible pool on the nights the rule
    traded) and the count-matched null (N random eligible names per night, 1000 draws)."""
    trade_dates = r_topn["bar_date"].unique()
    n_per_day = r_topn.groupby("bar_date").size()
    e_on_dates = e[e["bar_date"].isin(trade_dates)]
    universe_mean = e_on_dates.groupby("bar_date")["ret_on_next"].mean()
    rule_mean = r_topn.groupby("bar_date")["ret_net"].mean() if "ret_net" in r_topn else r_topn.groupby("bar_date")["ret_on_next"].mean() - COST_BPS[PRIMARY_COST]
    margin = (rule_mean - universe_mean).dropna()
    margin_bps = margin.mean() * 1e4
    if len(margin) > 1 and margin.std(ddof=1) > 0:
        margin_t = margin.mean() / (margin.std(ddof=1) / np.sqrt(len(margin)))
    else:
        margin_t = float("nan")

    rng = np.random.default_rng(rng_seed)
    by_day = {d: g["ret_on_next"].to_numpy() for d, g in e_on_dates.groupby("bar_date")}
    actual_net_bps = r_topn["ret_net"].mean() * 1e4 if "ret_net" in r_topn else float("nan")
    draw_means = np.empty(NULL_DRAWS)
    for k in range(NULL_DRAWS):
        vals = []
        for d in trade_dates:
            pool = by_day.get(d)
            n = int(n_per_day.get(d, 0))
            if pool is None or n == 0:
                continue
            n = min(n, len(pool))
            idx = rng.choice(len(pool), size=n, replace=False)
            vals.append(pool[idx].mean())
        draw_means[k] = (np.mean(vals) - COST_BPS[PRIMARY_COST]) * 1e4 if vals else np.nan
    pct = float((draw_means < actual_net_bps).mean() * 100) if not np.isnan(actual_net_bps) else float("nan")
    return {
        "universe_placebo_bps": universe_mean.mean() * 1e4,
        "placebo_margin_bps": margin_bps,
        "placebo_margin_t": margin_t,
        "null_mean_bps": float(np.nanmean(draw_means)),
        "null_pctile_of_actual": pct,
    }


def weekly_cadence(x):
    """Green-week share, weekly P10, strong-week (>=+5R) gap median/P90. R = pnl / $3000
    per name (risk == notional, PREREG's '$3K per name'), so R == ret_net directly."""
    if len(x) == 0:
        return {}
    trades = [{"date": pd.to_datetime(row.bar_date).date(), "r": row.ret_net}
              for row in x.itertuples()]
    lo = min(t["date"] for t in trades)
    hi = max(t["date"] for t in trades)
    weekly = build_weekly_series(trades, lo, hi)
    r_series = [r for _, r in weekly]
    green = sum(1 for r in r_series if r >= 0.5)
    red = sum(1 for r in r_series if r <= -0.5)
    flat = len(r_series) - green - red
    p10 = percentile(r_series, 10)
    cycles, strong_idx = compute_cycles(weekly, 5.0)
    gaps = [c["gap"] for c in cycles]
    return {
        "n_weeks": len(weekly),
        "green_frac": green / len(weekly) if weekly else float("nan"),
        "red": red, "flat": flat,
        "weekly_p10_R": p10,
        "n_strong_weeks": len(strong_idx),
        "gap_median_wk": percentile(gaps, 50) if gaps else float("nan"),
        "gap_p90_wk": percentile(gaps, 90) if gaps else float("nan"),
        "n_cycles": len(cycles),
    }


def run_sample(d, sample_name, split_fn):
    log(f"\n=== {sample_name}: {len(d):,} raw rows, {d['symbol'].nunique():,} symbols, "
        f"{d['bar_date'].min()}..{d['bar_date'].max()} ===")
    feats = build_features(d)
    feats["split"] = split_fn(feats["bar_date"])
    e = eligible_universe(feats)
    r = rule_hits(e)
    log(f"  eligible universe rows: {len(e):,}  rule-hit rows: {len(r):,} "
        f"over {r['bar_date'].nunique()} distinct nights")

    all_rows = []
    nights_rows = []
    for n in NS:
        cell = 1550 if n == 10 else 1551
        top = top_n_per_day(r, n)
        top["ret_net"] = top["ret_on_next"] - COST_BPS[PRIMARY_COST]
        for split in ("TRAIN", "VAL", "TEST"):
            x = top[top["split"] == split]
            if len(x) == 0:
                continue
            block = stats_block(x.copy(), f"{sample_name}/N{n}/{split}")
            block.update({"sample": sample_name, "N": n, "cell": cell, "split": split, "year": "ALL"})
            if len(x) >= 10:
                block.update(placebo_and_null(e[e["split"] == split], x.copy(), NULL_SEED + n))
                block.update(weekly_cadence(x))
            all_rows.append(block)
            for row in x.itertuples():
                nights_rows.append({
                    "sample": sample_name, "cell": cell, "split": split,
                    "date": row.bar_date, "symbol": row.symbol,
                    "ret_on_next": row.ret_on_next, "ret_net_5bp": row.ret_net,
                    "vol_ratio": row.vol_ratio,
                })
            # per-calendar-year slice (core metrics only, per the step-budget note in the report)
            x_yr = x.copy()
            x_yr["year"] = pd.to_datetime(x_yr["bar_date"]).dt.year
            for yr, xy in x_yr.groupby("year"):
                if len(xy) == 0:
                    continue
                yb = stats_block(xy.copy(), f"{sample_name}/N{n}/{split}/{yr}")
                yb.update({"sample": sample_name, "N": n, "cell": cell, "split": split, "year": str(yr)})
                all_rows.append(yb)
    return all_rows, nights_rows, feats


def split_extension(bar_date):
    # EXTENSION has no TRAIN/VAL/TEST distinction in the PREREG -- it is read whole as the
    # decisive out-of-sample block. Tag it all TRAIN so the stats machinery above is reused;
    # the report below calls it EXTENSION explicitly, never TRAIN/VAL/TEST.
    return pd.Series("TRAIN", index=bar_date.index)


def split_panel(bar_date):
    dt = pd.to_datetime(bar_date)
    return np.where(dt < pd.Timestamp("2026-01-01"), "TRAIN",
                    np.where(dt < pd.Timestamp("2026-06-01"), "VAL", "TEST"))


def survivorship_counts(feats, sample_name):
    y = pd.to_datetime(feats["bar_date"]).dt.year
    counts = feats.groupby(y)["symbol"].nunique()
    log(f"  [{sample_name}] distinct symbols per calendar year:")
    for yr, c in counts.items():
        log(f"    {yr}: {c}")
    return counts


def main():
    log("Loading EXTENSION (Alpaca 2019-01-02..2024-06-28)...")
    ext_raw = load_extension()
    log("Loading PANEL (Databento 2024-07-01..2026-09-04)...")
    pan_raw = load_panel()

    ext_rows, ext_nights, ext_feats = run_sample(ext_raw, "EXTENSION", split_extension)
    ext_surv = survivorship_counts(ext_feats, "EXTENSION")

    pan_rows, pan_nights, pan_feats = run_sample(pan_raw, "PANEL", split_panel)

    all_rows = ext_rows + pan_rows
    nights = ext_nights + pan_nights
    stats_df = pd.DataFrame(all_rows)
    nights_df = pd.DataFrame(nights)
    stats_df.to_csv(f"{OUT}/rebuild_1550_stats.csv", index=False)
    nights_df.to_csv(f"{OUT}/rebuild_1550_nights.csv", index=False)
    log(f"\nWrote {OUT}/rebuild_1550_stats.csv ({len(stats_df)} rows), "
        f"{OUT}/rebuild_1550_nights.csv ({len(nights_df)} rows)")

    # Refuter: 15:49/15:45 executable variant cannot be tested from daily bars (no intraday
    # snapshot data in scope for this rebuild) -- flagged, not silently skipped.
    log("\nNOTE: the 15:49-order/15:45-signal executable variant requires intraday snapshots; "
        "this rebuild only has daily OHLCV and cannot test it -- reported as UNTESTED, not PASS.")

    pd.set_option("display.width", 220)
    core_cols = ["sample", "N", "split", "year", "n_nights", "n_days", "gross_bps",
                 "net_bps_5bp", "t_day_clustered", "t_naive", "ex_top5pct_bps",
                 "flagged_gt30pct"]
    log("\n" + stats_df[stats_df["year"] == "ALL"][core_cols].to_string(index=False))


if __name__ == "__main__":
    main()
