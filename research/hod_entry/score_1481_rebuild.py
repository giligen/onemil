"""
Independent scorer for cells 1,481 (retest-long entry) and 1,486 (1,481 x L3 top-tercile
join), per PREREG_1481.md and its Amendment 1.

This script does NOT read cell_1481.py (the builder). It scores the builder's already-
produced fill-level rebuild (research/hod_entry/rebuild_1481_fills.csv), which the caller
states is the independent rebuild's retest fills under the correct limit-price convention.
Day-clustered t, ex-top-5% mean, weeks-spanned and the first-12/day 4-concurrent slot
simulator are re-implemented here from the conventions already established in
research/hod_entry/cell_1445.py and research/hod_consol/run_consol.py (read for their
formulas only -- neither is the 1,481 builder) so this scorer has no import-time
dependency on that toolchain.

SEALED TEST: this script hard-stops if any split other than TRAIN / VAL is present in the
input file, and never computes, prints or writes a statistic on such rows.

Usage: python3 score_1481_rebuild.py
Writes: research/hod_entry/RESULT_1481_rebuild_score.md
"""
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

FILLS_CSV = "research/hod_entry/rebuild_1481_fills.csv"
L3_CSV = "research/hod_entry/model_1478_L3_predictions.csv"
OUT_MD = "research/hod_entry/RESULT_1481_rebuild_score.md"

L3_THRESHOLD = 0.3070  # frozen TRAIN-H2 top-tercile threshold, cell 1,478-L3 (HGB)
NULL_SEED = 1481
NULL_DRAWS = 1000
CONCURRENT_CAP = 4
DAILY_CAP = 12
SPREAD_RAIL_PCT = 0.5  # median R' must be >= 0.5% of price

NUMERIC_COLS = [
    "fill", "stop", "level", "fill_min", "base_net_R", "limit", "retest_minute",
    "retest_ts", "dip_low", "entry", "stop_used", "target", "Rp", "r_pct_price",
    "retest_delay_min", "exit_m", "exit_price", "raw_R", "cost_R", "net_R_prime",
]

HOLDOUT_LABEL = {"TRAIN": "TRAIN-H2", "VAL": "VAL"}


# --------------------------------------------------------------------------------------
# Shared statistical conventions (re-implemented, not imported -- see module docstring)
# --------------------------------------------------------------------------------------

def day_clustered_t(y, day):
    """statsmodels OLS on a constant, clustered by day -- the t-stat on the mean."""
    y = pd.Series(y).dropna()
    if len(y) < 2:
        return float("nan")
    d = pd.Series(day).loc[y.index]
    if d.nunique() < 2:
        return float("nan")
    X = np.ones((len(y), 1))
    model = sm.OLS(y.to_numpy(), X).fit(cov_type="cluster", cov_kwds={"groups": d.to_numpy()})
    return float(model.tvalues[0])


def ex_top5_mean(y):
    """Mean excluding the top 5% (by value) of a series -- tail-dependence check."""
    y = pd.Series(y).dropna().sort_values(ascending=False)
    n = len(y)
    if n == 0:
        return float("nan")
    k = int(round(0.05 * n))
    return float(y.iloc[k:].mean()) if k < n else float(y.mean())


def weeks_spanned(days):
    """Distinct ISO (year, week) count over a day-string series -- the fills/wk denominator."""
    iso = pd.to_datetime(pd.Series(days).unique())
    wk = {(d.isocalendar()[0], d.isocalendar()[1]) for d in iso}
    return max(len(wk), 1)


def simulate_slots(trades, concurrent_cap=CONCURRENT_CAP, daily_cap=DAILY_CAP):
    """Per day: order by entry_m, keep the first `daily_cap`, at most `concurrent_cap` open
    at once (an open slot frees once its exit_m has passed the candidate's entry_m)."""
    keep = pd.Series(False, index=trades.index)
    for _, g in trades.groupby("day"):
        g = g.sort_values("entry_m")
        open_exits, daily_count = [], 0
        for row in g.itertuples():
            open_exits = [x for x in open_exits if x > row.entry_m]
            if len(open_exits) < concurrent_cap and daily_count < daily_cap:
                keep.loc[row.Index] = True
                open_exits.append(row.exit_m)
                daily_count += 1
    return keep


def fills_per_week(retest_subset, weeks):
    """Fills/wk under the live cap, ordered by RETEST time (not the base fill time)."""
    if not len(retest_subset):
        return 0.0
    trades = retest_subset.rename(columns={"retest_minute": "entry_m"})[["day", "entry_m", "exit_m"]].copy()
    kept = simulate_slots(trades)
    return float(kept.sum()) / weeks


def stratified_null_percentile(population, retest, seed=NULL_SEED, n_draws=NULL_DRAWS):
    """1,000 draws that pick, per day, the same count of BASE fills as the cell has retest
    fills that day (sampled without replacement from that day's population base_net_R),
    mean of pooled base_net_R per draw; percentile of the cell's actual mean net R' within
    that null distribution."""
    actual_mean = retest.net_R_prime.mean()
    if len(retest) == 0 or np.isnan(actual_mean):
        return float("nan"), 0
    day_counts = retest.groupby("day").size()
    day_pools = {d: g.base_net_R.dropna().to_numpy() for d, g in population.groupby("day")}
    rng = np.random.RandomState(seed)
    draws = np.empty(n_draws)
    for i in range(n_draws):
        picked = []
        for d, k in day_counts.items():
            pool = day_pools.get(d, np.array([]))
            if len(pool) == 0:
                continue
            k_eff = min(int(k), len(pool))
            idx = rng.choice(len(pool), size=k_eff, replace=False)
            picked.append(pool[idx])
        draws[i] = np.concatenate(picked).mean() if picked else np.nan
    valid = draws[~np.isnan(draws)]
    pct = float((valid <= actual_mean).mean() * 100) if len(valid) else float("nan")
    return pct, len(valid)


# --------------------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------------------

def load_fills():
    df = pd.read_csv(FILLS_CSV, dtype={"day": str, "symbol": str, "split": str,
                                        "status": str, "why": str, "base_why": str, "wk": str})
    for c in NUMERIC_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["filled"] = df["filled"].astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "": False}
    )
    df["filled"] = df["filled"].fillna(False)

    bad_splits = sorted(set(df["split"].unique()) - {"TRAIN", "VAL"})
    if bad_splits:
        sys.stderr.write(
            f"SEALED TEST GUARD: split values outside TRAIN/VAL found: {bad_splits} -- "
            "refusing to compute or print any statistic. Aborting.\n"
        )
        sys.exit(1)
    return df


def load_l3():
    l3 = pd.read_csv(L3_CSV, dtype={"day": str, "symbol": str})
    l3["hgb_prob_L3"] = pd.to_numeric(l3["hgb_prob_L3"], errors="coerce")
    return l3[["day", "symbol", "hgb_prob_L3"]].drop_duplicates(subset=["day", "symbol"])


# --------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------

def score_cell(cell_id, population, holdout_label):
    """population: the base-fill rows this cell draws from (already split- and, for 1486,
    L3-restricted). Returns a dict of all required statistics for this cell x holdout."""
    n_base = len(population)
    retest = population[population["filled"]]
    n_retest = len(retest)
    fill_share = n_retest / n_base if n_base else float("nan")

    mean_R = float(retest.net_R_prime.mean()) if n_retest else float("nan")
    t_R = day_clustered_t(retest.net_R_prime, retest.day) if n_retest else float("nan")
    ex5_R = ex_top5_mean(retest.net_R_prime) if n_retest else float("nan")
    weeks = weeks_spanned(population.day)
    fwk = fills_per_week(retest, weeks)
    med_r_pct_price = float(retest.r_pct_price.median()) if n_retest else float("nan")
    dip_bps = (retest.level - retest.dip_low) / retest.level * 10000.0
    med_dip_bps = float(dip_bps.median()) if n_retest else float("nan")
    med_delay = float(retest.retest_delay_min.median()) if n_retest else float("nan")

    delta = retest.net_R_prime - retest.base_net_R
    delta_mean = float(delta.mean()) if n_retest else float("nan")
    delta_t = day_clustered_t(delta, retest.day) if n_retest else float("nan")
    delta_ex5 = ex_top5_mean(delta) if n_retest else float("nan")

    null_pct, null_n = stratified_null_percentile(population, retest)

    return dict(
        cell=cell_id, holdout=holdout_label, n_base=n_base, n_retest=n_retest,
        fill_share=fill_share, mean_R=mean_R, t_R=t_R, ex5_R=ex5_R, fills_wk=fwk,
        med_r_pct_price=med_r_pct_price, med_dip_bps=med_dip_bps, med_delay_min=med_delay,
        delta_mean=delta_mean, delta_t=delta_t, delta_ex5=delta_ex5,
        null_pct=null_pct, null_n=null_n,
    )


def pass_fail(val_row, train_row):
    """PREREG pass bar, VAL-gated with a TRAIN-H2 same-sign/paired-ΔR cross-check."""
    fails = []
    if not (val_row["mean_R"] >= 0.15):
        fails.append(f"VAL mean_R {val_row['mean_R']:.3f} < +0.150")
    if not (val_row["t_R"] >= 2.5):
        fails.append(f"VAL t {val_row['t_R']:.2f} < 2.5")
    if not (val_row["ex5_R"] > 0):
        fails.append(f"VAL ex-top-5% {val_row['ex5_R']:.3f} <= 0")
    if not (val_row["fills_wk"] >= 3.0):
        fails.append(f"VAL fills/wk {val_row['fills_wk']:.2f} < 3.0")
    if not (val_row["null_pct"] >= 99):
        fails.append(f"VAL null percentile {val_row['null_pct']:.1f} < 99")
    same_sign = (train_row["mean_R"] > 0) == (val_row["mean_R"] > 0) and val_row["mean_R"] > 0
    if not (same_sign and train_row["t_R"] >= 1.0):
        fails.append(f"TRAIN-H2 same-sign/t {train_row['mean_R']:.3f} (t={train_row['t_R']:.2f}) fails same-sign t>=1")
    if not (val_row["delta_mean"] >= 0.10):
        fails.append(f"VAL paired dR {val_row['delta_mean']:.3f} < +0.100")
    if not (train_row["delta_mean"] >= 0.10):
        fails.append(f"TRAIN-H2 paired dR {train_row['delta_mean']:.3f} < +0.100")
    return ("PASS" if not fails else "FAIL"), fails


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main():
    df = load_fills()
    n_rows = len(df)
    l3 = load_l3()

    merged = df.merge(l3, on=["day", "symbol"], how="left")
    join_matched = merged["hgb_prob_L3"].notna().sum()
    join_rate = join_matched / len(merged) * 100

    rows = []
    never_retest = {}
    dq_excluded = {}
    for split in ["TRAIN", "VAL"]:
        label = HOLDOUT_LABEL[split]
        pop_all = merged[merged["split"] == split]
        pop_1486 = pop_all[pop_all["hgb_prob_L3"] >= L3_THRESHOLD]

        rows.append(score_cell("1481", pop_all, label))
        rows.append(score_cell("1486", pop_1486, label))

        nr = pop_all[pop_all["status"] == "no_retest"]
        never_retest[label] = (len(nr), float(nr.base_net_R.mean()) if len(nr) else float("nan"))
        dq = pop_all[pop_all["status"] == "bar_tick_disagree"]
        dq_excluded[label] = len(dq)

    by_cell_holdout = {(r["cell"], r["holdout"]): r for r in rows}
    verdicts = {}
    for cell in ["1481", "1486"]:
        val_row = by_cell_holdout[(cell, "VAL")]
        train_row = by_cell_holdout[(cell, "TRAIN-H2")]
        verdicts[cell] = pass_fail(val_row, train_row)

    rail_flags = [r for r in rows if not np.isnan(r["med_r_pct_price"]) and r["med_r_pct_price"] < SPREAD_RAIL_PCT]

    write_report(n_rows, join_matched, join_rate, rows, never_retest, dq_excluded, verdicts, rail_flags)
    print_summary(rows, never_retest, verdicts, join_matched, len(merged), join_rate)


def fmt(x, nd=3):
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def write_report(n_rows, join_matched, join_rate, rows, never_retest, dq_excluded, verdicts, rail_flags):
    lines = []
    lines.append(
        f"Fills file: `research/hod_entry/rebuild_1481_fills.csv` -- {n_rows} rows "
        f"(splits present: TRAIN, VAL only; TEST was not read, per the sealed-test guard)."
    )
    lines.append("")
    lines.append(f"1,486 join (day+symbol) to `model_1478_L3_predictions.csv`, column `hgb_prob_L3`: "
                  f"{join_matched}/{n_rows} matched ({join_rate:.1f}%).")
    lines.append("")

    lines.append("## Per cell x holdout")
    lines.append("")
    header = ("| cell | holdout | n_base | n_retest | fill_share | mean_R | t | ex_top5_R | fills/wk | "
               "med_r_pct_price | med_dip_bps | med_delay_min | dR_mean | dR_t | dR_ex5 | null_pctile |")
    lines.append(header)
    lines.append("|" + "---|" * (header.count("|") - 1))
    for r in rows:
        lines.append(
            f"| {r['cell']} | {r['holdout']} | {r['n_base']} | {r['n_retest']} | "
            f"{r['fill_share']*100:.1f}% | {fmt(r['mean_R'])} | {fmt(r['t_R'],2)} | {fmt(r['ex5_R'])} | "
            f"{r['fills_wk']:.1f} | {fmt(r['med_r_pct_price'],1)}% | {fmt(r['med_dip_bps'],1)} | "
            f"{fmt(r['med_delay_min'],1)} | {fmt(r['delta_mean'])} | {fmt(r['delta_t'],2)} | "
            f"{fmt(r['delta_ex5'])} | {fmt(r['null_pct'],1)} |"
        )
    lines.append("")

    lines.append("## Never-retest cohort (base fills with status == `no_retest`; the runners)")
    lines.append("")
    lines.append("| holdout | n | mean base_net_R |")
    lines.append("|---|---|---|")
    for label, (n, m) in never_retest.items():
        lines.append(f"| {label} | {n} | {fmt(m)} |")
    lines.append("")
    lines.append(
        f"Excluded as indeterminate (`status == bar_tick_disagree`, not counted as retest or never-retest): "
        f"TRAIN-H2 {dq_excluded.get('TRAIN-H2', 0)}, VAL {dq_excluded.get('VAL', 0)}."
    )
    lines.append("")

    lines.append("## Pass bar verdicts (PREREG_1481.md, VAL-gated)")
    lines.append("")
    for cell, (verdict, fails) in verdicts.items():
        lines.append(f"**Cell {cell}: {verdict}**")
        if fails:
            for f in fails:
                lines.append(f"- FAIL: {f}")
        lines.append("")

    lines.append("## R-vs-spread rail (median R' must be >= 0.5% of price)")
    lines.append("")
    if rail_flags:
        for r in rail_flags:
            lines.append(f"- FLAG: cell {r['cell']} / {r['holdout']}: median R' = {fmt(r['med_r_pct_price'],1)}% of price")
    else:
        lines.append("- no cell/holdout below the 0.5% rail")
    lines.append("")

    with open(OUT_MD, "w") as f:
        f.write("\n".join(lines) + "\n")


def print_summary(rows, never_retest, verdicts, join_matched, join_total, join_rate):
    print(f"1,486 join match rate: {join_matched}/{join_total} ({join_rate:.1f}%)")
    for r in rows:
        print(f"cell {r['cell']} {r['holdout']}: n={r['n_retest']} mean_R={fmt(r['mean_R'])} "
              f"t={fmt(r['t_R'],2)} dR={fmt(r['delta_mean'])} null_pctile={fmt(r['null_pct'],1)}")
    for label, (n, m) in never_retest.items():
        print(f"never-retest {label}: n={n} mean_base_net_R={fmt(m)}")
    for cell, (verdict, fails) in verdicts.items():
        print(f"cell {cell}: {verdict} ({len(fails)} failing criteria)")


if __name__ == "__main__":
    main()
