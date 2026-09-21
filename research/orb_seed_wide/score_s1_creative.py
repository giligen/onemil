"""Score the no-walk cells of PREREG_S1_CREATIVE.md (1,322-1,326) + diagnostics on the walked S1 book.

GAP-RANK (top-10 by gap among the wide universe that day), PM-$VOL (TRAIN-median split; VOID unless
coverage >= 80 % and winner/loser missingness gap <= 5 pp), RVOL-5m (TRAIN-median split), HMM-CALM,
SPY-GAP-UP. Diagnostics: float (current), price buckets, rank top-5/top-20. Same statistics as
score_s1_filters.py plus the exploration-tier line.

Usage: python3 research/orb_seed_wide/score_s1_creative.py
"""
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_s1_filters import R_USD, OUT, split_stats, fmt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
WIDE = OUT / "orb_features_20260920_2142.csv"


def load_book() -> pd.DataFrame:
    """Entered S1 rows with R, split, day-rank by gap, HMM state, RVOL, PM $vol, float."""
    b = pd.read_csv(OUT / "runS1_true.csv")
    b = b[b.entered == 1].copy()
    b["date"] = pd.to_datetime(b["date"])
    b["R"] = b["_sized_pnl"] / R_USD
    b["split"] = np.where(b.date.dt.year == 2025, "TRAIN", "VAL")
    assert (b.date < "2026-06-01").all()
    # day rank by gap among the wide universe (gap>=3, $3-50, pv>=500K), rank 1 = largest gap
    w = pd.read_csv(WIDE, usecols=["symbol", "date", "gap_pct"])
    w["date"] = pd.to_datetime(w["date"])
    w["gap_rank"] = w.groupby("date").gap_pct.rank(ascending=False, method="min")
    b = b.merge(w[["symbol", "date", "gap_rank"]], on=["symbol", "date"], how="left")
    # HMM: calm = state with the lowest mean vol20
    h = pd.read_csv(ROOT / "research/regime/hmm_labels.csv")
    h["date"] = pd.to_datetime(h["bar_date"])
    calm = int(h.groupby("hmm_state").vol20.mean().idxmin())
    b = b.merge(h[["date", "hmm_state"]], on="date", how="left")
    b["calm"] = b.hmm_state == calm
    print(f"HMM calm state = {calm}; label coverage on book {b.hmm_state.notna().mean():.0%}")
    b["rvol5"] = b.range_total_volume / b.avg_daily_volume_20d
    b["pm_dollar_vol"] = pm_dollar_volume(b)
    con = sqlite3.connect(ROOT / "data/cache.db")
    fl = pd.read_sql("select symbol, float_shares from universe", con)
    con.close()
    b = b.merge(fl, on="symbol", how="left")
    return b


def pm_dollar_volume(b: pd.DataFrame) -> pd.Series:
    """Sum of close*volume over 04:00-09:29 ET bars per (symbol, date) from intraday_bars_1min; NaN if none."""
    con = sqlite3.connect(ROOT / "data/cache.db")
    keys = b[["symbol", "date"]].drop_duplicates()
    keys["bar_date"] = keys.date.dt.strftime("%Y-%m-%d")
    keys[["symbol", "bar_date"]].to_sql("_keys", con, if_exists="replace", index=False)
    q = """select k.symbol, k.bar_date, sum(i.close * i.volume) as pmdv
           from _keys k join intraday_bars_1min i on i.symbol = k.symbol and i.bar_date = k.bar_date
           where time(i.timestamp) < '13:30:00' and time(i.timestamp) >= '08:00:00'
           group by k.symbol, k.bar_date"""
    try:
        pm = pd.read_sql(q, con)
    finally:
        con.execute("drop table if exists _keys")
        con.close()
    pm["date"] = pd.to_datetime(pm.bar_date)
    m = b[["symbol", "date"]].merge(pm[["symbol", "date", "pmdv"]], on=["symbol", "date"], how="left")
    return m.pmdv.values


def report_cell(name: str, b: pd.DataFrame, keep_mask: pd.Series) -> None:
    """Print TRAIN/VAL kept vs dropped stats, halves, and the pass / exploration-tier line."""
    print(f"=== {name} ===")
    res = {}
    for split in ("TRAIN", "VAL"):
        d = b[b.split == split]
        wk = d.date.dt.to_period("W").nunique()
        k, dr = d[keep_mask[d.index]], d[~keep_mask[d.index]]
        s = split_stats(k, wk)
        res[split] = (s, dr.R.mean() if len(dr) else float("nan"))
        print(f"{split:5s} keep {fmt(s)}   drop n={len(dr)} R={res[split][1]:+.3f}")
        if split == "TRAIN":
            h1, h2 = k[k.date.dt.month <= 6], k[k.date.dt.month > 6]
            res["halves"] = (h1.R.mean() if len(h1) else np.nan, h2.R.mean() if len(h2) else np.nan)
            print(f"      halves H1 n={len(h1)} R={res['halves'][0]:+.3f}  H2 n={len(h2)} R={res['halves'][1]:+.3f}")
    tr, va = res["TRAIN"][0], res["VAL"][0]
    if tr["n"] and va["n"]:
        passed = (tr["meanR"] >= 0.15 and va["meanR"] >= 0.15 and va["t_clu"] >= 2 and min(res["halves"]) >= 0
                  and res["TRAIN"][1] < 0 and res["VAL"][1] < 0 and va["fills_wk"] >= 3)
        explore = (tr["meanR"] > 0 and va["meanR"] > 0 and min(res["halves"]) > 0 and va["fills_wk"] >= 3
                   and res["VAL"][1] <= 0)
        print(f"      PASS={passed}  EXPLORATION-TIER={explore}")
    print()


def main() -> None:
    b = load_book()
    tr = b[b.split == "TRAIN"]
    report_cell("1,322 GAP-RANK top-10", b, b.gap_rank <= 10)
    cov = b.pm_dollar_vol.notna()
    if cov.mean() >= 0.8:
        gap_pp = abs(cov[b.R > 0].mean() - cov[b.R <= 0].mean()) * 100
        print(f"PM coverage {cov.mean():.0%}, winner/loser missingness gap {gap_pp:.1f} pp")
        if gap_pp <= 5:
            med = tr.pm_dollar_vol.median()
            report_cell(f"1,323 PM-$VOL >= TRAIN median ${med:,.0f}", b, b.pm_dollar_vol >= med)
        else:
            print("1,323 PM-$VOL: VOID (missingness gap > 5 pp)\n")
    else:
        print(f"1,323 PM-$VOL: VOID (coverage {cov.mean():.0%} < 80 %) — run the backfill first\n")
    med = tr.rvol5.median()
    report_cell(f"1,324 RVOL-5m >= TRAIN median {med:.4f}", b, b.rvol5 >= med)
    report_cell("1,325 HMM-CALM", b, b.calm.fillna(False))
    report_cell("1,326 SPY-GAP-UP", b, b.spy_gap_pct >= 0)
    print("== diagnostics (report-only) ==")
    for name, m in (("rank top-5", b.gap_rank <= 5), ("rank top-20", b.gap_rank <= 20),
                    ("price $3-10", b.entry_price < 10), ("price $10-30", b.entry_price >= 10),
                    ("float < 20M (current)", b.float_shares < 2e7), ("float >= 20M (current)", b.float_shares >= 2e7)):
        for split in ("TRAIN", "VAL"):
            d = b[(b.split == split) & m.fillna(False)]
            print(f"  {name:24s} {split:5s} n={len(d):3d} R={d.R.mean() if len(d) else float('nan'):+.3f}")
    print(f"  float coverage {b.float_shares.notna().mean():.0%}; gap_rank coverage {b.gap_rank.notna().mean():.0%}")


if __name__ == "__main__":
    main()
