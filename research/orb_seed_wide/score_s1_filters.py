"""Score the three pre-registered S1 loser filters (PREREG_S1_FILTERS.md, cells 1,316-1,318).

Reads the walked S1 book, applies each filter to entered rows, prints per split: n, mean R,
iid t, day-clustered t, ex-top-5 %, TRAIN halves, fills/week, removed-cohort mean R, and the
cadence-bar block via scripts/cadence_bar.py. Writes per-cell trade CSVs next to the book.

Usage: python3 research/orb_seed_wide/score_s1_filters.py
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R_USD = 375.0
ROOT = Path(__file__).resolve().parents[2]
BOOK = ROOT / "research/orb_seed_wide/out/runS1_true.csv"
OUT = ROOT / "research/orb_seed_wide/out"

CELLS = {
    "F-A gap>=4.0": lambda d: d.gap_pct >= 4.0,
    "F-B rr<=2.0": lambda d: d.range_return_pct <= 2.0,
    "F-AB both": lambda d: (d.gap_pct >= 4.0) & (d.range_return_pct <= 2.0),
}


def clustered_t(df: pd.DataFrame) -> float:
    """Day-clustered t-stat of mean R: cluster sums over dates, t = mean / SE of cluster means."""
    daily = df.groupby("date").R.sum()
    n_days = len(daily)
    if n_days < 3:
        return float("nan")
    mean_per_trade = df.R.mean()
    # variance of the total via cluster sums, scaled to the per-trade mean
    var_total = daily.var(ddof=1) * n_days
    return mean_per_trade / (np.sqrt(var_total) / len(df))


def split_stats(df: pd.DataFrame, weeks: int) -> dict:
    """Summary statistics for one split of one cell."""
    if len(df) == 0:
        return {"n": 0}
    k = max(1, int(round(len(df) * 0.05)))
    ex_top = df.R.drop(df.R.nlargest(k).index).mean()
    t_iid = df.R.mean() / (df.R.std(ddof=1) / np.sqrt(len(df))) if len(df) > 1 else float("nan")
    return {"n": len(df), "meanR": df.R.mean(), "t_iid": t_iid, "t_clu": clustered_t(df),
            "ex_top5": ex_top, "fills_wk": len(df) / weeks, "usd": df["_sized_pnl"].sum()}


def fmt(s: dict) -> str:
    if s["n"] == 0:
        return "n=0"
    return (f"n={s['n']:3d} R={s['meanR']:+.3f} t_iid={s['t_iid']:+.2f} t_clu={s['t_clu']:+.2f} "
            f"exTop5={s['ex_top5']:+.3f} fills/wk={s['fills_wk']:.2f} ${s['usd']:,.0f}")


def cadence(df: pd.DataFrame, tag: str, split: str) -> None:
    """Write the trade CSV and print the cadence-bar block for one split."""
    p = OUT / f"s1_{tag}_{split}.csv"
    df[["date", "R", "symbol"]].rename(columns={"R": "pnl_R"}).to_csv(p, index=False)
    r = subprocess.run([sys.executable, str(ROOT / "scripts/cadence_bar.py"), "--trades", str(p),
                        "--split", split], capture_output=True, text=True, cwd=ROOT)
    print(r.stdout.strip() or r.stderr.strip()[-600:])


def main() -> None:
    book = pd.read_csv(BOOK)
    book = book[book.entered == 1].copy()
    book["date"] = pd.to_datetime(book["date"])
    book["R"] = book["_sized_pnl"] / R_USD
    tr = book[book.date.dt.year == 2025]
    va = book[(book.date.dt.year == 2026) & (book.date < "2026-06-01")]
    assert (book.date < "2026-06-01").all(), "TEST rows present in the S1 book"
    wk_tr = tr.date.dt.to_period("W").nunique() if len(tr) else 1
    wk_va = va.date.dt.to_period("W").nunique() if len(va) else 1
    print(f"baseline S1  TRAIN {fmt(split_stats(tr, wk_tr))}\n             VAL   {fmt(split_stats(va, wk_va))}")
    print(f"weeks with a fill: TRAIN {wk_tr}  VAL {wk_va}  (fills/wk uses these)\n")
    for name, rule in CELLS.items():
        tag = name.split()[0].replace("-", "").lower()
        print(f"=== {name} ===")
        for split, d, wk in (("TRAIN", tr, wk_tr), ("VAL", va, wk_va)):
            keep, drop = d[rule(d)], d[~rule(d)]
            s = split_stats(keep, wk)
            print(f"{split:5s} keep {fmt(s)}")
            print(f"      drop n={len(drop):3d} R={drop.R.mean() if len(drop) else float('nan'):+.3f}")
            if split == "TRAIN":
                h1, h2 = keep[keep.date.dt.month <= 6], keep[keep.date.dt.month > 6]
                print(f"      halves H1 n={len(h1)} R={h1.R.mean():+.3f}  H2 n={len(h2)} R={h2.R.mean():+.3f}")
            cadence(keep, tag, split)
        print()


if __name__ == "__main__":
    main()
