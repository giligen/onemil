"""Score the S1 exit-pass variants (PREREG_S1_EXIT.md, cells 1,319-1,321) against the baseline.

For baseline (runS1_true) and each variant book (runS1_E1/E2/E3): per split n, mean R, iid and
day-clustered t, ex-top-5 %, TRAIN halves, fills/wk, weekly MDD in R; plus the trade-by-trade delta
vs baseline on (date, symbol) and the cadence-bar block. Reuses score_s1_filters helpers.

Usage: python3 research/orb_seed_wide/score_s1_exit.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_s1_filters import R_USD, OUT, split_stats, fmt, cadence  # noqa: E402

VARIANTS = {"true": "baseline", "E1": "touchgo off", "E2": "lock arm 1.0R", "E3": "scale 50% @2R"}


def load(tag: str) -> pd.DataFrame:
    """Entered rows of one walked S1 book with R and a (date, symbol) key."""
    p = OUT / f"runS1_{tag}.csv"
    if not p.exists():
        return pd.DataFrame()
    b = pd.read_csv(p)
    b = b[b.entered == 1].copy()
    b["date"] = pd.to_datetime(b["date"])
    b["R"] = b["_sized_pnl"] / R_USD
    assert (b.date < "2026-06-01").all(), f"TEST rows in {p}"
    return b


def weekly_mdd(df: pd.DataFrame) -> float:
    """Max drawdown, in R, of the cumulative weekly P&L series."""
    w = df.groupby(df.date.dt.to_period("W")).R.sum().cumsum()
    return float((w - w.cummax()).min()) if len(w) else float("nan")


def main() -> None:
    base = load("true")
    splits = {"TRAIN": lambda d: d[d.date.dt.year == 2025],
              "VAL": lambda d: d[d.date.dt.year == 2026]}
    weeks = {s: f(base).date.dt.to_period("W").nunique() for s, f in splits.items()}
    for tag, label in VARIANTS.items():
        b = load(tag)
        print(f"=== {tag} {label} ===" if len(b) else f"=== {tag} {label}: NOT WALKED YET ===")
        if not len(b):
            continue
        for s, f in splits.items():
            d = f(b)
            st = split_stats(d, weeks[s])
            print(f"{s:5s} {fmt(st)}  wkMDD={weekly_mdd(d):+.2f}R (base {weekly_mdd(f(base)):+.2f}R)")
            if s == "TRAIN":
                h1, h2 = d[d.date.dt.month <= 6], d[d.date.dt.month > 6]
                print(f"      halves H1 n={len(h1)} R={h1.R.mean():+.3f}  H2 n={len(h2)} R={h2.R.mean():+.3f}")
            if tag != "true":
                # changed = exit path differs (pnl_pct), not the compounding-sizing drift in _sized_pnl
                m = f(base)[["date", "symbol", "R", "pnl_pct", "exit_reason"]].merge(
                    d[["date", "symbol", "R", "pnl_pct", "exit_reason"]], on=["date", "symbol"],
                    how="outer", suffixes=("_b", "_v"), indicator=True)
                ch = m[(m._merge == "both") & ((m.pnl_pct_b - m.pnl_pct_v).abs() > 1e-6)]
                print(f"      delta vs base: {len(ch)} trades changed, sum {(ch.R_v - ch.R_b).sum():+.2f}R; "
                      f"only-in-base {int((m._merge == 'left_only').sum())}, only-in-variant "
                      f"{int((m._merge == 'right_only').sum())}")
                print("      exit mix:", d.exit_reason.value_counts().to_dict())
            cadence(d, f"exit_{tag}", s)
        print()


if __name__ == "__main__":
    main()
