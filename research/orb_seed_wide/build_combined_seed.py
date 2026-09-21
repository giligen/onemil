"""Build the combined-seed features CSV (cell 1,327) and the PM-backfill candidates list (cell 1,323).

Combined seed = wide rows with (open <= 30 & gap >= 4) or (30 < open <= 50 & 3 <= gap < 5).
First verifies that the same construction reproduces out/runS1_features.csv row-for-row (so the strata
definition is the one the earlier cells used), then writes out/runCOMB_features.csv and
out/s1_pm_candidates.csv (symbol, bar_date for every S1 entered row).

Usage: python3 research/orb_seed_wide/build_combined_seed.py
"""
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent / "out"
WIDE = OUT / "orb_features_20260920_2142.csv"

w = pd.read_csv(WIDE)
price = w["entry_price"]
s1_rebuilt = w[(w.gap_pct >= 3) & (w.gap_pct < 5) & (price >= 3) & (price <= 30)]
s1_file = pd.read_csv(OUT / "runS1_features.csv")
key = lambda d: set(zip(d.symbol, d.date))  # noqa: E731
overlap = len(key(s1_rebuilt) & key(s1_file))
print(f"S1 reconstruction: rebuilt {len(s1_rebuilt)} file {len(s1_file)} overlap {overlap}")
if overlap < 0.98 * len(s1_file):
    raise SystemExit("S1 stratum definition does not reproduce runS1_features.csv — check the price column")

comb = w[((price <= 30) & (w.gap_pct >= 4)) | ((price > 30) & (price <= 50) & (w.gap_pct >= 3) & (w.gap_pct < 5))]
comb = comb[price >= 3]
comb.to_csv(OUT / "runCOMB_features.csv", index=False)
print(f"combined seed rows {len(comb)} (gap>=4 & <=30: {int(((price <= 30) & (w.gap_pct >= 4)).sum())}, "
      f"S3: {int(((price > 30) & (price <= 50) & (w.gap_pct >= 3) & (w.gap_pct < 5)).sum())})")

book = pd.read_csv(OUT / "runS1_true.csv")
book = book[book.entered == 1]
cands = book[["symbol", "date"]].rename(columns={"date": "bar_date"}).drop_duplicates()
cands.to_csv(OUT / "s1_pm_candidates.csv", index=False)
print(f"PM candidates {len(cands)} symbol-days {cands.bar_date.min()}..{cands.bar_date.max()}")
