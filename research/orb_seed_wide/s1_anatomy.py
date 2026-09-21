"""S1 (gap 3-5 %, open $3-30) loser anatomy on TRAIN 2025 ONLY.

Exploration stage of the bottom-up-losers method (H/METHOD.md): find which
features separate losers from winners on the walked S1 book, on TRAIN only.
VAL is never touched here; the filters chosen from this print go into a
PREREG and are then scored on VAL by score_s1_filters.py.

Usage: python3 research/orb_seed_wide/s1_anatomy.py [runS1_true.csv]
"""
import sys
import numpy as np
import pandas as pd

R_USD = 375.0
path = sys.argv[1] if len(sys.argv) > 1 else "research/orb_seed_wide/out/runS1_true.csv"
df = pd.read_csv(path)
df = df[df.entered == 1].copy()
df["date"] = pd.to_datetime(df["date"])
df["R"] = df["_sized_pnl"] / R_USD
tr = df[df.date.dt.year == 2025].copy()
print(f"S1 entered rows: all={len(df)} TRAIN={len(tr)} VAL(untouched)={len(df)-len(tr)}")
print(f"TRAIN mean R={tr.R.mean():+.3f}  median={tr.R.median():+.3f}  win%={100*(tr.R>0).mean():.0f}")

print("\n== exit_reason (TRAIN) ==")
g = tr.groupby("exit_reason").R.agg(["count", "mean", "sum"])
print(g.round(3).to_string())

print("\n== R distribution (TRAIN) ==")
print(tr.R.quantile([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]).round(2).to_string())
print(f"top-5% share of gross wins: {100*tr.R[tr.R>0].nlargest(max(1,len(tr)//20)).sum()/tr.R[tr.R>0].sum():.0f}%")

print("\n== TRAIN halves ==")
h1 = tr[tr.date.dt.month <= 6]; h2 = tr[tr.date.dt.month > 6]
print(f"H1 n={len(h1)} R={h1.R.mean():+.3f}   H2 n={len(h2)} R={h2.R.mean():+.3f}")

feats = [c for c in df.columns if c not in
         {"symbol", "date", "entry_price", "pnl", "pnl_pct", "exit_reason", "win", "entered",
          "_rp_position", "_rp_pnl", "_sized_pnl", "month", "_anchor", "R", "_q_rank"}]
print("\n== per-feature terciles on TRAIN: mean R low/mid/high, spread, |t| of high-vs-low ==")
rows = []
for f in feats:
    s = pd.to_numeric(tr[f], errors="coerce")
    if s.nunique() < 3:
        # binary / categorical: group by value
        for v, grp in tr.groupby(s):
            rows.append((f"{f}={v}", len(grp), grp.R.mean(), np.nan, np.nan, np.nan))
        continue
    try:
        q = pd.qcut(s, 3, labels=False, duplicates="drop")
    except ValueError:
        continue
    m = tr.groupby(q).R.agg(["mean", "count"])
    if len(m) < 3:
        continue
    lo, hi = tr[q == 0].R, tr[q == 2].R
    t = (hi.mean() - lo.mean()) / np.sqrt(hi.var() / len(hi) + lo.var() / len(lo))
    rows.append((f, len(tr), m["mean"].iloc[0], m["mean"].iloc[1], m["mean"].iloc[2], t))
out = pd.DataFrame(rows, columns=["feature", "n", "lo", "mid", "hi", "t_hi_lo"])
out["spread"] = out.hi - out.lo
print(out.sort_values("t_hi_lo", key=lambda s: s.abs(), ascending=False).round(3).to_string(index=False))

print("\n== worst 10 TRAIN trades ==")
print(tr.nsmallest(10, "R")[["date", "symbol", "entry_price", "gap_pct", "range_size_pct",
                            "range_total_volume", "exit_reason", "R"]].round(3).to_string(index=False))
print("\n== best 10 TRAIN trades ==")
print(tr.nlargest(10, "R")[["date", "symbol", "entry_price", "gap_pct", "range_size_pct",
                           "range_total_volume", "exit_reason", "R"]].round(3).to_string(index=False))
