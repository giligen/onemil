"""Score cell 1,327 (PREREG_S1_CREATIVE.md): combined seed walk vs production alone (runB_true).

Per split: total $, mean R, fills/wk, weekly MDD (R), share of production's filled (date, symbol) preserved
in the combined book, the added trades' own mean R, and the cadence-bar block.

Usage: python3 research/orb_seed_wide/score_comb.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_s1_filters import R_USD, OUT, split_stats, fmt, cadence  # noqa: E402
from score_s1_exit import weekly_mdd  # noqa: E402


def load(name: str) -> pd.DataFrame:
    b = pd.read_csv(OUT / f"{name}.csv")
    b = b[b.entered == 1].copy()
    b["date"] = pd.to_datetime(b["date"])
    b["R"] = b["_sized_pnl"] / R_USD
    assert (b.date < "2026-06-01").all(), name
    return b


def main() -> None:
    base, comb = load("runB_true"), load("runCOMB_true")
    for split, sel in (("TRAIN", lambda d: d[d.date.dt.year == 2025]), ("VAL", lambda d: d[d.date.dt.year == 2026])):
        b, c = sel(base), sel(comb)
        wk = b.date.dt.to_period("W").nunique()
        kb, kc = set(zip(b.date, b.symbol)), set(zip(c.date, c.symbol))
        preserved = len(kb & kc) / len(kb) if kb else float("nan")
        added = c[[k not in kb for k in zip(c.date, c.symbol)]]
        mdd_b, mdd_c = weekly_mdd(b), weekly_mdd(c)
        print(f"=== {split} ===")
        print(f"production {fmt(split_stats(b, wk))}  wkMDD={mdd_b:+.2f}R")
        print(f"combined   {fmt(split_stats(c, wk))}  wkMDD={mdd_c:+.2f}R")
        print(f"production picks preserved {preserved:.0%}; added trades n={len(added)} R={added.R.mean() if len(added) else float('nan'):+.3f} "
              f"${added['_sized_pnl'].sum():+,.0f}")
        ok = (c["_sized_pnl"].sum() >= b["_sized_pnl"].sum() and mdd_c >= 1.25 * mdd_b and preserved >= 0.9
              and len(c) / wk >= 3)
        print(f"cell 1,327 {split}: {'PASS' if ok else 'FAIL'} ($ up={c['_sized_pnl'].sum() >= b['_sized_pnl'].sum()}, "
              f"MDD ok={mdd_c >= 1.25 * mdd_b}, preserved ok={preserved >= 0.9}, fills/wk={len(c) / wk:.2f})")
        cadence(c, "comb", split)
        print()


if __name__ == "__main__":
    main()
