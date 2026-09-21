"""Week-by-week past-quarter (TEST, owner-unsealed 9/21) table: production pool vs union of pools.

Reads out/run{prodQ,gap4Q,p30Q}_true.csv (each pool walked alone on the June-Sep 2026 wide features),
builds the union (production first, add-ons up to 8 slots/day), prints weekly fills and $ at the $375 stage
and at 4x, plus totals and the add-on cohort. Usage: python3 research/orb_seed_wide/quarter_weekly.py
"""
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent / "out"
R_USD, SLOTS = 375.0, 8


def load(tag: str) -> pd.DataFrame:
    p = OUT / f"run{tag}_true.csv"
    if not p.exists():
        raise SystemExit(f"missing {p} — walk not finished")
    b = pd.read_csv(p, keep_default_na=False, na_values=[""])
    b["date"] = pd.to_datetime(b["date"])
    b["pool"] = tag
    return b


def main() -> None:
    prod = load("prodQ")
    adds = pd.concat([load("gap4Q"), load("p30Q")]).sort_values(["date", "_composite"], ascending=[True, False])
    used = prod.groupby("date").size().to_dict()
    keep = [g.head(max(0, SLOTS - used.get(d, 0))) for d, g in adds.groupby("date")]
    union = pd.concat([prod] + keep, ignore_index=True)
    for name, b in (("PRODUCTION", prod), ("UNION", union)):
        e = b[b.entered == 1]
        w = e.groupby(e.date.dt.to_period("W")).agg(fills=("symbol", "size"), usd=("_sized_pnl", "sum"))
        w["x4"] = w.usd * 4
        print(f"=== {name}: fills {len(e)}, total ${e._sized_pnl.sum():+,.0f} at $375, ${4 * e._sized_pnl.sum():+,.0f} at 4x, "
              f"mean R {e._sized_pnl.mean() / R_USD:+.3f}, green weeks {(w.usd > 0).sum()}/{len(w)}, worst week ${w.usd.min():+,.0f}")
        print(w.round(0).to_string())
    a = union[(union.pool != "prodQ") & (union.entered == 1)]
    print(f"add-on cohort: n={len(a)} mean R {a._sized_pnl.mean() / R_USD:+.3f} ${a._sized_pnl.sum():+,.0f}; "
          f"by pool {a.groupby('pool')._sized_pnl.agg(['size', 'sum']).round(0).to_dict()}")


if __name__ == "__main__":
    main()
