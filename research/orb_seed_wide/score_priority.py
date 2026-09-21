"""Cell 1,328 PRIORITY SLOTS (PREREG_S1_CREATIVE.md addendum): production first, add-on fills leftover slots.

Book = all runB_true picks (production alone, 8 slots) + per day the add-on picks from runCOMB_true
(rows not in runB_true) in composite order, up to 8 - (production picks that day). Each trade keeps its own
walked _sized_pnl. Prints production vs priority book per split, MDD, added cohort, cadence block.

Usage: python3 research/orb_seed_wide/score_priority.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_s1_filters import R_USD, OUT, split_stats, fmt, cadence  # noqa: E402
from score_s1_exit import weekly_mdd  # noqa: E402

SLOTS = 8


def load(name: str) -> pd.DataFrame:
    b = pd.read_csv(OUT / f"{name}.csv", keep_default_na=False, na_values=[""])
    b["date"] = pd.to_datetime(b["date"])
    b["R"] = b["_sized_pnl"] / R_USD
    assert (b.date < "2026-06-01").all(), name
    return b


def build_priority(base: pd.DataFrame, comb: pd.DataFrame) -> pd.DataFrame:
    """Production picks + leftover-slot add-ons per day (composite descending)."""
    kb = set(zip(base.date, base.symbol))
    addons = comb[[k not in kb for k in zip(comb.date, comb.symbol)]].copy()
    addons = addons.sort_values(["date", "_composite"], ascending=[True, False])
    used = base.groupby("date").size().to_dict()
    keep = []
    for d, grp in addons.groupby("date"):
        room = SLOTS - used.get(d, 0)
        if room > 0:
            keep.append(grp.head(room))
    add = pd.concat(keep) if keep else addons.iloc[0:0]
    add["_addon"] = True
    base = base.copy()
    base["_addon"] = False
    return pd.concat([base, add], ignore_index=True)


def main() -> None:
    base, comb = load("runB_true"), load("runCOMB_true")
    prio = build_priority(base, comb)
    print(f"picks: production {len(base)}, add-on candidates {len(comb) - len(set(zip(comb.date, comb.symbol)) & set(zip(base.date, base.symbol)))}, "
          f"add-ons admitted {int(prio._addon.sum())}")
    for split, sel in (("TRAIN", lambda d: d[d.date.dt.year == 2025]), ("VAL", lambda d: d[d.date.dt.year == 2026])):
        b = sel(base[base.entered == 1]); p = sel(prio[prio.entered == 1]); a = p[p._addon]
        wk = b.date.dt.to_period("W").nunique()
        mdd_b, mdd_p = weekly_mdd(b), weekly_mdd(p)
        print(f"=== {split} ===")
        print(f"production {fmt(split_stats(b, wk))}  wkMDD={mdd_b:+.2f}R")
        print(f"priority   {fmt(split_stats(p, wk))}  wkMDD={mdd_p:+.2f}R")
        print(f"added (entered) n={len(a)} R={a.R.mean() if len(a) else float('nan'):+.3f} ${a['_sized_pnl'].sum():+,.0f}  "
              f"ex-top-5% of added={a.R.drop(a.R.nlargest(max(1, len(a) // 20)).index).mean() if len(a) else float('nan'):+.3f}")
        ok = (p["_sized_pnl"].sum() >= b["_sized_pnl"].sum() and mdd_p >= 1.25 * mdd_b and len(p) / wk >= 3
              and len(a) > 0 and a.R.mean() > 0)
        print(f"cell 1,328 {split}: {'PASS' if ok else 'FAIL'} ($ up={p['_sized_pnl'].sum() >= b['_sized_pnl'].sum()}, "
              f"MDD ok={mdd_p >= 1.25 * mdd_b}, fills/wk={len(p) / wk:.2f}, added R>0={len(a) > 0 and a.R.mean() > 0})")
        cadence(p, "prio", split)
        print()
    prio.to_csv(OUT / "runPRIO_book.csv", index=False)


if __name__ == "__main__":
    main()
