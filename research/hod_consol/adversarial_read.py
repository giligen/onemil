"""Adversarial read of the base-under-the-high cells (main session, 2026-09-23).

The scorer's val_t is a mean-of-day-means t (days weighted equally). This prints, per cell and split:
trade-weighted mean, iid t, trade-weighted cluster-robust t (clusters = days), the day-equal-weighted
mean and t, the share of total R made on the top 10 % of days, and the SAME statistics on the tradable
SLOTTED book (first 12 per day, 4 concurrent — run_consol.simulate_slots), which is what live would trade.

Usage: python3 research/hod_consol/adversarial_read.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_consol as rc  # noqa: E402


def stats(x: pd.Series, day: pd.Series) -> dict:
    """Trade-weighted mean, iid t, cluster-robust t (clusters = days), day-weighted mean/t, top-day share."""
    n = len(x)
    if n < 3:
        return dict(n=n)
    m = x.mean()
    t_iid = m / (x.std(ddof=1) / np.sqrt(n))
    resid_sum = (x - m).groupby(day.values).sum()
    g = len(resid_sum)
    se_cl = np.sqrt((resid_sum ** 2).sum() * g / max(g - 1, 1)) / n
    dm = x.groupby(day.values).mean()
    t_day = dm.mean() / (dm.std(ddof=1) / np.sqrt(len(dm))) if len(dm) > 1 else np.nan
    day_sum = x.groupby(day.values).sum().sort_values(ascending=False)
    k = max(1, int(round(0.1 * len(day_sum))))
    top_share = day_sum.iloc[:k].sum() / x.sum() if x.sum() != 0 else np.nan
    return dict(n=n, days=g, mean=m, t_iid=t_iid, t_cluster=m / se_cl if se_cl else np.nan,
                day_mean=dm.mean(), t_day=t_day, top10pct_days_share=top_share)


def main() -> None:
    for cid in ("1400", "1401", "1402"):
        p = HERE / "trades" / f"{cid}_all.csv"
        if not p.exists():
            print(f"{cid}: missing {p}")
            continue
        w = pd.read_csv(p, keep_default_na=False, na_values=[""])
        w["net_R"] = pd.to_numeric(w["net_R"], errors="coerce")
        w = w.dropna(subset=["net_R"])
        print(f"=== {cid} {rc.CELL_NAME[cid]} ===")
        for split in ("TRAIN", "VAL"):
            s = w[w.split == split].copy()
            if s.empty:
                continue
            a = stats(s.net_R, s.day)
            keep = rc.simulate_slots(s)
            sl = s.loc[keep.index[keep]]
            b = stats(sl.net_R, sl.day)
            f = lambda d: (f"n={d.get('n')} days={d.get('days')} mean={d.get('mean', np.nan):+.3f} "  # noqa: E731
                           f"t_iid={d.get('t_iid', np.nan):+.2f} t_cluster={d.get('t_cluster', np.nan):+.2f} "
                           f"day_mean={d.get('day_mean', np.nan):+.3f} t_day={d.get('t_day', np.nan):+.2f} "
                           f"top10%days_share={d.get('top10pct_days_share', np.nan):.0%}")
            print(f"  {split:5s} ALL     {f(a)}")
            print(f"  {split:5s} SLOTTED {f(b)}")


if __name__ == "__main__":
    main()
