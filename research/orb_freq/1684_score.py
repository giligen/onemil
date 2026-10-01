"""Cell 1,684 scoring: per-pool per-window reads (n, fills/wk, mean R, iid t, day-clustered t,
ex-top-5%, MDE, weekly P10, worst week), the union with production (independent_1328.py's method:
pool rows not already in production that day, admitted up to 8-prod_count_that_day slots by
_composite desc), and the PREREG's pass bar. Writes 1684_pool_books.csv (every scored trade row,
tagged) and prints the material for RESULT_1684.md.

MDE convention (research/meta_label/mde.py): 2.802 * std(R, ddof=1) / sqrt(n) -- 80% power,
two-sided 5%. Day-clustered t: one-sample t on the per-day MEAN R (one observation per day), not
per-trade -- the standard clustered-SE simplification when the number of trading days, not the
trade count, is the independent unit.
"""
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/ec2-user/onemil')
sys.path.insert(0, str(ROOT / 'scripts'))
import cadence_bar as cb  # noqa: E402 -- scripts/cadence_bar.py's own SPLIT_RANGES/TEST_START are
# hardcoded to the production calendar (TRAIN 2025, VAL Jan-May 2026, TEST sealed >=Jun 2026) and
# its CLI --split/--include-test CANNOT express this cell's windows (in-regime runs through
# 2026-09-18, out-of-regime is entirely 2024H2) -- both fall outside every CLI split's fixed
# (lo,hi). We therefore call its own C1/C4 functions directly on OUR (lo,hi), same formulas/
# thresholds, bypassing only the hardcoded-range gate (verified: calling the CLI on a 2024H2 book
# zeroed every week because hi was clamped to 2026-05-31 and lo to 2025-01-01).
OUT = ROOT / 'research/orb_freq'
R_USD = 375.0
Z80 = 2.802
MAX_SLOTS = 8


def load_book(path, entered_only=True):
    if not Path(path).exists():
        return None
    df = pd.read_csv(path, keep_default_na=False, na_values=[''])
    if entered_only:
        df = df[df['entered'].astype(str).isin(['1', 'True', 'true'])].copy()
    df['date'] = pd.to_datetime(df['date'])
    df['R'] = df['_sized_pnl'].astype(float) / R_USD
    return df.sort_values('date').reset_index(drop=True)


def stats(df, label):
    if df is None or len(df) == 0:
        return dict(label=label, n=0)
    r = df['R'].to_numpy(float)
    n = len(r)
    weeks = df['date'].dt.to_period('W')
    n_weeks = weeks.nunique()
    mean_r = r.mean()
    sd = r.std(ddof=1) if n > 1 else float('nan')
    iid_t = mean_r / (sd / np.sqrt(n)) if n > 1 and sd > 0 else float('nan')
    daily = df.groupby(df['date'].dt.date)['R'].mean()
    n_days = len(daily)
    dc_t = (daily.mean() / (daily.std(ddof=1) / np.sqrt(n_days))
            if n_days > 1 and daily.std(ddof=1) > 0 else float('nan'))
    k = max(1, int(np.ceil(n * 0.05)))
    thresh = pd.Series(r).nlargest(k).min()
    ex_top5 = r[r < thresh].mean() if (r < thresh).any() else float('nan')
    mde = Z80 * sd / np.sqrt(n) if n > 1 else float('nan')
    weekly_sum = df.groupby(weeks)['R'].sum()
    p10 = weekly_sum.quantile(0.10)
    worst = weekly_sum.min()
    return dict(label=label, n=n, fills_wk=n / n_weeks if n_weeks else float('nan'),
                mean_r=mean_r, iid_t=iid_t, dc_t=dc_t, ex_top5=ex_top5, mde=mde,
                weekly_p10=p10, worst_week=worst, n_weeks=n_weeks, total_usd=df['_sized_pnl'].sum())


def fmt(s):
    if s.get('n', 0) == 0:
        return f"{s['label']}: n=0 (no fills)"
    return (f"{s['label']}: n={s['n']} fills/wk={s['fills_wk']:.2f} meanR={s['mean_r']:+.3f} "
            f"iid_t={s['iid_t']:.2f} dc_t={s['dc_t']:.2f} exTop5={s['ex_top5']:+.3f} "
            f"MDE={s['mde']:.3f} wkP10={s['weekly_p10']:+.2f}R worstWk={s['worst_week']:+.2f}R "
            f"${s['total_usd']:+,.0f}")


def union_book(prod, pool):
    """independent_1328.py's method: pool rows not on a (date,symbol) already in prod that day,
    admitted by _composite desc up to 8 - (prod picks that day)."""
    if pool is None or len(pool) == 0:
        return prod.copy(), pool
    prod_set = set(zip(prod['date'], prod['symbol'])) if prod is not None else set()
    addon = pool[~pool.apply(lambda r: (r['date'], r['symbol']) in prod_set, axis=1)].copy()
    raw_overlap = 1 - len(addon) / len(pool)
    addon = addon.sort_values(['date', '_composite'], ascending=[True, False])
    prod_per_day = prod.groupby('date').size() if prod is not None and len(prod) else pd.Series(dtype=int)
    admitted = []
    for d, g in addon.groupby('date'):
        n_prod = prod_per_day.get(d, 0)
        take = max(0, MAX_SLOTS - n_prod)
        if take > 0:
            admitted.append(g.head(take))
    admitted = pd.concat(admitted, ignore_index=True) if admitted else addon.iloc[0:0]
    union = pd.concat([prod, admitted], ignore_index=True) if prod is not None else admitted
    return union.sort_values('date').reset_index(drop=True), admitted, raw_overlap


def write_trades_csv(df, path):
    out = df[['date', 'symbol', 'R']].copy()
    out['date'] = out['date'].dt.strftime('%Y-%m-%d')
    out = out.rename(columns={'R': 'pnl_R'})
    out.to_csv(path, index=False)


def cadence_report(df, book_name, lo, hi, seed=0):
    """C1 (strong-week gap) + C4 (green weeks vs count-matched null) + weekly P10/worst, using
    scripts/cadence_bar.py's own functions on OUR (lo,hi) window -- see the import-time note."""
    if df is None or len(df) == 0:
        return f"{book_name}: n=0, no cadence report"
    trades = [{'date': d.date(), 'r': r, 'symbol': s}
              for d, r, s in zip(df['date'], df['R'], df['symbol'])]
    weekly = cb.build_weekly_series(trades, lo, hi)
    cycles, strong_idx = cb.compute_cycles(weekly, strong_r=5.0)
    c1 = cb.score_c1(cycles, gap_median_thresh=3.0, gap_p90_thresh=6.0)
    c4 = cb.score_c4(weekly, trades, green_thresh=0.55, green_margin=0.10, rng=random.Random(seed))
    weekly_r = [r for _, r in weekly]
    p10 = cb.percentile(weekly_r, 10)
    worst = min(weekly_r) if weekly_r else float('nan')
    return (f"{book_name} [{lo}..{hi}]: weeks={len(weekly)} strong-weeks={len(strong_idx)} "
            f"C1 gap median={c1['median']} wk P90={c1['p90']} wk [{'pass' if c1['pass'] else 'fail'}]  "
            f"C4 green={100*(c4['green'] or 0):.0f}% null={100*(c4['null'] or 0):.0f}% "
            f"[{'pass' if c4['pass'] else 'fail'}]  weeklyP10={p10:+.2f}R worst={worst:+.2f}R")


if __name__ == '__main__':
    print(__doc__)
