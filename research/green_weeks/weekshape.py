#!/usr/bin/env python3
"""green_weeks — the ONE week-shape scorer, shared by all three books.

PREREG §2. A book is a table of trades with a `day` (YYYY-MM-DD) and a value
column (`pnl` in $ or `rr` in R).  Weeks are `W-FRI` periods; the denominator is
every market week in the split, whether or not the book traded (a no-trade week
is FLAT, counted in the denominator).  Green = week value > 0.

Nothing here ranks on total P&L (PREREG §2 item 7); the caller does the ranking.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# PREREG §3 — identical splits for the three books; the TEST end is per book.
SPLITS = {
    'TRAIN': ('2025-01-01', '2025-12-31'),
    'VAL':   ('2026-01-01', '2026-05-31'),
    'TEST':  ('2026-06-01', None),        # per-book end, supplied by the caller
}


def week_index(a: str, b: str) -> pd.PeriodIndex:
    """Every W-FRI week touching [a, b] — the green-week DENOMINATOR."""
    return pd.period_range(a, b, freq='W-FRI')


def _max_drawdown(daily_cum: np.ndarray) -> float:
    if len(daily_cum) == 0:
        return 0.0
    peak = np.maximum.accumulate(daily_cum)
    return float((daily_cum - peak).min())


def _longest_red_streak(vals: np.ndarray) -> int:
    best = cur = 0
    for v in vals:
        if v < 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def score(trades: pd.DataFrame, split: str, start: str, end: str,
          value: str = 'pnl', day: str = 'day') -> dict:
    """Week-shape metrics for one cell on one split.  `trades` may be empty."""
    weeks = week_index(start, end)
    nw = len(weeks)
    d = trades[(trades[day] >= start) & (trades[day] <= end)].copy()
    out = {
        'split': split, 'n': len(d), 'n_weeks': nw,
        'tr_per_wk': len(d) / nw if nw else np.nan,
    }
    if len(d) == 0:
        out.update({'green_pct': 0.0, 'flat_pct': 100.0, 'red_pct': 0.0,
                    'red_streak': 0, 'worst_wk': 0.0, 'best_wk': 0.0,
                    'mo_green_pct': np.nan, 'worst_mo': 0.0, 'mdd': 0.0,
                    'pnl': 0.0, 'wr': np.nan, 'top1_share': np.nan,
                    'top5_share': np.nan, 'top10_share': np.nan,
                    'green_wks': 0, 'flat_wks': nw, 'wk_traded_pct': 0.0})
        out['_wk'] = pd.Series(0.0, index=weeks)
        return out

    dt = pd.to_datetime(d[day])
    d['_wk'] = dt.dt.to_period('W-FRI')
    d['_mo'] = dt.dt.to_period('M')

    wk = d.groupby('_wk')[value].sum().reindex(weeks, fill_value=0.0)
    n_traded_wk = d['_wk'].nunique()
    green = int((wk > 0).sum())
    red = int((wk < 0).sum())
    flat = nw - green - red                       # no-trade weeks + exact zeros

    mo = d.groupby('_mo')[value].sum()
    dayv = d.groupby(day)[value].sum().sort_index()
    cum = dayv.cumsum().to_numpy()

    v = d[value].to_numpy(dtype=float)
    tot = float(v.sum())
    srt = np.sort(v)[::-1]

    def _share(k):
        k = max(1, int(round(len(v) * k)))
        return float(srt[:k].sum() / tot) if tot else np.nan

    out.update({
        'green_pct': 100.0 * green / nw,
        'flat_pct': 100.0 * flat / nw,
        'red_pct': 100.0 * red / nw,
        'green_wks': green, 'flat_wks': flat,
        'red_streak': _longest_red_streak(wk.to_numpy()),
        'worst_wk': float(wk.min()), 'best_wk': float(wk.max()),
        'mo_green_pct': 100.0 * float((mo > 0).mean()),
        'worst_mo': float(mo.min()),
        'mdd': _max_drawdown(cum),
        'pnl': tot,
        'wr': 100.0 * float((v > 0).mean()),
        'top1_share': _share(0.01), 'top5_share': _share(0.05),
        'top10_share': _share(0.10),
        'wk_traded_pct': 100.0 * n_traded_wk / nw,
    })
    out['_wk'] = wk
    return out


def discordant(wk_a: pd.Series, wk_b: pd.Series) -> tuple:
    """PREREG §7 — paired power.  Weeks green in exactly one of the two cells.
    Returns (n_discordant, n_b_green_only, n_a_green_only)."""
    ga, gb = (wk_a > 0), (wk_b > 0)
    disc = int((ga != gb).sum())
    return disc, int((gb & ~ga).sum()), int((ga & ~gb).sum())


def score_all(trades: pd.DataFrame, test_end: str, value: str = 'pnl',
              day: str = 'day', reveal_test: bool = False) -> dict:
    """TRAIN/VAL always; TEST only when the FREEZE seal is explicitly lifted."""
    res = {}
    for s, (a, b) in SPLITS.items():
        if s == 'TEST':
            if not reveal_test:
                continue
            b = test_end
        res[s] = score(trades, s, a, b, value=value, day=day)
    return res
