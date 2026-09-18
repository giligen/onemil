#!/usr/bin/env python3
"""Meta-label study — book statistics, identical in convention to Stage Q.

R per pick = _sized_pnl / (shares * (range_high - range_low)), shares =
_rp_position / entry_price (Q_fill/rescore_q.py::book_stats).  A no-fill pick
books R = 0 and still spends its slot.

Splits.  The walk-forward cannot score a month until it has 50 trailing training
sessions, so 2025-01..2025-03 are UNSCORED in every model cell and the pipeline
falls back to the shipped ranking there — those months are byte-identical to M0
in all nine cells.  They are reported as `PRE` and excluded from the TRAIN
comparison so the gate is measured only where a model is actually deciding.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
from trading.orb_csv import read_orb_csv  # noqa: E402

PC = f'{ROOT}/research/fuckup_audit/P_cost'
_RNG = None


def split_of(day: str) -> str:
    if day < '2025-04-01':
        return 'PRE'
    if day < '2026-01-01':
        return 'TRAIN'
    if day < '2026-06-01':
        return 'VAL'
    return 'TEST'


def _mdd(daily: pd.Series) -> float:
    cum = daily.cumsum()
    return float((cum - cum.cummax()).min())


def range_lookup():
    global _RNG
    if _RNG is None:
        x = pd.read_csv(f'{PC}/exit_times.csv', keep_default_na=False, na_values=[''],
                        dtype={'symbol': str, 'date': str},
                        usecols=['symbol', 'date', 'range_high', 'range_low'])
        _RNG = dict(zip(zip(x.symbol, x.date), x.range_high - x.range_low))
    return _RNG


def load_book(path: str) -> pd.DataFrame:
    b = read_orb_csv(path)
    b['date'] = pd.to_datetime(b['date']).dt.strftime('%Y-%m-%d')
    b['split'] = b.date.map(split_of)
    b['month'] = b.date.str[:7]
    rl = range_lookup()
    rng = np.array([rl.get((s, dt), np.nan) for s, dt in zip(b.symbol, b.date)])
    shares = b['_rp_position'] / b['entry_price']
    b['R'] = np.where(rng > 0, b['_sized_pnl'] / (shares * rng), 0.0)
    return b


def stats(g: pd.DataFrame) -> dict:
    mo = g.groupby('month')._sized_pnl.sum()
    r = g.R.to_numpy(float)
    t = (r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))) if len(r) > 1 and r.std() > 0 else np.nan
    return dict(picks=len(g), fills=int((g.entered.astype(float) != 0).sum()),
                pnl=round(float(g._sized_pnl.sum()), 0),
                r_per_pick=round(float(r.mean()), 4),
                t=round(float(t), 2),
                mdd=round(_mdd(g.groupby('date')._sized_pnl.sum()), 0),
                worst_mo=round(float(mo.min()), 0) if len(mo) else np.nan,
                red_mo=int((mo < 0).sum()), n_mo=int(len(mo)))


def book_stats(path: str, splits=('TRAIN', 'VAL')) -> dict:
    b = load_book(path)
    out = {}
    for sp in splits:
        g = b[b.split == sp]
        if len(g):
            out[sp] = stats(g)
    return out


def paired_vs_m0(path: str, m0_path: str, split: str) -> dict:
    """Day-paired difference in the split's daily P&L (the book is a daily
    object: slots, dedup and the no-refill vetoes all live inside one day)."""
    a = load_book(path)
    b = load_book(m0_path)
    a, b = a[a.split == split], b[b.split == split]
    da = a.groupby('date')._sized_pnl.sum()
    db = b.groupby('date')._sized_pnl.sum()
    idx = sorted(set(da.index) | set(db.index))
    d = da.reindex(idx).fillna(0) - db.reindex(idx).fillna(0)
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 and d.std() > 0 else np.nan
    return dict(n_days=len(d), d_pnl=round(float(d.sum()), 0),
                d_mean_day=round(float(d.mean()), 2), t_day=round(float(t), 2))


if __name__ == '__main__':
    for p in sys.argv[1:]:
        print(p, book_stats(p))
