#!/usr/bin/env python3
"""Scoring for research/orb_frequency — the owner's metric first.

PRIMARY = % of GREEN WEEKS over EVERY market week in the split (weeks with no
pick are in the denominator).  Flat-week share is printed beside it on every
row.  Total P&L is TERTIARY and is a floor condition only (PREREG §3, §7).

TEST (2026-06-01+) is SEALED: nothing from that split is returned unless
--reveal-test is passed AND FREEZE.md already carries the recommendation
commit.  See FREEZE.md.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
sys.path.insert(0, ROOT)
from trading.orb_csv import read_orb_csv  # noqa: E402

D = f'{ROOT}/research/orb_frequency'
FEATURES = f'{ROOT}/analysis_results/orb_features_20260916_2053.csv'
MIN_STOP_PCT = 1.0          # orb.yaml, via study_orb_pipeline_static_lock

SPLITS = {
    'TRAIN': ('2025-01-01', '2025-12-31'),
    'VAL':   ('2026-01-01', '2026-05-31'),
    'TEST':  ('2026-06-01', '2026-12-31'),
}


# --------------------------------------------------------------------------
# the market-week calendar: the 427 trading days of the candidate population
# --------------------------------------------------------------------------
_CAL = None


def calendar() -> pd.DataFrame:
    """Trading days of the population, with their ISO week and month."""
    global _CAL
    if _CAL is None:
        d = read_orb_csv(FEATURES)[['date']].drop_duplicates()
        d['date'] = pd.to_datetime(d['date'])
        d = d.sort_values('date').reset_index(drop=True)
        d['wk'] = d['date'].dt.strftime('%G-W%V')
        d['mo'] = d['date'].dt.to_period('M').astype(str)
        _CAL = d
    return _CAL


def split_of(day: pd.Timestamp) -> str:
    s = str(day)[:10]
    for name, (a, b) in SPLITS.items():
        if a <= s <= b:
            return name
    return 'OUT'


def r_of(df: pd.DataFrame) -> pd.Series:
    """R = pnl_pct / max(range_size_pct, MIN_STOP_PCT).  PREREG §1."""
    stop = df['range_size_pct'].clip(lower=MIN_STOP_PCT)
    return df['pnl_pct'].astype(float) / stop


def _mdd(daily: pd.Series) -> float:
    if daily.empty:
        return 0.0
    cum = daily.cumsum()
    return float((cum - cum.cummax()).min())


def _streak(seq) -> int:
    best = cur = 0
    for v in seq:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def week_frame(book: pd.DataFrame, split: str) -> pd.DataFrame:
    """One row per MARKET WEEK of the split (no-pick weeks included)."""
    cal = calendar()
    cal = cal[cal['date'].map(split_of) == split]
    weeks = cal[['wk']].drop_duplicates().sort_values('wk').reset_index(drop=True)
    b = book[book['date'].map(split_of) == split]
    agg = b.groupby('wk')['_sized_pnl'].sum() if len(b) else pd.Series(dtype=float)
    n = b.groupby('wk').size() if len(b) else pd.Series(dtype=int)
    weeks['pnl'] = weeks['wk'].map(agg).fillna(0.0)
    weeks['n'] = weeks['wk'].map(n).fillna(0).astype(int)
    return weeks


def month_frame(book: pd.DataFrame, split: str) -> pd.DataFrame:
    cal = calendar()
    cal = cal[cal['date'].map(split_of) == split]
    months = cal[['mo']].drop_duplicates().sort_values('mo').reset_index(drop=True)
    b = book[book['date'].map(split_of) == split]
    agg = b.groupby('mo')['_sized_pnl'].sum() if len(b) else pd.Series(dtype=float)
    months['pnl'] = months['mo'].map(agg).fillna(0.0)
    return months


def load_book(path: str) -> pd.DataFrame:
    b = read_orb_csv(path)
    b['date'] = pd.to_datetime(b['date'])
    b['wk'] = b['date'].dt.strftime('%G-W%V')
    b['mo'] = b['date'].dt.to_period('M').astype(str)
    b['R'] = r_of(b)
    return b


def stats(path_or_book, split: str, reveal_test: bool = False) -> dict:
    """Every metric PREREG §5c asks for, for one book on one split."""
    if split == 'TEST' and not reveal_test:
        raise SystemExit('TEST is SEALED (FREEZE.md) — pass reveal_test=True '
                         'only after the recommendation is committed.')
    b = load_book(path_or_book) if isinstance(path_or_book, str) else path_or_book
    s = b[b['date'].map(split_of) == split]
    w = week_frame(b, split)
    m = month_frame(b, split)
    nw = len(w)
    green = int((w['pnl'] > 0).sum())
    red = int((w['pnl'] < 0).sum())
    flat = int((w['pnl'] == 0).sum())
    R = s['R'].values
    n = len(R)
    pnl = float(s['_sized_pnl'].sum())
    daily = s.groupby('date')['_sized_pnl'].sum().sort_index()
    top = np.sort(s['_sized_pnl'].values)[::-1]
    out = {
        'split': split, 'weeks': nw, 'picks': n,
        'picks_per_wk': n / nw if nw else 0.0,
        'fills': int(s['entered'].sum()) if 'entered' in s else n,
        'green_wk_pct': 100.0 * green / nw if nw else 0.0,
        'flat_wk_pct': 100.0 * flat / nw if nw else 0.0,
        'red_wk_pct': 100.0 * red / nw if nw else 0.0,
        'red_streak': _streak(w['pnl'] < 0),
        'worst_wk': float(w['pnl'].min()) if nw else 0.0,
        'best_wk': float(w['pnl'].max()) if nw else 0.0,
        'green_mo_pct': 100.0 * float((m['pnl'] > 0).mean()) if len(m) else 0.0,
        'worst_mo': float(m['pnl'].min()) if len(m) else 0.0,
        'mdd': _mdd(daily),
        'pnl': pnl,
        'totR': float(R.sum()),
        'R_pick': float(R.mean()) if n else 0.0,
        'wr_fills': 100.0 * float((s.loc[s['entered'] == 1, '_sized_pnl'] > 0).mean())
        if 'entered' in s and int(s['entered'].sum()) else 0.0,
        't': float(R.mean() / (R.std(ddof=1) / np.sqrt(n))) if n > 1 and R.std(ddof=1) else 0.0,
    }
    for q, lab in ((0.01, 1), (0.05, 5)):
        k = int(np.ceil(q * n))
        out[f'R_pick_ex{lab}'] = float(np.sort(R)[::-1][k:].mean()) if n - k > 0 else 0.0
    for k, lab in ((1, 'top1'), (5, 'top5'), (10, 'top10')):
        out[f'{lab}_share'] = 100.0 * float(top[:k].sum() / pnl) if pnl > 0 and n >= k else np.nan
    # MDE80 on the green-week share, unpaired two-proportion normal approx
    p = green / nw if nw else 0.0
    se = np.sqrt(2 * p * (1 - p) / nw) if nw and 0 < p < 1 else np.nan
    out['mde80_green_pp'] = 100.0 * 2.80 * se if se == se else np.nan
    return out


COLS = ['split', 'weeks', 'picks', 'picks_per_wk', 'fills', 'green_wk_pct',
        'flat_wk_pct', 'red_wk_pct', 'red_streak', 'worst_wk', 'green_mo_pct',
        'worst_mo', 'mdd', 'pnl', 'totR', 'R_pick', 'R_pick_ex1', 'R_pick_ex5',
        'wr_fills', 't', 'top1_share', 'top5_share', 'top10_share',
        'mde80_green_pp']


def score_book(path: str, tag: str, splits=('TRAIN', 'VAL'),
               reveal_test: bool = False) -> pd.DataFrame:
    rows = []
    for sp in splits:
        r = stats(path, sp, reveal_test)
        r['cell'] = tag
        rows.append(r)
    return pd.DataFrame(rows)[['cell'] + COLS]


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    reveal = '--reveal-test' in sys.argv
    sp = ('TRAIN', 'VAL', 'TEST') if reveal else ('TRAIN', 'VAL')
    for p in args:
        print(score_book(p, os.path.basename(p), sp, reveal).to_string(index=False))
