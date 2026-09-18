#!/usr/bin/env python3
"""Scoring for research/orb_multiwindow — one book CSV in, the PREREG cells out.

Conventions (stated so they can be checked):
  * R per pick. The pipeline sizes every position at
    `_rp_position = min(risk / stop_frac, account/N)` and the $10K-stage cap
    binds on ~100% of picks, so the realized dollar risk of a pick is
    `_rp_position * stop_frac`, `stop_frac = max(range_size_pct, 1.0)/100`.
    R_i = `_sized_pnl_i / (that risk)`. A no-fill row books 0 P&L and 0 R but
    IS a pick (it spent a slot) — it is counted in the denominator.
  * Splits (PLAN §1): TRAIN 2025-01-02..2025-12-31, VAL 2026-01-01..2026-05-31,
    TEST 2026-06-01..2026-09-30.
  * MDD is on the daily cumulative curve of the book (same convention as the
    pipeline's monthly `intra_dd`, but over the whole window, not per month).
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from trading.orb_csv import read_orb_csv

SPLITS = [('TRAIN', '2025-01-01', '2025-12-31'),
          ('VAL', '2026-01-01', '2026-05-31'),
          ('TEST', '2026-06-01', '2026-12-31')]
MIN_STOP_PCT = 1.0


def load(path: str) -> pd.DataFrame:
    df = read_orb_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    stop_frac = df['range_size_pct'].clip(lower=MIN_STOP_PCT) / 100.0
    df['_risk'] = df['_rp_position'] * stop_frac
    df['_R'] = df['_sized_pnl'] / df['_risk']
    return df


def mdd(df: pd.DataFrame) -> float:
    d = df.groupby('date')['_sized_pnl'].sum().sort_index()
    cum = d.cumsum()
    return float((cum - cum.cummax()).min()) if len(cum) else 0.0


def monthly(df: pd.DataFrame) -> pd.Series:
    return df.groupby(df['date'].dt.to_period('M'))['_sized_pnl'].sum()


def stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return dict(picks=0, fills=0, pnl=0.0, r_pick=float('nan'), t=float('nan'),
                    mdd=0.0, worst_mo=0.0, red_mo=0, wk_green=float('nan'))
    m = monthly(df)
    wk = df.groupby(df['date'].dt.to_period('W'))['_sized_pnl'].sum()
    n = len(df)
    r = df['_R']
    t = float(r.mean() / (r.std(ddof=1) / math.sqrt(n))) if n > 1 and r.std(ddof=1) > 0 else float('nan')
    fills = int((df['entered'] == 1).sum()) if 'entered' in df else n
    return dict(picks=n, fills=fills, pnl=float(df['_sized_pnl'].sum()),
                r_pick=float(r.mean()), t=t, mdd=mdd(df),
                worst_mo=float(m.min()), red_mo=int((m < 0).sum()),
                wk_green=float((wk > 0).mean() * 100))


def split_rows(df: pd.DataFrame, label: str, tag: str = '',
               with_all: bool = True) -> list:
    rows = []
    heads = ([('ALL', '2000-01-01', '2100-01-01')] if with_all else []) + SPLITS
    for name, lo, hi in heads:
        s = df[(df['date'] >= lo) & (df['date'] <= hi)]
        rows.append(dict(book=label, split=name, tag=tag, **stats(s)))
    return rows


def tails(df: pd.DataFrame) -> dict:
    out = {}
    r = df['_R'].sort_values()
    for q, nm in ((0.01, 'ex_top1'), (0.05, 'ex_top5')):
        k = max(1, int(round(len(df) * q)))
        out[nm] = float(r.iloc[:-k].mean())
    out['cap_3R'] = float(df['_R'].clip(upper=3.0).mean())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--book', action='append', required=True,
                    metavar='LABEL=PATH')
    ap.add_argument('--base', default=None,
                    help='LABEL of the 5-min reference book (for added-picks)')
    ap.add_argument('--out', default=None)
    ap.add_argument('--no-test', action='store_true',
                    help='honesty rail: hide the TEST split (and the ALL row, '
                         'which contains it) until FREEZE.md is written')
    a = ap.parse_args()
    global SPLITS
    if a.no_test:
        SPLITS = [s for s in SPLITS if s[0] != 'TEST']
    books = {}
    for spec in a.book:
        label, path = spec.split('=', 1)
        books[label] = load(path)
    rows = []
    base_keys = None
    if a.base:
        b = books[a.base]
        base_keys = set(zip(b['symbol'], b['date']))
    for label, df in books.items():
        rows += split_rows(df, label, with_all=not a.no_test)
        if base_keys is not None and label != a.base:
            add = df[[k not in base_keys
                      for k in zip(df['symbol'], df['date'])]]
            rows += split_rows(add, label, tag='ADDED', with_all=not a.no_test)
    out = pd.DataFrame(rows)
    pd.set_option('display.width', 250)
    print(out.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    if a.out:
        out.to_csv(a.out, index=False)
        print(f"\nsaved {a.out}")
    print("\n--- tails (whole window, R/pick) ---")
    for label, df in books.items():
        if len(df) > 20:
            print(label, {k: round(v, 3) for k, v in tails(df).items()})
    return 0


if __name__ == '__main__':
    sys.exit(main())
