#!/usr/bin/env python3
"""Re-derive the ORB range-size veto threshold for an opening-range width W.

Repeats the V1 veto study's derivation (research/orb_veto_study/DESIGN.md):
the candidate is the WORST QUINTILE of `range_size_pct` among the ENTERED raw
candidates, and it is adopted only if that quintile is the worst in BOTH
years. The threshold is that quintile's upper edge. `range_size_pct` is a
percentage of price over a W-minute range, so it CANNOT be inherited from
W=5 — it is re-derived per window, before any book is scored.

Raw R (the detector's own R, no pipeline selection and no static lock — the
features CSV's fixed +2R / -1R proxy, exactly what the V1 scan used):

    R_i = pnl_pct_i / 100 * entry_i / (entry_i - range_low_i)

`range_low` is not a column; it is recovered EXACTLY from the columns that are
(the features are all built from the same range bars):

    K        = (1 + range_return_pct/100) + (range_size_pct/100)*(1 - range_close_position)
    open     = entry_price / (ENTRY_SLIP * K)          # entry = range_high * ENTRY_SLIP
    rs       = range_size_pct/100 * open
    close    = open * (1 + range_return_pct/100)
    range_low = close - range_close_position * rs

ENTRY_SLIP is 1.003 for entered rows (study_orb.simulate_orb_trade applies
30 bps) and 1.0 for the `entered=0` rows (trade_row stores the bare
range_high). Only entered rows are scored here.

Usage:
    python3 derive_range_size_veto.py FEATURES.csv [--label W15]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from trading.orb_csv import read_orb_csv

ENTRY_SLIP = 1.003


def raw_r(df: pd.DataFrame) -> pd.Series:
    k = ((1 + df['range_return_pct'] / 100.0)
         + (df['range_size_pct'] / 100.0) * (1 - df['range_close_position']))
    open_p = df['entry_price'] / (ENTRY_SLIP * k)
    rs = df['range_size_pct'] / 100.0 * open_p
    close_p = open_p * (1 + df['range_return_pct'] / 100.0)
    range_low = close_p - df['range_close_position'] * rs
    denom = df['entry_price'] - range_low
    return (df['pnl_pct'] / 100.0) * df['entry_price'] / denom.replace(0, np.nan)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('features')
    ap.add_argument('--label', default='')
    ap.add_argument('--feature', default='range_size_pct')
    a = ap.parse_args()
    df = read_orb_csv(a.features)
    df['date'] = pd.to_datetime(df['date'])
    if 'entered' in df.columns:
        df = df[df['entered'] == 1].copy()
    df = df.dropna(subset=[a.feature, 'pnl_pct', 'entry_price',
                           'range_return_pct', 'range_close_position'])
    df['_R'] = raw_r(df)
    df = df[np.isfinite(df['_R'])]
    df['_year'] = df['date'].dt.year
    edges = [float(df[a.feature].quantile(q)) for q in (0.2, 0.4, 0.6, 0.8)]
    df['_q'] = pd.cut(df[a.feature], [-np.inf] + edges + [np.inf],
                      labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    print(f"{a.label or a.features}: {len(df)} entered raw candidates; "
          f"{a.feature} quintile edges = {[round(e, 4) for e in edges]}")
    tbl = df.pivot_table(index='_q', columns='_year', values='_R',
                         aggfunc=['mean', 'count'], observed=True)
    print(tbl.to_string(float_format=lambda v: f"{v:,.3f}"))
    means = df.pivot_table(index='_q', columns='_year', values='_R',
                           aggfunc='mean', observed=True)
    years = [y for y in means.columns if y in (2025, 2026)]
    worst = {y: means[y].idxmin() for y in years}
    print(f"worst quintile per year: {worst}")
    if len(set(worst.values())) == 1 and list(worst.values())[0] == 'Q1':
        print(f"ADOPT: bottom quintile is worst in BOTH years -> "
              f"{a.feature} <= {edges[0]:.3f}")
    else:
        print(f"NO ERA-CONSISTENT BOTTOM QUINTILE for {a.feature} "
              f"(worst = {worst}); the W=5 rule does not transfer — report "
              f"the veto as NOT adopted for this window.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
