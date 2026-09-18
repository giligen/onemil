#!/usr/bin/env python3
"""Meta-label study, step 0 — the FEATURE-AVAILABILITY AUDIT.

PLAN S1 standing rule (from D1, 2026-09-17): every field used in a decision is
traced to its construction and shown to be computable at the decision instant
(09:35:00 ET) from data timestamped <= 09:35:00, and every field with < 100%
coverage gets a missingness table per split and per outcome BEFORE it is used.
D1's +0.415R was an availability leak (premarket dollars backfilled only for
symbol-days that signalled later), so this runs before any model is fit.

Writes research/meta_label/availability_audit.csv and prints the DROP list.
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
from trading.orb_csv import read_orb_csv  # noqa: E402

M = f'{ROOT}/research/meta_label'
D1 = f'{ROOT}/research/fuckup_audit/D1_orb'
PC = f'{ROOT}/research/fuckup_audit/P_cost'

# The 24 candidate features carried by candidates_dump.csv, with the trace of
# WHERE each is built in study_orb_features.py::compute_features and the latest
# timestamp its inputs carry.  `t<=0935` is the audit verdict.
TRACE = {
    # --- opening-range block: bars [09:30, 09:35), i.e. rb ---
    'entry_price':             ('study_orb_features.trade_row: range_high x (1+30bps) — the ORDER level', '09:35:00'),
    'range_size_pct':          ('L268 (range_high-range_low)/open, rb = bars 09:30-09:34', '09:35:00'),
    'range_total_volume':      ('L269 rb.volume.sum()', '09:35:00'),
    'range_avg_bar_range_pct': ('L271 mean (h-l)/c over rb', '09:35:00'),
    'range_volume_stddev_pct': ('L274 std of rb.volume', '09:35:00'),
    'bars_green_in_range':     ('L277 count(close>open) over rb', '09:35:00'),
    'range_close_position':    ('L278 (close-range_low)/range_size', '09:35:00'),
    'range_return_pct':        ('L279 (close-open)/open over rb', '09:35:00'),
    'last_bar_green':          ('L280 rb.close[-1] > rb.open[-1]', '09:35:00'),
    'range_vwap_distance_pct': ('L286 rb close vs rb VWAP', '09:35:00'),
    # --- prior-day / 20-day block: daily bars strictly BEFORE today ---
    'gap_pct':                 ('L305 (today 09:30 open - prev close)/prev close', '09:30:00'),
    'prev_day_range_pct':      ('L306 prev daily bar only (bar_date < today)', 'prev close'),
    'prev_day_close_position': ('L308 prev daily bar only', 'prev close'),
    'avg_daily_volume_20d':    ('L322 prior_df = daily bars < today, tail(20)', 'prev close'),
    'avg_daily_range_pct_20d': ('L324 prior_df', 'prev close'),
    'price_vs_20d_high_pct':   ('L326 today 09:30 open vs prior_df high', '09:30:00'),
    'return_volatility_20d':   ('L331 std of prior_df daily returns', 'prev close'),
    'prev_day_volume_vs_20d':  ('L334 prev volume / prior_df mean volume', 'prev close'),
    # --- SPY block ---
    'spy_range_pct_5min':      ('L363 SPY bars [09:30,09:35)', '09:35:00'),
    'spy_return_5min_pct':     ('L364 SPY bars [09:30,09:35)', '09:35:00'),
    'spy_gap_pct':             ('L378 SPY today OPEN vs SPY prev close — value is the 09:30 print', '09:30:00'),
    'spy_3d_range_pct':        ('L385 SPY prior 3 daily bars (bar_date < today)', 'prev close'),
    # --- calendar ---
    'day_of_week':             ('L390 calendar', 'known ex ante'),
    'days_since_month_start':  ('L391 calendar', 'known ex ante'),
}

OUTCOME_COLS = ['pnl', 'pnl_pct', 'exit_reason', 'win', 'entered', '_rp_pnl', '_rp_position']

# The sentinel branch: < 5 prior daily bars => the whole 20-day block is written
# as 0.0 rather than NaN (study_orb_features L315-319).  That is the G1
# short-history marker; it is a legitimate 09:35-knowable state, but it must be
# counted so a model cannot read "0.0" as a volatility level.
SENTINEL_BLOCK = ['avg_daily_volume_20d', 'avg_daily_range_pct_20d',
                  'price_vs_20d_high_pct', 'return_volatility_20d',
                  'prev_day_volume_vs_20d']

# Fields where 0.0 is written as the MISSING-DATA sentinel by an explicit
# `else: feat[...] = 0.0` branch in study_orb_features.compute_features.  For
# every other field 0 is a legitimate value (no green bars, a red last bar,
# a flat gap) and its rate must NOT be read as missingness.
SENTINEL_FIELDS = set(SENTINEL_BLOCK) | {
    'spy_range_pct_5min', 'spy_return_5min_pct',   # L366-370: no SPY 5-min bars
    'spy_gap_pct', 'spy_3d_range_pct',             # L380/387: no SPY daily bars
}


def split_of(day: str) -> str:
    return 'TRAIN' if day < '2026-01-01' else ('VAL' if day < '2026-06-01' else 'TEST')


def main() -> None:
    d = read_orb_csv(f'{D1}/candidates_dump.csv')
    d['date'] = pd.to_datetime(d['date']).dt.strftime('%Y-%m-%d')
    d['split'] = d['date'].map(split_of)
    print(f'population: {len(d)} candidates, {d.date.nunique()} days, '
          f'{d.date.min()} -> {d.date.max()}, entered={int(d.entered.sum())}')

    # ---- population hygiene (F6 reconciliation standing rule) -------------
    tt = [s for s in d.symbol.unique() if re.match(r'^Z[A-Z]ZZT$', str(s))]
    print(f'NASDAQ test tickers (^Z[A-Z]ZZT$) in the population: {tt or "none"}')

    # ---- entry_price obtainability: is it the ORDER level, not a fill? ----
    ex = pd.read_csv(f'{PC}/exit_times.csv', keep_default_na=False, na_values=[''],
                     dtype={'symbol': str, 'date': str},
                     usecols=['symbol', 'date', 'range_high', 'range_low'])
    rl = dict(zip(zip(ex.symbol, ex.date), zip(ex.range_high, ex.range_low)))
    rh = np.array([rl.get((s, dt), (np.nan, np.nan))[0] for s, dt in zip(d.symbol, d.date)])
    ratio = d.entry_price.to_numpy(float) / rh
    cov = np.isfinite(ratio)
    print(f'entry_price / range_high: covered {cov.sum()}/{len(d)}; '
          f'max |ratio-1.003| = {np.nanmax(np.abs(ratio[cov] - 1.003)):.3e} '
          f'(entry_price IS the 09:35 order level, not a realised fill)')

    rows = []
    for f, (trace, ts) in TRACE.items():
        v = pd.to_numeric(d[f], errors='coerce')
        r = {'feature': f, 'source': trace, 'latest_input_ts': ts,
             'computable_at_0935': True,
             'coverage_pct': 100.0 * v.notna().mean(),
             'n_nan': int(v.isna().sum()),
             'sentinel_zero_pct': 100.0 * float((v == 0).mean()),
             'nunique': int(v.nunique())}
        # missingness / sentinel rate conditioned on the OUTCOME — the D1 leak
        # signature (a field present only for rows that went on to do something)
        for lbl, m in (('entered', d.entered == 1), ('nofill', d.entered != 1)):
            r[f'nan_pct_{lbl}'] = 100.0 * float(v[m].isna().mean())
            r[f'zero_pct_{lbl}'] = 100.0 * float((v[m] == 0).mean())
        for sp in ('TRAIN', 'VAL', 'TEST'):
            m = d.split == sp
            r[f'nan_pct_{sp}'] = 100.0 * float(v[m].isna().mean())
            r[f'zero_pct_{sp}'] = 100.0 * float((v[m] == 0).mean())
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(f'{M}/availability_audit.csv', index=False)

    pd.set_option('display.width', 250)
    print('\n=== availability / coverage (all 24) ===')
    print(t[['feature', 'latest_input_ts', 'coverage_pct', 'sentinel_zero_pct',
             'zero_pct_entered', 'zero_pct_nofill', 'nan_pct_TRAIN', 'nan_pct_VAL']]
          .to_string(index=False, float_format=lambda x: f'{x:.2f}'))

    # DROP rule, pre-registered: drop a field iff its value is NOT computable at
    # 09:35:00 from data timestamped <= 09:35:00, OR its missingness/sentinel
    # rate differs by more than 5 points between entered and no-fill rows (a
    # cohort-availability signature — the D1 leak).
    drop = []
    for _, r in t.iterrows():
        if not r['computable_at_0935']:
            drop.append((r['feature'], 'not computable at 09:35'))
        elif abs(r['nan_pct_entered'] - r['nan_pct_nofill']) > 5.0:
            drop.append((r['feature'], 'missingness conditioned on the fill outcome'))
        elif (r['feature'] in SENTINEL_FIELDS
              and abs(r['zero_pct_entered'] - r['zero_pct_nofill']) > 5.0):
            drop.append((r['feature'], 'sentinel rate conditioned on the fill outcome'))
        elif (r['feature'] in SENTINEL_FIELDS
              and max(r[f'zero_pct_{s}'] for s in ('TRAIN', 'VAL', 'TEST')) > 20.0):
            # The value in THIS dump is a cache-coverage marker, not the field:
            # live computes it from the tape at 09:35 and would never see the
            # sentinel.  Fitting on it fits the coverage era, not the market.
            drop.append((r['feature'], 'missing-data sentinel on >20% of a split '
                                       '(coverage marker, not the field live computes)'))
    # per-split sentinel rate for the sentinel-bearing fields (the era check)
    print('\n=== sentinel (0.0-on-missing) rate per split ===')
    print(t[t.feature.isin(SENTINEL_FIELDS)][
        ['feature', 'zero_pct_TRAIN', 'zero_pct_VAL', 'zero_pct_TEST']]
        .to_string(index=False, float_format=lambda x: f'{x:.2f}'))

    print('\n=== DROP list ===')
    print(drop or 'none — all 24 are computable at 09:35 and their missingness '
                  'does not depend on the outcome')

    # sentinel block co-occurrence (the G1 short-history marker)
    sen = (pd.to_numeric(d['return_volatility_20d'], errors='coerce') == 0)
    print(f'\nshort-history sentinel (return_volatility_20d == 0): {int(sen.sum())} rows '
          f'({100*sen.mean():.2f}%); the other four 20-day fields are 0 on '
          f'{int((pd.to_numeric(d[SENTINEL_BLOCK[0]], errors="coerce")[sen] == 0).sum())} of them')

    # outcome columns must never enter the feature matrix
    print(f'\nEXCLUDED as outcomes (never features): {OUTCOME_COLS}')


if __name__ == '__main__':
    main()
