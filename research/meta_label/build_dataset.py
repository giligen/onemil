#!/usr/bin/env python3
"""Meta-label study, step 1 — the modelling table.

Rows: the 13,033 entered-inclusive ORB candidates of
`research/fuckup_audit/D1_orb/candidates_dump.csv` (the SAME population D1's
`book_n8_q1on.csv` is selected from).

Features: the 22 survivors of `audit_availability.py` (24 minus the two SPY
5-minute fields, which are a 0.0 cache-coverage sentinel on 100% of every month
from 2026-04 on).  Plus, as a DECLARED CELL DIMENSION, the five shipped veto
flags -- each recomputed here through the SAME shared helper the live engine and
the pipeline call, so the flag a model sees is the flag that fires.

Labels: the candidate's own realised R under the shipped exit spec,
    R = (pnl_pct / 100) * entry_price / (range_high - range_low),
which is share-count invariant and therefore identical to Stage Q's
`_sized_pnl / (shares * range)` before the quintile multiplier.  Computed under
TWO fill/cost arms: `meas_cost` (Stage Q's measured model -- capped limit,
min(ask, cap), ask > cap => rest-then-fill-or-never, measured per-trade NBBO on
both legs; the PRIMARY arm of this study) and `asis` (D1's arm; the M0
reproduction gate).  A no-fill row books R = 0 and still spends its slot.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
from trading.orb_csv import read_orb_csv                      # noqa: E402
from study_orb_filter import FILTER_FEATURES, composite_score  # noqa: E402
from study_orb_sizing import assign_quintile                   # noqa: E402

M = f'{ROOT}/research/meta_label'
D1 = f'{ROOT}/research/fuckup_audit/D1_orb'
QF = f'{ROOT}/research/fuckup_audit/Q_fill'
PC = f'{ROOT}/research/fuckup_audit/P_cost'

DROPPED = ['spy_range_pct_5min', 'spy_return_5min_pct']

FEATURES = [
    'entry_price', 'range_size_pct', 'range_total_volume',
    'range_avg_bar_range_pct', 'range_volume_stddev_pct', 'bars_green_in_range',
    'range_close_position', 'range_return_pct', 'last_bar_green',
    'range_vwap_distance_pct', 'gap_pct', 'prev_day_range_pct',
    'prev_day_close_position', 'avg_daily_volume_20d', 'avg_daily_range_pct_20d',
    'price_vs_20d_high_pct', 'return_volatility_20d', 'prev_day_volume_vs_20d',
    'spy_gap_pct', 'spy_3d_range_pct', 'day_of_week', 'days_since_month_start',
]

# feature families, for the ablation (PLAN S1 / PREREG)
FAMILIES = {
    'opening_range': ['range_size_pct', 'range_total_volume',
                      'range_avg_bar_range_pct', 'range_volume_stddev_pct',
                      'bars_green_in_range', 'range_close_position',
                      'range_return_pct', 'last_bar_green',
                      'range_vwap_distance_pct', 'entry_price'],
    'prev_day': ['gap_pct', 'prev_day_range_pct', 'prev_day_close_position',
                 'prev_day_volume_vs_20d'],
    'twenty_day': ['avg_daily_volume_20d', 'avg_daily_range_pct_20d',
                   'price_vs_20d_high_pct', 'return_volatility_20d'],
    'market': ['spy_gap_pct', 'spy_3d_range_pct'],
    'calendar': ['day_of_week', 'days_since_month_start'],
}

VETO_FLAGS = ['veto_q1', 'veto_pdr', 'veto_g1', 'veto_rs', 'veto_catalyst']


def _label_R(dump_path: str, rng: np.ndarray) -> np.ndarray:
    d = read_orb_csv(dump_path)
    d['date'] = pd.to_datetime(d['date']).dt.strftime('%Y-%m-%d')
    d = d.sort_values(['date', 'symbol']).reset_index(drop=True)
    r = (d.pnl_pct.astype(float) / 100.0) * d.entry_price.astype(float) / rng
    return np.where(np.isfinite(r), r, 0.0)


def main() -> None:
    import yaml
    cfg = yaml.safe_load(open(f'{ROOT}/orb.yaml'))
    filt = cfg.get('filter') or {}
    g1c = filt.get('g1_veto') or {}

    d = read_orb_csv(f'{D1}/candidates_dump.csv')
    d['date'] = pd.to_datetime(d['date']).dt.strftime('%Y-%m-%d')
    d = d.sort_values(['date', 'symbol']).reset_index(drop=True)

    # ---- range (the R unit), from Stage P's exit-time table --------------
    ex = pd.read_csv(f'{PC}/exit_times.csv', keep_default_na=False, na_values=[''],
                     dtype={'symbol': str, 'date': str},
                     usecols=['symbol', 'date', 'range_high', 'range_low'])
    rl = dict(zip(zip(ex.symbol, ex.date), ex.range_high - ex.range_low))
    # non-fill rows have no exit_times row; their entry_price IS range_high*1.003
    # so the R unit is reconstructed exactly from the shipped stop (range_low is
    # unavailable there, but R is 0 for a $0 row, so only fills need the unit).
    rng = np.array([rl.get((s, dt), np.nan) for s, dt in zip(d.symbol, d.date)])
    cov = np.isfinite(rng) & (rng > 0)
    print(f'R unit (range_high-range_low) covered on {cov.sum()}/{len(d)} rows '
          f'({int((d.entered == 1).sum())} entered); uncovered rows are no-fills '
          f'(R=0 by construction): '
          f'{int(((~cov) & (d.entered == 1)).sum())} entered rows uncovered')

    out = d[['symbol', 'date'] + FEATURES].copy()
    out['entered'] = d['entered'].astype(int)

    for arm in ('meas_cost', 'asis'):
        out[f'R_{arm}'] = _label_R(f'{QF}/dump_{arm}.csv', rng)
        out.loc[out.entered != 1, f'R_{arm}'] = 0.0
    for arm in ('meas_cost', 'asis'):
        out[f'y_{arm}'] = (out[f'R_{arm}'] > 0).astype(int)

    # ---- the five shipped vetoes, through the shipped helpers -------------
    from trading.orb_pdr_veto import pdr_veto_applies, DEFAULT_MIN_PDR_PCT
    from trading.orb_g1_veto import g1_reject, DEFAULT_PDR_MIN, DEFAULT_RV20_MIN
    from trading.orb_range_size_veto import (range_size_veto_applies,
                                             DEFAULT_MIN_RANGE_SIZE_PCT)
    pdr_min = float((filt.get('prev_day_range_veto') or {}).get(
        'min_prev_day_range_pct', DEFAULT_MIN_PDR_PCT))
    rs_min = float((filt.get('range_size_veto') or {}).get(
        'min_range_size_pct', DEFAULT_MIN_RANGE_SIZE_PCT))
    g1_rv = float(g1c.get('return_volatility_20d_min', DEFAULT_RV20_MIN))
    g1_pdr = float(g1c.get('prev_day_range_pct_min', DEFAULT_PDR_MIN))
    g1_sh = bool(g1c.get('short_history_veto', False))

    out['veto_pdr'] = [int(pdr_veto_applies(None if pd.isna(v) else float(v), pdr_min))
                       for v in d.prev_day_range_pct]
    out['veto_g1'] = [int(g1_reject(rv, p, g1_rv, g1_pdr,
                                    short_history_veto=g1_sh) is not None)
                      for rv, p in zip(d.return_volatility_20d, d.prev_day_range_pct)]
    out['veto_rs'] = [int(range_size_veto_applies(v, rs_min)) for v in d.range_size_pct]

    # Q1: the composite z-score + the frozen orb.yaml quintile cutoffs, exactly
    # as the pipeline computes them (live-parity literals, never a refit).
    feats = filt.get('features') or {}
    params = {f: {'mean': float(feats[f]['mean']), 'std': float(feats[f]['std']),
                  'sign': int(feats[f]['sign'])} for f, _s in FILTER_FEATURES}
    cutoffs = [float(x) for x in (cfg.get('quintile_cutoffs') or [])]
    comp = composite_score(d, params)
    out['_composite'] = comp
    out['veto_q1'] = (assign_quintile(comp, cutoffs) == 'Q1').astype(int)

    # Catalyst: own-ticker premarket news OR an anchor cohort >= 2 that morning.
    from trading.orb_asset_class import (DEFAULT_CLASS_MAP, underlying_anchor,
                                         load_class_map)
    from trading.orb_catalyst_veto import (DEFAULT_MIN_COHORT,
                                           anchor_cohort_counts,
                                           catalyst_veto_applies)
    import csv as _csv
    import glob as _glob
    names = {}
    with open(DEFAULT_CLASS_MAP, newline='') as fh:
        for row in _csv.DictReader(fh):
            names[row['symbol']] = row.get('name', '')
    cmap = load_class_map()
    anchors = {s: underlying_anchor(s, names.get(s), cmap) for s in set(d.symbol)}
    d['_a'] = d.symbol.map(anchors)
    cohorts = {day: anchor_cohort_counts(g['_a']) for day, g in d.groupby('date')}
    raw_news = {}
    for p in sorted(_glob.glob('data/research/orb_news_catalyst_*.csv')):
        for _, r in read_orb_csv(p).iterrows():
            raw_news[(r['symbol'], r['day'])] = (r['n_articles'] or 0) > 0
    hn = [raw_news.get((s, dt)) for s, dt in zip(d.symbol, d.date)]
    out['veto_catalyst'] = [int(catalyst_veto_applies(h, a, cohorts.get(dt, {}),
                                                      DEFAULT_MIN_COHORT))
                            for h, a, dt in zip(hn, d['_a'], d.date)]
    out['news_known_pct'] = None
    print('news coverage: {:.1f}% of candidate rows have a news row '
          '(unknown fails OPEN, as in production)'.format(
              100.0 * np.mean([h is not None for h in hn])))

    for c in VETO_FLAGS:
        print(f'  {c}: fires on {out[c].sum()} / {len(out)} candidates '
              f'({100*out[c].mean():.1f}%)')

    out = out.drop(columns=['news_known_pct'])
    out.to_csv(f'{M}/meta_dataset.csv', index=False)
    print(f'\nWrote {M}/meta_dataset.csv  {out.shape}')
    print(f'features used: {len(FEATURES)}  (dropped by the audit: {DROPPED})')
    for arm in ('meas_cost', 'asis'):
        f = out[out.entered == 1]
        print(f'  R_{arm}: mean over entered {f[f"R_{arm}"].mean():+.4f}, '
              f'over all candidates {out[f"R_{arm}"].mean():+.4f}, '
              f'win rate {100*f[f"y_{arm}"].mean():.1f}%')


if __name__ == '__main__':
    main()
