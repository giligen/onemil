#!/usr/bin/env python3
"""lev_rebalance PASS 3 — flow proxy correction (F3). See PREREG.md Pass 3
addendum. Rescores the SAME pass-2 qualified population (p2_signal_days_qualified.csv)
with F3 = 2*|r|*wrapper_same_day_$vol/underlying_dollar_ADV20 (with the
ADV20->expanding(min5) fallback), then ranks top-decile vs rest, LONG and
COMBINED, per split. No new candidate-day selection, no re-pull.
"""
import json
import os
import sqlite3
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = '/home/ec2-user/onemil'
OUT = os.path.join(ROOT, 'research', 'lev_rebalance')
CACHE_DB = f"file:{ROOT}/data/cache.db?mode=ro"
BARS_DB = os.path.join(OUT, 'bars_1500_1600.db')

LOOKBACK_LO = '2024-11-01'
RUN_HI = '2026-05-31'


def log(*a):
    print(*a, flush=True)


def day_means(df, date_col='date', val_col='pnl_R'):
    return df.groupby(date_col)[val_col].mean()


def one_sample_day_clustered_t(df, date_col='date', val_col='pnl_R'):
    dm = day_means(df, date_col, val_col)
    if len(dm) < 2:
        return (dm.mean() if len(dm) else np.nan), np.nan, len(dm)
    t, p = stats.ttest_1samp(dm.values, 0.0)
    return dm.mean(), t, len(dm)


def main():
    universe_map = json.load(open(os.path.join(OUT, 'universe_map.json')))
    qual = pd.read_csv(os.path.join(OUT, 'p2_signal_days_qualified.csv'), parse_dates=['date'])
    qual = qual[qual['sho_gate'] == True].copy()
    log(f'pass-2 qualified, sho-clean population: {len(qual)}')

    cache_conn = sqlite3.connect(CACHE_DB, uri=True, timeout=30)
    bars_conn = sqlite3.connect(BARS_DB)

    wrappers_all = sorted({w for v in universe_map.values() for w in v})
    syms_in_bars = set(r[0] for r in bars_conn.execute('SELECT DISTINCT symbol FROM bars_1500_1600'))
    wrappers_in_bars = sorted(set(wrappers_all) & syms_in_bars)
    log(f'wrapper tickers referenced: {len(wrappers_all)}; present in bars_1500_1600.db: {len(wrappers_in_bars)}')

    # underlyings + wrappers daily bars, for the same-day $vol numerator and the ADV20 denom
    need_syms = sorted(set(qual['underlying'].unique().tolist()) | set(wrappers_all))
    parts = []
    CH = 400
    for i in range(0, len(need_syms), CH):
        chunk = need_syms[i:i + CH]
        q = (f"SELECT symbol, bar_date, close, volume FROM daily_bars WHERE symbol IN "
             f"({','.join('?' * len(chunk))}) AND bar_date >= ? AND bar_date <= ?")
        parts.append(pd.read_sql_query(q, cache_conn, params=[*chunk, LOOKBACK_LO, RUN_HI]))
    daily = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    daily['bar_date'] = pd.to_datetime(daily['bar_date']).dt.date
    daily['dollar_vol'] = daily['close'] * daily['volume']
    daily = daily.sort_values(['symbol', 'bar_date'])
    daily['adv20_strict'] = (daily.groupby('symbol')['dollar_vol']
                              .transform(lambda s: s.shift(1).rolling(20, min_periods=20).mean()))
    daily['adv_fallback'] = (daily.groupby('symbol')['dollar_vol']
                              .transform(lambda s: s.shift(1).expanding(min_periods=5).mean()))
    daily['adv20_und'] = daily['adv20_strict'].fillna(daily['adv_fallback'])
    by_sym = {s: g.set_index('bar_date') for s, g in daily.groupby('symbol')}
    wrapper_first_bar = {s: (by_sym[s].index.min() if s in by_sym and len(by_sym[s]) else None)
                          for s in wrappers_all}

    qual['date_d'] = qual['date'].dt.date
    f3_vals = []
    f3_missing_reason = []
    for _, row in qual.iterrows():
        und, d, r = row['underlying'], row['date_d'], row['r']
        wraps = universe_map.get(und, [])
        wrap_dvol_today = 0.0
        any_wrap_data = False
        for w in wraps:
            fb = wrapper_first_bar.get(w)
            if fb is None or d < fb:
                continue
            if w in by_sym and d in by_sym[w].index:
                wrap_dvol_today += float(by_sym[w].loc[d, 'dollar_vol'])
                any_wrap_data = True
        adv20_und = np.nan
        if und in by_sym and d in by_sym[und].index:
            adv20_und = by_sym[und].loc[d, 'adv20_und']
        if not any_wrap_data:
            f3_vals.append(np.nan)
            f3_missing_reason.append('no_wrapper_daily_vol')
        elif pd.isna(adv20_und) or adv20_und <= 0:
            f3_vals.append(np.nan)
            f3_missing_reason.append('no_underlying_adv')
        else:
            f3_vals.append(2.0 * abs(r) * wrap_dvol_today / float(adv20_und))
            f3_missing_reason.append('')
    qual['F3'] = f3_vals
    qual['F3_missing_reason'] = f3_missing_reason

    qual['pnl_frac'] = (qual['moc_close'] / qual['eff_entry'] - 1.0) * qual['side']
    qual['pnl_R'] = qual['pnl_frac'] / 0.02

    n_missing = int(qual['F3'].isna().sum())
    log(f'F3 missing: {n_missing}/{len(qual)} ({n_missing/max(len(qual),1):.1%})')
    reason_counts = qual.loc[qual['F3'].isna(), 'F3_missing_reason'].value_counts().to_dict()
    log('missing reasons:', reason_counts)

    qual.to_csv(os.path.join(OUT, 'p3_signal_days_scored.csv'), index=False)

    out = {'n_qualified_sho_clean': int(len(qual)), 'n_f3_missing': n_missing,
           'f3_missing_share': n_missing / max(len(qual), 1),
           'f3_missing_reasons': reason_counts}

    for split in ['TRAIN', 'VAL']:
        out[split] = {}
        for pop_name, pop_df in [('LONG', qual[(qual['split'] == split) & (qual['side'] == 1)]),
                                  ('COMBINED', qual[qual['split'] == split])]:
            pop = pop_df.dropna(subset=['F3']).copy()
            if len(pop) < 10:
                out[split][pop_name] = {'n': len(pop), 'note': 'too few for decile'}
                continue
            thresh = pop['F3'].quantile(0.90)
            top = pop[pop['F3'] >= thresh]
            rest = pop[pop['F3'] < thresh]
            top_mean, top_t, top_ndays = one_sample_day_clustered_t(top)
            rest_mean, rest_t, rest_ndays = one_sample_day_clustered_t(rest)
            out[split][pop_name] = dict(
                n_top=len(top), n_top_days=int(top_ndays), top_mean_R=float(top_mean),
                top_t=float(top_t) if not np.isnan(top_t) else None,
                top_pct_price=float(top_mean) * 0.02 * 100,
                n_rest=len(rest), n_rest_days=int(rest_ndays), rest_mean_R=float(rest_mean),
                rest_t=float(rest_t) if not np.isnan(rest_t) else None,
            )
        # quintiles on COMBINED population
        pop = qual[(qual['split'] == split)].dropna(subset=['F3']).copy()
        try:
            pop['F3_q'] = pd.qcut(pop['F3'], 5, labels=False, duplicates='drop')
            qtab = pop.groupby('F3_q')['pnl_R'].mean()
            out[split]['quintiles'] = {str(k): float(v) for k, v in qtab.items()}
            vals = [out[split]['quintiles'].get(str(i)) for i in range(5)]
            out[split]['monotone_q5_gt_q1_desc'] = (
                len([v for v in vals if v is not None]) == 5 and
                all(vals[i] > vals[i + 1] for i in range(4))
            )
        except Exception as e:
            out[split]['quintiles'] = {}
            out[split]['monotone_q5_gt_q1_desc'] = False
            out[split]['quintile_error'] = str(e)

    with open(os.path.join(OUT, 'p3_stats.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)
    log('SUMMARY', json.dumps(out, indent=2, default=str))


if __name__ == '__main__':
    main()
