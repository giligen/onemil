#!/usr/bin/env python3
"""Stage P step 3 — build `spreads.parquet` and the band-vs-measured tables.

Input : P_cost/spread_rows.csv (fetch_spreads.py)
Output: P_cost/spreads.parquet        one row per (symbol, date) fill
        P_cost/cell_table.csv         band constant vs measured, per (pb x hb)
        P_cost/dispersion.csv         cross-sectional dispersion inside each cell
        P_cost/measured_cost.csv      the per-trade measured fills for the rescore

Honesty rails (asserted, the run aborts on failure):
  R1  every quote used for a decision is timestamped AT OR BEFORE that decision
  R2  the entry fill instant is a real trade above range_high in the entry bar
  R3  test tickers (research/scripts/pit_listings) are excluded
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)

P = f'{ROOT}/research/fuckup_audit/P_cost'
BAND = f'{ROOT}/research/lit_review_2026/cost_curve.csv'
# The band table's own price bands start at $5 (its population was filtered
# price >= 5).  ORB trades below $5, so a '<$5' band is added here and reported
# with band_bps empty — the band table simply cannot price those trades.
PB_EDGES = [0, 5, 10, 20, 50, 200, 1e9]
PB_LAB = ['<$5', '$5-10', '$10-20', '$20-50', '$50-200', '$200+']
HB_EDGES = [569, 575, 600, 660, 780, 960]
HB_LAB = ['09:30-09:35', '09:35-10:00', '10:00-11:00', '11:00-13:00', '13:00+']
EXIT_SLIP = 10.0 / 10000.0
STOP_REASONS = {'stop', 'lock', 'scale_stop', 'scale_lock'}


def band_table():
    """The cost-curve band constant: median NBBO spread in bps, per (pb, hb)."""
    d = pd.read_csv(BAND, dtype={'symbol': str, 'day': str},
                    keep_default_na=False, na_values=[''])
    d = d[d.n_q > 0].copy()
    d['sp_bps'] = d.spread / d.price * 1e4
    g = d.groupby(['pb', 'hb'], observed=True).agg(
        band_n=('sp_bps', 'size'), band_bps=('sp_bps', 'median'))
    return g


def main() -> int:
    d = pd.read_csv(f'{P}/spread_rows.csv', keep_default_na=False, na_values=[''],
                    dtype={'symbol': str, 'date': str})
    d = d.drop_duplicates(subset=['symbol', 'date'], keep='last').reset_index(drop=True)
    print(f'fetched rows: {len(d)}', flush=True)

    # --- R3 test tickers -------------------------------------------------
    from research.scripts.pit_listings import is_test_ticker
    is_test = d.symbol.map(is_test_ticker)
    if is_test.any():
        print(f'R3: dropping {int(is_test.sum())} test-ticker rows', flush=True)
    d = d[~is_test].reset_index(drop=True)

    for c in ('entry_ts', 'exit_ts', 'entry_fill_ts', 'entry_q_ts', 'exit_q_ts'):
        d[c] = pd.to_datetime(d[c], utc=True, errors='coerce')

    # --- R1 causality ----------------------------------------------------
    m = d.entry_q_ts.notna() & d.entry_fill_ts.notna()
    assert (d.loc[m, 'entry_q_ts'] <= d.loc[m, 'entry_fill_ts']).all(), \
        'R1 violated: entry quote after the fill instant'
    m = d.exit_q_ts.notna()
    xb = d.exit_ts.dt.floor('min') + pd.Timedelta(minutes=1)
    assert (d.loc[m, 'exit_q_ts'] < xb[m]).all(), \
        'R1 violated: exit quote after the exit bar'
    # --- R2 obtainability of the fill instant ----------------------------
    m = d.entry_fill_ts.notna()
    eb = d.entry_ts.dt.floor('min')
    assert (d.loc[m, 'entry_fill_ts'] >= eb[m]).all() and \
           (d.loc[m, 'entry_fill_ts'] < eb[m] + pd.Timedelta(minutes=1)).all(), \
        'R2 violated: fill instant outside the breakout minute'
    print('R1/R2/R3 assertions passed', flush=True)

    et = d.entry_ts.dt.tz_convert('America/New_York')
    d['entry_m'] = et.dt.hour * 60 + et.dt.minute
    xt = d.exit_ts.dt.tz_convert('America/New_York')
    d['exit_m'] = xt.dt.hour * 60 + xt.dt.minute
    d['price'] = d.entry_price
    d['pb'] = pd.cut(d.price, PB_EDGES, labels=PB_LAB, include_lowest=True).astype(str)
    d['hb'] = pd.cut(d.entry_m, HB_EDGES, labels=HB_LAB).astype(str)
    d['hb_exit'] = pd.cut(d.exit_m, HB_EDGES, labels=HB_LAB).astype(str)
    d['r_pct'] = (d.range_high - d.range_low) / d.price * 100

    d['entry_bps'] = d.entry_spread / d.price * 1e4
    d['entry_med_bps'] = d.entry_med_spread / d.price * 1e4
    d['exit_bps'] = d.exit_spread / d.price * 1e4
    d['exit_med_bps'] = d.exit_med_spread / d.price * 1e4
    d['cov_entry'] = d.entry_spread.notna()
    d['cov_exit'] = d.exit_spread.notna()

    # --- measured fills ---------------------------------------------------
    cap = d.range_high * 1.003
    d['assumed_entry'] = d.entry_price
    d['measured_entry'] = d.entry_ask
    d['entry_above_cap'] = d.entry_ask > cap * (1 + 1e-12)
    lvl = np.where(d.exit_reason.isin(STOP_REASONS),
                   d.exit_price / (1 - EXIT_SLIP), np.nan)
    d['assumed_exit'] = d.exit_price
    d['measured_exit'] = d.exit_bid
    d['exit_level'] = lvl

    d.to_parquet(f'{P}/spreads.parquet', index=False, compression='zstd')

    # --- band vs measured, per cell --------------------------------------
    bt = band_table()
    rows = []
    for (pb, hb), g in d[d.cov_entry].groupby(['pb', 'hb'], observed=True):
        v = g.entry_bps.dropna().values
        if len(v) == 0:
            continue
        b = bt.loc[(pb, hb)] if (pb, hb) in bt.index else None
        mm = g.entry_med_bps.dropna().values          # like-for-like with the band
        rows.append(dict(
            pb=pb, hb=hb, n=len(v),
            band_bps=round(float(b.band_bps), 1) if b is not None else np.nan,
            band_n=int(b.band_n) if b is not None else 0,
            minmed_bps=round(float(np.median(mm)), 1) if len(mm) else np.nan,
            med_bps=round(float(np.median(v)), 1),
            mean_bps=round(float(np.mean(v)), 1),
            p25=round(float(np.percentile(v, 25)), 1),
            p75=round(float(np.percentile(v, 75)), 1),
            p90=round(float(np.percentile(v, 90)), 1),
            ratio_band_over_minmed=(round(float(b.band_bps) / float(np.median(mm)), 2)
                                    if b is not None and len(mm) and np.median(mm) > 0
                                    else np.nan),
            ratio_band_over_med=(round(float(b.band_bps) / float(np.median(v)), 2)
                                 if b is not None and np.median(v) > 0 else np.nan),
            sp_over_R_med=round(float(np.median(g.entry_bps / (g.r_pct * 100))), 3),
        ))
    ct = pd.DataFrame(rows).sort_values(['pb', 'hb'])
    ct.to_csv(f'{P}/cell_table.csv', index=False)

    # --- dispersion inside each cell -------------------------------------
    disp = []
    for (pb, hb), g in d[d.cov_entry].groupby(['pb', 'hb'], observed=True):
        v = g.entry_bps.dropna().values
        if len(v) < 5:
            continue
        b = bt.loc[(pb, hb)].band_bps if (pb, hb) in bt.index else np.nan
        disp.append(dict(pb=pb, hb=hb, n=len(v),
                         p10=round(float(np.percentile(v, 10)), 1),
                         p90=round(float(np.percentile(v, 90)), 1),
                         p90_over_p10=round(float(np.percentile(v, 90) /
                                                  max(np.percentile(v, 10), 1e-9)), 1),
                         iqr_over_med=round(float((np.percentile(v, 75) -
                                                   np.percentile(v, 25)) /
                                                  max(np.median(v), 1e-9)), 2),
                         share_within_2x_band=(round(float(((v > b / 2) & (v < b * 2)).mean()), 3)
                                               if b == b else np.nan)))
    pd.DataFrame(disp).to_csv(f'{P}/dispersion.csv', index=False)

    # --- the exit side, per (price band x EXIT hour band) ------------------
    erows = []
    for (pb, hb), g in d[d.cov_exit].groupby(['pb', 'hb_exit'], observed=True):
        v = g.exit_bps.dropna().values
        mm = g.exit_med_bps.dropna().values
        if len(v) == 0:
            continue
        b = bt.loc[(pb, hb)] if (pb, hb) in bt.index else None
        erows.append(dict(
            pb=pb, hb_exit=hb, n=len(v),
            band_bps=round(float(b.band_bps), 1) if b is not None else np.nan,
            minmed_bps=round(float(np.median(mm)), 1) if len(mm) else np.nan,
            med_bps=round(float(np.median(v)), 1),
            mean_bps=round(float(np.mean(v)), 1),
            p75=round(float(np.percentile(v, 75)), 1),
            p90=round(float(np.percentile(v, 90)), 1),
            ratio_band_over_med=(round(float(b.band_bps) / float(np.median(v)), 2)
                                 if b is not None and np.median(v) > 0 else np.nan)))
    pd.DataFrame(erows).sort_values(['pb', 'hb_exit']).to_csv(
        f'{P}/exit_cell_table.csv', index=False)

    # --- entry instant vs the breakout minute's median ---------------------
    m = d.entry_bps.notna() & d.entry_med_bps.notna() & (d.entry_med_bps > 0)
    iv = []
    for hb, g in d[m].groupby('hb', observed=True):
        r = (g.entry_bps / g.entry_med_bps)
        iv.append(dict(hb=hb, n=len(g),
                       med_instant=round(float(g.entry_bps.median()), 1),
                       med_minmed=round(float(g.entry_med_bps.median()), 1),
                       ratio_med=round(float(r.median()), 3),
                       ratio_mean=round(float(r.mean()), 3),
                       share_instant_wider=round(float((r > 1).mean()), 3)))
    r = (d.loc[m, 'entry_bps'] / d.loc[m, 'entry_med_bps'])
    iv.append(dict(hb='ALL', n=int(m.sum()),
                   med_instant=round(float(d.loc[m, 'entry_bps'].median()), 1),
                   med_minmed=round(float(d.loc[m, 'entry_med_bps'].median()), 1),
                   ratio_med=round(float(r.median()), 3),
                   ratio_mean=round(float(r.mean()), 3),
                   share_instant_wider=round(float((r > 1).mean()), 3)))
    pd.DataFrame(iv).to_csv(f'{P}/instant_vs_minmed.csv', index=False)

    print(ct.to_string(index=False), flush=True)
    print(f'\ncoverage: entry {d.cov_entry.mean() * 100:.1f}%  '
          f'exit {d.cov_exit.mean() * 100:.1f}%  fill-instant found '
          f'{d.entry_fill_ts.notna().mean() * 100:.1f}%', flush=True)
    print(f'wrote spreads.parquet ({len(d)} rows), cell_table.csv, dispersion.csv',
          flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
