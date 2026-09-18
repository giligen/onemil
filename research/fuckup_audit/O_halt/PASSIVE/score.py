#!/usr/bin/env python3
"""S1-PASSIVE step 4 — score the 6 pre-declared cells. Contract frozen in PREREG.md.

net_R = raw_pct / 2.0  -  0.0 * half_entry  -  1.0 * half_exit      (PREREG §4)
  raw_pct    = (fill_px - cover_px) / fill_px * 100          (SHORT)
  half_exit  = 0.5 * (exit_spread / cover_px * 100) / 2.0
  half_entry = 0.5 * (entry_mean_spread / fill_px * 100) / 2.0   (charged 0; reported)
The O_halt charge (0.25*half_entry + 0.875*half_exit) and the O_halt cover convention
(horizon bar CLOSE instead of the next bar's OPEN) are carried as comparison arms.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
P = f'{ROOT}/research/fuckup_audit/O_halt/PASSIVE'
R_PCT = 2.0
SPLITS = [('TRAIN', '2025-01-01', '2025-12-31'), ('VAL', '2026-01-01', '2026-05-31'),
          ('TEST', '2026-06-01', '2026-12-31')]


def split_of(day):
    return np.where(day <= '2025-12-31', 'TRAIN', np.where(day <= '2026-05-31', 'VAL', 'TEST'))


def stats(x):
    n = len(x)
    if n < 3:
        return dict(n=n, mean=np.nan, t=np.nan, mde=np.nan)
    se = float(np.std(x, ddof=1)) / np.sqrt(n)
    return dict(n=n, mean=float(np.mean(x)), t=float(np.mean(x) / se) if se else np.nan,
                mde=float(2.8 * se))


def weeks_green(df, col='net_R'):
    if df.empty:
        return np.nan, 0
    g = df.groupby(pd.to_datetime(df['day']).dt.to_period('W'))[col].sum()
    return float((g > 0).mean()), len(g)


def main() -> int:
    s = pd.read_parquet(f'{P}/sim_rows.parquet')
    cq = pd.read_csv(f'{P}/cover_nbbo.csv', keep_default_na=False, na_values=[''],
                     dtype={'symbol': str, 'day': str})
    cq['cov_spread'] = pd.to_numeric(cq['spread'], errors='coerce')
    cq = cq.drop_duplicates(subset=['symbol', 'day', 'ts'], keep='last')

    s['cov_h5_t'] = pd.to_datetime(s['cov_h5_t'], utc=True, errors='coerce')
    s['ts'] = s['cov_h5_t'].apply(lambda x: x.isoformat() if pd.notna(x) else '')
    s = s.merge(cq[['symbol', 'day', 'ts', 'cov_spread']], on=['symbol', 'day', 'ts'], how='left')

    bf = pd.read_csv(f'{P}/borrow_flags.csv', keep_default_na=False, na_values=[''],
                     dtype={'symbol': str})
    bf['tradeable'] = bf['shortable'].astype(str).str.lower().eq('true') & \
        bf['easy_to_borrow'].astype(str).str.lower().eq('true')
    s = s.merge(bf[['symbol', 'tradeable']], on='symbol', how='left')
    s['tradeable'] = s['tradeable'].fillna(False)

    s['split'] = split_of(s['day'].values)
    f = s[s['filled'] == 1].copy()
    # cover-spread fallback: the entry-minute mean spread, flagged
    f['cov_spread_src'] = np.where(f['cov_spread'].notna(), 'measured', 'entry_fallback')
    f['cov_spread'] = f['cov_spread'].fillna(f['mean_spread'])

    f['half_entry'] = 0.5 * (f['mean_spread'] / f['fill_px'] * 100.0) / R_PCT
    f['half_exit'] = 0.5 * (f['cov_spread'] / f['cov_h5'] * 100.0) / R_PCT
    f['raw_pct'] = (f['fill_px'] - f['cov_h5']) / f['fill_px'] * 100.0
    f['gross_R'] = f['raw_pct'] / R_PCT
    f['net_R'] = f['gross_R'] - 1.0 * f['half_exit']
    # comparison arms
    f['net_R_ohalt_cost'] = f['gross_R'] - 0.25 * f['half_entry'] - 0.875 * f['half_exit']
    f['raw_close'] = (f['fill_px'] - f['close_h5']) / f['fill_px'] * 100.0
    f['net_R_closecov'] = f['raw_close'] / R_PCT - 1.0 * f['half_exit']
    for hz in ('h30', 'eod'):
        col = f'cov_{hz}'
        f[f'net_R_{hz}'] = (f['fill_px'] - f[col]) / f['fill_px'] * 100.0 / R_PCT - 1.0 * f['half_exit']

    f.to_parquet(f'{P}/scored_trades.parquet', index=False)

    rows = []
    for (b, arm), g in s.groupby(['b', 'arm']):
        for split, a, z in SPLITS:
            gs = g[g['split'] == split]
            fs = f[(f['b'] == b) & (f['arm'] == arm) & (f['split'] == split)]
            st = stats(fs['net_R'].values)
            wg, nw = weeks_green(fs)
            x = np.sort(fs['net_R'].values)
            ex5 = float(np.mean(x[:int(len(x) * 0.95)])) if len(x) > 20 else np.nan
            cap3 = float(np.mean(np.minimum(x, 3.0))) if len(x) > 2 else np.nan
            tr = fs[fs['tradeable']]
            rows.append(dict(
                b=b, arm=arm, split=split, candidates=len(gs),
                filled=int(gs['filled'].sum()),
                fill_rate=round(float(gs['filled'].mean()), 4) if len(gs) else np.nan,
                n=st['n'], mean_netR=st['mean'], t=st['t'], mde=st['mde'],
                gross_R=float(fs['gross_R'].mean()) if len(fs) else np.nan,
                cost_R=float(fs['half_exit'].mean()) if len(fs) else np.nan,
                weeks_green=wg, weeks=nw, ex_top5=ex5, cap3R=cap3,
                trades_wk=round(st['n'] / nw, 2) if nw else np.nan,
                netR_ohalt_cost=float(fs['net_R_ohalt_cost'].mean()) if len(fs) else np.nan,
                netR_closecov=float(fs['net_R_closecov'].mean()) if len(fs) else np.nan,
                netR_h30=float(fs['net_R_h30'].mean()) if len(fs) else np.nan,
                netR_eod=float(fs['net_R_eod'].mean()) if len(fs) else np.nan,
                tradeable_share=round(float(fs['tradeable'].mean()), 4) if len(fs) else np.nan,
                n_trade=len(tr), mean_netR_trade=float(tr['net_R'].mean()) if len(tr) else np.nan,
                t_trade=stats(tr['net_R'].values)['t'],
            ))
    res = pd.DataFrame(rows).sort_values(['b', 'arm', 'split'])
    res.to_csv(f'{P}/cells.csv', index=False)
    with pd.option_context('display.width', 260, 'display.max_columns', 40):
        for split, _, _ in SPLITS:
            print(f'\n===== {split} =====')
            print(res[res.split == split].drop(columns=['split']).round(4).to_string(index=False))

    # search-adjusted permutation on TRAIN, max |t| over the 6 cells
    rng = np.random.default_rng(7)
    cells = [f[(f['b'] == b) & (f['arm'] == arm) & (f['split'] == 'TRAIN')]['net_R'].values
             for b in sorted(s['b'].unique()) for arm in ('touch', 'strict')]
    obs = max(abs(stats(c)['t']) for c in cells if len(c) > 2)
    B, hits = 2000, 0
    for _ in range(B):
        mx = 0.0
        for c in cells:
            if len(c) < 3:
                continue
            y = c * rng.choice([-1.0, 1.0], size=len(c))
            se = np.std(y, ddof=1) / np.sqrt(len(y))
            mx = max(mx, abs(np.mean(y) / se) if se else 0.0)
        hits += mx >= obs
    perm_p = (hits + 1) / (B + 1)
    print(f'\nTRAIN max|t| = {obs:.2f}  search-adjusted permutation p = {perm_p:.4f} (6 cells, B={B})')

    # per-month, per cell
    f['month'] = f['day'].str.slice(0, 7)
    mt = f.groupby(['b', 'arm', 'month'])['net_R'].agg(['size', 'sum', 'mean']).round(3)
    mt.to_csv(f'{P}/monthly.csv')

    # capacity: shares = 1% of the fill-bar volume
    f['shares_cap'] = 0.01 * f['fill_vol']
    f['notional_cap'] = f['shares_cap'] * f['fill_px']
    cap = f.groupby(['b', 'arm'])[['shares_cap', 'notional_cap']].median().round(1)
    cap.to_csv(f'{P}/capacity.csv')
    print('\nmedian capacity (1% of fill-bar volume):')
    print(cap.to_string())

    json.dump({'train_max_abs_t': round(float(obs), 4), 'perm_p_6cells': round(float(perm_p), 4),
               'n_sim_rows': int(len(s)),
               'cov_spread_measured_share': round(float((f['cov_spread_src'] == 'measured').mean()), 4)},
              open(f'{P}/score_summary.json', 'w'), indent=1)
    print('cover-spread measured share by split:')
    print(f.groupby('split')['cov_spread_src'].value_counts(normalize=True).round(4).to_string())
    return 0


if __name__ == '__main__':
    sys.exit(main())
