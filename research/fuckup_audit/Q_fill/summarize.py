#!/usr/bin/env python3
"""Stage Q step 5 — the per-trade deliverable and the book-level decomposition."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
from trading.orb_csv import read_orb_csv          # noqa: E402

Q = f'{ROOT}/research/fuckup_audit/Q_fill'
PC = f'{ROOT}/research/fuckup_audit/P_cost'
OLD_POS = 50_000.0


def main() -> int:
    sp = pd.read_parquet(f'{PC}/spreads.parquet')
    sp['key'] = sp.symbol + '|' + sp.date
    w = pd.read_csv(f'{Q}/walk_rows.csv', keep_default_na=False, na_values=[''],
                    dtype={'symbol': str, 'date': str})
    w['key'] = w.symbol + '|' + w.date
    r = pd.read_csv(f'{Q}/delayed_resim.csv', keep_default_na=False, na_values=[''],
                    dtype={'symbol': str, 'date': str})
    r['key'] = r.symbol + '|' + r.date
    pt = pd.read_csv(f'{Q}/per_trade_arms.csv', keep_default_na=False,
                     na_values=[''])

    t = (sp[sp.cov_entry][['key', 'symbol', 'date', 'entry_price', 'range_high',
                           'range_low', 'entry_ts', 'entry_ask', 'entry_bid',
                           'exit_reason', 'entry_m']]
         .merge(w[['key', 'filled_later', 'fill_ts', 'secs_to_fill', 'min_ask',
                   'life_s']], on='key', how='left')
         .merge(r[['key', 'lag_min', 'new_exit_reason', 'new_pnl', 'tg_pnl',
                   'asis_pnl']], on='key', how='left')
         .merge(pt, on='key', how='left'))
    t['cap_raw'] = t.entry_price
    t['cap_live'] = t.entry_price.round(2)
    t['flag_raw'] = (t.entry_ask > t.cap_raw * (1 + 1e-12)).astype(int)
    t['flag_live'] = (t.entry_ask > t.cap_live + 1e-9).astype(int)
    t['fill_model'] = np.where(
        t.flag_raw == 0, 'marketable_at_trigger',
        np.where(t.filled_later == 1,
                 np.where(t.lag_min.fillna(0) < 0.5, 'rested_same_bar',
                          'rested_later_bar'), 'never_filled'))
    t.drop(columns=['entered']).to_csv(f'{Q}/per_trade_fill_models.csv', index=False)
    print(t.fill_model.value_counts().to_dict(), flush=True)
    n = len(t)
    for lab, m in [('flagged raw cap', t.flag_raw == 1),
                   ('flagged live cap (round 2dp)', t.flag_live == 1)]:
        g = t[m]
        fl = g.filled_later == 1
        print(f'{lab}: {m.sum()} of {n} ({m.mean()*100:.1f}%) | filled later '
              f'{int(fl.sum())} ({fl.mean()*100:.1f}%) | median s to fill '
              f'{g.loc[fl, "secs_to_fill"].median():.1f} | later BAR '
              f'{int((g.lag_min >= 0.5).sum())} | never {int((~fl).sum())}',
              flush=True)

    # ---- book-level decomposition (8 and 3 slots) -------------------------
    for nslot in (8, 3):
        b = read_orb_csv(f'{Q}/book_asis_n{nslot}.csv')
        b['date'] = pd.to_datetime(b['date']).dt.strftime('%Y-%m-%d')
        b['key'] = b.symbol + '|' + b.date
        m = b[['key', '_rp_position', 'entry_price', 'entered']].merge(
            t[['key', 'fill_model', 'pnl_asis', 'pnl_strict', 'pnl_measured']],
            on='key', how='left')
        m['scale'] = m._rp_position / OLD_POS
        print(f'\n--- {nslot}-slot book: {len(b)} picks ---', flush=True)
        for mod in ('marketable_at_trigger', 'rested_same_bar',
                    'rested_later_bar', 'never_filled'):
            g = m[m.fill_model == mod]
            d = ((g.pnl_measured - g.pnl_asis) * g.scale).sum()
            print(f'  {mod:<24} picks {len(g):>4}  sized measured-vs-asis '
                  f'{d:>10,.0f}', flush=True)
        miss = m[m.fill_model.isna()]
        print(f'  (no measured quote / no fill row): {len(miss)} picks, carried at as-is',
              flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
