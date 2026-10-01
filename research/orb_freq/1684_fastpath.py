"""Cell 1,684 fast path: production reference + idea1 + idea2, IN-REGIME ONLY, built by filtering
the ALREADY-BUILT wide-seed features CSVs (gap>=3%, $3-50, research/orb_seed_wide/out/
orb_features_20260920_2142.csv Jan25-May26 + orb_features_20260921_1842.csv Jun-Sep26) instead of
running study_orb_features again. No new minute-bar work: idea2's test (gap vs prior HIGH >= 3%)
is ALWAYS a subset of gap vs prior CLOSE >= 3% (prior_high >= prior_close algebraically), and
idea1's [3,5)% gap slice is already inside the wide seed's >=3% net; idea1's <3% extension and both
ideas' out-of-regime half are out of scope for this budget (stated in RESULT_1684.md).

Joins data/cache.db::daily_bars (read-only) by (symbol,date) onto the wide CSV rows to get the
actual open/prev_close/prev_high/prev_volume (the wide CSV itself carries only derived %s), then
writes one features-CSV subset per {prod, idea1, idea2} and runs study_orb_pipeline_static_lock.py
on each (ORB_CATALYST_VETO=0, ORB_BT_BARS_DB left at its default data/cache.db -- these are all
gap-up names inside the wide-seed's own already-working population, unlike idea10/idea11).
"""
import logging
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path('/home/ec2-user/onemil')
OUT = ROOT / 'research/orb_freq'
CACHE_DB = ROOT / 'data/cache.db'
WIDE_CSVS = [ROOT / 'research/orb_seed_wide/out/orb_features_20260920_2142.csv',
             ROOT / 'research/orb_seed_wide/out/orb_features_20260921_1842.csv']

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger('1684fast')


def main():
    frames = [pd.read_csv(p, keep_default_na=False, na_values=['']) for p in WIDE_CSVS]
    wide = pd.concat(frames, ignore_index=True).drop_duplicates(['symbol', 'date'])
    log.info('wide seed rows (gap>=3%%, $3-50): %d  (%s..%s)', len(wide), wide.date.min(), wide.date.max())

    syms = sorted(wide.symbol.unique())
    con = sqlite3.connect(f'file:{CACHE_DB}?mode=ro', uri=True)
    ph = ','.join('?' * len(syms))
    daily = pd.read_sql_query(
        f"SELECT symbol, bar_date, open, high, low, close, volume FROM daily_bars "
        f"WHERE symbol IN ({ph}) AND bar_date >= '2024-11-01'", con, params=syms)
    con.close()
    daily = daily.sort_values(['symbol', 'bar_date']).reset_index(drop=True)
    g = daily.groupby('symbol')
    daily['prev_close'] = g['close'].shift(1)
    daily['prev_high'] = g['high'].shift(1)
    daily['prev_volume'] = g['volume'].shift(1)
    daily['gap_pct_check'] = (daily['open'] - daily['prev_close']) / daily['prev_close'] * 100
    daily['gap_vs_high_pct'] = (daily['open'] - daily['prev_high']) / daily['prev_high'] * 100

    m = wide.merge(daily[['symbol', 'bar_date', 'open', 'prev_close', 'prev_high', 'prev_volume',
                           'gap_pct_check', 'gap_vs_high_pct']],
                    left_on=['symbol', 'date'], right_on=['symbol', 'bar_date'], how='left')
    n_nojoin = m['open'].isna().sum()
    log.info('join: %d/%d rows matched daily_bars (%d unmatched)', len(m) - n_nojoin, len(m), n_nojoin)
    drift = (m['gap_pct'] - m['gap_pct_check']).abs()
    log.info('gap_pct sanity vs re-derived: median abs diff %.4f, max %.4f, >1pp: %d',
              drift.median(), drift.max(), (drift > 1.0).sum())
    m = m.dropna(subset=['open', 'prev_close'])

    band = (m['open'] >= 3) & (m['open'] <= 30) & (m['prev_volume'] >= 500_000)
    is_prod = band & (m['gap_pct'] >= 5.0)
    idea1 = band & (m['gap_pct'] >= 3.0) & (m['gap_pct'] < 5.0) & ~is_prod \
        & ((m['entry_price'] - m['prev_close']) / m['prev_close'] * 100 >= 5.0)
    idea2 = band & (m['gap_vs_high_pct'] >= 3.0) & ~is_prod
    log.info('prod=%d idea1(gap[3,5)+range>=5pct-by-0935)=%d idea2(gap-vs-prior-high>=3pct)=%d '
              'overlap idea1&idea2=%d', is_prod.sum(), idea1.sum(), idea2.sum(), (idea1 & idea2).sum())

    pools_dir = OUT / 'fastpath'
    pools_dir.mkdir(exist_ok=True)
    orig_cols = wide.columns.tolist()
    for name, mask in [('prod', is_prod), ('idea1', idea1), ('idea2', idea2)]:
        sub = m.loc[mask, orig_cols]
        fp = pools_dir / f'{name}_features.csv'
        sub.to_csv(fp, index=False)
        log.info('%s: %d candidate rows -> %s', name, len(sub), fp)

    for name in ('prod', 'idea1', 'idea2'):
        fp = pools_dir / f'{name}_features.csv'
        outp = pools_dir / f'{name}_true.csv'
        env = dict(os.environ, ORB_BT_FEATURES_CSV=str(fp), ORB_BT_BOOK_OUT=str(outp),
                   ORB_CATALYST_VETO='0', PYTHONPATH=str(ROOT))
        log_path = pools_dir / f'{name}_true.log'
        log.info('running pipeline for %s -> %s', name, outp)
        with open(log_path, 'w') as lf:
            rc = subprocess.run(['nice', '-n', '10', 'python3', 'study_orb_pipeline_static_lock.py'],
                                 cwd=str(ROOT), env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
        log.info('%s pipeline rc=%d (log: %s)', name, rc, log_path)


if __name__ == '__main__':
    main()
