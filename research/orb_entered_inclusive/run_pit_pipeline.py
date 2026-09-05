"""ORB — run the pipeline on production features ∪ point-in-time additions (2026-09-05).

Steps:
  1. union the latest production features CSV (analysis_results/orb_features_*.csv,
     the entered-inclusive rebuild) with the PIT features CSV produced by
     `ORB_CACHE_DB=pit_cache.db ORB_FEATURES_OUT_DIR=<dir> study_orb_features.py`
     (dedup on symbol+date, production row wins),
  2. run study_orb_pipeline_static_lock on the union with every output redirected
     (ORB_BT_FEATURES_CSV / ORB_BT_BOOK_OUT / ORB_BT_MONTHLY_OUT) so nothing in
     analysis_results/ changes,
  3. print the compare_books diff vs the production book.

Usage: python3 run_pit_pipeline.py PIT_FEATURES.csv [PROD_FEATURES.csv]
"""
import glob
import os
import subprocess
import sys

import pandas as pd
sys.path.insert(0, '/home/ec2-user/onemil')
from trading.orb_csv import read_orb_csv  # noqa: E402  ticker-safe reader (NA/NAN/NULL are symbols)

ROOT = '/home/ec2-user/onemil'
D = f'{ROOT}/research/orb_entered_inclusive'


def latest_prod_features() -> str:
    """Most recent production features CSV (corrmatrix sidecar excluded)."""
    return sorted(p for p in glob.glob(f'{ROOT}/analysis_results/orb_features_*.csv')
                  if 'corrmatrix' not in p)[-1]


def main() -> None:
    pit_csv = sys.argv[1]
    prod_csv = sys.argv[2] if len(sys.argv) > 2 else latest_prod_features()
    prod, pit = read_orb_csv(prod_csv), read_orb_csv(pit_csv)
    key_prod = set(zip(prod['symbol'], prod['date'].astype(str)))
    new = pit[[k not in key_prod for k in zip(pit['symbol'], pit['date'].astype(str))]]
    union = pd.concat([prod, new], ignore_index=True)
    union_csv = f'{D}/orb_features_union_pit.csv'
    union.to_csv(union_csv, index=False)
    print(f"prod {prod_csv}: {len(prod):,} rows | PIT {pit_csv}: {len(pit):,} rows, "
          f"{len(new):,} new keys | union -> {union_csv} ({len(union):,})")

    env = dict(os.environ, ORB_BT_FEATURES_CSV=union_csv,
               ORB_BT_BOOK_OUT=f'{D}/orb_bplus_book_pit.csv',
               ORB_BT_MONTHLY_OUT=f'{D}/orb_monthly_static_lock_pit.csv')
    with open(f'{ROOT}/logs/orb_pipeline_pit_20260905.log', 'w') as log:
        rc = subprocess.call([sys.executable, 'study_orb_pipeline_static_lock.py'],
                             cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    print(f"pipeline exit {rc} (log logs/orb_pipeline_pit_20260905.log)")
    if rc != 0:
        sys.exit(rc)
    subprocess.call([sys.executable, f'{D}/compare_books.py',
                     f'{ROOT}/analysis_results/orb_bplus_book.csv', f'{D}/orb_bplus_book_pit.csv',
                     'prod_entered_inclusive', 'prod+PIT'], cwd=ROOT)


if __name__ == '__main__':
    main()
