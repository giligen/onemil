"""Cell 1,684: split a fresh-path features CSV (built by 1684_features.py) into its idea pools and
run study_orb_pipeline_static_lock.py on each, ORB_BT_BARS_DB=research/bf_zero/bars_sip.db (read-only;
the fresh builds' minute bars live there) + ORB_BT_DAILY_SOURCE=the matching daily_source_<window>.parquet
dumped by 1684_features.py (bars_sip.db has no daily_bars table for ATR14 to fall back to).

Usage: python3 research/orb_freq/1684_pipeline.py --window in_regime|out_regime
"""
import argparse
import logging
import os
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path('/home/ec2-user/onemil')
OUT = ROOT / 'research/orb_freq'
BARS_SIP = ROOT / 'research/bf_zero/bars_sip.db'

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger('1684pipe')

POOLS_BY_WINDOW = {'in_regime': ['idea10', 'idea11'],
                   'out_regime': ['idea1', 'idea2', 'idea10', 'idea11']}
FLAG_COL = {'idea1': 'idea1_pre', 'idea2': 'idea2', 'idea10': 'idea10', 'idea11': 'idea11'}


def latest_features_csv(window: str) -> Path:
    d = OUT / f'out_{window}'
    cands = sorted(p for p in d.glob('orb_features_*.csv') if 'corrmatrix' not in p.name)
    if not cands:
        raise SystemExit(f"FATAL: no orb_features_*.csv under {d} -- run 1684_features.py first")
    return cands[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--window', required=True, choices=['in_regime', 'out_regime'])
    a = ap.parse_args()
    window = a.window

    feat_csv = latest_features_csv(window)
    feat = pd.read_csv(feat_csv, keep_default_na=False, na_values=[''])
    log.info('%s features: %d rows from %s', window, len(feat), feat_csv)

    reads = pd.read_csv(OUT / '1684_reads.csv', keep_default_na=False, na_values=[''])
    reads = reads[reads['window'] == window][['symbol', 'bar_date'] + list(FLAG_COL.values())]
    reads = reads.rename(columns={'bar_date': 'date'})

    m = feat.merge(reads, on=['symbol', 'date'], how='left', suffixes=('', '_flag'))
    n_unflagged = m[list(set(FLAG_COL.values()))].isna().any(axis=1).sum()
    if n_unflagged:
        log.warning('%d/%d feature rows did not match a 1684_reads.csv flag row (symbol/date join miss)',
                     n_unflagged, len(m))

    daily_source = OUT / f'daily_source_{window}.parquet'
    pools_dir = OUT / f'fresh_{window}'
    pools_dir.mkdir(exist_ok=True)
    for pool in POOLS_BY_WINDOW[window]:
        col = FLAG_COL[pool]
        mask = m[col].fillna(False).astype(bool)
        sub = m.loc[mask, feat.columns]
        fp = pools_dir / f'{pool}_features.csv'
        sub.to_csv(fp, index=False)
        log.info('%s/%s: %d candidate rows -> %s', window, pool, len(sub), fp)
        if len(sub) == 0:
            log.warning('%s/%s has ZERO rows -- skipping pipeline run', window, pool)
            continue
        outp = pools_dir / f'{pool}_true.csv'
        env = dict(os.environ, ORB_BT_FEATURES_CSV=str(fp), ORB_BT_BOOK_OUT=str(outp),
                   ORB_BT_BARS_DB=str(BARS_SIP), ORB_BT_DAILY_SOURCE=str(daily_source),
                   ORB_CATALYST_VETO='0', PYTHONPATH=str(ROOT))
        log_path = pools_dir / f'{pool}_true.log'
        log.info('running pipeline for %s/%s -> %s', window, pool, outp)
        with open(log_path, 'w') as lf:
            rc = subprocess.run(['nice', '-n', '10', 'python3', 'study_orb_pipeline_static_lock.py'],
                                 cwd=str(ROOT), env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
        log.info('%s/%s pipeline rc=%d (log: %s)', window, pool, rc, log_path)


if __name__ == '__main__':
    main()
