#!/usr/bin/env python3
"""Stage N1 step 3b — search-adjusted permutation p for the best cell.

Null: the feature carries no information about which candidate is which. It is
imposed by shuffling the feature column WITHIN each date (day structure, the
feature's marginal distribution and the candidate set all preserved), then
re-running the SAME cell through the SAME pipeline path. The statistic is the
cell's TRAIN t of R/pick. p = (1 + #{perm t >= observed t}) / (1 + n_perm),
and it is reported against the whole 12-cell grid (max-t over the grid is the
search adjustment discussed in the report).

Usage: python3 perm.py --cell comp_ofi_range --n 200
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '/home/ec2-user/onemil')
os.chdir('/home/ec2-user/onemil')
from trading.orb_csv import read_orb_csv  # noqa: E402

sys.path.insert(0, '/home/ec2-user/onemil/research/fuckup_audit/N_databento/N1')
from cells import FEATS, N1, D1, BOOKS, enrich, split_stats, SPLITS  # noqa: E402

TMP = f'{N1}/perm_tmp'


def env_for(cell: str) -> dict:
    kind, feat = cell.split('_', 1)
    sign = dict(FEATS)[feat]
    if kind == 'comp':
        return {'ORB_N1_COMPOSITE_FEATURE': feat, 'ORB_N1_COMPOSITE_SIGN': str(sign)}
    return {'ORB_N1_VETO_FEATURE': feat,
            'ORB_N1_VETO_SIDE': 'low' if sign > 0 else 'high',
            'ORB_N1_VETO_Q': '0.2'}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', required=True)
    ap.add_argument('--n', type=int, default=200)
    a = ap.parse_args()
    os.makedirs(TMP, exist_ok=True)
    feat = a.cell.split('_', 1)[1]
    base_sc = pd.read_csv(f'{N1}/sidecar.csv', keep_default_na=False, na_values=[''])
    out_csv = f'{N1}/perm_{a.cell}.csv'
    done = pd.read_csv(out_csv)['t'].tolist() if os.path.exists(out_csv) else []
    rng = np.random.RandomState(20260918 + len(done))
    lo, hi = SPLITS['TRAIN']
    for i in range(len(done), a.n):
        sc = base_sc.copy()
        sc[feat] = sc.groupby('date')[feat].transform(
            lambda s: s.to_numpy()[rng.permutation(len(s))])
        p_sc = f'{TMP}/sidecar_{a.cell}.csv'
        sc.to_csv(p_sc, index=False)
        book = f'{TMP}/book.csv'
        if os.path.exists(book):
            os.remove(book)
        env = dict(os.environ)
        env.update({
            'ORB_BT_FEATURES_CSV': 'analysis_results/orb_features_20260916_2053.csv',
            'ORB_BT_RESIM_CACHE': f'{D1}/candidates_dump.csv',
            'ORB_BT_RISK': '375', 'ORB_BT_N': '8',
            'ORB_BT_ACCOUNT': repr(3333.333333333333 * 8), 'ORB_SKIP_Q1': '1',
            'ORB_BT_BOOK_OUT': book, 'ORB_BT_MONTHLY_OUT': f'{TMP}/monthly.csv',
            'ORB_BT_SIDECAR_CSV': p_sc})
        env.update(env_for(a.cell))
        with open(f'{TMP}/log.txt', 'w') as fh:
            rc = subprocess.call(['nice', '-n', '10', 'python3', '-u',
                                  'study_orb_pipeline_static_lock.py'],
                                 env=env, stdout=fh, stderr=subprocess.STDOUT)
        if rc != 0:
            raise SystemExit(f"perm {i} failed — see {TMP}/log.txt")
        t = split_stats(enrich(read_orb_csv(book)), lo, hi)['t']
        done.append(t)
        pd.DataFrame({'t': done}).to_csv(out_csv, index=False)
        if (i + 1) % 10 == 0:
            print(f"  perm {i + 1}/{a.n}  last t={t:.3f}", flush=True)
    obs = split_stats(enrich(read_orb_csv(f'{BOOKS}/book_{a.cell}.csv')), lo, hi)['t']
    arr = np.array([x for x in done if np.isfinite(x)])
    p = (1 + int((arr >= obs).sum())) / (1 + len(arr))
    print(f"CELL {a.cell}: observed TRAIN t={obs:.3f}; {len(arr)} perms; "
          f"perm t mean={arr.mean():.3f} sd={arr.std(ddof=1):.3f} "
          f"p95={np.percentile(arr, 95):.3f}; p={p:.4f}", flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
