#!/usr/bin/env python3
"""Cell runner for research/orb_gates2 (PREREG §2).

Every cell is the SHIPPED pipeline (`study_orb_pipeline_static_lock.py`) replayed
off a candidate dump with `ORB_BT_RESIM_CACHE`; only documented env overrides move.
No threshold, z-param, quintile cutoff or adaptive mult is ever refit.  Nothing
outside research/orb_gates2/ is written.

G6 and G8 are NOT run here — they are row-exact derivations of G3 and G5 under the
shipped catalyst helper at min_cohort=1 (see partial_catalyst.py), because the
catalyst veto is post-selection with no refill.
"""
from __future__ import annotations

import os
import subprocess
import sys

ROOT = '/home/ec2-user/onemil'
D = f'{ROOT}/research/orb_gates2'
FEATURES = f'{ROOT}/analysis_results/orb_features_20260916_2053.csv'

DUMPS = {
    'meas': f'{ROOT}/research/fuckup_audit/Q_fill/dump_measured.csv',    # PRIMARY
    'asis': f'{ROOT}/research/fuckup_audit/D1_orb/candidates_dump.csv',  # bracket
}
PER_POS = 3333.333333333333          # invariant per-position cap (D1 convention)
N_SLOTS = 8                          # settled in stage 1; not re-opened

CELLS = {
    'G0': {},                                                    # shipped B+
    'G1': {'ORB_SKIP_Q1': '0'},                                  # Q1 filter OFF
    'G2': {'ORB_RANGE_SIZE_VETO': '0'},                          # range-size OFF
    'G3': {'ORB_CATALYST_VETO': '0'},                            # catalyst OFF
    'G4': {'ORB_SKIP_Q1': '0', 'ORB_RANGE_SIZE_VETO': '0'},      # both wrong-side
    'G5': {'ORB_SKIP_Q1': '0', 'ORB_RANGE_SIZE_VETO': '0',
           'ORB_CATALYST_VETO': '0'},                            # everything flagged
    'G7': {'ORB_PDR_VETO': '0'},                                 # PDR redundancy
    'G9': {'ORB_SKIP_Q1': '0', 'ORB_RANGE_SIZE_VETO': '0',
           'ORB_PDR_VETO': '0'},                                 # defect cleanup
}


def run(tag: str, env: dict, dump: str) -> str:
    book = f'{D}/book_{tag}_{dump}.csv'
    e = dict(os.environ)
    e.update({
        'ORB_BT_FEATURES_CSV': FEATURES,
        'ORB_BT_RESIM_CACHE': DUMPS[dump],
        'ORB_BT_RISK': '375',
        'ORB_BT_N': str(N_SLOTS),
        'ORB_BT_ACCOUNT': repr(PER_POS * N_SLOTS),
        'ORB_SKIP_Q1': '1',
        'ORB_BT_BOOK_OUT': book,
        'ORB_BT_MONTHLY_OUT': f'{D}/monthly_{tag}_{dump}.csv',
    })
    e.update({k: str(v) for k, v in env.items()})
    with open(f'{D}/log_{tag}_{dump}.txt', 'w') as fh:
        rc = subprocess.call(['nice', '-n', '10', 'python3', '-u',
                              'study_orb_pipeline_static_lock.py'],
                             cwd=ROOT, env=e, stdout=fh, stderr=subprocess.STDOUT)
    print(f'  {tag}_{dump:5s} rc={rc}', flush=True)
    if rc:
        raise SystemExit(f'cell {tag}_{dump} failed — see log_{tag}_{dump}.txt')
    return book


if __name__ == '__main__':
    only = sys.argv[1:] or list(CELLS)
    for dump in ('meas', 'asis'):
        for tag in only:
            run(tag, CELLS[tag], dump)
    print('GRID DONE')
