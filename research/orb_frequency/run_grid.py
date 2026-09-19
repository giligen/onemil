#!/usr/bin/env python3
"""Cell runner for research/orb_frequency.

Every cell is the SHIPPED pipeline (`study_orb_pipeline_static_lock.py`) replayed
off a candidate dump with `ORB_BT_RESIM_CACHE`; only documented env overrides
move.  No threshold, z-param, quintile cutoff or adaptive mult is ever refit.
Nothing outside research/orb_frequency/ is written.

Usage:  python3 run_grid.py ladders | frontier | slots | all
"""
from __future__ import annotations

import os
import subprocess
import sys

ROOT = '/home/ec2-user/onemil'
D = f'{ROOT}/research/orb_frequency'
FEATURES = f'{ROOT}/analysis_results/orb_features_20260916_2053.csv'

DUMPS = {
    'meas': f'{ROOT}/research/fuckup_audit/Q_fill/dump_measured.csv',   # PRIMARY
    'asis': f'{ROOT}/research/fuckup_audit/D1_orb/candidates_dump.csv',  # bracket
}
PER_POS = 3333.333333333333          # invariant per-position cap (D1 convention)

SHIPPED = {}                          # every gate at its orb.yaml value

# ---- PREREG §5a: single-gate ladders, everything else shipped -------------
LADDERS = {
    'L_base':      {},
    'L_q1off':     {'ORB_SKIP_Q1': '0'},
    'L_pdr8':      {'ORB_PDR_VETO_MIN_PCT': '8.0'},
    'L_pdr6':      {'ORB_PDR_VETO_MIN_PCT': '6.0'},
    'L_pdroff':    {'ORB_PDR_VETO': '0'},
    'L_g1off':     {'ORB_G1_VETO': '0'},
    'L_rs15':      {'ORB_RANGE_SIZE_VETO_MIN_PCT': '1.5'},
    'L_rs10':      {'ORB_RANGE_SIZE_VETO_MIN_PCT': '1.0'},
    'L_rsoff':     {'ORB_RANGE_SIZE_VETO': '0'},
    'L_catoff':    {'ORB_CATALYST_VETO': '0'},
    'L_thr05':     {'ORB_BT_THRESHOLD': '-0.5'},
    'L_thrall':    {'ORB_BT_THRESHOLD': '-99'},
}

# ---- PREREG §5b: declared combined points --------------------------------
FRONTIER = {
    'F0': {},
    'F1': {'ORB_RANGE_SIZE_VETO': '0'},
    'F2': {'ORB_RANGE_SIZE_VETO': '0', 'ORB_PDR_VETO_MIN_PCT': '8.0'},
    'F3': {'ORB_RANGE_SIZE_VETO': '0', 'ORB_G1_VETO': '0'},
    'F4': {'ORB_RANGE_SIZE_VETO': '0', 'ORB_G1_VETO': '0', 'ORB_PDR_VETO': '0'},
    'F5': {'ORB_RANGE_SIZE_VETO': '0', 'ORB_G1_VETO': '0', 'ORB_PDR_VETO': '0',
           'ORB_CATALYST_VETO': '0'},
    'F6': {'ORB_RANGE_SIZE_VETO': '0', 'ORB_G1_VETO': '0', 'ORB_PDR_VETO': '0',
           'ORB_CATALYST_VETO': '0', 'ORB_SKIP_Q1': '0',
           'ORB_BT_THRESHOLD': '-99'},
}


def run(tag: str, env: dict, dump: str = 'meas', n: int = 8,
        per_pos: float = PER_POS) -> str:
    """Run one cell; returns the book CSV path."""
    book = f'{D}/book_{tag}.csv'
    e = dict(os.environ)
    e.update({
        'ORB_BT_FEATURES_CSV': FEATURES,
        'ORB_BT_RESIM_CACHE': DUMPS[dump],
        'ORB_BT_RISK': '375',
        'ORB_BT_N': str(n),
        'ORB_BT_ACCOUNT': repr(per_pos * n),
        'ORB_SKIP_Q1': '1',
        'ORB_BT_BOOK_OUT': book,
        'ORB_BT_MONTHLY_OUT': f'{D}/monthly_{tag}.csv',
    })
    e.update({k: str(v) for k, v in env.items()})
    with open(f'{D}/log_{tag}.txt', 'w') as fh:
        rc = subprocess.call(['nice', '-n', '10', 'python3', '-u',
                              'study_orb_pipeline_static_lock.py'],
                             cwd=ROOT, env=e, stdout=fh, stderr=subprocess.STDOUT)
    print(f'  {tag:22s} rc={rc}')
    if rc:
        raise SystemExit(f'cell {tag} failed — see log_{tag}.txt')
    return book


def ladders():
    for dump in ('meas', 'asis'):
        for tag, env in LADDERS.items():
            run(f'{tag}_{dump}', env, dump)


def frontier():
    for dump in ('meas', 'asis'):
        for tag, env in FRONTIER.items():
            if tag == 'F0' and f'L_base_{dump}' :
                pass
            run(f'{tag}_{dump}', env, dump)


# the two highest green-week cells of the declared grid (PREREG §6) plus the
# POST-HOC combinations the separation map suggests.  Post-hoc cells are
# labelled as such everywhere and are NOT eligible for the recommendation.
EXTRA = {
    'Fcat': {'ORB_CATALYST_VETO': '0'},                      # L_catoff
    'P1_broken': {'ORB_SKIP_Q1': '0', 'ORB_RANGE_SIZE_VETO': '0'},   # POST-HOC
    'P2_catq1': {'ORB_CATALYST_VETO': '0', 'ORB_SKIP_Q1': '0',
                 'ORB_RANGE_SIZE_VETO': '0'},                # POST-HOC
}
ALL_CFG = {**FRONTIER, **EXTRA}


def slots(best: list[str]):
    """PREREG §6: N in {3,8,12,16} x {F0 + the best two frontier points}."""
    for cfg in ['F0'] + best:
        for n in (3, 8, 12, 16):
            run(f'S{n}_{cfg}_meas', ALL_CFG[cfg], 'meas', n=n)


def posthoc():
    for tag in ('P1_broken', 'P2_catq1'):
        for dump in ('meas', 'asis'):
            run(f'{tag}_{dump}', EXTRA[tag], dump)


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if what in ('ladders', 'all'):
        ladders()
    if what in ('frontier', 'all'):
        frontier()
    if what == 'posthoc':
        posthoc()
    if what == 'slots':
        slots(sys.argv[2:])
    print('GRID DONE')
