#!/usr/bin/env python3
"""Meta-label study — the driver.  One python process, xgboost n_jobs=1.

  python3 run_study.py select     # the declared grid, TRAIN only -> chosen config
  python3 run_study.py cells      # the 9 cells, TRAIN+VAL, both fill arms
  python3 run_study.py shuffle    # the shuffled-label control
  python3 run_study.py ablate     # per-feature-family ablation
  python3 run_study.py test       # TEST, refuses to run without FREEZE.md
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/research/meta_label')
from analyze import book_stats, paired_vs_m0  # noqa: E402

M = f'{ROOT}/research/meta_label'
SC = f'{M}/scores'
BK = f'{M}/books'
os.makedirs(SC, exist_ok=True)
os.makedirs(BK, exist_ok=True)

GRID = [(3, 200), (3, 500), (5, 200), (5, 500)]
MODELS = ('m1', 'm2', 'm3')
FLAGS = (0, 1)
ARM = 'meas_cost'          # PRIMARY: Stage Q's measured fill + measured cost
ARM2 = 'asis'              # secondary: D1's arm (the M0 reproduction gate)


def wf(model, flags, depth, trees, out, last_month, arm=ARM, shuffle=False,
       drop_family=''):
    cmd = [sys.executable, f'{M}/walkforward.py', '--model', model,
           '--flags', str(flags), '--depth', str(depth), '--trees', str(trees),
           '--arm', arm, '--last-month', last_month, '--out', out]
    if shuffle:
        cmd.append('--shuffle')
    if drop_family:
        cmd += ['--drop-family', drop_family]
    subprocess.run(cmd, check=True)


def book(tag, arm, sidecar='', rank='', veto='', thr='0.5'):
    subprocess.run(['bash', f'{M}/run_book.sh', tag, arm, sidecar, rank, veto, thr],
                   check=True)
    return f'{BK}/book_{tag}_{arm}.csv'


def cell_tag(model, flags, suffix=''):
    return f'{model}_f{flags}{suffix}'


# --------------------------------------------------------------------------
def cmd_select():
    """The single declared grid, scored on TRAIN ONLY (2025-04..2025-12)."""
    rows = []
    for model in MODELS:
        for flags in FLAGS:
            for depth, trees in GRID:
                tag = f'{cell_tag(model, flags)}_d{depth}t{trees}_sel'
                s = f'{SC}/{tag}.csv'
                wf(model, flags, depth, trees, s, '2025-12')
                p = book(tag, ARM, s, 'meta_score')
                st = book_stats(p, ('TRAIN',)).get('TRAIN', {})
                rows.append(dict(model=model, flags=flags, depth=depth,
                                 trees=trees, **st))
                print('  ', rows[-1], flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(f'{M}/grid_train.csv', index=False)
    best = {}
    for (model, flags), g in t.groupby(['model', 'flags']):
        b = g.sort_values('r_per_pick', ascending=False).iloc[0]
        best[f'{model}_f{flags}'] = dict(depth=int(b.depth), trees=int(b.trees),
                                         train_r=float(b.r_per_pick))
    json.dump(best, open(f'{M}/chosen_config.json', 'w'), indent=1)
    print('\nCHOSEN (TRAIN only):', json.dumps(best, indent=1))


def _chosen():
    return json.load(open(f'{M}/chosen_config.json'))


def cmd_cells(last_month='2026-05', splits=('TRAIN', 'VAL'), shuffle=False,
              out_name='cells'):
    best = _chosen()
    rows = []
    for arm in (ARM, ARM2):
        p0 = book('M0', arm)
        for sp, st in book_stats(p0, splits).items():
            rows.append(dict(cell='M0', arm=arm, split=sp, **st,
                             d_pnl=0, d_mean_day=0.0, t_day=float('nan')))
    for model in MODELS:
        for flags in FLAGS:
            c = best[f'{model}_f{flags}']
            sfx = '_shuf' if shuffle else ''
            tag = cell_tag(model, flags, sfx)
            s = f'{SC}/{tag}.csv'
            wf(model, flags, c['depth'], c['trees'], s, last_month,
               shuffle=shuffle)
            for arm in (ARM, ARM2):
                p = book(tag, arm, s, 'meta_score')
                for sp, st in book_stats(p, splits).items():
                    rows.append(dict(cell=f'{model.upper()} flags={flags}',
                                     arm=arm, split=sp, **st,
                                     **paired_vs_m0(p, f'{BK}/book_M0_{arm}.csv', sp)))
            # M4 = the M1 model used ONLY to veto; the shipped ranking stands.
            if model == 'm1':
                tag4 = cell_tag('m4', flags, sfx)
                for arm in (ARM, ARM2):
                    p = book(tag4, arm, s, '', 'meta_score', '0.5')
                    for sp, st in book_stats(p, splits).items():
                        rows.append(dict(cell=f'M4 flags={flags}', arm=arm,
                                         split=sp, **st,
                                         **paired_vs_m0(p, f'{BK}/book_M0_{arm}.csv', sp)))
    t = pd.DataFrame(rows)
    t.to_csv(f'{M}/{out_name}.csv', index=False)
    pd.set_option('display.width', 250)
    print(t.to_string(index=False))


def cmd_shuffle():
    cmd_cells(shuffle=True, out_name='cells_shuffled')


def cmd_ablate():
    from build_dataset import FAMILIES
    best = _chosen()
    rows = []
    for model in MODELS:
        for flags in FLAGS:
            c = best[f'{model}_f{flags}']
            for fam in FAMILIES:
                tag = f'{cell_tag(model, flags)}_abl_{fam}'
                s = f'{SC}/{tag}.csv'
                wf(model, flags, c['depth'], c['trees'], s, '2026-05',
                   drop_family=fam)
                p = book(tag, ARM, s, 'meta_score')
                for sp, st in book_stats(p, ('TRAIN', 'VAL')).items():
                    rows.append(dict(cell=f'{model.upper()} flags={flags}',
                                     dropped=fam, split=sp, **st))
                    print('  ', rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(f'{M}/ablation.csv', index=False)


def cmd_m4thr():
    """M4 sensitivity: the veto cut at the ranked pool's OWN base rate of a
    winner (0.145 over the model-active TRAIN window) instead of the 0.5 I wrote
    into the harness.  Declared AFTER seeing that 0.5 vetoes 98% of the book at a
    14.5% base rate; reported, counted in the cell count, and NOT gate-eligible."""
    rows = []
    for flags in FLAGS:
        s = f'{SC}/{cell_tag("m1", flags)}.csv'
        for arm in (ARM, ARM2):
            p = book(f'm4b_f{flags}', arm, s, '', 'meta_score', '0.145')
            for sp, st in book_stats(p, ('TRAIN', 'VAL')).items():
                rows.append(dict(cell=f'M4b(base-rate cut) flags={flags}', arm=arm,
                                 split=sp, **st,
                                 **paired_vs_m0(p, f'{BK}/book_M0_{arm}.csv', sp)))
    t = pd.DataFrame(rows)
    t.to_csv(f'{M}/cells_m4b.csv', index=False)
    print(t.to_string(index=False))


def cmd_test():
    if not os.path.exists(f'{M}/FREEZE.md'):
        raise SystemExit('TEST is SEALED: write research/meta_label/FREEZE.md '
                         'naming the G1+G2 survivors first (PREREG).')
    cmd_cells(last_month='2026-09', splits=('TRAIN', 'VAL', 'TEST'),
              out_name='cells_test')


if __name__ == '__main__':
    {'select': cmd_select, 'cells': cmd_cells, 'shuffle': cmd_shuffle,
     'ablate': cmd_ablate, 'm4thr': cmd_m4thr, 'test': cmd_test}[sys.argv[1]]()
