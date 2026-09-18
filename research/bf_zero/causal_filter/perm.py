#!/usr/bin/env python3
"""CAUSAL_FILTER — search-adjusted permutation p across the 12 cells.

Null: the survivors carry no information about the outcome. The survivors' feature block is shuffled
as ONE row block among the TRAIN signals (so the features keep their joint distribution and their
correlation with each other, and only the feature->outcome link is broken), the 12 cells are rebuilt
and booked, and the MAX TRAIN mean net R across the 12 is recorded. p = share of permutations whose
max is >= the observed max. B = 200 (seeded).

Output: causal_filter/perm.json
"""
import json, os, sys
import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/research/bf_zero/causal_filter')
from cells import load, build_cells, book                        # noqa: E402

D = f'{ROOT}/research/bf_zero/causal_filter'
B = int(os.environ.get('PERM_B', '200'))
COL = 'net_meas'


def max_cell(c, sel):
    cells, _ = build_cells(c, sel)
    best = -9.0
    for name, mask in cells.items():
        d = c[mask.fillna(False)]
        d = d[d.obtainable == True]                              # noqa: E712
        t = book(d[d.split == 'TRAIN'], COL)
        if t is not None and len(t) / 53 >= 5.0:                  # the G1 frequency floor
            best = max(best, float(t.net.mean()))
    return best


def main():
    sel = json.load(open(f'{D}/selection.json'))['survivors']
    if not sel:
        json.dump(dict(note='no survivors — permutation not run'), open(f'{D}/perm.json', 'w'))
        print('no survivors', flush=True)
        return
    c = load()
    c = c[c.split != 'TEST'].reset_index(drop=True)
    obs = max_cell(c, sel)
    feats = [s['feat'] for s in sel]
    rng = np.random.default_rng(11)
    tr_idx = c.index[c.split == 'TRAIN'].to_numpy()
    nulls = []
    for b in range(B):
        cc = c.copy()
        perm = rng.permutation(tr_idx)
        cc.loc[tr_idx, feats] = c.loc[perm, feats].to_numpy()
        nulls.append(max_cell(cc, sel))
        if (b + 1) % 25 == 0:
            print(f'{b + 1}/{B} obs {obs:+.3f} null p95 {np.percentile(nulls, 95):+.3f}', flush=True)
    nulls = np.array(nulls)
    p = float((nulls >= obs).mean())
    json.dump(dict(B=B, observed_max_TRAIN_meanR=round(obs, 4), p=p,
                   null_mean=round(float(nulls.mean()), 4),
                   null_p95=round(float(np.percentile(nulls, 95)), 4)),
              open(f'{D}/perm.json', 'w'), indent=1)
    print(f'observed {obs:+.4f} | null mean {nulls.mean():+.4f} p95 {np.percentile(nulls, 95):+.4f} '
          f'| search-adjusted p = {p:.3f}', flush=True)


if __name__ == '__main__':
    main()
