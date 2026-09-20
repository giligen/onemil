#!/usr/bin/env python3
"""F39 supplement — the PAIRED signal-minus-control difference, day-clustered.

The decomposition table prints four unpaired levels.  The honest inference (pass 6 §1.x: "the
paired, day-clustered t is the correct statistic") is the PAIRED difference: each booked trade
minus the mean net of ITS OWN matched controls.  Run for every cell; the overnight cells are the
reason it exists — under an overnight exit the CONTROLS are positive too, so the unpaired level
flatters the detector.

  python3 sup39.py > sup39.log
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c12 import (D12, S, SPLITS, attach_cost12, book_ranked, build_cost_model,   # noqa: E402
                 clustered_t)
from s39 import CELLS                                                    # noqa: E402


def main() -> int:
    build_cost_model()
    s = pd.read_csv(f'{D12}/sig39.csv', dtype={'day': str, 'symbol': str, 'lvl': str, 'why': str},
                    keep_default_na=False, na_values=[''])
    c = pd.read_csv(f'{D12}/w39c.csv', dtype={'day': str, 'symbol': str, 'ctrl': str, 'arm': str},
                    keep_default_na=False, na_values=[''])
    c['split'] = S.split_of(c.day.values)
    c = c[c.split.isin(SPLITS)]
    print('| cell | split | n | signal net R | CB paired diff (tc) | CA paired diff (tc) | MDE |')
    print('|---|---|---|---|---|---|---|')
    rows = []
    for name, lv, st, ex in CELLS:
        x = s[(s.lvl == lv) & (s.stop_tag == st) & (s.exit_tag == ex)]
        if not len(x):
            continue
        x = x.sort_values(['day', 'entry_m', 'symbol'], kind='mergesort').reset_index(drop=True)
        b = book_ranked(x, 12, 4)
        cc = c.copy()
        cc['rr'] = cc[f'{ex}_rr']; cc['why'] = cc[f'{ex}_why']
        cc = cc[cc.rr.notna() & (cc.why != '')]
        cc = cc.rename(columns={'symbol': 'key_sym'})
        cc['symbol'] = cc.ctrl; cc['entry_m2'] = cc.ctrl_m
        cc = cc.rename(columns={'entry_m': 'key_m', 'entry_m2': 'entry_m'})
        cc = attach_cost12(cc)
        for sp in SPLITS:
            d = b[b.split == sp]
            if len(d) < 5:
                continue
            out = []
            for arm in ('CB', 'CA'):
                a = cc[(cc.arm == arm) & (cc.split == sp)]
                g = a.groupby(['day', 'key_sym', 'key_m']).net.mean()
                m = d.set_index(['day', 'symbol', 'entry_m']).index.map(g)
                dd = d.assign(diff=d.net.values - np.asarray(m, dtype=float))
                dd = dd[dd['diff'].notna()]
                out.append((float(dd['diff'].mean()) if len(dd) else np.nan,
                            clustered_t(dd, 'diff'), len(dd)))
            mde = 2.80 * float(d.net.std(ddof=1) / np.sqrt(len(d)))
            print(f'| {name} | {sp} | {len(d)} | {d.net.mean():+.3f} | '
                  f'{out[0][0]:+.3f} ({out[0][1]:+.2f}, n {out[0][2]}) | '
                  f'{out[1][0]:+.3f} ({out[1][1]:+.2f}, n {out[1][2]}) | {mde:.3f} |', flush=True)
            rows.append(dict(cell=name, split=sp, n=len(d), net=d.net.mean(),
                             cb_diff=out[0][0], cb_tc=out[0][1],
                             ca_diff=out[1][0], ca_tc=out[1][1], mde=mde))
    pd.DataFrame(rows).to_csv(f'{D12}/paired39.csv', index=False)
    return 0


if __name__ == '__main__':
    sys.exit(main())
