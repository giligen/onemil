#!/usr/bin/env python3
"""Stage R_daily step 4 — the two data-integrity control books.

The daily files are UNADJUSTED and the ITCH-era daily bar aggregates the extended session, so two
known defects can fabricate a return:

  (1) a split inside the hold window (`split_candidates.csv`, seam.md §4);
  (2) a bogus extended-hours print used as the entry open.  MPWR 2022-10-14 is the worked example:
      the panel's open is $4.26 against a signal-day close of $310 — one trade worth +7,178% in
      `K3_h1_n10`, i.e. more than that cell's entire TRAIN gross mean.

Neither is corrected in the primary numbers (they are the pre-registered rule executing on the
bought data).  Both are reported here, and the pre-registered tail tests already neutralise (2).

Control A: drop trades whose [entry_date, exit_date] window contains a split candidate.
Control B: drop trades whose entry price is more than 50% away from the signal day's close — on a
           $10M+/day name that is a bad print or a corporate action, never a tradable open.
Control C: both.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '/home/ec2-user/onemil/research/fuckup_audit/K')
os.chdir('/home/ec2-user/onemil')
R = 'research/fuckup_audit/R_daily'

from build_k import tstat  # noqa: E402


def main(phase='A'):
    sc = pd.read_csv(f'{R}/split_candidates.csv', keep_default_na=False, na_values=[''])
    by_sym = {}
    for s, d in zip(sc.symbol.astype(str), sc.bar_date.astype(str)):
        by_sym.setdefault(s, []).append(d)
    rows = []
    for fn in sorted(os.listdir(f'{R}/trades')):
        if not fn.endswith('.csv') or fn.endswith('_sec.csv'):
            continue
        t = pd.read_csv(f'{R}/trades/{fn}', keep_default_na=False, na_values=[''])
        if not len(t):
            continue
        split_hit = np.array([any(a <= d <= b for d in by_sym.get(str(s), []))
                              for s, a, b in zip(t.symbol, t.entry_date.astype(str),
                                                 t.exit_date.astype(str))])
        badprint = (t.entry / t.sig_close - 1.0).abs() > 0.5
        for sp in (('TRAIN', 'VAL') if phase == 'A' else ('TRAIN', 'VAL', 'TEST')):
            m = (t.split == sp).to_numpy()
            if not m.sum():
                continue
            x = t[m]
            rows.append(dict(
                cell=fn[:-4], split=sp, n=int(m.sum()),
                net_bps=x.net.mean() * 1e4, t=tstat(x.net),
                n_split=int(split_hit[m].sum()),
                netA_bps=t.net[m & ~split_hit].mean() * 1e4,
                n_badprint=int(badprint.to_numpy()[m].sum()),
                netB_bps=t.net[m & ~badprint.to_numpy()].mean() * 1e4,
                tB=tstat(t.net[m & ~badprint.to_numpy()]),
                netC_bps=t.net[m & ~split_hit & ~badprint.to_numpy()].mean() * 1e4,
                tC=tstat(t.net[m & ~split_hit & ~badprint.to_numpy()])))
    out = pd.DataFrame(rows)
    out.to_csv(f'{R}/controls.csv', index=False, float_format='%.6g')
    print(out.to_string(index=False, float_format='%.1f'), flush=True)
    print(f'\ntotal trades dropped: split {out.n_split.sum():,}  bad print {out.n_badprint.sum():,}')
    return out


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'A')
