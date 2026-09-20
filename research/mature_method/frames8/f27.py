#!/usr/bin/env python3
"""frames8 / F27 — THE POWER FLOOR.  Arithmetic only: what the live books can EVER prove.

For ORB and BF-P1: the per-trade SD of (a) the raw R and (b) pass 7's control-differenced
statistic; the trade count needed to detect THAT BOOK'S OWN measured effect at 80 % power; and the
calendar time that implies at the shipped frequency.  Then the same for the two books POOLED.

Reads only `frames7/`'s own artifacts.  Writes one CSV inside `frames8/`.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, f'{ROOT}/research/mature_method/frames7')

D7 = f'{ROOT}/research/mature_method/frames7'
D8 = f'{ROOT}/research/mature_method/frames8'
Z = 2.80                      # two-sided 80 % power at alpha = 0.05
FREQ = {'orb': 6.5, 'bf': 0.65}      # trades per WEEK, shipped (ORB 5-8; BF 2.8/month)


def need(effect, sd):
    """Trades needed to detect `effect` at 80 % power with per-trade SD `sd`."""
    if not (effect and np.isfinite(effect)) or effect == 0:
        return np.nan
    return (Z * sd / abs(effect)) ** 2


def main():
    p = pd.read_csv(f'{D7}/p24.csv', keep_default_na=False, na_values=[''])
    print('p24 columns:', p.columns.tolist(), flush=True)
    print(p.groupby(['book', 'arm', 'split']).size().head(30), flush=True)
    rows = []
    for book in ('orb', 'bf'):
        for split in ('TRAIN', 'VAL'):
            sg = p[(p.book == book) & (p.arm == 'sig') & (p.split == split)]
            if not len(sg):
                continue
            keys = ['day', 'symbol', 'entry_m']
            raw = sg.rr.astype(float)
            out = dict(book=book, split=split, n=len(sg), raw_mean=float(raw.mean()),
                       raw_sd=float(raw.std(ddof=1)))
            for arm, lab in (('b', 'b'), ('a2', 'a'), ('u', 'u')):
                ct = p[(p.book == book) & (p.arm == arm) & (p.split == split)]
                if not len(ct):
                    continue
                g = ct.groupby(keys).rr.mean().rename('c')
                m = sg.merge(g, on=keys, how='inner')
                dif = (m.rr.astype(float) - m.c.astype(float))
                out[f'n_{lab}'] = len(m)
                out[f'diff_{lab}'] = float(dif.mean())
                out[f'sd_{lab}'] = float(dif.std(ddof=1))
            rows.append(out)
    R = pd.DataFrame(rows)
    print('\n=== per-trade dispersion, raw vs control-differenced ===', flush=True)
    print(R.to_string(index=False), flush=True)

    print('\n=== the power floor ===', flush=True)
    lines = []
    for r in R.itertuples():
        f = FREQ[r.book]
        for lab, eff, sd in (('raw', r.raw_mean, r.raw_sd),
                             ('diff_b', getattr(r, 'diff_b', np.nan), getattr(r, 'sd_b', np.nan)),
                             ('diff_a', getattr(r, 'diff_a', np.nan), getattr(r, 'sd_a', np.nan))):
            n = need(eff, sd)
            wk = n / f if np.isfinite(n) else np.nan
            lines.append(dict(book=r.book, split=r.split, stat=lab, effect=eff, sd=sd,
                              n_needed=n, weeks=wk, years=wk / 52.0 if np.isfinite(wk) else np.nan))
            print(f'{r.book:4s} {r.split:5s} {lab:7s} effect={eff:+.3f} sd={sd:.3f} '
                  f'n80={n:9.0f} weeks={wk:9.0f} years={wk / 52.0 if np.isfinite(wk) else np.nan:7.1f}',
                  flush=True)
    L = pd.DataFrame(lines)
    L.to_csv(f'{D8}/cells27.csv', index=False)

    print('\n=== pooled (BF + ORB share the account, the R unit and the session) ===', flush=True)
    for split in ('TRAIN', 'VAL'):
        s = p[(p.arm == 'sig') & (p.split == split)]
        if not len(s):
            continue
        f = FREQ['orb'] + FREQ['bf']
        raw = s.rr.astype(float)
        n = need(raw.mean(), raw.std(ddof=1))
        print(f'POOL {split:5s} raw     effect={raw.mean():+.3f} sd={raw.std(ddof=1):.3f} '
              f'n80={n:9.0f} weeks={n / f:9.0f} years={n / f / 52:7.1f}', flush=True)
        keys = ['book', 'day', 'symbol', 'entry_m']
        ct = p[(p.arm == 'b') & (p.split == split)].groupby(keys).rr.mean().rename('c')
        m = s.merge(ct, on=keys, how='inner')
        dif = m.rr.astype(float) - m.c.astype(float)
        n2 = need(dif.mean(), dif.std(ddof=1))
        print(f'POOL {split:5s} diff_b  effect={dif.mean():+.3f} sd={dif.std(ddof=1):.3f} '
              f'n80={n2:9.0f} weeks={n2 / f:9.0f} years={n2 / f / 52:7.1f}', flush=True)
    print('F27 DONE', flush=True)


if __name__ == '__main__':
    main()
