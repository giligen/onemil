#!/usr/bin/env python3
"""Stage E — parity of E/candidates_causal*.csv against B/candidates4.csv.

WHAT IS COMPARABLE.  Stage B's `build_candidates.load_bars` reads `data/cache.db` FIRST and only
falls through to `research/bf_zero/bars_sip.db`; this stage never reads cache.db. So a key is
tape-identical between the two files ONLY when cache.db holds nothing for it and bars_sip.db holds
something. Those keys must agree EXACTLY — same family code, same walks, same fill models, one
tape. Keys served by cache.db in Stage B are reported separately as a tape difference, never
folded into the parity number.

Compared on (day, symbol, fam, cfg): sig_m, level, stop, range_so_far_pct, next_entry, next_rr_2r,
rest_entry, rest_rr_2r — the task's bar is entry_next / rr_2r to 1e-6.

Read-only everywhere; prints a table and writes E/parity_causal.md.

RUN:  E_TAG=_smoke python3 research/fuckup_audit/E/parity_causal.py
"""
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
E = f'{ROOT}/research/fuckup_audit/E'
TAG = os.environ.get('E_TAG', '')
MINE = f'{E}/candidates_causal{TAG}.csv'
THEIRS = f'{ROOT}/research/fuckup_audit/B/candidates4.csv'
SIP = f'{ROOT}/research/bf_zero/bars_sip.db'
CACHE = f'{ROOT}/data/cache.db'
OUT = f'{E}/parity_causal{TAG}.md'

KEYCOLS = ['day', 'symbol', 'fam', 'cfg']
NUM = ['sig_m', 'level', 'stop', 'range_so_far_pct',
       'next_entry', 'next_entry_m', 'next_rr_2r',
       'rest_entry', 'rest_entry_m', 'rest_rr_2r']


def read_mine():
    d = pd.read_csv(MINE, dtype={'symbol': str, 'day': str, 'fam': str, 'cfg': str},
                    keep_default_na=False, na_values=[''])
    for c in NUM:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    return d


def read_theirs(days, fams):
    keep = []
    for ch in pd.read_csv(THEIRS, dtype={'symbol': str, 'day': str, 'fam': str, 'cfg': str},
                          keep_default_na=False, na_values=[''], chunksize=200_000,
                          usecols=KEYCOLS + NUM):
        keep.append(ch[ch.day.isin(days) & ch.fam.isin(fams)])
    d = pd.concat(keep, ignore_index=True) if keep else pd.DataFrame(columns=KEYCOLS + NUM)
    for c in NUM:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    return d


def main():
    mine = read_mine()
    days = sorted(set(mine.day))
    fams = sorted(set(mine.fam))
    theirs = read_theirs(set(days), set(fams))
    print(f'mine {len(mine):,} rows / theirs {len(theirs):,} rows over {len(days)} days '
          f'{days[0]}..{days[-1]}', flush=True)

    # which keys did Stage B serve from cache.db (and are therefore NOT a like-for-like tape)?
    keys = sorted(set(zip(theirs.day, theirs.symbol)) | set(zip(mine.day, mine.symbol)))
    cc = sqlite3.connect(f'file:{CACHE}?mode=ro', uri=True, timeout=180)
    sp = sqlite3.connect(f'file:{SIP}?mode=ro', uri=True, timeout=180)
    in_cache, in_sip = set(), set()
    for d, s in keys:
        if cc.execute('select 1 from intraday_bars_1min where symbol=? and bar_date=? limit 1',
                      (s, d)).fetchone():
            in_cache.add((d, s))
        if sp.execute('select 1 from bars where symbol=? and day=? limit 1',
                      (s, d)).fetchone():
            in_sip.add((d, s))
    cc.close()
    sp.close()
    same_tape = (in_sip - in_cache)
    print(f'keys {len(keys):,} | cache.db {len(in_cache):,} | bars_sip.db {len(in_sip):,} '
          f'| tape-identical {len(same_tape):,}', flush=True)

    m = mine.set_index(KEYCOLS).sort_index()
    t = theirs.set_index(KEYCOLS).sort_index()
    common = m.index.intersection(t.index)
    mk = pd.Index([(d, s) for d, s, _, _ in common])
    mask = np.array([k in same_tape for k in mk])
    cmp_idx = common[mask]
    a, b = m.loc[cmp_idx], t.loc[cmp_idx]

    L = ['# Stage E — parity vs `B/candidates4.csv`\n',
         f'`{MINE}` vs `{THEIRS}`, days {days[0]}..{days[-1]} ({len(days)}), '
         f'families {", ".join(fams)}.\n',
         f'- rows: mine **{len(mine):,}**, Stage B (same days+families) **{len(theirs):,}**',
         f'- symbol-days: {len(keys):,}; served by `data/cache.db` in Stage B {len(in_cache):,}; '
         f'present in `bars_sip.db` {len(in_sip):,}; **tape-identical {len(same_tape):,}**',
         f'- signal rows on tape-identical keys present in BOTH files: **{len(cmp_idx):,}**',
         f'- signal rows only in mine: {len(m.index.difference(t.index)):,}; '
         f'only in Stage B: {len(t.index.difference(m.index)):,} '
         '(the two universes are different by design — this is not an error)\n',
         '| column | n compared | max abs diff | n diff > 1e-6 |', '|---|---:|---:|---:|']
    worst = 0.0
    bad = 0
    for c in NUM:
        x, y = a[c].values, b[c].values
        ok = ~(np.isnan(x) & np.isnan(y))
        dif = np.abs(np.nan_to_num(x[ok], nan=0.0) - np.nan_to_num(y[ok], nan=0.0))
        nb = int((dif > 1e-6).sum())
        L.append(f'| `{c}` | {int(ok.sum()):,} | {dif.max() if len(dif) else 0:.3g} | {nb:,} |')
        if c in ('next_entry', 'next_rr_2r'):
            worst = max(worst, float(dif.max()) if len(dif) else 0.0)
            bad += nb
    L.append('')
    L.append(f'**Task bar (entry_next / rr_2r to 1e-6): max abs diff {worst:.3g}, '
             f'{bad} row(s) over tolerance.**\n')
    with open(OUT, 'w') as f:
        f.write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
