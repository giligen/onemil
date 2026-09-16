#!/usr/bin/env python3
"""D0 step 1 — the candidate table for the feature-selection pilot.

Population = Stage A's / score4's scoring population (price >= 5, entry_m <= 841, r_pct >= 1,
range_so_far_pct >= 5 for the F5-F10 group), restricted to the three base families Stage A adopted
(F6 {}, F8 {"N": 30}, F8 {"N": 15}) and to entries >= 10:00 (entry_m >= 600).

Nothing is recomputed here: the columns are copied verbatim from research/bf_zero2/candidates3.csv.
Read chunked (the source is 678 MB) with usecols, keep_default_na=False.
Output: research/fuckup_audit/D/table.csv
"""
import os, sys, time
import pandas as pd

ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT)
SRC = 'research/bf_zero2/candidates3.csv'
OUT = 'research/fuckup_audit/D/table.csv'
USE = ['day', 'symbol', 'fam', 'cfg', 'sig_m', 'entry_m', 'level', 'entry', 'stop', 'r_pct', 'price',
       'dist_open_pct', 'range_so_far_pct', 'rv_adv', 'gap_pct', 'adv20', 'spread_pct',
       'rr_hold', 'why_hold', 'exit_m_hold', 'rr_2r', 'why_2r', 'exit_m_2r']
KEEP = {('F6', '{}'), ('F8', '{"N": 30}'), ('F8', '{"N": 15}')}

if os.path.exists(OUT):
    os.remove(OUT)
n_in = n_out = 0
t0 = time.time()
for i, ch in enumerate(pd.read_csv(SRC, usecols=USE, dtype={'symbol': str, 'day': str, 'fam': str, 'cfg': str},
                                   keep_default_na=False, na_values=[''], chunksize=400_000, low_memory=True)):
    n_in += len(ch)
    ch = ch[(ch.price >= 5) & (ch.entry_m <= 841) & (ch.r_pct >= 1.0) & (ch.entry_m >= 600)]
    ch = ch[ch.range_so_far_pct >= 5]                      # F6/F8 are both in the F5-F10 causal-floor group
    ch = ch[[(f, c) in KEEP for f, c in zip(ch.fam, ch.cfg)]]
    ch.reindex(columns=USE).to_csv(OUT, mode='a', header=not os.path.exists(OUT), index=False)
    n_out += len(ch)
    print(f'chunk {i} in {n_in:,} kept {n_out:,} | {(time.time()-t0)/60:.1f} min', flush=True)
print(f'DONE in {n_in:,} kept {n_out:,}', flush=True)
d = pd.read_csv(OUT, dtype={'symbol': str, 'day': str, 'fam': str, 'cfg': str},
                keep_default_na=False, na_values=[''])
d['key'] = d.fam + ' ' + d.cfg
print(d.groupby('key').size().to_string(), flush=True)
print('symbol-days (union over families):', d.drop_duplicates(['day', 'symbol']).shape[0], flush=True)
print('days:', d.day.nunique(), flush=True)
