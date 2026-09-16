#!/usr/bin/env python3
"""Stage A step 1 — chunked extraction of score4's scoring population from the 678 MB candidates3.csv.

Population, verbatim from score4.py: price >= 5, entry_m <= 841, r_pct >= 1, and range_so_far_pct >= 5 for every
family that is not F1-F4 (the causal universe floor). All 26 family-configs kept.
Columns kept are the ones Stage A needs; nothing is recomputed here.
Output: research/fuckup_audit/A/pop_a.csv
"""
import os, sys, time
import pandas as pd
ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT)
SRC = 'research/bf_zero2/candidates3.csv'
OUT = 'research/fuckup_audit/A/pop_a.csv'
USE = ['day', 'symbol', 'fam', 'cfg', 'sig_m', 'entry_m', 'r_pct', 'price', 'dist_open_pct', 'range_so_far_pct',
       'spread_pct', 'rr_hold', 'why_hold', 'exit_m_hold', 'rr_2r', 'why_2r', 'exit_m_2r']
if os.path.exists(OUT): os.remove(OUT)
n_in = n_out = 0; t0 = time.time()
for i, ch in enumerate(pd.read_csv(SRC, usecols=USE, dtype={'symbol': str, 'day': str, 'fam': str, 'cfg': str},
                                   keep_default_na=False, na_values=[''], chunksize=400_000, low_memory=True)):
    n_in += len(ch)
    ch = ch[(ch.price >= 5) & (ch.entry_m <= 841) & (ch.r_pct >= 1.0)]
    need = ~ch.fam.isin(['F1', 'F2', 'F3', 'F4'])
    ch = ch[~(need & ~(ch.range_so_far_pct >= 5))]
    ch.reindex(columns=USE).to_csv(OUT, mode='a', header=not os.path.exists(OUT), index=False)
    n_out += len(ch)
    print(f'chunk {i} in {n_in:,} kept {n_out:,} | {(time.time()-t0)/60:.1f} min', flush=True)
print(f'DONE in {n_in:,} kept {n_out:,}', flush=True)
