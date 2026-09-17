#!/usr/bin/env python3
"""R1 - extract implementation A's F6-PDR candidate rows (post-gate, pre-book) from C/pop_c.csv.

A = research/fuckup_audit/H/F6/f6_pdr_book.py scored off C/pop_c.csv (built by B/build_candidates4.py,
which imports research/bf_zero/build_candidates.py's fam_r2g / load_bars with BFZ_SLIP=0).
Writes a_cands.csv with every field needed to reconcile against implementation B.
"""
import os, sys
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT)
D = 'research/fuckup_audit'
OUT = f'{D}/H/F6_reconcile'
USE = ['day', 'symbol', 'fam', 'sig_m', 'level', 'stop', 'price', 'range_so_far_pct', 'spread_cc_bps',
       'prev_day_range_pct', 'prev_close', 'sig_h',
       'next_entry', 'next_entry_m', 'next_r_pct',
       'next_rr_hold', 'next_why_hold', 'next_exit_m_hold',
       'next_rr_2r', 'next_why_2r', 'next_exit_m_2r']
parts = []
for ch in pd.read_csv(f'{D}/C/pop_c.csv', usecols=USE, chunksize=400_000,
                      keep_default_na=False, na_values=[''], dtype={'symbol': str, 'day': str}):
    parts.append(ch[ch.fam == 'F6'])
raw = pd.concat(parts, ignore_index=True)
print('A raw F6 signal rows', len(raw), flush=True)
raw.to_csv(f'{OUT}/a_signals_all.csv', index=False)

c = raw[raw.next_entry.notna()]
c = c[(c.price >= 5) & (c.next_r_pct >= 1) & (c.next_entry_m <= 841) & (c.range_so_far_pct >= 5)]
c = c[c.prev_day_range_pct >= 8].reset_index(drop=True)
print('A candidates after gates (PDR>=8)', len(c), flush=True)
c.to_csv(f'{OUT}/a_cands.csv', index=False)
