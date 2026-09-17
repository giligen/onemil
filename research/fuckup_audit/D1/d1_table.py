#!/usr/bin/env python3
"""D1 step 1 — the candidate table (D0's d0_table.py, re-pointed at candidates4 via C/pop_c.csv).

Population = D1/PREREG.md §0.1: next-open fill present, fill >= $5, entry_m <= 841, r_pct of the variant >= 1,
range_so_far_pct >= 5 (asserted for every family here — F6 implies it), ALL DAY (no entry-minute floor),
families F6 {} / F8 N=30 / F14 N=15 / F11 base F6, plus the declared extra F8 N=5 for the two-leg cell only.

Source is `C/pop_c.csv`, the lossless row-subset of `B/candidates4.csv` (identity re-verified in C/REPORT.md §0):
its keep rule is the union over fills of the scorer's filter, hence a strict superset of this population.
Nothing is recomputed here — columns are copied verbatim.

Output: research/fuckup_audit/D1/table.csv
"""
import os, sys, time
import pandas as pd

ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT)
SRC = 'research/fuckup_audit/C/pop_c.csv'
OUT = 'research/fuckup_audit/D1/table.csv'
BASE = ['day', 'symbol', 'fam', 'cfg', 'sig_m', 'minutes_since_open', 'level', 'stop', 'price', 'dist_open_pct',
        'range_so_far_pct', 'rv_adv', 'gap_pct', 'adv20', 'spread_pct', 'spread_cc_bps',
        'sig_o', 'sig_h', 'sig_l', 'sig_c', 'sig_v', 'n_touches', 'consol_bars', 'consol_vol_ratio',
        'cum_dollar_vol', 'vwap_dist_pct', 'close_confirm', 'pm_dollar_vol', 'prev_day_range_pct',
        'prev_close', 'asset_class']
NXT = ['next_entry', 'next_entry_m', 'next_r_pct', 'next_r_pct_m1', 'next_mae_pct', 'next_mfe_r',
       'next_rr_hold', 'next_why_hold', 'next_exit_m_hold',
       'next_rr_2r_stopm1', 'next_why_2r_stopm1', 'next_exit_m_2r_stopm1']
USE = BASE + NXT
KEEP = {('F6', '{}'), ('F8', '{"N": 30}'), ('F14', '{"N": 15}'), ('F11', '{"base": "F6"}'), ('F8', '{"N": 5}')}

if os.path.exists(OUT):
    os.remove(OUT)
n_in = n_out = n_pricediff = 0
t0 = time.time()
first = True
for i, ch in enumerate(pd.read_csv(SRC, usecols=USE, dtype={'symbol': str, 'day': str, 'fam': str, 'cfg': str},
                                   keep_default_na=False, na_values=[''], chunksize=250_000, low_memory=True)):
    n_in += len(ch)
    ch = ch[[(f, c) in KEEP for f, c in zip(ch.fam, ch.cfg)]]
    ch = ch[ch.next_entry.notna() & (ch.next_entry >= 5) & (ch.next_entry_m <= 841)
            & ((ch.next_r_pct >= 1.0) | (ch.next_r_pct_m1 >= 1.0))
            & (ch.range_so_far_pct >= 5)]
    n_pricediff += int(((ch.price >= 5) != (ch.next_entry >= 5)).sum())
    ch.reindex(columns=USE).to_csv(OUT, mode='a', header=first, index=False)
    first = False
    n_out += len(ch)
    print(f'chunk {i} in {n_in:,} kept {n_out:,} | {(time.time()-t0)/60:.1f} min', flush=True)
print(f'DONE in {n_in:,} kept {n_out:,} | rows where (price>=5) != (fill>=5): {n_pricediff}', flush=True)

d = pd.read_csv(OUT, dtype={'symbol': str, 'day': str, 'fam': str, 'cfg': str},
                keep_default_na=False, na_values=[''])
d['key'] = d.fam + ' ' + d.cfg
d['split'] = pd.cut(d.day, bins=[], labels=[]) if False else None
import numpy as np
d['split'] = np.where(d.day < '2026-01-01', 'TRAIN', np.where(d.day < '2026-06-01', 'VAL', 'TEST'))
print(d.groupby(['key', 'split']).size().unstack(fill_value=0).to_string(), flush=True)
print('primary population (r_pct >= 1):', int((d.next_r_pct >= 1).sum()), flush=True)
print('secondary population (r_pct_m1 >= 1):', int((d.next_r_pct_m1 >= 1).sum()), flush=True)
print('symbol-days (union):', d.drop_duplicates(['day', 'symbol']).shape[0], 'days:', d.day.nunique(), flush=True)
print('entry-minute bands:', d.groupby(pd.cut(d.sig_m, [569, 600, 720, 841])).size().to_dict(), flush=True)
