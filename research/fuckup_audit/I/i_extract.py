#!/usr/bin/env python3
"""Stage I step 1 — extract the four declared families out of C/pop_c.csv under the PREREG population.

Population (PREREG.md Part A): next_entry present, price >= 5, next_r_pct >= 1, next_entry_m <= 841,
range_so_far_pct >= 5, prev_day_range_pct >= 8.  No selection, no new rule.
Output: I/pop_i.csv (one row per SIGNAL, all four families, un-deduped).
"""
import os, time
import pandas as pd

ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT)
D = 'research/fuckup_audit/I'
SRC = 'research/fuckup_audit/C/pop_c.csv'
OUT = f'{D}/pop_i.csv'

FAMS = {('F6', '{}'): 'F6', ('F14', '{"N": 15}'): 'F14',
        ('F11', '{"base": "F6"}'): 'F11', ('F8', '{"N": 30}'): 'F8N30'}

KEEP = ['day', 'symbol', 'fam', 'cfg', 'sig_m', 'level', 'stop', 'price', 'dist_open_pct', 'range_so_far_pct',
        'rv_adv', 'gap_pct', 'adv20', 'spread_pct', 'spread_cc_bps', 'prev_day_range_pct', 'asset_class',
        'next_entry', 'next_entry_m', 'next_r_pct',
        'next_rr_2r', 'next_why_2r', 'next_exit_m_2r',
        'next_rr_hold', 'next_why_hold', 'next_exit_m_hold']

parts, n_in, t0 = [], 0, time.time()
for ch in pd.read_csv(SRC, usecols=KEEP, dtype={'symbol': str, 'day': str, 'fam': str, 'cfg': str},
                      keep_default_na=False, na_values=[''], chunksize=200_000, low_memory=True):
    n_in += len(ch)
    key = list(zip(ch.fam, ch.cfg))
    ch = ch.assign(tag=[FAMS.get(k) for k in key])
    ch = ch[ch.tag.notna()]
    if not len(ch):
        continue
    ok = (ch.range_so_far_pct >= 5) & ch.next_entry.notna() & (ch.price >= 5) \
        & (ch.next_entry_m <= 841) & (ch.next_r_pct >= 1.0) & (ch.prev_day_range_pct >= 8)
    ch = ch[ok]
    if len(ch):
        parts.append(ch)
    print(f'{n_in:,} read | kept {sum(len(p) for p in parts):,} | {(time.time()-t0)/60:.1f}m', flush=True)

d = pd.concat(parts, ignore_index=True)
d.to_csv(OUT, index=False)
print('DONE', len(d), '->', OUT, flush=True)
print(d.groupby('tag').size().to_string(), flush=True)
