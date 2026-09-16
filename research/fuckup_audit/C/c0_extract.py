#!/usr/bin/env python3
"""Stage C, step 0 — a compact, LOSSLESS row-subset of candidates4.csv so the later passes do not each re-parse 2.0 GB.

It keeps EVERY column (so the output has candidates4's header verbatim and any scorer reads it unchanged) and drops
only rows that CANNOT enter any pre-registered cell under any fill:

    keep  =  (fam in F1-F4  OR  range_so_far_pct >= 5)                       # the causal membership guarantee
             AND  for SOME fill t in {next, rest}:
                  t_entry is present AND t_entry >= 5 AND t_entry_m <= 841
                  AND (t_r_pct >= 1.0 OR t_r_pct_m1 >= 1.0)

That is the union over fills of score5's own per-fill filter, i.e. a strict SUPERSET of every scored population, so
scoring the extract must give the identical table as scoring the full file. That identity is not assumed — it is
re-run and checked (C/REPORT.md §0).

Usage: ulimit -v 1800000; nice -n 10 python3 research/fuckup_audit/C/c0_extract.py
"""
import os, time
import pandas as pd

ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT)
SRC = 'research/fuckup_audit/B/candidates4.csv'
OUT = 'research/fuckup_audit/C/pop_c.csv'
EXEMPT = ('F1', 'F2', 'F3', 'F4')

t0 = time.time()
n_in = n_out = 0
first = True
if os.path.exists(OUT):
    os.remove(OUT)
for ch in pd.read_csv(SRC, dtype={'symbol': str, 'day': str, 'fam': str, 'cfg': str},
                      keep_default_na=False, na_values=[''], chunksize=200_000, low_memory=True):
    n_in += len(ch)
    ok = ch.fam.isin(EXEMPT) | (ch.range_so_far_pct >= 5)
    any_fill = False
    for t in ('next', 'rest'):
        e, em = ch[f'{t}_entry'], ch[f'{t}_entry_m']
        f = (e.notna() & (e >= 5) & (em <= 841)
             & ((ch[f'{t}_r_pct'] >= 1.0) | (ch[f'{t}_r_pct_m1'] >= 1.0)))
        any_fill = f if any_fill is False else (any_fill | f)
    ch = ch[ok & any_fill]
    n_out += len(ch)
    ch.to_csv(OUT, index=False, header=first, mode='w' if first else 'a')
    first = False
    print(f'{n_in:,} read -> {n_out:,} kept | {(time.time()-t0)/60:.1f} min', flush=True)
print(f'DONE  in {n_in:,}  out {n_out:,} ({n_out/max(n_in,1):.1%})  {(time.time()-t0)/60:.1f} min -> {OUT}')
