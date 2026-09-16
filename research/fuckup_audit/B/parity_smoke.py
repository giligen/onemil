#!/usr/bin/env python3
"""Smoke-test parity: on the smoke days, every candidates3 row of the families candidates4 shares with it
(F1 {"P":0.12}, F6 {}, F8 {"N":5|15|30}) must appear in candidates4 with entry_next == entry and
rr_2r_next == rr_2r to 1e-6. candidates3 is 678 MB — read with usecols + chunksize only.
Usage: ulimit -v 1500000; nice -n 15 python3 research/fuckup_audit/B/parity_smoke.py [TAG=_smoke]"""
import os, sys
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT)
TAG = sys.argv[1] if len(sys.argv) > 1 else '_smoke'
C4 = f'research/fuckup_audit/B/candidates4{TAG}.csv'
C3 = 'research/bf_zero2/candidates3.csv'
KEYS = {('F1', '{"P": 0.12}'), ('F6', '{}'), ('F8', '{"N": 5}'), ('F8', '{"N": 15}'), ('F8', '{"N": 30}')}

c4 = pd.read_csv(C4, dtype={'symbol': str, 'day': str, 'fam': str, 'cfg': str}, keep_default_na=False, na_values=[''],
                 usecols=['day', 'symbol', 'fam', 'cfg', 'sig_m', 'level', 'stop', 'next_entry', 'next_entry_m',
                          'next_rr_2r', 'next_why_2r', 'next_exit_m_2r'])
days = sorted(c4.day.unique())
c4 = c4[[(f, g) in KEYS for f, g in zip(c4.fam, c4.cfg)]]
print(f'candidates4 rows on {len(days)} days, shared families: {len(c4):,}', flush=True)

got = []
for ch in pd.read_csv(C3, dtype={'symbol': str, 'day': str, 'fam': str, 'cfg': str}, keep_default_na=False,
                      na_values=[''], chunksize=400_000,
                      usecols=['day', 'symbol', 'fam', 'cfg', 'sig_m', 'entry_m', 'level', 'stop', 'entry',
                               'rr_2r', 'why_2r', 'exit_m_2r']):
    ch = ch[ch.day.isin(days)]
    if len(ch):
        got.append(ch[[(f, g) in KEYS for f, g in zip(ch.fam, ch.cfg)]])
c3 = pd.concat(got, ignore_index=True) if got else pd.DataFrame()
print(f'candidates3 rows on the same days/families: {len(c3):,}', flush=True)

k = ['day', 'symbol', 'fam', 'cfg']
mg = c3.merge(c4, on=k, how='left', suffixes=('_3', '_4'))
missing = mg[mg.next_entry.isna()]
print(f'candidates3 rows with NO candidates4 fill: {len(missing)}')
if len(missing):
    print(missing.head(10).to_string(index=False))
ok = mg[mg.next_entry.notna()].copy()
for a, b in (('entry', 'next_entry'), ('rr_2r', 'next_rr_2r')):
    d = (ok[a].astype(float) - ok[b].astype(float)).abs()
    print(f'{a} vs {b}: n={len(d):,} max|diff|={d.max():.3e} n>1e-6={int((d > 1e-6).sum())}')
for a, b in (('sig_m', 'sig_m_4'), ('entry_m', 'next_entry_m'), ('exit_m_2r', 'next_exit_m_2r')):
    aa = 'sig_m_3' if a == 'sig_m' else a
    d = (ok[aa].astype(float) - ok[b].astype(float)).abs()
    print(f'{a} vs {b}: max|diff|={d.max():.0f} mismatches={int((d > 0).sum())}')
print('why_2r mismatches:', int((ok.why_2r != ok.next_why_2r).sum()))
extra = c4.merge(c3[k].assign(_h=1), on=k, how='left')
extra = extra[extra._h.isna()]
print(f'candidates4 signal rows with no candidates3 counterpart: {len(extra):,} '
      f'(of which no next-open fill: {int(extra.next_entry.isna().sum()):,})')
print('PARITY OK' if (len(missing) == 0 and (ok.entry.astype(float) - ok.next_entry.astype(float)).abs().max() <= 1e-6
                      and (ok.rr_2r.astype(float) - ok.next_rr_2r.astype(float)).abs().max() <= 1e-6) else 'PARITY FAIL')
