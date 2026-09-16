#!/usr/bin/env python3
"""Stage C verification item (4): a 30-row hand sanity on THREE RANDOM DAYS — for the five families candidates3 and
candidates4 share, `entry_next` / `rr_2r_next` must equal candidates3's `entry` / `rr_2r` EXACTLY (not to 1e-6).
This is the small, readable twin of the full-file parity run (research/fuckup_audit/C/parity_full.log), which
compares all 913,985 shared rows over all 420 days; the point of this one is that the 30 rows are printed.
Usage: ulimit -v 1800000; nice -n 10 python3 research/fuckup_audit/C/verify30.py
"""
import os
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT)
C4 = 'research/fuckup_audit/B/candidates4.csv'
C3 = 'research/bf_zero2/candidates3.csv'
KEYS = {('F1', '{"P": 0.12}'), ('F6', '{}'), ('F8', '{"N": 5}'), ('F8', '{"N": 15}'), ('F8', '{"N": 30}')}
RD = ('symbol', 'day', 'fam', 'cfg')
rng = np.random.default_rng(2026)

days_all = sorted(pd.read_json('research/fuckup_audit/B/build4_state.json')['done'].astype(str).unique())
DAYS = sorted(rng.choice(days_all, 3, replace=False).tolist())
print('random days:', DAYS, flush=True)


def grab(path, cols, dayfilter):
    out = []
    for ch in pd.read_csv(path, usecols=cols, dtype={k: str for k in RD}, keep_default_na=False, na_values=[''],
                          chunksize=400_000, low_memory=True):
        ch = ch[ch.day.isin(dayfilter)]
        if len(ch):
            out.append(ch[[(f, g) in KEYS for f, g in zip(ch.fam, ch.cfg)]])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=cols)


c4 = grab(C4, ['day', 'symbol', 'fam', 'cfg', 'sig_m', 'level', 'stop', 'next_entry', 'next_entry_m', 'next_rr_2r',
               'next_why_2r', 'next_exit_m_2r'], DAYS)
c3 = grab(C3, ['day', 'symbol', 'fam', 'cfg', 'sig_m', 'entry', 'entry_m', 'rr_2r', 'why_2r', 'exit_m_2r'], DAYS)
print(f'candidates4 rows (shared families, 3 days): {len(c4):,} | candidates3 rows: {len(c3):,}', flush=True)

m = c3.merge(c4, on=list(RD), how='left', suffixes=('_3', '_4'))
print('candidates3 rows with no candidates4 row:', int(m.next_entry.isna().sum()))
m = m[m.next_entry.notna()]
s = m.sample(min(30, len(m)), random_state=7)
bad = 0
for r in s.itertuples():
    e_ok = float(r.entry) == float(r.next_entry)
    r_ok = float(r.rr_2r) == float(r.next_rr_2r)
    w_ok = (r.why_2r == r.next_why_2r) and int(r.exit_m_2r) == int(r.next_exit_m_2r)
    bad += (not e_ok) + (not r_ok) + (not w_ok)
    print(f'{"OK " if (e_ok and r_ok and w_ok) else "BAD"} {r.day} {r.symbol:<6} {r.fam} {r.cfg:<12} '
          f'entry c3={r.entry!r} c4={r.next_entry!r} | rr_2r c3={r.rr_2r!r} c4={r.next_rr_2r!r} | '
          f'why {r.why_2r}/{r.next_why_2r} exit_m {r.exit_m_2r}/{int(r.next_exit_m_2r)}')
print(f'\n{len(s)} rows checked — {"ALL EXACT" if bad == 0 else str(bad) + " MISMATCHES"}')
print('whole-3-day max |d entry| =', float((m.entry.astype(float) - m.next_entry.astype(float)).abs().max()),
      '| max |d rr_2r| =', float((m.rr_2r.astype(float) - m.next_rr_2r.astype(float)).abs().max()))
