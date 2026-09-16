#!/usr/bin/env python3
"""Extract the F6 (red-to-green) candidate POOL that feeds the claimed book, and prove parity with
research/bf_zero2/f6_2r_book.csv (the +2R close-fill book, 4 slots / 4 per day).

Pool filters are the ones the claim uses (score3.py / verify_f6.py): fam F6, price >= 5, entry_m <= 841,
r_pct >= 1, range_so_far_pct >= 5. Cost: half of a 40 bps spread in R units, charged on every exit whose
reason is not 'target'.  Output: audit_fills/pool.csv
"""
import os, sys
import numpy as np, pandas as pd
ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT); sys.path.insert(0, ROOT)
from trading.hod_break import run_book

D = 'research/bf_zero2'; A = f'{D}/audit_fills'
USE = ['day', 'symbol', 'fam', 'entry_m', 'entry', 'stop', 'r_pct', 'price', 'rr_e1c', 'why_e1c', 'exit_m_e1c',
       'rr_e4', 'range_so_far_pct', 'adv20', 'gap_pct', 'rv_profile', 'is_wrapper', 'n_bars_at_entry']
chunks = []
n = 0
for ch in pd.read_csv(f'{D}/candidates_full.csv', usecols=USE, dtype={'symbol': str, 'day': str, 'fam': str},
                      keep_default_na=False, na_values=[''], chunksize=500_000, low_memory=True):
    n += len(ch)
    k = ch[(ch.fam == 'F6') & (ch.price >= 5) & (ch.entry_m <= 841) & (ch.r_pct >= 1) & (ch.range_so_far_pct >= 5)]
    if len(k): chunks.append(k)
    print(f'scanned {n:,} rows, pool {sum(len(x) for x in chunks):,}', flush=True)
p = pd.concat(chunks, ignore_index=True)
p['split'] = np.where(p.day < '2026-01-01', 'TRAIN', np.where(p.day < '2026-06-01', 'VAL', 'TEST'))
p['wk'] = pd.to_datetime(p.day).dt.to_period('W-FRI').astype(str)
p['mo'] = p.day.str[:7]
p.to_csv(f'{A}/pool.csv', index=False)
print(f'POOL {len(p):,} rows | days {p.day.nunique()} | symbol-days {p.groupby(["day","symbol"]).ngroups}', flush=True)

# --- parity: rebuild the claimed book from the pool ---
half = 0.5 * 0.40 / p.r_pct.clip(lower=0.05)
p['net'] = p.rr_e1c - np.where(p.why_e1c == 'target', 0.0, half)
rows = [(r.day, int(r.entry_m), int(r.exit_m_e1c), r.symbol, r.net, r.wk) for r in p.itertuples()]
bk = pd.DataFrame(run_book(rows, 4, 4), columns=['day', 'em', 'xm', 'symbol', 'net', 'wk'])
bk['split'] = np.where(bk.day < '2026-01-01', 'TRAIN', np.where(bk.day < '2026-06-01', 'VAL', 'TEST'))
print('\nREBUILT book:'); print(bk.groupby('split').net.agg(['count', 'mean']).round(3).to_string())
claim = pd.read_csv(f'{D}/f6_2r_book.csv')
print('\nCLAIMED book:'); print(claim.groupby('split').net.agg(['count', 'mean']).round(3).to_string())
m = claim[['day', 'symbol', 'net']].merge(bk[['day', 'symbol', 'net']], on=['day', 'symbol'], how='outer', suffixes=('_claim', '_mine'), indicator=True)
print('\nmerge:', m._merge.value_counts().to_dict(), '| max |dnet| =',
      float((m.net_claim - m.net_mine).abs().max()) if (m._merge == 'both').all() else 'n/a')
print('DONE', flush=True)
