#!/usr/bin/env python3
"""A0 — every family-config x exit x split under the six cost contracts, booked with run_book(12, 4).

Also the parity anchor: contract (b) must reproduce research/bf_zero2/score4_results.csv cell for cell.
Output: A/a0_cells.csv (long), A/a0_parity.csv, A/a0_tables.md
"""
import os, sys
import numpy as np, pandas as pd
ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT); sys.path.insert(0, f'{ROOT}/research/fuckup_audit/A')
import acore

A = 'research/fuckup_audit/A'
c = acore.load()
WK = acore.week_index(c)
print(f'rows {len(c):,} | keys {c.key.nunique()} | weeks ' +
      str({k: len(v) for k, v in WK.items()}) +
      f" | exit mix {c.why_2r.value_counts(normalize=True).round(3).to_dict()}", flush=True)
print('corrected spread table (bps):'); print(acore.corrected_spread_table().round(1).to_string(), flush=True)
print('gate keeps %.1f%% of candidates (spread/R <= %.2f)' % ((c.sp_over_r_c <= acore.GATE_SP_OVER_R).mean() * 100,
                                                              acore.GATE_SP_OVER_R), flush=True)

out = []
for key, dk in c.groupby('key'):
    for tag in ('hold', '2r'):
        for sp in ('TRAIN', 'VAL', 'TEST'):
            x = dk[dk.split == sp]
            t = acore.book_rows(x, tag)
            if t is None: continue
            row = dict(key=key, exit=tag, split=sp)
            for cc, lab in (('a', 'gross'), ('b', 's4'), ('c', 'corr'), ('cp', 'corrp')):
                for k, v in acore.stats(t, cc, WK[sp]).items():
                    row[f'{lab}_{k}'] = v
            mix = t.why.value_counts(normalize=True)
            for w in ('stop', 'target', 'eod'):
                row[f'mix_{w}'] = round(float(mix.get(w, 0.0)), 3)
            g = acore.book_rows(x[x.sp_over_r_c <= acore.GATE_SP_OVER_R], tag)
            if g is not None:
                for cc, lab in (('c', 'gate'), ('cp', 'gatep')):
                    for k, v in acore.stats(g, cc, WK[sp]).items():
                        row[f'{lab}_{k}'] = v
            out.append(row)
T = pd.DataFrame(out)
T.to_csv(f'{A}/a0_cells.csv', index=False)
print(f'cells written {len(T)}', flush=True)

# ---- parity anchor against score4_results.csv (contract b, TRAIN) ----
s4 = pd.read_csv('research/bf_zero2/score4_results.csv', keep_default_na=False, na_values=[''])
s4['exit'] = s4.exit.map({'hold-to-close -1R': 'hold', '+2R close-fill': '2r'})
m = T[T.split == 'TRAIN'].merge(s4, on=['key', 'exit'], how='outer', indicator=True)
m['d_n'] = m.s4_n - m.TRAIN_n; m['d_mean'] = m.s4_meanR - m.TRAIN_meanR; m['d_wk'] = m.s4_wkR - m.TRAIN_wkR
m[['key', 'exit', '_merge', 's4_n', 'TRAIN_n', 'd_n', 's4_meanR', 'TRAIN_meanR', 'd_mean', 'd_wk']].to_csv(f'{A}/a0_parity.csv', index=False)
print('PARITY vs score4 (TRAIN, contract b): merged %d | max |dn| %s | max |dmeanR| %s | max |dwkR| %s' % (
    len(m), m.d_n.abs().max(), round(float(m.d_mean.abs().max()), 4), round(float(m.d_wk.abs().max()), 2)), flush=True)

# ---- tables ----
pd.set_option('display.width', 400); pd.set_option('display.max_columns', 80)
def fmt(sp):
    x = T[T.split == sp].copy()
    x = x[['key', 'exit', 'corr_n', 'corr_tpw', 'gross_meanR', 's4_meanR', 'corr_meanR', 'corrp_meanR', 'gate_meanR',
           'gatep_meanR', 'corr_t', 'corr_WR', 'corr_wkR', 'corr_green', 'corr_worst', 'gate_n', 'gate_tpw', 'gate_wkR',
           'mix_stop', 'mix_target', 'mix_eod']].sort_values('corr_meanR', ascending=False)
    return x.to_string(index=False)
L = ['# A0 — the corrected cost contract, every cell', '',
     f'population rows {len(c):,} | keys {c.key.nunique()} | cells {len(T)} (26 keys x 2 exits x 3 splits)', '',
     'corrected spread table (median signal-minute NBBO, bps of price):', '',
     acore.corrected_spread_table().round(1).to_string(), '',
     'gate (spread/R <= 0.15) keeps %.1f%% of candidates' % ((c.sp_over_r_c <= acore.GATE_SP_OVER_R).mean() * 100), '']
for sp in ('TRAIN', 'VAL', 'TEST'):
    L += [f'## {sp}', '', fmt(sp), '']
open(f'{A}/a0_tables.md', 'w').write('\n'.join(L))
print('\n'.join(L[:12]))
print('DONE', flush=True)
