#!/usr/bin/env python3
"""D1 step 8 — assemble the numeric tables (cells, gates, calibration, importance, per-month) into D1/results.md.

`python3 d1_tables.py [periods...]` — default trainpred val. Adding `test` only works after cells_test.csv exists.
Nothing is decided here; the gate arithmetic is printed exactly as PREREG §0.8 defines it.
"""
import os, sys
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT)
sys.path.insert(0, f'{ROOT}/research/fuckup_audit/D1')
import d1_core as K

D1 = 'research/fuckup_audit/D1'
TAG = os.environ.get('D1_TAG', '')
PER = sys.argv[1:] or ['trainpred', 'val']
pd.set_option('display.width', 260); pd.set_option('display.max_rows', 900)
L = ['# D1 — results tables (assembled by d1_tables.py; the report reads from here)', '']

cells = {}
for per in PER:
    f = f'{D1}/cells_{per}{TAG}.csv'
    if not os.path.exists(f):
        continue
    c = pd.read_csv(f, keep_default_na=False, na_values=[''])
    cells[per] = c
    show = [x for x in ['tgt', 'key', 'model', 'sel', 'tape', 'n', 'tpw', 'meanR', 'se', 't', 'mde', 'WR', 'wkR',
                        'green', 'worst', 'mix_stop', 'mix_target'] if x in c.columns]
    L += [f'## cells — {per}', '', c[show].to_string(index=False), '']

# ---- gates
if 'trainpred' in cells and 'val' in cells:
    tp = cells['trainpred']; va = cells['val']
    m = tp[(tp.tape == 'real') & tp.model.isin(['reg', 'clf', 'base'])].merge(
        va[(va.tape == 'real')], on=['tgt', 'key', 'model', 'sel'], suffixes=('_tp', '_val'))
    m['G1'] = (m.meanR_tp > 0) & (m.t_tp >= 2.0) & (m.tpw_tp >= 5.0)
    g1n = int(m.G1.sum())
    bar = (g1n // 10)
    m['G2'] = (m.meanR_val > 0) & (m.t_val >= 1.0) & (m.green_val >= 0.55)
    # the VAL bar is raised by 1 SE of weekly R per 10 cells that passed G1 — weekly SE is not in cells_*.csv,
    # so the raised bar is reported in the report text from the booked weekly series when it binds (bar = 0 here
    # whenever fewer than 10 cells pass G1).
    L += ['## gates (real tape, the 48 declared model cells)', '',
          f'G1 (TRAINPRED 2025-10..12: meanR>0, t>=2, >=5 trades/wk): {g1n} of {len(m)}',
          f'G2 (VAL: meanR>0, t>=1, >=55% weeks green; bar raised by {bar} x weekly SE): '
          f'{int((m.G1 & m.G2).sum())} of the G1 passes, {int(m.G2.sum())} of {len(m)} on the arithmetic alone', '',
          m[['tgt', 'key', 'model', 'sel', 'n_tp', 'meanR_tp', 't_tp', 'tpw_tp', 'G1',
             'n_val', 'meanR_val', 't_val', 'green_val', 'wkR_val', 'G2']].to_string(index=False), '']
    # reversed-tape gate on the same cells
    rv = va[va.tape == 'rev'][['tgt', 'key', 'model', 'sel', 'meanR', 't', 'green', 'wkR']]
    rv.columns = ['tgt', 'key', 'model', 'sel', 'rev_meanR', 'rev_t', 'rev_green', 'rev_wkR']
    mm = m.merge(rv, on=['tgt', 'key', 'model', 'sel'], how='left')
    mm['REV_FAIL'] = mm.rev_meanR > 0
    L += ['## Nagel reversed-tape gate on VAL (a cell with rev_meanR > 0 FAILS)', '',
          mm[['tgt', 'key', 'model', 'sel', 'meanR_val', 't_val', 'rev_meanR', 'rev_t', 'rev_green',
              'REV_FAIL']].to_string(index=False), '']
    sh = va[va.tape == 'shuf'][['tgt', 'key', 'model', 'sel', 'meanR', 't']]
    sh.columns = ['tgt', 'key', 'model', 'sel', 'shuf_meanR', 'shuf_t']
    L += ['## shuffled-target twin on VAL', '', sh.to_string(index=False), '']
    # tail tests on the real-tape VAL cells
    tl = [x for x in ['tgt', 'key', 'model', 'sel', 'meanR', 'cut1_meanR', 'cut5_meanR', 'cap3_meanR', 'wkR',
                      'cut5_wkR', 'cap3_wkR'] if x in va.columns]
    L += ['## tail test on VAL (top 1% / top 5% removed, winners capped at +3R)', '',
          va[va.tape == 'real'][tl].to_string(index=False), '']

# ---- calibration
for per in PER:
    f = f'{D1}/calib_{per}{TAG}.csv'
    if os.path.exists(f):
        c = pd.read_csv(f)
        r = c.groupby(['tgt', 'key', 'model']).rho.first().reset_index()
        L += [f'## decile calibration rho — {per} (predicted decile vs realised mean net R)', '',
              r.to_string(index=False), '']

# ---- importance
f = f'{D1}/importance{TAG}.csv'
if os.path.exists(f):
    imp = pd.read_csv(f)
    g = imp.groupby(['key', 'feat']).imp.agg(['mean', 'std', 'size']).reset_index()
    g['rank'] = g.groupby('key')['mean'].rank(ascending=False)
    top = g.sort_values(['key', 'mean'], ascending=[True, False]).groupby('key').head(10)
    L += ['## permutation importance (primary target, real tape) — top 10 per family, mean over the monthly refits',
          '', top.to_string(index=False), '']
    # stability: the rank of each feature across refits
    rk = imp.copy()
    rk['r'] = rk.groupby(['key', 'month']).imp.rank(ascending=False)
    st = rk.groupby(['key', 'feat']).r.agg(['mean', 'std']).reset_index().sort_values(['key', 'mean'])
    L += ['## importance rank stability across the monthly refits (mean and sd of the rank)', '',
          st.groupby('key').head(8).to_string(index=False), '']

# ---- per-month of the booked real-tape cells
for per in PER:
    f = f'{D1}/booked_{per}{TAG}.csv'
    if os.path.exists(f):
        b = pd.read_csv(f, dtype={'day': str, 'month': str}, keep_default_na=False, na_values=[''])
        mm = b.groupby(['tgt', 'key', 'model', 'sel', 'month']).net.agg(['size', 'sum', 'mean']).round(3)
        L += [f'## per-month booked net R — {per}', '', mm.to_string(), '']

open(f'{D1}/results{TAG}.md', 'w').write('\n'.join(L))
print('wrote', f'{D1}/results{TAG}.md', len(L), 'lines', flush=True)
