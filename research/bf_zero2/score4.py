#!/usr/bin/env python3
"""score4 — the first scoring of this project that satisfies every correction found in the 2026-09-15/16 audits.

What is different from every previous number, and why:
  1. FILL. Reads candidates3.csv, built by build_candidates3.py: the signal is a bar's high reaching the level, the FILL
     is the NEXT bar's open and only if that open is at or under level x 1.006. That is what the live engine does. The
     old pass 1 filled at the touch, and 41% of its trades were filled below the low of the bar that filled them.
  2. COST, per OUTCOME, not flat (the owner's correction, measured 2026-09-16 on 874 trades with quotes at BOTH ends):
     entry always pays half a spread; a TARGET exit rests on a limit and pays nothing; a stop pays half a spread at the
     exit minute, where spreads run 0.875x the entry spread; a 15:55 close pays 0.412x. Spread itself is the measured
     per-price-band figure carried in the candidate row, not the 0.40% that was used before.
  3. BOOK. run_book(rows, 12, 4) - twelve a day, four concurrent - as DESIGN.md pre-registered. The 4/day used before was
     a selection rule that took only the first four signals of the day.
  4. CAUSAL UNIVERSE. F5-F10 require range-so-far >= 5% computed on bars STRICTLY BEFORE the signal bar.
  5. GATES as pre-registered: TRAIN weekly net R >= +10 AND >= 5 trades/week; VAL weekly >= +7 and >= 60% weeks green;
     TEST read once and reported whatever it says. Every cell looked at is counted.
Output: score4_tables.md"""
import os, sys, numpy as np, pandas as pd
ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT); sys.path.insert(0, ROOT)
from trading.hod_break import run_book
D = 'research/bf_zero2'
EXIT_SPREAD_RATIO = {'stop': 0.875, 'eod': 0.412, 'target': 0.0}      # measured 2026-09-16, cost_by_outcome.md
MIN_TPW, TRAIN_WK, VAL_WK, VAL_GREEN = 5.0, 10.0, 7.0, 0.60
c = pd.read_csv(f'{D}/candidates3.csv', dtype={'symbol': str, 'day': str, 'fam': 'category', 'cfg': 'category'},
                keep_default_na=False, na_values=[''], low_memory=True)
c = c[(c.price >= 5) & (c.entry_m <= 841) & (c.r_pct >= 1.0)]
NEED = ~c.fam.isin(['F1', 'F2', 'F3', 'F4'])
c = c[~(NEED & ~(c.range_so_far_pct >= 5))].reset_index(drop=True)
c['split'] = np.where(c.day < '2026-01-01', 'TRAIN', np.where(c.day < '2026-06-01', 'VAL', 'TEST'))
c['wk'] = pd.to_datetime(c.day).dt.to_period('W-FRI').astype(str)
c['key'] = c.fam.astype(str) + ' ' + c.cfg.astype(str)
half = 0.5 * c.spread_pct / c.r_pct.clip(lower=0.05)                  # half a spread, in R units
for tag in ('hold', '2r'):
    why = c[f'why_{tag}']
    c[f'net_{tag}'] = c[f'rr_{tag}'] - half - half * why.map(EXIT_SPREAD_RATIO).fillna(0.875)
WEEKS = {s: c[c.split == s].wk.nunique() for s in ('TRAIN', 'VAL', 'TEST')}
print(f'rows {len(c):,} | keys {c.key.nunique()} | weeks {WEEKS} | exit mix {c.why_2r.value_counts(normalize=True).round(3).to_dict()}', flush=True)


def book(d, tag, split):
    x = d[(d.split == split) & d[f'net_{tag}'].notna()]
    if len(x) < 40: return None
    rows = [(r.day, int(r.entry_m), int(getattr(r, f'exit_m_{tag}')), r.symbol, getattr(r, f'net_{tag}'), r.wk) for r in x.itertuples()]
    t = pd.DataFrame(run_book(rows, 12, 4), columns=['day', 'em', 'xm', 'symbol', 'net', 'wk'])
    if not len(t): return None
    w = t.groupby('wk').net.sum().reindex(sorted(c[c.split == split].wk.unique())).fillna(0)
    return dict(n=len(t), tpw=round(len(t) / WEEKS[split], 1), meanR=round(t.net.mean(), 3),
                t=round(t.net.mean() / (t.net.std() / np.sqrt(len(t))), 2), WR=round((t.net > 0).mean() * 100, 1),
                wkR=round(float(w.mean()), 1), green=round(float((w > 0).mean()), 2), worst=round(float(w.min()), 1))


rows = []; cells = 0
for key, dk in c.groupby('key', observed=True):
    for tag in ('hold', '2r'):
        cells += 1
        r = {sp: book(dk, tag, sp) for sp in ('TRAIN', 'VAL', 'TEST')}
        if r['TRAIN']:
            rows.append(dict(key=key, exit={'hold': 'hold-to-close -1R', '2r': '+2R close-fill'}[tag],
                             **{f'TRAIN_{k}': v for k, v in r['TRAIN'].items()},
                             **{f'VAL_{k}': (r['VAL'] or {}).get(k) for k in ('tpw', 'meanR', 'wkR', 'green')},
                             **{f'TEST_{k}': (r['TEST'] or {}).get(k) for k in ('tpw', 'meanR', 'wkR', 'green')}))
T = pd.DataFrame(rows).sort_values('TRAIN_wkR', ascending=False)
pd.set_option('display.width', 250)
gate1 = T[(T.TRAIN_wkR >= TRAIN_WK) & (T.TRAIN_tpw >= MIN_TPW)]
gate2 = gate1[(gate1.VAL_wkR >= VAL_WK) & (gate1.VAL_green >= VAL_GREEN)]
L = ['# score4 — honest fills, per-outcome costs, the pre-registered book and gates', '',
     f'cells looked at: {cells} | rows {len(c):,}', '',
     '## every family-config x exit, ranked by TRAIN weekly net R', T.head(30).to_string(index=False), '',
     f'## GATE 1 — TRAIN weekly net R >= +{TRAIN_WK} and >= {MIN_TPW} trades/week: {len(gate1)} of {len(T)} pass',
     gate1.to_string(index=False) if len(gate1) else '(none)', '',
     f'## GATE 2 — VAL weekly >= +{VAL_WK} and >= {VAL_GREEN:.0%} weeks green: {len(gate2)} pass',
     gate2.to_string(index=False) if len(gate2) else '(none)', '',
     '## TEST, read once, for anything that cleared both gates',
     gate2[['key', 'exit', 'TEST_tpw', 'TEST_meanR', 'TEST_wkR', 'TEST_green']].to_string(index=False) if len(gate2) else '(nothing cleared the gates — TEST not consulted)']
open(f'{D}/score4_tables.md', 'w').write('\n'.join(L)); print('\n'.join(L[:8])); print('\nGATE1', len(gate1), 'GATE2', len(gate2), flush=True)
T.to_csv(f'{D}/score4_results.csv', index=False); print('DONE', flush=True)
