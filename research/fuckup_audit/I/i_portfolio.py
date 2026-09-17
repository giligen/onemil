#!/usr/bin/env python3
"""Stage I Part B — the portfolio of sleeves, monthly $. No new simulation: each sleeve's own declared artefact.

(i)   intraday stack  — this stage's frozen cell, liquidity-capped twin at $300 risk (I/cap_monthly.csv)
(ii)  QQQ noise band  — H/QQQ/final_book_days.csv `r1x` x $60,000 (unfiltered base book, 0.5 bp/leg)
(iii) ORB B+          — analysis_results/orb_bplus_book.csv `_sized_pnl` (orb.yaml B+ stage: $10K / 3 / $375)
(iv)  TQQQ 1x         — Q/step3_months.csv book 'TQQQ 1x notional (0.5bp/leg)' (monthly only, $60K notional)
"""
import os, sys
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT); sys.path.insert(0, ROOT)
D = 'research/fuckup_audit/I'
OUT = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    OUT.append(s); print(s, flush=True)


# ---- (i) intraday stack, capped twin
cap = pd.read_csv(f'{D}/cap_monthly.csv', keep_default_na=False, na_values=[''])
intraday = cap.set_index('mo').usd

# ---- (ii) QQQ
q = pd.read_csv(f'{D}/../H/QQQ/final_book_days.csv', keep_default_na=False, na_values=[''])
q['mo'] = q.date.str[:7]
qqq = (q.groupby('mo').r1x.sum() * 60_000)
qqq_all = qqq.copy()

# ---- (iii) ORB
o = pd.read_csv('analysis_results/orb_bplus_book.csv', keep_default_na=False, na_values=[''])
orb = o.groupby('month')._sized_pnl.sum()

# ---- (iv) TQQQ
t3 = pd.read_csv(f'{D}/../Q/step3_months.csv', keep_default_na=False, na_values=[''])
tq = t3[t3.book == 'TQQQ 1x notional (0.5bp/leg)'].set_index('month').usd

MOS = [f'{y}-{m:02d}' for y in (2025, 2026) for m in range(1, 13)]
MOS = [m for m in MOS if '2025-01' <= m <= '2026-09']
P = pd.DataFrame(index=MOS)
P['intraday_stack'] = intraday.reindex(MOS).fillna(0.0)
P['qqq_noise_60k'] = qqq.reindex(MOS).fillna(0.0)
P['orb_bplus_10k'] = orb.reindex(MOS).fillna(0.0)
P['tqqq_1x_60k'] = tq.reindex(MOS).fillna(0.0)
P['SUM_3'] = P.intraday_stack + P.qqq_noise_60k + P.orb_bplus_10k
P['SUM_4'] = P.SUM_3 + P.tqqq_1x_60k
P.round(0).to_csv(f'{D}/b_portfolio_months.csv')

say('# Stage I Part B — the portfolio of sleeves, month by month ($)')
say('')
say('Window = 2025-01 .. 2026-09 (the months all three primary sleeves cover). No forecast: this is the '
    'historical arithmetic of already-declared books, each at its own declared sizing.')
say('')
say('| month | intraday stack ($300 risk, capped) | QQQ noise ($60K 1x) | ORB B+ ($10K stage) | SUM of 3 | '
    'TQQQ 1x ($60K) | SUM of 4 |')
say('|---|---:|---:|---:|---:|---:|---:|')
for m, r in P.iterrows():
    say(f'| {m} | {r.intraday_stack:+,.0f} | {r.qqq_noise_60k:+,.0f} | {r.orb_bplus_10k:+,.0f} | '
        f'{r.SUM_3:+,.0f} | {r.tqqq_1x_60k:+,.0f} | {r.SUM_4:+,.0f} |')
say('')

say('## Per-sleeve summary over the 21 months')
say('')
say('| sleeve | total $ | mean $/month | median $/month | months green | worst month | best month | '
    'max drawdown of the monthly curve | 2025 total | 2026-YTD total |')
say('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
for col in ['intraday_stack', 'qqq_noise_60k', 'orb_bplus_10k', 'SUM_3', 'tqqq_1x_60k', 'SUM_4']:
    v = P[col]
    cs = v.cumsum()
    mdd = float((cs - cs.cummax()).min())
    y25 = v[[m for m in MOS if m.startswith('2025')]].sum()
    y26 = v[[m for m in MOS if m.startswith('2026')]].sum()
    say(f'| {col} | {v.sum():+,.0f} | {v.mean():+,.0f} | {v.median():+,.0f} | {int((v > 0).sum())}/{len(v)} | '
        f'{v.min():+,.0f} ({v.idxmin()}) | {v.max():+,.0f} ({v.idxmax()}) | {mdd:,.0f} | {y25:+,.0f} | '
        f'{y26:+,.0f} |')
say('')

say('## Pairwise monthly correlations (21 months)')
say('')
cols = ['intraday_stack', 'qqq_noise_60k', 'orb_bplus_10k', 'tqqq_1x_60k']
cm = P[cols].corr()
say('| | ' + ' | '.join(cols) + ' |')
say('|---|' + '---:|' * len(cols))
for a in cols:
    say(f'| {a} | ' + ' | '.join(f'{cm.loc[a, b]:+.2f}' for b in cols) + ' |')
say('')
say(f'Sum of the three sleeves\' standalone monthly standard deviations: '
    f'${P[["intraday_stack","qqq_noise_60k","orb_bplus_10k"]].std().sum():,.0f}; '
    f'standard deviation of their SUM: ${P.SUM_3.std():,.0f} '
    f'(diversification ratio {P.SUM_3.std() / P[["intraday_stack","qqq_noise_60k","orb_bplus_10k"]].std().sum():.2f}).')
say('')

# QQQ over its own full history, for context
say('## Context: the QQQ sleeve over its whole file (2016-01 .. 2026-09), $60K 1x')
say('')
qa = qqq_all
cs = qa.cumsum()
say(f'months {len(qa)} | total ${qa.sum():+,.0f} | mean ${qa.mean():+,.0f}/month | '
    f'months green {int((qa > 0).sum())}/{len(qa)} | worst month ${qa.min():+,.0f} ({qa.idxmin()}) | '
    f'monthly-curve MDD ${float((cs - cs.cummax()).min()):,.0f}')
say('')
open(f'{D}/b_portfolio.md', 'w').write('\n'.join(OUT) + '\n')
print('wrote', f'{D}/b_portfolio.md', flush=True)
