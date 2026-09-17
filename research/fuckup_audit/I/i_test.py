#!/usr/bin/env python3
"""Stage I — the TEST read (once), after FREEZE.md was closed. S0/S1/S2/S3 on the `hold` exit only.

Plus the tail tests on the frozen cell (S0 hold) across all three splits, and the frozen stack's per-trade CSV.
"""
import numpy as np, pandas as pd
from icore import D, STACKS, load, weeks_of, dedupe, book, stats

OUT = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    OUT.append(s); print(s, flush=True)


c = load()
W = weeks_of(c)
say('# Stage I Part A — TEST (read once, per FREEZE.md; descriptive, no selection)')
say('')
say('| stack | exit | split | n | tr/wk | mean net R | gross | t | WR% | wkR | weeks green | worst wk | MDD | '
    'months green | ex-top5% | cap+3R | family mix |')
say('|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|')
rows = []
for sname, fams in STACKS.items():
    dd = dedupe(c, fams)
    t = book(dd[dd.split == 'TEST'], 'hold')
    st = stats(t, 'TEST', W); st.update(stack=sname, exit='hold')
    rows.append(st)
    say(f"| {sname} | hold | TEST | {st['n']} | {st['tpw']} | {st['meanR']:+.3f} | {st['gross']:+.3f} | "
        f"{st['t']:.2f} | {st['WR']} | {st['wkR']:+.2f} | {st['green']:.2f} | {st['worst']:.1f} | {st['mdd']:.1f} | "
        f"{st['moG']} | {st['ex5']:+.3f} | {st['cap3']:+.3f} | {st['mix']} |")
pd.DataFrame(rows).to_csv(f'{D}/a_test_cells.csv', index=False)
say('')
say('Reference (already public, `H/F6/f6_pdr_book.md` exit hold | PDR>=8, TEST): n 376, +0.104 R, t 0.74, '
    '26.9 tr/wk, 43% weeks green, worst week -13.3, MDD -26.9.')
say('')

# ---------- tails on the frozen cell ----------
say('## Tail tests on the FROZEN cell (S0 = F6 alone, hold)')
say('')
say('| split | n | mean net R | top-1% removed | top-5% removed | winners capped at +3R | '
    'share of total R in the top 5% of trades |')
say('|---|---:|---:|---:|---:|---:|---:|')
s0 = dedupe(c, STACKS['S0'])
frozen = []
for sp in ('TRAIN', 'VAL', 'TEST'):
    t = book(s0[s0.split == sp], 'hold')
    t['split'] = sp
    frozen.append(t)
    v = np.sort(t.net.values); n = len(v)
    k1, k5 = int(n * 0.99), int(n * 0.95)
    top5 = v[k5:].sum()
    say(f'| {sp} | {n} | {v.mean():+.3f} | {v[:k1].mean():+.3f} | {v[:k5].mean():+.3f} | '
        f'{np.minimum(v, 3).mean():+.3f} | {top5 / v.sum() * 100:,.0f}% |')
say('')
f = pd.concat(frozen, ignore_index=True)
f.to_csv(f'{D}/frozen_stack_trades.csv', index=False)
say(f'Frozen per-trade CSV: `{D}/frozen_stack_trades.csv` ({len(f)} booked trades, all three splits).')
say('')

# monthly R of the frozen cell
m = f.groupby('mo').net.agg(['sum', 'count'])
say('monthly net R (frozen cell): ' + ' | '.join(f"{i} {r['sum']:+.1f} ({int(r['count'])})" for i, r in m.iterrows()))
say(f"months green {int((m['sum'] > 0).sum())}/{len(m)}")
say('')

open(f'{D}/a_test.md', 'w').write('\n'.join(OUT) + '\n')
print('wrote', f'{D}/a_test.md', flush=True)
