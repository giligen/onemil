#!/usr/bin/env python3
"""a4 — follow-ups the six questions imply:
  (1) the honest (bootstrap, non-normal) per-split p-value of the book's t;
  (2) the first-come effect measured WITHIN day (day fixed effects) and within R-size strata —
      is "the first four" a signal or a proxy for a tight stop?
  (3) how much of the TRAIN->VAL->TEST rise is the changing wrapper mix;
  (4) cost / stop-slippage stress in bps of PRICE (the book lives in tight-R trades, where a fixed
      % slip is a large number of R);
  (5) how many target->stop conversions zero out each split;
  (the fill model is a5/a6.)
"""
import os, sys, sqlite3
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT); sys.path.insert(0, ROOT)
A = 'research/bf_zero2/audit_stats'
rng = np.random.default_rng(31337)
OUT = []
def P(*a):
    s = ' '.join(str(x) for x in a); OUT.append(s); print(s, flush=True)

book = pd.read_csv('research/bf_zero2/f6_2r_book.csv', keep_default_na=False, na_values=[''])
book['day'] = book.day.astype(str); book['symbol'] = book.symbol.astype(str)
pool = pd.read_parquet(f'{A}/pool_f6.parquet'); pool['day'] = pool.day.astype(str); pool['symbol'] = pool.symbol.astype(str)
SPL = ('TRAIN', 'VAL', 'TEST')
alldays = {s: sorted(pool.day[pool.split == s].unique()) for s in SPL}

# ---------------------------------------------------------------- 1
P('# 1 — the honest p-value: bootstrap null of t, not the normal table\n')
P("H0 imposed by centering each split's trade R at zero; days resampled in 5-day circular blocks, 20,000 reps.")
P(f"{'split':6} {'obs t':>7} {'normal p':>10} {'bootstrap p':>12} {'null t 95%':>11} {'null t 99%':>11} {'ratio':>7}")
from scipy import stats as st
NB = 20_000
for s in SPL:
    b = book[book.split == s]
    g = b.groupby('day').net.agg(['size', 'sum', lambda v: (v ** 2).sum()])
    g.columns = ['n', 'sm', 'ss']
    idxmap = {d: i for i, d in enumerate(alldays[s])}
    nd = len(alldays[s])
    arr = np.zeros((3, nd)); ii = np.array([idxmap[d] for d in g.index])
    arr[0, ii] = g.n.values; arr[1, ii] = g.sm.values; arr[2, ii] = g.ss.values
    mu = arr[1].sum() / arr[0].sum()
    arr = np.stack([arr[0], arr[1] - arr[0] * mu, arr[2] - 2 * mu * arr[1] + arr[0] * mu ** 2])
    blk = 5; nblk = int(np.ceil(nd / blk))
    stt = rng.integers(0, nd, size=(NB, nblk))
    idx = (stt[:, :, None] + np.arange(blk)[None, None, :]).reshape(NB, -1)[:, :nd] % nd
    n = arr[0][idx].sum(1); sm = arr[1][idx].sum(1); ss = arr[2][idx].sum(1)
    m = sm / n; v = (ss - sm ** 2 / n) / (n - 1); tb = m / np.sqrt(v / n)
    obs = b.net.mean() / (b.net.std() / np.sqrt(len(b)))
    pnorm = 1 - st.norm.cdf(obs)
    P(f'{s:6} {obs:7.2f} {pnorm:10.5f} {(tb >= obs).mean():12.5f} {np.percentile(tb,95):11.2f} '
      f'{np.percentile(tb,99):11.2f} {(tb>=obs).mean()/max(pnorm,1e-12):7.1f}x')
P('  ("ratio" = how many times larger the honest p is than the normal-table p. > 1 means the t-test')
P('   overstates the evidence, because the R distribution is skewed and the days are dependent.)')

# ---------------------------------------------------------------- 2
P('\n\n# 2 — is "first four of the day" a signal, or a proxy for a tight stop?\n')
p = pool.copy()
p['rk'] = p.groupby('day', observed=True).entry_m.rank(method='first').astype(int)
p['top4'] = p.rk <= 4
P('## within-day contrast: mean R of the day\'s first four MINUS mean R of the same day\'s others')
P("   (day fixed effects — every market-wide regime, volatility and selection-count effect is differenced out)")
P(f"{'split':6} {'days':>5} {'diff':>8} {'se':>7} {'t':>7}")
for s in SPL:
    x = p[p.split == s]
    d = x.groupby('day', observed=True).apply(
        lambda g: pd.Series({'a': g.net_e1c[g.top4].mean(), 'b': g.net_e1c[~g.top4].mean(), 'nb': (~g.top4).sum()}), include_groups=False)
    d = d[d.nb >= 4].dropna()
    dif = (d.a - d.b).values
    P(f'{s:6} {len(dif):5d} {dif.mean():+8.3f} {dif.std(ddof=1)/np.sqrt(len(dif)):7.3f} {dif.mean()/(dif.std(ddof=1)/np.sqrt(len(dif))):7.2f}')

P('\n## the same contrast INSIDE R-size strata (R as % of price) — does it survive the tight-stop control?')
p['rq'] = pd.cut(p.r_pct, [1, 2, 3, 4.5, 7, 1e9], labels=['1-2%', '2-3%', '3-4.5%', '4.5-7%', '7%+'])
for s in SPL:
    x = p[p.split == s]
    line = []
    for q, g in x.groupby('rq', observed=True):
        a, b = g.net_e1c[g.top4], g.net_e1c[~g.top4]
        if len(a) < 20: continue
        se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        line.append(f'{q} {a.mean()-b.mean():+.3f} (t {(a.mean()-b.mean())/se:+.1f}, n{len(a)})')
    P(f'  {s:6} ' + ' | '.join(line))

P('\n## population mean R by R-size, ALL candidates (the mechanical gradient the selection rides)')
for s in SPL:
    x = pool[pool.split == s]
    P(f'  {s:6} ' + ' | '.join(f'{q} {g.net_e1c.mean():+.3f} (n{len(g)})'
                              for q, g in x.groupby(pd.cut(x.r_pct, [1, 2, 3, 4.5, 7, 1e9]), observed=True)))

# ---------------------------------------------------------------- 3
P('\n\n# 3 — how much of the TRAIN -> VAL -> TEST rise is the changing wrapper mix?\n')
w = pool.drop_duplicates(['day', 'symbol'])[['day', 'symbol', 'is_wrapper', 'adv20']]
b = book.merge(w, on=['day', 'symbol'], how='left')
P(f"{'split':6} {'wrapper share':>14} {'wrapper R':>10} {'other R':>9} {'overall':>8} {'TRAIN-mix counterfactual':>26}")
base = b[b.split == 'TRAIN']
wt0 = base.is_wrapper.mean()
for s in SPL:
    x = b[b.split == s]
    mw = x.net[x.is_wrapper == 1].mean(); mo = x.net[x.is_wrapper == 0].mean()
    P(f'{s:6} {x.is_wrapper.mean():14.1%} {mw:10.3f} {mo:9.3f} {x.net.mean():8.3f} {wt0*mw+(1-wt0)*mo:26.3f}')
P('  (the last column re-weights each split to TRAIN\'s wrapper share: what is left is the genuine change)')

# ---------------------------------------------------------------- 4
P('\n\n# 4 — cost stress in bps of PRICE (the book is concentrated in tight-R trades)\n')
P('extra slippage charged on every non-target exit (stops and 15:55 closes), on top of the 40 bps already charged.')
P(f"{'extra bps':>10} " + ' '.join(f'{s:>18}' for s in SPL))
for extra in (0, 10, 20, 30, 50, 75, 100):
    cells = []
    for s in SPL:
        x = book[book.split == s]
        adj = np.where(x.why_e1c == 'target', 0.0, extra / 100.0 / x.r_pct.clip(lower=0.05))
        v = x.net.values - adj
        nwk = pool[pool.split == s].wk.nunique()
        cells.append(f'{v.mean():+.3f}R {v.sum()/nwk:+5.1f}/wk')
    P(f'{extra:10d} ' + ' '.join(f'{c:>18}' for c in cells))
P('\nmedian R as % of price in the book: ' + ' '.join(f'{s} {book[book.split==s].r_pct.median():.2f}%' for s in SPL))
P('  -> 25 bps of extra stop slippage costs about ' +
  f'{0.25/book.r_pct.median():.2f}R on the median trade.')

# ---------------------------------------------------------------- 5
P('\n\n# 5 — how thin is the margin? target->stop conversions that zero out a split\n')
P(f"{'split':6} {'n':>5} {'targets':>8} {'hit rate':>9} {'total R':>8} {'R per conversion':>17} {'conversions to zero':>20} {'= pct-points of hit rate':>25}")
for s in SPL:
    x = book[book.split == s]
    tg = (x.why_e1c == 'target').sum()
    stop_mean = x.net[x.why_e1c == 'stop'].mean()
    per = 2.0 - stop_mean
    k = x.net.sum() / per
    P(f'{s:6} {len(x):5d} {tg:8d} {tg/len(x):9.1%} {x.net.sum():8.1f} {per:17.2f} {k:20.1f} {k/len(x):25.1%}')

open(f'{A}/followups.md', 'w').write('\n'.join(OUT))
print('DONE', flush=True)
