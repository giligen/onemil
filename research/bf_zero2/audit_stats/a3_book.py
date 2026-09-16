#!/usr/bin/env python3
"""a3 — Q3 (block bootstrap), Q4 (stability), Q5 (selection), Q6 (power) on the F6 +2R book."""
import os, sys
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT); sys.path.insert(0, ROOT)
A = 'research/bf_zero2/audit_stats'
rng = np.random.default_rng(20260916)
OUT = []
def P(*a):
    s = ' '.join(str(x) for x in a); OUT.append(s); print(s, flush=True)

book = pd.read_csv('research/bf_zero2/f6_2r_book.csv')
book['day'] = book.day.astype(str)
pool = pd.read_parquet(f'{A}/pool_f6.parquet')
pool['day'] = pool.day.astype(str)
SPL = ('TRAIN', 'VAL', 'TEST')
NWK = {s: pool[pool.split == s].wk.nunique() for s in SPL}
NDAY = {s: pool[pool.split == s].day.nunique() for s in SPL}

# =====================================================================================
P('# Q3 — block bootstrap of the weekly and daily series\n')
P(f'weeks per split {NWK} | trading days {NDAY}')

def series(df, unit, split, allunits):
    g = df[df.split == split].groupby(unit, observed=True).net.sum()
    return g.reindex(sorted(allunits)).fillna(0.0).values

allwk = {s: sorted(pool[pool.split == s].wk.unique()) for s in SPL}
allday = {s: sorted(pool[pool.split == s].day.unique()) for s in SPL}
NB = 10_000

def boot(vals, nb, block):
    """Circular block bootstrap of the mean of `vals`; returns nb resampled means."""
    n = len(vals)
    nblk = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(nb, nblk))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(nb, -1) % n
    return vals[idx[:, :n]].mean(axis=1)

res = []
for unit, allu, blk in (('week', allwk, 2), ('day', allday, 5)):
    for s in SPL:
        v = series(book, 'wk' if unit == 'week' else 'day', s, allu[s])
        # convert daily to a weekly-equivalent scale for comparability
        scale = 1.0 if unit == 'week' else (len(allday[s]) / len(allwk[s]))
        m = boot(v, NB, blk) * scale
        res.append(dict(unit=unit, split=s, n_units=len(v), obs=v.mean() * scale,
                        lo=np.percentile(m, 2.5), hi=np.percentile(m, 97.5),
                        p_below_0=(m < 0).mean(), p_below_2=(m < 2).mean(),
                        p_below_5=(m < 5).mean(), sd=m.std()))
    # pooled
    v = np.concatenate([series(book, 'wk' if unit == 'week' else 'day', s, allu[s]) for s in SPL])
    scale = 1.0 if unit == 'week' else (sum(NDAY.values()) / sum(NWK.values()))
    m = boot(v, NB, blk) * scale
    res.append(dict(unit=unit, split='POOLED', n_units=len(v), obs=v.mean() * scale,
                    lo=np.percentile(m, 2.5), hi=np.percentile(m, 97.5),
                    p_below_0=(m < 0).mean(), p_below_2=(m < 2).mean(),
                    p_below_5=(m < 5).mean(), sd=m.std()))
B = pd.DataFrame(res)
P('\nmean R per week (daily series rescaled to a week), 10,000 circular block resamples')
P('  (block = 2 weeks for the weekly series, 5 days for the daily series)')
P(B.round(3).to_string(index=False))

# =====================================================================================
P('\n\n# Q4 — stability\n')
b = book.copy()
b['hour'] = (b.em // 60)
b['m5'] = pd.cut(b.em, [569, 575, 580, 585, 590, 600, 842])
b['pband'] = pd.cut(b.price, [5, 10, 20, 50, 1e9])
wrap = pool.drop_duplicates(['day', 'symbol'])[['day', 'symbol', 'is_wrapper', 'adv20', 'rv_profile', 'gap_pct']]
b = b.merge(wrap, on=['day', 'symbol'], how='left')

def tbl(col, label):
    P(f'\n## by {label}')
    for s in SPL:
        x = b[b.split == s]
        P(f'  {s:6} ' + ' | '.join(f'{str(i)} {g.net.mean():+.3f} (n{len(g)}, {g.net.sum():+.0f}R)'
                                   for i, g in x.groupby(col, observed=True)))

tbl('mo', 'month')
tbl('pband', 'price band')
tbl('m5', 'entry-minute band')
tbl('is_wrapper', 'leveraged-ETF wrapper flag')
tbl(pd.cut(b.r_pct, [1, 2, 4, 8, 1e9]), 'R size (% of price)')

P('\n## concentration — largest single symbol / single day as a share of the split profit')
P(f"{'split':6} {'total R':>9} {'top sym':>10} {'sym R':>8} {'share':>7} {'top day':>12} {'day R':>8} {'share':>7} {'top trade share':>16}")
for s in SPL:
    x = b[b.split == s]; tot = x.net.sum()
    sy = x.groupby('symbol').net.sum().sort_values(ascending=False)
    dy = x.groupby('day').net.sum().sort_values(ascending=False)
    P(f'{s:6} {tot:9.1f} {sy.index[0]:>10} {sy.iloc[0]:8.1f} {sy.iloc[0]/tot:7.1%} '
      f'{dy.index[0]:>12} {dy.iloc[0]:8.1f} {dy.iloc[0]/tot:7.1%} {x.net.max()/tot:16.1%}')
    P(f'       top-5 symbols {sy.head(5).sum()/tot:.1%} of profit ({list(sy.head(5).index)}); '
      f'top-5 days {dy.head(5).sum()/tot:.1%}; symbols with >10% of profit: '
      f'{list(sy[sy/tot > 0.10].index)}; days with >10%: {list(dy[dy/tot > 0.10].index)}')

P('\n## trimming the best trades (and, for contrast, the worst)')
P(f"{'split':6} {'n':>5} {'total R':>9} | {'drop best 1%':>13} {'5%':>8} {'10%':>8} | {'drop worst 1%':>14} {'5%':>8} {'10%':>8}")
for s in SPL:
    x = b[b.split == s].net.values; tot = x.sum(); n = len(x)
    srt = np.sort(x)
    row = [f'{s:6} {n:5d} {tot:9.1f} |']
    for q in (0.01, 0.05, 0.10):
        k = int(np.ceil(n * q)); row.append(f'{srt[:n-k].sum():13.1f}' if q == 0.01 else f'{srt[:n-k].sum():8.1f}')
    row.append('|')
    for q in (0.01, 0.05, 0.10):
        k = int(np.ceil(n * q)); row.append(f'{srt[k:].sum():14.1f}' if q == 0.01 else f'{srt[k:].sum():8.1f}')
    P(' '.join(row))
P(f'  (share of profit surviving a 10% best-trim: ' +
  ', '.join(f'{s} {np.sort(b[b.split==s].net.values)[:len(b[b.split==s])-int(np.ceil(len(b[b.split==s])*0.10))].sum()/b[b.split==s].net.sum():.0%}' for s in SPL) + ')')
P('\n  exit-reason mix and the left tail')
for s in SPL:
    x = b[b.split == s]
    P(f'  {s:6} ' + ' | '.join(f'{k} {len(g)} ({g.net.mean():+.2f}R)' for k, g in x.groupby('why_e1c'))
      + f' | worst trade {x.net.min():.2f}R | trades < -2R: {(x.net<-2).sum()} ({(x.net<-2).mean():.1%})')

# =====================================================================================
P('\n\n# Q5 — does the result depend on WHICH four of the day\'s candidates are taken?\n')

def run_order(sub, order, max_day=4, max_conc=4):
    """Greedy book in the given admission order; a candidate is admitted if fewer than
    max_conc already-admitted trades overlap its [entry, exit] interval and the day is not full."""
    take = np.zeros(len(sub), dtype=bool)
    em = sub['entry_m'].values; xm = sub['exit_m'].values
    dcode = sub['dcode'].values
    cur_day = -1; nd = 0; ivs = []
    for i in order:
        d = dcode[i]
        if d != cur_day:
            cur_day = d; nd = 0; ivs = []
        if nd >= max_day:
            continue
        e, x = em[i], xm[i]
        if sum(1 for (a, bb) in ivs if a <= x and e <= bb) >= max_conc:
            continue
        take[i] = True; ivs.append((e, x)); nd += 1
    return take

p = pool[['day', 'symbol', 'entry_m', 'exit_m_e1c', 'net_e1c', 'split', 'wk']].copy()
p = p.rename(columns={'exit_m_e1c': 'exit_m', 'net_e1c': 'net'})
p['symbol'] = p.symbol.astype(str)
p = p.sort_values(['day', 'entry_m', 'symbol']).reset_index(drop=True)
p['dcode'] = pd.factorize(p.day)[0]
p['exit_m'] = p.exit_m.fillna(955).astype(int)
p['entry_m'] = p.entry_m.astype(int)
day_start = p.groupby('dcode', observed=True).indices

def split_means(take):
    out = {}
    for s in SPL:
        m = take & (p.split.values == s)
        out[s] = p.net.values[m].mean() if m.sum() else np.nan
        out[s + '_n'] = int(m.sum())
    return out

# A — the claim: entry minute ascending, alphabetical tie-break
takeA = run_order(p, np.arange(len(p)))
A_ = split_means(takeA)
P('A  first-come, alphabetical tie-break (THE CLAIM): ' +
  ' '.join(f'{s} {A_[s]:+.3f} (n{A_[s+"_n"]})' for s in SPL))

# D — last-come control
ordD = np.lexsort((p.symbol.values, -p.entry_m.values, p.dcode.values))
D_ = split_means(run_order(p, ordD))
P('D  LAST-come control (latest four of the day):   ' +
  ' '.join(f'{s} {D_[s]:+.3f} (n{D_[s+"_n"]})' for s in SPL))

# E — population, no selection
P('E  the whole qualifying population, no book:     ' +
  ' '.join(f'{s} {p.net.values[p.split.values==s].mean():+.3f} (n{(p.split.values==s).sum()})' for s in SPL))

NREP = 200
rows_B, rows_C = [], []
for rep in range(NREP):
    # B — first-come, RANDOM tie-break inside the same minute
    u = rng.random(len(p))
    ordB = np.lexsort((u, p.entry_m.values, p.dcode.values))
    rows_B.append(split_means(run_order(p, ordB)))
    # C — four uniformly random qualifying candidates per day
    u = rng.random(len(p))
    ordC = np.lexsort((u, p.dcode.values))
    rows_C.append(split_means(run_order(p, ordC)))
    if rep % 50 == 0: print(f'  rep {rep}', flush=True)
Bd, Cd = pd.DataFrame(rows_B), pd.DataFrame(rows_C)
for name, df in (('B  first-come, RANDOM tie-break', Bd), ('C  four RANDOM candidates per day', Cd)):
    P(f'\n{name}  ({NREP} draws)')
    P(f"  {'split':6} {'mean':>8} {'sd':>7} {'p2.5':>8} {'p97.5':>8} {'claim':>8} {'pctile of claim':>16} {'n/draw':>8}")
    for s in SPL:
        v = df[s].values
        P(f'  {s:6} {v.mean():+8.3f} {v.std():7.3f} {np.percentile(v,2.5):+8.3f} {np.percentile(v,97.5):+8.3f} '
          f'{A_[s]:+8.3f} {(v < A_[s]).mean():15.1%} {df[s+"_n"].mean():8.0f}')

P('\n## the same thing as R per week (multiply mean R by trades/week)')
for name, df in (('C random-4', Cd),):
    for s in SPL:
        tpw = df[s + '_n'].mean() / NWK[s]
        P(f'  {s:6} random-4: {df[s].mean()*tpw:+.2f} R/wk   vs claim {A_[s]*A_[s+"_n"]/NWK[s]:+.2f} R/wk')

# what the first-come rule is actually selecting
P('\n## what "first-come" selects (population mean R by within-day entry rank)')
p['rk'] = p.groupby('dcode', observed=True).entry_m.rank(method='first').astype(int)
for s in SPL:
    x = p[p.split == s]
    bb = pd.cut(x.rk, [0, 4, 8, 16, 32, 64, 10 ** 6])
    P(f'  {s:6} ' + ' | '.join(f'{str(i)} {g.net.mean():+.3f} (n{len(g)})' for i, g in x.groupby(bb, observed=True)))
P('\n## and the mechanical correlate: R size (% of price) by within-day entry rank')
for s in SPL:
    x = pool[pool.split == s].copy()
    x['rk'] = x.groupby('day', observed=True).entry_m.rank(method='first').astype(int)
    bb = pd.cut(x.rk, [0, 4, 8, 16, 32, 64, 10 ** 6])
    P(f'  {s:6} ' + ' | '.join(f'{str(i)} r_pct {g.r_pct.median():.2f}%' for i, g in x.groupby(bb, observed=True)))

# =====================================================================================
P('\n\n# Q6 — power\n')
P(f"{'split':6} {'n':>5} {'sd(R)':>7} {'se':>7} {'observed mean':>14} {'MDE 80% power':>14} {'MDE in R/wk':>12} {'obs/MDE':>8}")
for s in SPL:
    x = book[book.split == s].net.values
    n, sd = len(x), x.std(ddof=1)
    se = sd / np.sqrt(n)
    mde = (1.959964 + 0.8416212) * se
    tpw = n / NWK[s]
    P(f'{s:6} {n:5d} {sd:7.3f} {se:7.4f} {x.mean():+14.4f} {mde:14.4f} {mde*tpw:12.2f} {x.mean()/mde:8.2f}')
x = book.net.values
se = x.std(ddof=1) / np.sqrt(len(x))
P(f'{"POOLED":6} {len(x):5d} {x.std(ddof=1):7.3f} {se:7.4f} {x.mean():+14.4f} {(1.959964+0.8416212)*se:14.4f}')
P('\n  MDE = (z_.975 + z_.80) x se, two-sided 5%. Weekly-series power (the quantity the claim is stated in):')
for s in SPL:
    w = book[book.split == s].groupby('wk').net.sum().reindex(sorted(pool[pool.split == s].wk.unique())).fillna(0).values
    se = w.std(ddof=1) / np.sqrt(len(w))
    P(f'  {s:6} weeks {len(w):3d} sd {w.std(ddof=1):6.2f} se {se:5.2f} observed {w.mean():+6.2f} R/wk  MDE {(1.959964+0.8416212)*se:5.2f} R/wk')

open(f'{A}/q3456_book.md', 'w').write('\n'.join(OUT))
Bd.to_csv(f'{A}/sel_random_tiebreak.csv', index=False); Cd.to_csv(f'{A}/sel_random4.csv', index=False)
B.to_csv(f'{A}/bootstrap_weekly.csv', index=False)
print('DONE', flush=True)
