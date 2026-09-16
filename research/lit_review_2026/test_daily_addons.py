#!/usr/bin/env python3
"""RUNBOOK row 15 — the cheap daily add-ons that need no news / float / halt / quote data.
 M30 beta night-day: long the top-beta decile close→open (rolling 60-day beta vs SPY, causal).
 M31 tug-of-war: monthly count of days with overnight>0 and intraday<0; long the top decile next month (close-to-close).
 M38 MAX+loser: Friday close, among the top-decile MAX (max daily return over 20 d) buy the bottom-decile past-week losers, hold 5 d.
 M39 weekend speculation: Thursday close → Friday close in high-volatility $5-20 names.
 M40 turn of month on SPY: buy close T-4, sell close T+3 (and the classical T-1..T+3).
 M41 new 252-day high on >=1.5x volume: buy the close, hold 5 and 10 days.
Long-only everywhere (we cannot short cheaply). Costs by dollar-volume band, round trip, charged once per trade.
Splits fixed; a 4-name book where the rule is cross-sectional. Output daily_addons.md."""
import os, re, numpy as np, pandas as pd
ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT)
d = pd.read_parquet('research/lit_review_2026/daily_panel.parquet',
                    columns=['symbol', 'bar_date', 'close', 'high', 'volume', 'adv20', 'dvol20', 'ret_on', 'ret_id', 'ret_cc', 'ret_on_next', 'vol_ratio', 'dow', 'high52'])
bad = [c for c in d.symbol.cat.categories if re.match(r'^Z[VWX]ZZ|^ZZ', str(c))] if hasattr(d.symbol, 'cat') else []
if bad: d = d[~d.symbol.isin(bad)]
d = d[(d.close >= 5) & d.dvol20.notna()]
for k in ('ret_on', 'ret_id', 'ret_cc', 'ret_on_next'): d = d[d[k].isna() | (d[k].abs() <= 0.5)]
d['symbol'] = d.symbol.astype(str)
d = d.sort_values(['symbol', 'bar_date']).reset_index(drop=True)
g = d.groupby('symbol', observed=True)
for n in (5, 10, 20):
    d[f'fwd{n}'] = g.close.shift(-n) / d.close - 1                      # forward n-day close-to-close
d['split'] = np.where(d.bar_date < '2026-01-01', 'TRAIN', np.where(d.bar_date < '2026-06-01', 'VAL', 'TEST'))
d['cost'] = np.where(d.dvol20 >= 5e7, 6, np.where(d.dvol20 >= 1e7, 12, np.where(d.dvol20 >= 5e6, 25, 40))) / 1e4
L = ['# RUNBOOK row 15 — cheap daily add-ons (long-only, net of costs)', '']

def book(x, ret, name, topn=4, desc=True, score='score'):
    if not len(x): L.append(f'{name}: no rows'); return
    for sp in ('TRAIN', 'VAL', 'TEST'):
        y = x[(x.split == sp) & x[ret].notna()]
        if len(y) < 30: L.append(f'{name} [{sp}]: n {len(y)} — underpowered'); continue
        y = y.sort_values(['bar_date', score], ascending=[True, not desc]).groupby('bar_date').head(topn)
        net = y[ret] - y.cost
        L.append(f'{name} [{sp}]: n {len(y):5d} gross {y[ret].mean()*1e4:+7.1f} bps net {net.mean()*1e4:+7.1f} t {net.mean()/(net.std()/np.sqrt(len(net))):+5.2f} hit {(net>0).mean()*100:4.1f}%')

# M30 beta
spy = d[d.symbol == 'SPY'][['bar_date', 'ret_cc']].rename(columns={'ret_cc': 'mkt'})
b = d[d.dvol20 >= 1e7].merge(spy, on='bar_date', how='left').sort_values(['symbol', 'bar_date'])
gg = b.groupby('symbol', observed=True)
b['cov'] = gg.apply(lambda x: x.ret_cc.rolling(60, min_periods=40).cov(x.mkt)).reset_index(level=0, drop=True)
b['var'] = b.mkt.rolling(60, min_periods=40).var()
b['beta'] = (b['cov'] / b['var']).groupby(b.symbol, observed=True).shift(1)                    # causal
b['score'] = b.beta
book(b[b.beta.notna()], 'ret_on_next', 'M30 top-beta decile, close→open', topn=4)
# M31 tug of war
d['ym'] = d.bar_date.str[:7]; tw = d.assign(tug=((d.ret_on > 0) & (d.ret_id < 0)).astype(int)).groupby(['symbol', 'ym']).agg(tug=('tug', 'sum'), n=('tug', 'size'), dv=('dvol20', 'last')).reset_index()
tw = tw[(tw.n >= 15) & (tw.dv >= 5e6)]; tw['ymn'] = tw.ym.map({m: n for m, n in zip(sorted(tw.ym.unique())[:-1], sorted(tw.ym.unique())[1:])})
mret = d.groupby(['symbol', 'ym']).agg(first=('close', 'first'), last=('close', 'last')).reset_index(); mret['mret'] = mret['last'] / mret['first'] - 1
tw = tw.merge(mret[['symbol', 'ym', 'mret']].rename(columns={'ym': 'ymn', 'mret': 'fwd_m'}), on=['symbol', 'ymn'], how='left')
tw['split'] = np.where(tw.ymn < '2026-01', 'TRAIN', np.where(tw.ymn < '2026-06', 'VAL', 'TEST')); tw['cost'] = 0.0025; tw['bar_date'] = tw.ymn; tw['score'] = tw.tug
book(tw[tw.fwd_m.notna()], 'fwd_m', 'M31 tug-of-war top names, hold next month', topn=10)
# M38 MAX + weekly loser (Friday)
d['max20'] = g.ret_cc.transform(lambda s: s.rolling(20, min_periods=15).max()); d['ret_wk'] = g.close.transform(lambda s: s / s.shift(5) - 1)
fri = d[(d.dow == 4) & d.max20.notna() & d.ret_wk.notna() & (d.adv20 >= 1e5)].copy()
thr = fri[fri.split == 'TRAIN'].max20.quantile(0.9); hi = fri[fri.max20 >= thr].copy(); hi['score'] = hi.ret_wk
book(hi, 'fwd5', 'M38 high-MAX weekly losers, hold 5d', topn=4, desc=False)
# M39 Thursday→Friday speculative
d['vol20'] = g.ret_cc.transform(lambda s: s.rolling(20, min_periods=15).std())
th = d[(d.dow == 3) & (d.close.between(5, 20)) & d.vol20.notna() & (d.dvol20 >= 2e6)].copy()
th['fwd1'] = g.close.shift(-1) / d.close - 1; th['score'] = th.vol20
book(th[th.fwd1.notna()], 'fwd1', 'M39 speculative Thu→Fri', topn=4)
# M40 turn of month, SPY
s = d[d.symbol == 'SPY'][['bar_date', 'close', 'split']].sort_values('bar_date').reset_index(drop=True)
s['mo'] = s.bar_date.str[:7]; s['idx_from_end'] = s.groupby('mo').cumcount(ascending=False); s['idx_from_start'] = s.groupby('mo').cumcount()
for lo, hi_, nm in ((4, 3, 'T-4→T+3'), (1, 3, 'T-1→T+3')):
    ent = s[s.idx_from_end == lo - 1].bar_date.tolist(); rr = []
    for e in ent:
        i = s.index[s.bar_date == e][0]; j = min(i + lo + hi_, len(s) - 1)
        rr.append(dict(bar_date=e, r=s.close.iloc[j] / s.close.iloc[i] - 1, split=s.split.iloc[i]))
    R = pd.DataFrame(rr)
    for sp in ('TRAIN', 'VAL', 'TEST'):
        x = R[R.split == sp]
        if len(x) < 4: continue
        net = x.r - 0.0002
        L.append(f'M40 SPY turn of month {nm} [{sp}]: n {len(x)} mean {net.mean()*1e4:+6.1f} bps hit {(net>0).mean()*100:.0f}%')
# M41 new 252-day high on volume
nh = d[(d.high52.notna()) & (d.close >= d.high52) & (d.vol_ratio >= 1.5) & (d.dvol20 >= 5e6)].copy(); nh['score'] = nh.vol_ratio
book(nh, 'fwd5', 'M41 new 252d high on 1.5x volume, hold 5d', topn=4)
book(nh, 'fwd10', 'M41 new 252d high on 1.5x volume, hold 10d', topn=4)
open('research/lit_review_2026/daily_addons.md', 'w').write('\n'.join(L)); print('\n'.join(L)); print('DONE', flush=True)
