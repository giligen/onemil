#!/usr/bin/env python3
"""PART 1 — does the bull-flag conviction score predict anything?

Descriptive only. Every cell is declared in PREREG.md §3. TEST is never touched
here (the diagnostic runs on the whole 2025-01..2026-08 cache and is reported per
YEAR, which is the split convention PREREG §3 declares for Part 1).

Read-only on the cache, config and cache.db. Writes only into research/bf_sizing/.
"""
import sys
import math
import json

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, '/home/ec2-user/onemil')
ROOT = '/home/ec2-user/onemil'
OUT = f'{ROOT}/research/bf_sizing'
CACHE = f'{ROOT}/data/bull_flag_cache_causal_full_20260905.csv'
RUNS = f'{ROOT}/research/bf_frequency/runs'

RNG = np.random.default_rng(20260919)
COMPONENTS = ['conv_pole_gain', 'conv_flag_tightness', 'conv_vol_ratio',
              'conv_spy_regime', 'conv_retracement', 'conv_vwap_dist',
              'conv_gap_fading']

NUM = ('entry_price', 'stop_loss', 'pnl', 'shares', 'conviction_mult',
       'macd_zone_mult', 'avg_volume_20d', 'qf_pole_gain_pct', 'qf_pole_bars',
       'intraday_change_at_entry', 'conv_raw_score') + tuple(COMPONENTS)


def load(path):
    d = pd.read_csv(path, keep_default_na=False, na_values=[''],
                    dtype={'symbol': str})
    for c in NUM:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors='coerce')
    d['rps'] = d['entry_price'] - d['stop_loss']
    return d


df = load(CACHE)
df = df[(df['date'] >= '2025-01-01') & (df['date'] <= '2026-08-31')].copy()
df['R'] = df['pnl'] / (df['rps'] * df['shares'])
df = df[np.isfinite(df['R'])].copy()
df['year'] = df['date'].str[:4]
df['risk'] = df['rps'] * df['shares']
df['m'] = (df['conviction_mult'].fillna(1.0) * df['macd_zone_mult'].fillna(1.0))
df['base_shares'] = df['shares'] / df['m']
df['base_risk'] = df['base_shares'] * df['rps']

log = []


def P(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    log.append(s)


# ---------------------------------------------------------------- §7 sanity --
P('=== PREREG §7 sanity: is base_shares = shares/m usable? ===')
nom = np.floor(2000.0 / df['rps'])
over = (df['base_shares'] > nom * 1.01).mean() * 100
P(f'  picks whose base_shares exceeds floor($2000/rps) by >1%: {over:.1f}% '
  f'(a cap bound AFTER the multiplier in the build for these rows)')
P(f'  base_risk: median ${df["base_risk"].median():,.0f}  '
  f'p25 ${df["base_risk"].quantile(.25):,.0f}  '
  f'p75 ${df["base_risk"].quantile(.75):,.0f}  max ${df["base_risk"].max():,.0f}')
P(f'  shipped risk (cache, multipliers IN): median ${df["risk"].median():,.0f}  '
  f'max ${df["risk"].max():,.0f}')

# --------------------------------------------------------------- populations --
from trading.bf_universe_filter import filter_trades as bf_univ
recs = df.to_dict('records')
keep_ids = {id(x) for x in bf_univ(recs)}
df['univ_ok'] = [id(r) in keep_ids for r in recs]

P_all = df
P_univ = df[df['univ_ok']].copy()
# the population actually entering the conviction gate in the live cascade
P_gate = P_univ[(P_univ['avg_volume_20d'].fillna(0) >= 200000)
                & (P_univ['entry_price'] <= 20.0)
                & (P_univ['qf_pole_gain_pct'] >= 5.0)].copy()


def joinR(run):
    d = load(f'{RUNS}/{run}.csv')
    k = ['symbol', 'date', 'entry_time_et']
    m = d.merge(df[k + ['R', 'base_shares', 'base_risk', 'm', 'year'] + COMPONENTS
                    + ['conv_raw_score', 'avg_volume_20d']],
                on=k, how='left', suffixes=('', '_c'))
    m['risk'] = m['rps'] * m['shares']
    return m


P_pick = joinR('P1')
P_pickF7 = joinR('F7')

POPS = [('P_all', P_all), ('P_univ', P_univ), ('P_gate', P_gate),
        ('P_pick(P1)', P_pick), ('P_pickF7', P_pickF7)]
P('')
P('populations: ' + ' · '.join(f'{n}={len(d)}' for n, d in POPS))


def spear(x, y, nboot=10000):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[ok], np.asarray(y)[ok]
    if len(x) < 8 or np.std(x) == 0:
        return float('nan'), float('nan'), (float('nan'), float('nan')), len(x)
    rho, p = stats.spearmanr(x, y)
    n = len(x)
    bs = np.empty(nboot)
    for i in range(nboot):
        idx = RNG.integers(0, n, n)
        xs, ys = x[idx], y[idx]
        bs[i] = stats.spearmanr(xs, ys)[0] if np.std(xs) > 0 else np.nan
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return float(rho), float(p), (float(lo), float(hi)), n


# ------------------------------------------------------- D1 decile table -----
def deciles(d, col, label, nb=10):
    s = d[np.isfinite(d[col])].copy()
    if len(s) < nb * 2:
        nb = max(3, len(s) // 8)
    try:
        s['b'] = pd.qcut(s[col].rank(method='first'), nb, labels=False)
    except ValueError:
        return
    g = s.groupby('b').agg(n=('R', 'size'), lo=(col, 'min'), hi=(col, 'max'),
                           meanR=('R', 'mean'), medR=('R', 'median'),
                           wr=('R', lambda v: (v > 0).mean() * 100),
                           risk=('risk', 'mean'), pnl=('pnl', 'sum'))
    P(f'\n--- D1 deciles of {col} on {label} (n={len(s)}) ---')
    P('  bin |   n |     range      | meanR | medR  |  WR%  | mean$risk |   sum$')
    for b, r in g.iterrows():
        P(f'  {int(b):3d} | {int(r.n):3d} | {r.lo:6.2f}..{r.hi:6.2f} | '
          f'{r.meanR:+.2f} | {r.medR:+.2f} | {r.wr:5.1f} | {r.risk:9,.0f} | '
          f'{r.pnl:9,.0f}')


P('\n############ D1 — conviction deciles ############')
for nm, d in [('P_gate', P_gate), ('P_univ', P_univ)]:
    deciles(d, 'conviction_mult', nm)
    for y in ('2025', '2026'):
        deciles(d[d.year == y], 'conviction_mult', f'{nm} {y}', nb=5)

# ------------------------------------------------------- D2 conviction ρ -----
P('\n############ D2 — Spearman rho(conviction, R) ############')
P('  population        window     n     rho      p       95% CI')
d2 = []
for nm, d in POPS:
    for w, sub in [('pooled', d), ('2025', d[d.year == '2025']),
                   ('2026', d[d.year == '2026'])]:
        rho, p, ci, n = spear(sub['conviction_mult'].values, sub['R'].values)
        P(f'  {nm:16s} {w:8s} {n:4d}  {rho:+.3f}  {p:6.3f}  '
          f'[{ci[0]:+.3f},{ci[1]:+.3f}]')
        d2.append(dict(pop=nm, window=w, n=n, rho=rho, p=p, lo=ci[0], hi=ci[1]))
pd.DataFrame(d2).to_csv(f'{OUT}/d2_conviction_rho.csv', index=False)

# ------------------------------------------------------- D3 the headline -----
P('\n############ D3 — HEADLINE: Spearman rho(position size $risk, R) ############')
P('  population        window     n   rho(risk,R)   p      95% CI        rho(risk,sign R)')
d3 = []
for nm, d in POPS:
    for w, sub in [('pooled', d), ('2025', d[d.year == '2025']),
                   ('2026', d[d.year == '2026'])]:
        rho, p, ci, n = spear(sub['risk'].values, sub['R'].values)
        rho2, p2, _, _ = spear(sub['risk'].values,
                               np.sign(sub['R'].values), nboot=1)
        P(f'  {nm:16s} {w:8s} {n:4d}  {rho:+.3f}      {p:6.3f}  '
          f'[{ci[0]:+.3f},{ci[1]:+.3f}]   {rho2:+.3f} (p {p2:.3f})')
        d3.append(dict(pop=nm, window=w, n=n, rho_risk_R=rho, p=p, lo=ci[0],
                       hi=ci[1], rho_risk_signR=rho2, p_sign=p2))
pd.DataFrame(d3).to_csv(f'{OUT}/d3_size_rho.csv', index=False)

P('\n--- D3b the $/R reconciliation: cov(R, risk) IS the sizer, in dollars ---')
P('  population        window     n    sum$      n*meanR*meanRisk   cov term $   sizer verdict')
d3b = []
for nm, d in POPS:
    for w, sub in [('pooled', d), ('2025', d[d.year == '2025']),
                   ('2026', d[d.year == '2026'])]:
        if len(sub) < 3:
            continue
        r, k = sub['R'].values, sub['risk'].values
        tot = float((r * k).sum())
        naive = float(len(r) * r.mean() * k.mean())
        cov = tot - naive
        P(f'  {nm:16s} {w:8s} {len(r):4d}  {tot:10,.0f}  {naive:14,.0f}  '
          f'{cov:11,.0f}   {"HELPS" if cov > 0 else "HURTS"}')
        d3b.append(dict(pop=nm, window=w, n=len(r), total=tot, naive=naive,
                        cov_term=cov))
pd.DataFrame(d3b).to_csv(f'{OUT}/d3b_cov_reconciliation.csv', index=False)

# ------------------------------------------------------- D4 components -------
P('\n############ D4 — components: rho(component, R), 2025 / 2026 / pooled ############')
P('  population   component             n    rho25    rho26   rho_pool   p_pool   verdict')
d4 = []
for nm, d in [('P_gate', P_gate), ('P_univ', P_univ), ('P_pickF7', P_pickF7)]:
    for c in COMPONENTS + ['macd_zone_mult', 'conv_raw_score']:
        if c not in d.columns:
            continue
        r25, _, _, n25 = spear(d[d.year == '2025'][c].values,
                               d[d.year == '2025']['R'].values, nboot=1)
        r26, _, _, n26 = spear(d[d.year == '2026'][c].values,
                               d[d.year == '2026']['R'].values, nboot=1)
        rp, pp, cip, n = spear(d[c].values, d['R'].values)
        v = ('ANTI-PREDICTIVE' if (r25 < 0 and r26 < 0)
             else ('predictive' if (r25 > 0 and r26 > 0) else 'noise'))
        P(f'  {nm:11s} {c:20s} {n:4d}  {r25:+.3f}  {r26:+.3f}   {rp:+.3f}   '
          f'{pp:6.3f}   {v}')
        d4.append(dict(pop=nm, comp=c, n=n, rho25=r25, rho26=r26, rho=rp,
                       p=pp, lo=cip[0], hi=cip[1], verdict=v))
pd.DataFrame(d4).to_csv(f'{OUT}/d4_components.csv', index=False)

P('\n--- D4b component deciles on P_gate (the population the gate acts on) ---')
for c in COMPONENTS + ['macd_zone_mult']:
    deciles(P_gate, c, 'P_gate', nb=5)

# ------------------------------------------------------- D5 August 2026 ------
P('\n############ D5 — the +3.0R / -$5,068 August-2026 case, trade by trade ############')
aug = P_pickF7[(P_pickF7['date'] >= '2026-08-01')
               & (P_pickF7['date'] <= '2026-08-31')].copy()
aug = aug.sort_values('R')
P(f'  F7 August-2026: {len(aug)} trades, sum R {aug["R"].sum():+.2f}, '
  f'sum $ {aug["pnl"].sum():+,.0f}')
P('  symbol   date        conv  macd   R      $risk      $pnl')
for r in aug.itertuples():
    P(f'  {r.symbol:8s} {r.date}  {r.conviction_mult:.2f}  '
      f'{r.macd_zone_mult:.2f}  {r.R:+6.2f}  {r.risk:8,.0f}  {r.pnl:9,.0f}')
rr, kk = aug['R'].values, aug['risk'].values
P(f'  mean R {rr.mean():+.3f}  mean $risk {kk.mean():,.0f}  '
  f'n*meanR*meanRisk = {len(rr)*rr.mean()*kk.mean():+,.0f}  '
  f'cov term = {(rr*kk).sum() - len(rr)*rr.mean()*kk.mean():+,.0f}')
rho, p, ci, n = spear(kk, rr)
P(f'  rho(size, R) in August-2026 = {rho:+.3f} (p {p:.3f}, n {n})')
wi = rr > 0
P(f'  mean $risk on WINNERS {kk[wi].mean():,.0f} (n {wi.sum()}) vs '
  f'LOSERS {kk[~wi].mean():,.0f} (n {(~wi).sum()})')
aug.to_csv(f'{OUT}/d5_august2026_F7.csv', index=False)

# same for the P1 arm's August
augp = P_pick[(P_pick['date'] >= '2026-08-01') & (P_pick['date'] <= '2026-08-31')]
if len(augp):
    P(f'\n  P1 August-2026: {len(augp)} trades, sum R {augp["R"].sum():+.2f}, '
      f'sum $ {augp["pnl"].sum():+,.0f}')

# winners-vs-losers size test across the whole book (the anti-predictive test)
P('\n--- D5b mean dollar risk on winners vs losers, every population ---')
P('  population        window    n   $risk winners   $risk losers   ratio   Mann-Whitney p')
for nm, d in POPS:
    for w, sub in [('pooled', d), ('2025', d[d.year == '2025']),
                   ('2026', d[d.year == '2026'])]:
        if len(sub) < 10:
            continue
        w_ = sub['R'] > 0
        a, b = sub.loc[w_, 'risk'].values, sub.loc[~w_, 'risk'].values
        if len(a) < 3 or len(b) < 3:
            continue
        u = stats.mannwhitneyu(a, b, alternative='two-sided')[1]
        P(f'  {nm:16s} {w:8s} {len(sub):4d}  {a.mean():12,.0f}  '
          f'{b.mean():12,.0f}   {a.mean()/b.mean():5.2f}   {u:.3f}')

with open(f'{OUT}/part1_out.txt', 'w') as f:
    f.write('\n'.join(log) + '\n')
print('\nwrote research/bf_sizing/part1_out.txt', flush=True)
