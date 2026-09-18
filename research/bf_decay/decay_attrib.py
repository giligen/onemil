#!/usr/bin/env python3
"""BF decay diagnosis: attribute the 2025 -> 2026 change in R/pick layer by layer.

Reads the honest regen-7 cache (read-only), re-applies the P1 Stage-2 filter chain
in order using the SAME shared modules the BT and live engine use, and reports for
every layer: population, kept/rejected R, and the year-over-year delta.

Diagnosis only. Ships nothing, writes nothing outside research/bf_decay/.
"""
import os
import sys
import json
import math
import sqlite3
from collections import defaultdict

import pandas as pd
import numpy as np

sys.path.insert(0, '/home/ec2-user/onemil')
OUT = '/home/ec2-user/onemil/research/bf_decay'
CACHE = '/home/ec2-user/onemil/data/bull_flag_cache_causal_full_20260905.csv'

def log(*a):
    print(*a); sys.stdout.flush()

# ---------------------------------------------------------------- load
df = pd.read_csv(CACHE, keep_default_na=False, na_values=[''], dtype={'symbol': str})
df = df[(df['date'] >= '2025-01-01') & (df['date'] <= '2026-08-31')].copy()
df['year'] = df['date'].str[:4]
df['month'] = df['date'].str[:7]
for c in ['entry_price', 'stop_loss', 'pnl', 'shares', 'qf_pole_gain_pct',
          'qf_vwap_dist_pct', 'qf_gap_pct', 'conviction_mult', 'macd_zone_mult',
          'intraday_change_at_entry', 'avg_volume_20d', 'qf_pole_bars',
          'daily_range_pct', 'qf_fill_vwap_dist_pct', 'conv_raw_score',
          'qf_spy_return_pct', 'exit_price', 'partial_pnl']:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# R = dollars won / dollars risked at the PLANNED stop on the full position.
df['risk_ps'] = df['entry_price'] - df['stop_loss']
df['risk_usd'] = df['risk_ps'] * df['shares']
df['R'] = df['pnl'] / df['risk_usd']
df = df[np.isfinite(df['R'])].copy()
df['entry_min'] = df['entry_time_et'].str[:2].astype(int) * 60 + df['entry_time_et'].str[3:5].astype(int)

log(f"cache rows in window: {len(df)}  2025={sum(df.year=='2025')} 2026={sum(df.year=='2026')}")

# ---------------------------------------------------------------- regime
con = sqlite3.connect('file:/home/ec2-user/onemil/data/cache.db?mode=ro', uri=True)
spy = pd.read_sql_query(
    "select bar_date, open, high, low, close, volume from daily_bars "
    "where symbol='SPY' order by bar_date", con)
con.close()
spy['bar_date'] = pd.to_datetime(spy['bar_date'])
from trading.regime_helpers import build_regime_lookup, compute_regime_features
regime = build_regime_lookup(spy)
df['regime'] = df['date'].map(regime).fillna('unknown')

feats = compute_regime_features(spy)
feats['d'] = feats['bar_date'].astype(str).str[:10]
vol_map = dict(zip(feats['d'], feats['vol_20_ann']))
df['spy_vol20'] = df['date'].map(vol_map)

# ---------------------------------------------------------------- stats helpers
def stat(sub):
    if len(sub) == 0:
        return dict(n=0, meanR=float('nan'), sd=float('nan'), se=float('nan'),
                    wr=float('nan'), pnl=0.0)
    r = sub['R'].values
    return dict(n=len(r), meanR=float(r.mean()), sd=float(r.std(ddof=1)) if len(r) > 1 else float('nan'),
                se=float(r.std(ddof=1) / math.sqrt(len(r))) if len(r) > 1 else float('nan'),
                wr=float((sub['pnl'] > 0).mean() * 100), pnl=float(sub['pnl'].sum()))

def yr(sub, y):
    return stat(sub[sub.year == y])

def line(label, sub):
    a, b = yr(sub, '2025'), yr(sub, '2026')
    return dict(layer=label,
                n25=a['n'], R25=a['meanR'], WR25=a['wr'],
                n26=b['n'], R26=b['meanR'], WR26=b['wr'],
                dR=(b['meanR'] - a['meanR']) if a['n'] and b['n'] else float('nan'),
                se25=a['se'], se26=b['se'])

rows = []

# ---------------------------------------------------------------- LAYER CHAIN (P1 live config)
from config import Config
cfg = Config._load_yaml_only()
bf = cfg['trading']['bull_flag']
log(f"config: price_max={bf['max_entry_price']} pole={bf['min_pole_gain_pct']} "
    f"conv={cfg['trading']['conviction_scoring']['min_threshold']} "
    f"vwap_gate={bf['vwap_gate']['enabled']} ttf={bf['two_tier_filter']['enabled']}")

from trading.bf_universe_filter import filter_trades as bf_univ
recs = df.to_dict('records')

stages = []
stages.append(('L0 raw detections', df.copy()))

# L1 live universe rule by name
keep_idx = {id(r) for r in bf_univ(recs)}
df['univ_ok'] = [id(r) in keep_idx for r in recs]
cur = df[df.univ_ok].copy()
stages.append(('L1 universe (name rule)', cur))

# L2 min daily volume
min_vol = int(cfg['scanner']['min_daily_volume'])
cur2 = cur[cur['avg_volume_20d'].fillna(0) >= min_vol]
stages.append((f'L2 volume >= {min_vol:,}', cur2))

# L3 price cap
cur3 = cur2[cur2['entry_price'] <= float(bf['max_entry_price'])]
stages.append((f"L3 price <= ${bf['max_entry_price']:.0f}", cur3))

# L4 pole gain
cur4 = cur3[cur3['qf_pole_gain_pct'] >= float(bf['min_pole_gain_pct'])]
stages.append((f"L4 pole >= {bf['min_pole_gain_pct']:.0f}%", cur4))

# L5 conviction
cth = float(cfg['trading']['conviction_scoring']['min_threshold'])
cur5 = cur4[cur4['conviction_mult'].fillna(1.0) >= cth]
stages.append((f'L5 conviction >= {cth}', cur5))

# L6 pole bars
mpb = int(bf.get('max_pole_bars', 0))
cur6 = cur5[(cur5['qf_pole_bars'] > 0) & (cur5['qf_pole_bars'] <= mpb)] if mpb else cur5
stages.append((f'L6 pole_bars <= {mpb}', cur6))

# L7 VWAP gate (shared module)
from trading.bf_vwap_gate import load_vwap_gate_config, filter_trades as vwap_gate
r6 = cur6.to_dict('records')
kept7 = {id(r) for r in vwap_gate(r6, load_vwap_gate_config(bf))}
cur6 = cur6.copy(); cur6['vwap_ok'] = [id(r) in kept7 for r in r6]
cur7 = cur6[cur6.vwap_ok]
stages.append(('L7 VWAP gate', cur7))

# L8 two-tier filter (shared module)
from trading.two_tier_filter import classify_tier, build_features_from_trade, should_keep
tt = bf['two_tier_filter']
r7 = cur7.to_dict('records')
ok8, tiers = [], []
for r in r7:
    ic = r.get('intraday_change_at_entry')
    ic = None if (ic is None or (isinstance(ic, float) and math.isnan(ic))) else float(ic)
    tl = classify_tier(ic, a_tier_lower=float(tt.get('a_tier_lower', 20.0)),
                       extras_lower=float(tt.get('extras_lower', 10.0)))
    tiers.append(tl)
    mzm = float(r.get('macd_zone_mult') or 0.0)
    keep, _ = should_keep(tier=tl, macd_zone_mult=mzm,
                          features=build_features_from_trade(r), cfg=tt)
    ok8.append(keep)
cur7 = cur7.copy(); cur7['tier'] = tiers; cur7['tt_ok'] = ok8
cur8 = cur7[cur7.tt_ok]
stages.append(('L8 two-tier filter', cur8))

for name, s in stages:
    rows.append(line(name, s))

# ---------------------------------------------------------------- per-layer ISOLATED contribution
# For each gate: R of kept vs R of rejected, per year. If rejected > kept in 2026 the
# layer has INVERTED; if kept-minus-rejected shrank it has DECAYED.
gates = []
def gate(name, parent, mask):
    k, rj = parent[mask], parent[~mask]
    g = {'gate': name}
    for y in ('2025', '2026'):
        sk, sr = stat(k[k.year == y]), stat(rj[rj.year == y])
        g[f'keptR{y[2:]}'] = sk['meanR']; g[f'nk{y[2:]}'] = sk['n']
        g[f'rejR{y[2:]}'] = sr['meanR']; g[f'nr{y[2:]}'] = sr['n']
        g[f'sep{y[2:]}'] = sk['meanR'] - sr['meanR'] if sk['n'] and sr['n'] else float('nan')
    g['dsep'] = g['sep26'] - g['sep25']
    return g

gates.append(gate('price <= $20', cur2, cur2['entry_price'] <= 20))
gates.append(gate('pole >= 5%', cur3, cur3['qf_pole_gain_pct'] >= 5.0))
gates.append(gate('conviction >= 1.8', cur4, cur4['conviction_mult'].fillna(1.0) >= 1.8))
gates.append(gate('pole_bars <= 3', cur5, (cur5['qf_pole_bars'] > 0) & (cur5['qf_pole_bars'] <= 3)))
gates.append(gate('VWAP gate', cur6, cur6['vwap_ok']))
gates.append(gate('two-tier filter', cur7, cur7['tt_ok']))
# MACD zone multiplier is a SIZING layer, evaluate as high vs low
gates.append(gate('MACD zone mult >= 1.5', cur8, cur8['macd_zone_mult'].fillna(1.0) >= 1.5))
# conviction as a ranker WITHIN the survivors
gates.append(gate('conv >= median(1.9) within book', cur8, cur8['conviction_mult'].fillna(1.0) >= 1.9))

# ---------------------------------------------------------------- SELECTION INVERSION TEST
# Whole-stack: picked (cur8) vs rejected (everything in cur2 that the stack dropped)
pick_keys = set(zip(cur8['symbol'], cur8['date'], cur8['entry_time_et']))
cur2 = cur2.copy()
cur2['picked'] = [ (s,d,t) in pick_keys for s,d,t in zip(cur2['symbol'], cur2['date'], cur2['entry_time_et'])]
sel = gate('WHOLE STACK picked vs rejected', cur2, cur2['picked'])

# ---------------------------------------------------------------- population comparison
def popcmp(col, q=None):
    o = {}
    for y in ('2025', '2026'):
        s = df[df.year == y][col].dropna()
        o[y] = dict(n=len(s), mean=float(s.mean()), med=float(s.median()))
    return o

pop = {}
for c in ['entry_price', 'qf_pole_gain_pct', 'qf_gap_pct', 'qf_vwap_dist_pct',
          'avg_volume_20d', 'daily_range_pct', 'intraday_change_at_entry',
          'entry_min', 'conviction_mult', 'macd_zone_mult', 'qf_pole_bars',
          'spy_vol20', 'risk_ps']:
    pop[c] = popcmp(c)

# detections per month
det_month = df.groupby(['year']).agg(n=('R','size')).to_dict()
months25 = df[df.year=='2025']['month'].nunique()
months26 = df[df.year=='2026']['month'].nunique()

# regime mix
reg_mix = df.groupby(['year','regime']).size().unstack(fill_value=0)
reg_R = df.groupby(['year','regime'])['R'].agg(['size','mean']).unstack()

# ---------------------------------------------------------------- outcome side
# MFE proxy is not in the cache; use exit_reason mix + R distribution + bars held proxy
exit_mix = df.groupby(['year','exit_reason']).agg(n=('R','size'), meanR=('R','mean')).reset_index()
exit_mix_pick = cur8.groupby(['year','exit_reason']).agg(n=('R','size'), meanR=('R','mean')).reset_index()

def tail(sub):
    r = sub['R'].values
    if len(r)==0: return {}
    return dict(n=len(r), meanR=r.mean(), med=float(np.median(r)),
                p90=float(np.percentile(r,90)), max=float(r.max()),
                lossmean=float(r[r<0].mean()) if (r<0).any() else float('nan'),
                winmean=float(r[r>0].mean()) if (r>0).any() else float('nan'),
                wr=float((r>0).mean()*100),
                ge2R=float((r>=2).mean()*100),
                extop1=float(np.sort(r)[:-max(1,len(r)//100)].mean()) if len(r)>2 else float('nan'),
                extop5=float(np.sort(r)[:-max(1,len(r)//20)].mean()) if len(r)>2 else float('nan'))

tails = {'raw2025': tail(df[df.year=='2025']), 'raw2026': tail(df[df.year=='2026']),
         'pick2025': tail(cur8[cur8.year=='2025']), 'pick2026': tail(cur8[cur8.year=='2026'])}

# ---------------------------------------------------------------- MDE / noise
def mde(n1, n2, sd, alpha=0.05, power=0.80):
    if not n1 or not n2 or not np.isfinite(sd): return float('nan')
    return (1.96 + 0.84) * sd * math.sqrt(1/n1 + 1/n2)

p25, p26 = cur8[cur8.year=='2025'], cur8[cur8.year=='2026']
sd_pool = math.sqrt(((len(p25)-1)*p25['R'].var(ddof=1) + (len(p26)-1)*p26['R'].var(ddof=1)) /
                    (len(p25)+len(p26)-2))
obs = p26['R'].mean() - p25['R'].mean()
se_diff = math.sqrt(p25['R'].var(ddof=1)/len(p25) + p26['R'].var(ddof=1)/len(p26))
tstat = obs / se_diff
from math import erf
def norm_cdf(x): return 0.5*(1+erf(x/math.sqrt(2)))
pval = 2*(1-norm_cdf(abs(tstat)))

# bootstrap CI on the delta
rng = np.random.default_rng(7)
a, b = p25['R'].values, p26['R'].values
boot = np.array([rng.choice(b, len(b), True).mean() - rng.choice(a, len(a), True).mean()
                 for _ in range(20000)])
ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))

noise = dict(n25=len(a), n26=len(b), R25=float(a.mean()), R26=float(b.mean()),
             delta=float(obs), se=float(se_diff), t=float(tstat), p=float(pval),
             ci95=ci, sd_pool=float(sd_pool), mde80=float(mde(len(a), len(b), sd_pool)))

# raw-pool version (big n) for contrast
ra, rb = df[df.year=='2025']['R'].values, df[df.year=='2026']['R'].values
se_raw = math.sqrt(ra.var(ddof=1)/len(ra) + rb.var(ddof=1)/len(rb))
noise_raw = dict(n25=len(ra), n26=len(rb), R25=float(ra.mean()), R26=float(rb.mean()),
                 delta=float(rb.mean()-ra.mean()), se=float(se_raw),
                 t=float((rb.mean()-ra.mean())/se_raw))

# separation-spread noise: is the picked-vs-rejected spread significantly smaller in 26?
def sep_test(y):
    s = cur2[cur2.year==y]
    k, r = s[s.picked]['R'].values, s[~s.picked]['R'].values
    d = k.mean()-r.mean()
    se = math.sqrt(k.var(ddof=1)/len(k) + r.var(ddof=1)/len(r))
    return dict(year=y, nk=len(k), nr=len(r), keptR=float(k.mean()), rejR=float(r.mean()),
                sep=float(d), se=float(se), t=float(d/se))
seps = [sep_test('2025'), sep_test('2026')]

# ---------------------------------------------------------------- exits vs entries
# Same-entry counterfactual: R if the trade were exited at a fixed +2R target / -1R stop
# cannot be computed without bars. Instead: decompose R = WR-driven vs win-size-driven.
def decomp(sub):
    r = sub['R'].values
    w = r[r > 0]; l = r[r <= 0]
    return dict(n=len(r), wr=len(w)/len(r) if len(r) else float('nan'),
                avgwin=float(w.mean()) if len(w) else float('nan'),
                avgloss=float(l.mean()) if len(l) else float('nan'))
dc = {f'{k}{y}': decomp(s[s.year==y]) for k, s in [('raw', df), ('pick', cur8)] for y in ('2025','2026')}

# counterfactual: hold picked entries, swap the year's LOSS profile
def cf(pk_from, pk_to):
    """meanR of `to`-year picks if their losses had `from`-year's average loss."""
    r = pk_to['R'].values
    w = r[r > 0]; l = r[r <= 0]
    lf = pk_from['R'].values; lf = lf[lf <= 0]
    return float((w.sum() + len(l)*lf.mean())/len(r))

cf_loss = cf(p25, p26)   # 2026 picks with 2025 loss size
# 2026 picks with 2025 WR (resample)
def cf_wr(pk_to, wr_from):
    r = pk_to['R'].values; w = r[r>0]; l = r[r<=0]
    return float(wr_from*w.mean() + (1-wr_from)*l.mean())
cf_wrv = cf_wr(p26, dc['pick2025']['wr'])

# ---------------------------------------------------------------- regime split test
reg_pick = cur8.groupby(['year','regime'])['R'].agg(['size','mean'])
vol_split = {}
for y in ('2025','2026'):
    s = cur8[cur8.year==y]
    hi = s[s['spy_vol20'] >= 22]; lo = s[s['spy_vol20'] < 22]
    vol_split[y] = dict(hi_n=len(hi), hi_R=float(hi['R'].mean()) if len(hi) else float('nan'),
                        lo_n=len(lo), lo_R=float(lo['R'].mean()) if len(lo) else float('nan'))

# half-year splits
cur8 = cur8.copy()
cur8['half'] = cur8['date'].str[:4] + 'H' + ((cur8['date'].str[5:7].astype(int)-1)//6+1).astype(str)
halves = cur8.groupby('half')['R'].agg(['size','mean','sum']).to_dict('index')
df['half'] = df['date'].str[:4] + 'H' + ((df['date'].str[5:7].astype(int)-1)//6+1).astype(str)
halves_raw = df.groupby('half')['R'].agg(['size','mean']).to_dict('index')

# monthly picked
mon = cur8.groupby('month').agg(n=('R','size'), R=('R','mean'), pnl=('pnl','sum')).to_dict('index')

# ---------------------------------------------------------------- emit
out = dict(layers=rows, gates=gates, whole_stack=sel, seps=seps, pop=pop,
           months25=months25, months26=months26,
           det25=int((df.year=='2025').sum()), det26=int((df.year=='2026').sum()),
           regime_mix=reg_mix.to_dict(), regime_R=reg_R.to_string(),
           regime_pick=reg_pick.to_string(), vol_split=vol_split,
           tails=tails, noise=noise, noise_raw=noise_raw, decomp=dc,
           cf_loss_2026_with_2025_lossize=cf_loss, cf_2026_with_2025_wr=cf_wrv,
           halves=halves, halves_raw=halves_raw, monthly=mon)
with open(f'{OUT}/attrib.json', 'w') as f:
    json.dump(out, f, indent=1, default=float)

pd.DataFrame(rows).to_csv(f'{OUT}/layer_chain.csv', index=False)
pd.DataFrame(gates + [sel]).to_csv(f'{OUT}/gate_separation.csv', index=False)
cur8[['symbol','date','year','entry_time_et','entry_price','qf_pole_gain_pct',
      'qf_vwap_dist_pct','conviction_mult','macd_zone_mult','tier','R','pnl',
      'exit_reason','regime']].to_csv(f'{OUT}/picked_book.csv', index=False)

log("\n=== LAYER CHAIN (population R at each stage) ===")
log(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f'{x:7.3f}'))
log("\n=== GATE SEPARATION (kept R - rejected R) ===")
log(pd.DataFrame(gates + [sel]).to_string(index=False, float_format=lambda x: f'{x:7.3f}'))
log("\n=== SEP TESTS ==="); log(json.dumps(seps, indent=1, default=float))
log("\n=== NOISE ==="); log(json.dumps(noise, indent=1, default=float))
log("\n=== NOISE RAW ==="); log(json.dumps(noise_raw, indent=1, default=float))
log("\n=== DECOMP ==="); log(json.dumps(dc, indent=1, default=float))
log(f"\ncf: 2026 picks w/ 2025 loss size = {cf_loss:.3f}; w/ 2025 WR = {cf_wrv:.3f}")
log("\n=== TAILS ==="); log(json.dumps(tails, indent=1, default=float))
log("\n=== HALVES picked ==="); log(json.dumps(halves, indent=1, default=float))
log("\n=== HALVES raw ==="); log(json.dumps(halves_raw, indent=1, default=float))
log("\n=== POP ==="); log(json.dumps(pop, indent=1, default=float))
log("\n=== REGIME mix ==="); log(reg_mix.to_string())
log("\n=== REGIME R raw ==="); log(reg_R.to_string())
log("\n=== REGIME picked ==="); log(reg_pick.to_string())
log("\n=== VOL SPLIT picked ==="); log(json.dumps(vol_split, indent=1, default=float))
log(f"\ndetections: 2025={out['det25']} over {months25} mo; 2026={out['det26']} over {months26} mo")
log("\n=== EXIT MIX raw ==="); log(exit_mix.to_string(index=False))
log("\n=== EXIT MIX picked ==="); log(exit_mix_pick.to_string(index=False))
log("\n=== MONTHLY picked ==="); log(pd.DataFrame(mon).T.to_string())
