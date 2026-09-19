#!/usr/bin/env python3
"""A1 — per-gate separation map for EVERY gate in the live chain, incl. volume.

kept-R minus rejected-R, evaluated at the gate's OWN position in the live Stage-2
chain (the population actually entering it), for 2025 / 2026 / pooled, with n on
each side, se and t on the pooled separation.

Extends research/bf_decay/REPORT.md §1b, which never measured the ADV20 gate
(it was applied as layer L2 and its separation was skipped).

Read-only on the cache, the config and cache.db. Writes only into
research/bf_frequency/.
"""
import sys
import math

import numpy as np
import pandas as pd

sys.path.insert(0, '/home/ec2-user/onemil')
OUT = '/home/ec2-user/onemil/research/bf_frequency'
CACHE = '/home/ec2-user/onemil/data/bull_flag_cache_causal_full_20260905.csv'

df = pd.read_csv(CACHE, keep_default_na=False, na_values=[''], dtype={'symbol': str})
df = df[(df['date'] >= '2025-01-01') & (df['date'] <= '2026-08-31')].copy()
df['year'] = df['date'].str[:4]
for c in ('entry_price', 'stop_loss', 'pnl', 'shares', 'qf_pole_gain_pct',
          'conviction_mult', 'macd_zone_mult', 'intraday_change_at_entry',
          'avg_volume_20d', 'qf_pole_bars', 'qf_vwap_dist_pct',
          'qf_fill_vwap_dist_pct'):
    df[c] = pd.to_numeric(df[c], errors='coerce')
df['R'] = df['pnl'] / ((df['entry_price'] - df['stop_loss']) * df['shares'])
df = df[np.isfinite(df['R'])].copy()
df['risk_usd'] = (df['entry_price'] - df['stop_loss']) * df['shares']

from config import Config
cfg = Config._load_yaml_only()
bf = cfg['trading']['bull_flag']

from trading.bf_universe_filter import filter_trades as bf_univ
recs = df.to_dict('records')
df['univ_ok'] = [id(r) in {id(x) for x in bf_univ(recs)} for r in recs]

rows = []


def sep(name, parent, mask):
    k, r = parent[mask], parent[~mask]
    o = {'gate': name, 'n_kept': len(k), 'n_rej': len(r)}
    for y in ('2025', '2026'):
        ky, ry = k[k.year == y]['R'], r[r.year == y]['R']
        o[f'kept{y[2:]}'] = ky.mean() if len(ky) else np.nan
        o[f'rej{y[2:]}'] = ry.mean() if len(ry) else np.nan
        o[f'nk{y[2:]}'], o[f'nr{y[2:]}'] = len(ky), len(ry)
        o[f'sep{y[2:]}'] = (o[f'kept{y[2:]}'] - o[f'rej{y[2:]}']
                            if len(ky) and len(ry) else np.nan)
    ka, ra = k['R'].values, r['R'].values
    if len(ka) > 1 and len(ra) > 1:
        d = ka.mean() - ra.mean()
        se = math.sqrt(ka.var(ddof=1) / len(ka) + ra.var(ddof=1) / len(ra))
        o['sep_pool'], o['se_pool'], o['t_pool'] = d, se, d / se
    else:
        o['sep_pool'] = o['se_pool'] = o['t_pool'] = np.nan
    # dollar-risk asymmetry: does the sizer weight the two sides differently?
    o['risk$_kept'] = float(k['risk_usd'].median()) if len(k) else np.nan
    o['risk$_rej'] = float(r['risk_usd'].median()) if len(r) else np.nan
    rows.append(o)
    return k


cur = df
cur = sep('L1 live universe (name rule)', cur, cur['univ_ok'])
cur = sep('L2 ADV20 >= 200K  [NEVER MEASURED BEFORE]', cur,
          cur['avg_volume_20d'].fillna(0) >= 200000)
cur = sep('L3 entry price <= $20', cur, cur['entry_price'] <= 20.0)
cur = sep('L4 pole gain >= 5%', cur, cur['qf_pole_gain_pct'] >= 5.0)
cur = sep('L5 conviction >= 1.8', cur, cur['conviction_mult'].fillna(1.0) >= 1.8)
cur = sep('L6 pole_bars <= 3', cur,
          (cur['qf_pole_bars'] > 0) & (cur['qf_pole_bars'] <= 3))

from trading.bf_vwap_gate import load_vwap_gate_config, filter_trades as vwap_gate
r6 = cur.to_dict('records')
ok = {id(x) for x in vwap_gate(r6, load_vwap_gate_config(bf))}
cur = cur.copy(); cur['vwap_ok'] = [id(r) in ok for r in r6]
cur = sep('L7 VWAP gate', cur, cur['vwap_ok'])

from trading.two_tier_filter import classify_tier, build_features_from_trade, should_keep
tt = bf['two_tier_filter']
r7 = cur.to_dict('records')
keep, macd_leg, comp_leg = [], [], []
for r in r7:
    ic = r.get('intraday_change_at_entry')
    ic = None if ic is None or (isinstance(ic, float) and math.isnan(ic)) else float(ic)
    tl = classify_tier(ic, a_tier_lower=float(tt.get('a_tier_lower', 20.0)),
                       extras_lower=float(tt.get('extras_lower', 10.0)))
    k, reason = should_keep(tier=tl, macd_zone_mult=float(r.get('macd_zone_mult') or 0.0),
                            features=build_features_from_trade(r), cfg=tt)
    keep.append(k)
    macd_leg.append(k or reason != 'extras_macd_surgical_drop')
    comp_leg.append(k or reason != 'extras_composite_below_threshold')
cur = cur.copy()
cur['tt_ok'], cur['tt_macd_ok'], cur['tt_comp_ok'] = keep, macd_leg, comp_leg
sep('L8a two-tier: MACD surgical-drop leg', cur, pd.Series(macd_leg, index=cur.index))
sep('L8b two-tier: composite leg', cur, pd.Series(comp_leg, index=cur.index))
book = sep('L8 two-tier filter (both legs)', cur, cur['tt_ok'])
sep('L9 MACD zone mult >= 1.5 (SIZING, within book)', book,
    book['macd_zone_mult'].fillna(1.0) >= 1.5)

# whole stack: P1 picks vs everything the stack dropped, measured on the
# post-universe, post-volume population and on the post-universe population
keys = set(zip(book['symbol'], book['date'], book['entry_time_et']))
post_u = df[df.univ_ok].copy()
post_u['picked'] = [(s, d, t) in keys for s, d, t in
                    zip(post_u['symbol'], post_u['date'], post_u['entry_time_et'])]
sep('WHOLE P1 STACK picked vs rejected (post-universe)', post_u, post_u['picked'])

g = pd.DataFrame(rows)
g.to_csv(f'{OUT}/separation.csv', index=False)
pd.set_option('display.width', 250)
print(g[['gate', 'nk25', 'nr25', 'sep25', 'nk26', 'nr26', 'sep26',
         'sep_pool', 'se_pool', 't_pool', 'risk$_kept', 'risk$_rej']]
      .to_string(index=False, float_format=lambda x: f'{x:8.3f}'))

# ---- the sizer asymmetry that R hides: dollar risk by ADV20 bucket ----------
print('\n=== dollar risk per trade by ADV20 bucket (why R and $ disagree) ===')
b = df[df.univ_ok].copy()
b['bucket'] = pd.cut(b['avg_volume_20d'], [-1, 50e3, 100e3, 200e3, 500e3, 1e12],
                     labels=['<50K', '50-100K', '100-200K', '200-500K', '>500K'])
print(b.groupby('bucket', observed=False).agg(
    n=('R', 'size'), meanR=('R', 'mean'),
    med_risk_usd=('risk_usd', 'median'), sum_pnl=('pnl', 'sum')
).to_string(float_format=lambda x: f'{x:10.2f}'))
