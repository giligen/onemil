#!/usr/bin/env python3
"""Bull-flag from zero — pass 2: features that need cross-day or cross-symbol data (DESIGN.md).

  rv_clock    cumulative volume at entry ÷ this symbol's mean cumulative volume at the SAME clock
              checkpoint over its prior 20 held days (own-history baseline; n_prior reported —
              only days in our bar store, i.e. range >= 5% days: biased high, flagged)
  rv_profile  cumulative volume at entry ÷ (ADV20 × market-wide median fraction of the day's
              volume traded by that checkpoint)  — available everywhere
  coh_by_t    other symbols sharing the underlying anchor that fired the SAME family-config
              earlier the same day (causal; the day-level cohort is banned)
Output: candidates_full.csv (same rows as candidates.csv, columns added).
"""
import csv, os, sys
import numpy as np, pandas as pd
ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT); sys.path.insert(0, ROOT)
from trading.orb_asset_class import DEFAULT_CLASS_MAP, load_class_map, underlying_anchor
D = os.environ.get('BFZ_DIR', 'research/bf_zero')
VP_MIN = np.array([575, 585, 600, 630, 660, 720, 780, 840, 900])

c = pd.read_csv(f'{D}/candidates.csv', low_memory=False, dtype={'symbol': str}, keep_default_na=False)
for k in c.columns:
    if k not in ('day', 'symbol', 'fam', 'cfg', 'why_e1', 'why_e2', 'why_e3', 'why_e4'):
        c[k] = pd.to_numeric(c[k], errors='coerce')
c = c.drop_duplicates(['day', 'symbol', 'fam', 'cfg']).reset_index(drop=True)
N0 = len(c); print('candidates', N0, 'symbol-days', c[['day', 'symbol']].drop_duplicates().shape[0], flush=True)

# --- volume profile ---
vp = pd.read_csv(f'{D}/volume_profile.csv', dtype={'symbol': str}, keep_default_na=False).drop_duplicates(['day', 'symbol'])
cvcols = [f'cv_{m}' for m in VP_MIN]
for k in cvcols + ['day_vol', 'pm_vol']: vp[k] = pd.to_numeric(vp[k], errors='coerce')
vp = vp.sort_values(['symbol', 'day']).reset_index(drop=True)
# prior-20-held-day mean per checkpoint (shifted: strictly before the day)
g = vp.groupby('symbol')
sh = g[cvcols].shift(1)                                            # strictly before the day
roll = sh.groupby(vp.symbol).rolling(20, min_periods=5).mean().reset_index(level=0, drop=True).sort_index()
for k in cvcols: vp[f'{k}_prior'] = roll[k].values
vp['n_prior'] = sh[cvcols[0]].notna().astype(int).groupby(vp.symbol).rolling(20, min_periods=1).sum().reset_index(level=0, drop=True).sort_index().values
# market-wide fraction of the day's volume by checkpoint (median over held symbol-days)
frac = {m: float(np.nanmedian(vp[f'cv_{m}'] / vp.day_vol.replace(0, np.nan))) for m in VP_MIN}
print('median fraction of day volume by checkpoint:', {k: round(v, 3) for k, v in frac.items()}, flush=True)
c = c.merge(vp[['day', 'symbol', 'n_prior'] + [f'{k}_prior' for k in cvcols]], on=['day', 'symbol'], how='left')
# checkpoint at or before the entry minute
ck_idx = np.searchsorted(VP_MIN, c.entry_m.values, side='right') - 1
ck = np.where(ck_idx >= 0, VP_MIN[np.clip(ck_idx, 0, len(VP_MIN) - 1)], VP_MIN[0])
c['ck_min'] = ck
cumvol = c.rv_adv * c.adv20                                        # cumulative volume to the entry bar (pass 1 stored the ratio)
pri = c[[f'cv_{m}_prior' for m in VP_MIN]].values
prior = pri[np.arange(len(c)), np.clip(ck_idx, 0, len(VP_MIN) - 1)]
c['rv_clock'] = cumvol / pd.Series(prior).replace(0, np.nan).values
c['rv_profile'] = cumvol / (c.adv20 * pd.Series(ck).map(frac).values)
c = c.drop(columns=[f'{k}_prior' for k in cvcols])

# --- causal sibling cohort per family-config ---
names = {}
for p in (f'{ROOT}/data/research/alpaca_assets_all_20260905.csv', DEFAULT_CLASS_MAP):
    try:
        for r in csv.DictReader(open(p, newline='')):
            if r.get('symbol') and r.get('name') and r['symbol'] not in names: names[r['symbol']] = r['name']
    except FileNotFoundError: pass
cmap = load_class_map()
anchor = {s: underlying_anchor(s, names.get(s), cmap) for s in c.symbol.unique()}
c['anchor'] = c.symbol.map(anchor); c['is_wrapper'] = c.symbol.map(lambda s: cmap.get(s) == 'wrapper').astype(int)
c = c.sort_values(['day', 'fam', 'cfg', 'anchor', 'entry_m']).reset_index(drop=True)
# rank within (day, fam, cfg, anchor) by entry minute = number of siblings that fired strictly earlier (ties count as not-earlier)
has = c.anchor.notna()
c['coh_by_t'] = 0
c.loc[has, 'coh_by_t'] = c[has].groupby(['day', 'fam', 'cfg', 'anchor']).entry_m.rank(method='min').astype(int).values - 1
assert len(c) == N0, f'row fan-out {N0} -> {len(c)}'
c = c.sort_values(['day', 'symbol', 'fam', 'cfg']).reset_index(drop=True)
c.to_csv(f'{D}/candidates_full.csv', index=False)
print('DONE candidates_full', len(c), '| rv_clock present', int(c.rv_clock.notna().sum()), '| n_prior>=5', int((c.n_prior >= 5).sum()), '| anchored', int(c.anchor.notna().sum()), flush=True)
