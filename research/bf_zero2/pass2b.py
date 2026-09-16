#!/usr/bin/env python3
"""bf_zero2 pass 2 — the same features as research/bf_zero/pass2.py, memory-safe for 4.7M rows: chunked read of ONLY the
columns the scorer needs, the scoring population filter first (price >= 5, entry <= 14:01), float32. Writes candidates_full.csv."""
import csv, os, sys
import numpy as np, pandas as pd
ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT); sys.path.insert(0, ROOT)
from trading.orb_asset_class import DEFAULT_CLASS_MAP, load_class_map, underlying_anchor
D = 'research/bf_zero2'; VP_MIN = np.array([575, 585, 600, 630, 660, 720, 780, 840, 900])
KEEP = ['day', 'symbol', 'fam', 'cfg', 'entry_m', 'entry', 'stop', 'r_pct', 'price', 'dist_open_pct', 'rv_adv', 'bar_vol_x', 'above_vwap', 'gap_pct',
        'prev_range_pct', 'adv20', 'dist_20d_high_pct', 'spy_5m_ret', 'spy_range3', 'rr_e1', 'why_e1', 'rr_e2', 'rr_e3', 'rr_e4', 'rr_e1c', 'why_e1c',
        'exit_m_e1c', 'n_bars_at_entry', 'pm_vol', 'pm_high_pct', 'range_so_far_pct', 'bars_per_min']
STR = {'day', 'symbol', 'fam', 'cfg', 'why_e1', 'why_e1c'}
parts = []
for ch in pd.read_csv(f'{D}/candidates.csv', usecols=KEEP, dtype={k: str for k in STR}, keep_default_na=False, na_values=[''], chunksize=400_000, low_memory=True):
    for k in KEEP:
        if k not in STR: ch[k] = pd.to_numeric(ch[k], errors='coerce').astype('float32')
    ch = ch[(ch.price >= 5) & (ch.entry_m <= 841)]
    parts.append(ch); print(f'chunk kept {len(ch)}', flush=True)
c = pd.concat(parts, ignore_index=True); del parts
c = c.drop_duplicates(['day', 'symbol', 'fam', 'cfg']).reset_index(drop=True); N0 = len(c)
print('scoring population', N0, 'symbol-days', c[['day', 'symbol']].drop_duplicates().shape[0], flush=True)
vp = pd.read_csv(f'{D}/volume_profile.csv', dtype={'symbol': str}, keep_default_na=False).drop_duplicates(['day', 'symbol'])
cvcols = [f'cv_{m}' for m in VP_MIN]
for k in cvcols + ['day_vol', 'pm_vol']: vp[k] = pd.to_numeric(vp[k], errors='coerce').astype('float32')
vp = vp.sort_values(['symbol', 'day']).reset_index(drop=True); g = vp.groupby('symbol'); sh = g[cvcols].shift(1)
roll = sh.groupby(vp.symbol).rolling(20, min_periods=5).mean().reset_index(level=0, drop=True).sort_index()
for k in cvcols: vp[f'{k}_prior'] = roll[k].values.astype('float32')
vp['n_prior'] = sh[cvcols[0]].notna().astype(int).groupby(vp.symbol).rolling(20, min_periods=1).sum().reset_index(level=0, drop=True).sort_index().values
frac = {m: float(np.nanmedian(vp[f'cv_{m}'] / vp.day_vol.replace(0, np.nan))) for m in VP_MIN}
print('median fraction of day volume by checkpoint:', {k: round(v, 3) for k, v in frac.items()}, flush=True)
c = c.merge(vp[['day', 'symbol', 'n_prior'] + [f'{k}_prior' for k in cvcols]], on=['day', 'symbol'], how='left')
ck_idx = np.searchsorted(VP_MIN, c.entry_m.values, side='right') - 1
ck = np.where(ck_idx >= 0, VP_MIN[np.clip(ck_idx, 0, len(VP_MIN) - 1)], VP_MIN[0]); c['ck_min'] = ck
cumvol = c.rv_adv * c.adv20
pri = c[[f'cv_{m}_prior' for m in VP_MIN]].values; prior = pri[np.arange(len(c)), np.clip(ck_idx, 0, len(VP_MIN) - 1)]
c['rv_clock'] = (cumvol / pd.Series(prior).replace(0, np.nan).values).astype('float32')
c['rv_profile'] = (cumvol / (c.adv20 * pd.Series(ck).map(frac).values)).astype('float32')
c = c.drop(columns=[f'{k}_prior' for k in cvcols])
names = {}
for p in (f'{ROOT}/data/research/alpaca_assets_all_20260905.csv', DEFAULT_CLASS_MAP):
    try:
        for r in csv.DictReader(open(p, newline='')):
            if r.get('symbol') and r.get('name') and r['symbol'] not in names: names[r['symbol']] = r['name']
    except FileNotFoundError: pass
cmap = load_class_map()
anchor = {s: underlying_anchor(s, names.get(s), cmap) for s in c.symbol.unique()}
c['anchor'] = c.symbol.map(anchor); c['is_wrapper'] = c.symbol.map(lambda s: cmap.get(s) == 'wrapper').astype('int8')
c = c.sort_values(['day', 'fam', 'cfg', 'anchor', 'entry_m']).reset_index(drop=True)
has = c.anchor.notna(); c['coh_by_t'] = 0
c.loc[has, 'coh_by_t'] = c[has].groupby(['day', 'fam', 'cfg', 'anchor']).entry_m.rank(method='min').astype(int).values - 1
assert len(c) == N0, f'row fan-out {N0} -> {len(c)}'
c = c.sort_values(['day', 'symbol', 'fam', 'cfg']).reset_index(drop=True)
c.to_csv(f'{D}/candidates_full.csv', index=False)
print('DONE candidates_full', len(c), '| rv_clock present', int(c.rv_clock.notna().sum()), '| anchored', int(c.anchor.notna().sum()), flush=True)
