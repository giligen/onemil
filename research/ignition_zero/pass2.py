#!/usr/bin/env python3
"""Ignition-from-zero — pass 2: cross-symbol and external features joined onto
candidates.csv (DESIGN.md). Everything is keyed to the trigger minute or to
data published before the day:
  anchor / wrapper           trading.orb_asset_class (offline names + class map)
  coh_by_t                   other symbols sharing the anchor that triggered at
                             the SAME level at or before this trigger minute
  coh_day                    same, whole day (the BT annotate convention — a
                             mild lookahead; kept only for parity, not used)
  sympathy_lag_min           minutes since the earliest sibling trigger (H12)
  theme_n30 / theme_n30_l5   market-wide first +10% / +5% crosses in [t-30, t]
  news: n_pre, has_news_pre, news_recency_min, headline_class  (BT-trade
        symbol-days only — coverage reported; classes: dilution / positive /
        other by offline keyword lists)
  short interest: si_qty, si_dtc, si_ratio_adv20 (FINRA, latest report whose
        usable_from <= day)
  sector / float_shares      cache.db universe SNAPSHOT (not point-in-time —
        flagged; float rarely moves, sector never)
Output: candidates_full.csv
"""
import csv, glob, os, sqlite3, sys
import numpy as np, pandas as pd
ROOT = '/home/ec2-user/onemil'; sys.path.insert(0, ROOT); os.chdir(ROOT)
from trading.orb_asset_class import DEFAULT_CLASS_MAP, load_class_map, underlying_anchor
D = 'research/ignition_zero'
c = pd.read_csv(f'{D}/candidates.csv', low_memory=False)
c = c.drop_duplicates(['day', 'symbol', 'level']).sort_values(['day', 'level', 'trig_m']).reset_index(drop=True)
print('candidates', len(c), 'symbol-days', c[['day', 'symbol']].drop_duplicates().shape[0], flush=True)

# --- anchors / wrappers ---
names = {}
for p in (f'{ROOT}/data/research/alpaca_assets_all_20260905.csv', DEFAULT_CLASS_MAP):
    try:
        for r in csv.DictReader(open(p, newline='')):
            if r.get('symbol') and r.get('name') and r['symbol'] not in names: names[r['symbol']] = r['name']
    except FileNotFoundError: pass
cmap = load_class_map()
syms = c.symbol.unique()
anchor = {s: underlying_anchor(s, names.get(s), cmap) for s in syms}
c['anchor'] = c.symbol.map(anchor); c['is_wrapper'] = c.symbol.map(lambda s: cmap.get(s) == 'wrapper').astype(int)

# --- cohort / sympathy / theme (per day, per level) ---
coh_t = np.zeros(len(c), int); coh_d = np.zeros(len(c), int); symp = np.full(len(c), np.nan); th10 = np.zeros(len(c), int); th5 = np.zeros(len(c), int)
for (day, lvl), g in c.groupby(['day', 'level']):
    tm = g.trig_m.values; idx = g.index.values
    # theme: first crosses market-wide at this level within [t-30, t] (excluding self)
    for k, (i, t) in enumerate(zip(idx, tm)):
        n = int(((tm >= t - 30) & (tm <= t)).sum()) - 1
        if lvl == 10: th10[i] = n
        if lvl == 5: th5[i] = n
    for a, ga in g[g.anchor.notna()].groupby('anchor'):
        if len(ga) < 2: continue
        tma = ga.trig_m.values; ia = ga.index.values; first = tma.min()
        for i, t in zip(ia, tma):
            coh_t[i] = int((tma <= t).sum()) - 1; coh_d[i] = len(ga) - 1
            if t > first: symp[i] = t - first
c['coh_by_t'] = coh_t; c['coh_day'] = coh_d; c['sympathy_lag_min'] = symp; c['theme_n30'] = th10; c['theme_n30_l5'] = th5
# theme at level 10 should be visible on every level's row for the same symbol-day
t10 = c[c.level == 10].set_index(['day', 'symbol']).theme_n30
c['theme_n30'] = [t10.get((d, s), 0) if l != 10 else v for d, s, l, v in zip(c.day, c.symbol, c.level, c.theme_n30)]

# --- news (BT-trade symbol-days) ---
news = pd.concat([pd.read_csv(p, low_memory=False) for p in glob.glob(f'{ROOT}/research/ignition_news/news_*.csv')], ignore_index=True)
news = news[['day', 'symbol', 'n_pre', 'has_news_pre']].drop_duplicates(['day', 'symbol'])
arts = pd.concat([pd.read_csv(p, low_memory=False) for p in glob.glob(f'{ROOT}/research/ignition_news/articles_*.csv')], ignore_index=True)
arts['created_utc'] = pd.to_datetime(arts.created_utc, utc=True, errors='coerce')
DIL = ('offering', 'dilut', 'reverse split', 'reverse stock split', 'warrant', 'registered direct', 'at-the-market', 'atm program', 'convertible', 'private placement', 'shelf', 'pricing of', 'prices public', 'priced')
POS = ('fda', 'approval', 'approve', 'contract', 'award', 'partnership', 'collaborat', 'earnings', 'revenue', 'guidance', 'beats', 'record', 'acquisition', 'acquire', 'merger', 'buyout', 'to be acquired', 'strategic', 'patent', 'clearance', 'launch')
def classify(h):
    h = str(h).lower()
    if any(k in h for k in DIL): return 'dilution'
    if any(k in h for k in POS): return 'positive'
    return 'other'
arts['cls'] = arts.headline.map(classify)
c = c.merge(news, on=['day', 'symbol'], how='left')
c['trig_utc'] = [pd.Timestamp(d).tz_localize('America/New_York') + pd.Timedelta(minutes=int(m)) for d, m in zip(c.day, c.trig_m)]
c['trig_utc'] = c.trig_utc.dt.tz_convert('UTC')
m = c[['day', 'symbol', 'trig_utc']].merge(arts[['day', 'symbol', 'created_utc', 'cls']], on=['day', 'symbol'], how='inner')
m = m[m.created_utc <= m.trig_utc]
rec = m.groupby(['day', 'symbol', 'trig_utc']).agg(latest=('created_utc', 'max'), n_dil=('cls', lambda s: (s == 'dilution').sum()), n_pos=('cls', lambda s: (s == 'positive').sum())).reset_index()
rec['news_recency_min'] = (rec.trig_utc - rec.latest).dt.total_seconds() / 60
rec['headline_class'] = np.where(rec.n_dil > 0, 'dilution', np.where(rec.n_pos > 0, 'positive', 'other'))
c = c.merge(rec[['day', 'symbol', 'news_recency_min', 'headline_class']], on=['day', 'symbol'], how='left')
c['news_covered'] = c.has_news_pre.notna().astype(int)

# --- short interest (point-in-time via usable_from) ---
si = pd.read_csv(f'{D}/short_interest.csv', low_memory=False)
si = si[['symbolCode', 'usable_from', 'currentShortPositionQuantity', 'daysToCoverQuantity']].rename(columns={'symbolCode': 'symbol'}).sort_values(['symbol', 'usable_from'])
c = c.sort_values('day'); si = si.sort_values('usable_from')
c = pd.merge_asof(c, si, left_on='day', right_on='usable_from', by='symbol', direction='backward')
c = c.rename(columns={'currentShortPositionQuantity': 'si_qty', 'daysToCoverQuantity': 'si_dtc'})
c['si_ratio_adv20'] = c.si_qty / c.adv20.replace(0, np.nan)

# --- universe snapshot (sector, float) ---
u = pd.read_sql("select symbol, sector, float_shares from universe", sqlite3.connect(f'file:{ROOT}/data/cache.db?mode=ro', uri=True))
c = c.merge(u.drop_duplicates('symbol'), on='symbol', how='left')
c = c.sort_values(['day', 'level', 'trig_m']).reset_index(drop=True)
c.to_csv(f'{D}/candidates_full.csv', index=False)
print('DONE candidates_full', len(c), '| news covered', int(c.news_covered.sum()), '| SI present', int(c.si_qty.notna().sum()), '| float present', int((c.float_shares > 0).sum()), flush=True)
