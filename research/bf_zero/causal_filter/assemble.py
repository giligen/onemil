#!/usr/bin/env python3
"""CAUSAL_FILTER step 2 — assemble the feature table from the population, the signal-bar pass,
the SIP volume profile, the daily file, SPY, the asset-class map and the news file.

Adds ONE diagnostic column, `cohort` ('cache' = the symbol-day was served by data/cache.db, i.e. a
day the bull-flag scanner had already flagged as a mover; 'pit' = only the point-in-time top-up had
it). It is end-of-day information and is asserted out of every rule downstream.

Output: causal_filter/features.csv
"""
import csv, os, sqlite3, sys
import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
D = f'{ROOT}/research/bf_zero/causal_filter'
VP_MIN = np.array([575, 585, 600, 630, 660, 720, 780, 840, 900])

RD = dict(dtype={'symbol': str, 'day': str}, keep_default_na=False, na_values=[''])
pop = pd.read_csv(f'{D}/population.csv', **RD)
sb = pd.read_csv(f'{D}/sig_bars.csv', **RD).drop_duplicates(['day', 'symbol'])
vp = pd.read_csv(f'{D}/vprof_sip.csv', **RD).drop_duplicates(['day', 'symbol'])
print(f'pop {len(pop)} sig_bars {len(sb)} vprof {len(vp)}', flush=True)

c = pop.merge(sb.drop(columns=['entry_m']), on=['day', 'symbol'], how='left')

# ---- daily-file features (prev close/high/low, 20d high, SPY) -------------------------------
daily = pd.read_parquet(f'{ROOT}/data/research/databento/equs_daily_2025_2026.parquet',
                        columns=['symbol', 'bar_date', 'open', 'high', 'low', 'close'])
daily['bar_date'] = daily.bar_date.astype(str).str[:10]
daily = daily[daily.symbol.notna() & (daily.symbol.astype(str).str.strip() != '')]
keep = set(pop.symbol.unique()) | {'SPY'}
daily = daily[daily.symbol.isin(keep)].sort_values(['symbol', 'bar_date'])
g = daily.groupby('symbol')
daily['prev_close'] = g.close.shift(1)
daily['prev_high'] = g.high.shift(1)
daily['prev_low'] = g.low.shift(1)
daily['high20'] = g.high.transform(lambda s: s.shift(1).rolling(20, min_periods=5).max())
dk = daily.set_index(['symbol', 'bar_date'])[['prev_close', 'prev_high', 'prev_low', 'high20']]
c = c.merge(dk, left_on=['symbol', 'day'], right_index=True, how='left')
spyd = daily[daily.symbol == 'SPY'].set_index('bar_date')
spy_r3 = ((spyd.high - spyd.low) / spyd.close * 100).rolling(3).mean().shift(1)
c['spy_range3'] = c.day.map(spy_r3)

con = sqlite3.connect(f'file:{ROOT}/data/cache.db?mode=ro', uri=True, timeout=180)
# SPY 1-min: the Alpaca SIP pull (causal_filter/spy_1min.csv). cache.db's SPY minutes stop in 2026-05
# and would have left spy_5m_ret at 0% coverage on TEST — a feature that cannot be evaluated out of
# sample is not a feature (availability rule).
spy = pd.read_csv(f'{D}/spy_1min.csv', dtype={'bar_date': str})
spy_by_day = {d: dict(zip(gg.m, gg.close)) for d, gg in spy.groupby('bar_date')}


def spy5(day, m):
    dd = spy_by_day.get(day)
    if not dd:
        return np.nan
    a, b = dd.get(int(m)), dd.get(int(m) - 5)
    return (a / b - 1) * 100 if a and b else np.nan


c['spy_5m_ret'] = [spy5(d, m) for d, m in zip(c.day, c.entry_m)]
c['gap_pct'] = np.where(c.prev_close > 0, (c.open_px / c.prev_close - 1) * 100, np.nan)
c['prev_range_pct'] = np.where(c.prev_close > 0, (c.prev_high - c.prev_low) / c.prev_close * 100, np.nan)
c['dist_20d_high_pct'] = np.where(c.high20 > 0, (c.entry / c.high20 - 1) * 100, np.nan)
c['above_vwap'] = (c.entry > c.vwap_prev).astype(float).where(c.vwap_prev.notna())

# ---- rv_clock / n_prior from the SIP volume profile ----------------------------------------
cvc = [f'cv_{m}' for m in VP_MIN]
vp = vp.sort_values(['symbol', 'day']).reset_index(drop=True)
sh = vp.groupby('symbol')[cvc].shift(1)
roll = (sh.groupby(vp.symbol).rolling(20, min_periods=5).mean()
        .reset_index(level=0, drop=True).sort_index())
for k in cvc:
    vp[f'{k}_prior'] = roll[k].values
vp['n_prior'] = (sh[cvc[0]].notna().astype(int).groupby(vp.symbol).rolling(20, min_periods=1).sum()
                 .reset_index(level=0, drop=True).sort_index().values)
c = c.merge(vp[['day', 'symbol', 'n_prior'] + [f'{k}_prior' for k in cvc]], on=['day', 'symbol'], how='left')
ck_idx = np.clip(np.searchsorted(VP_MIN, c.entry_m.values, side='right') - 1, 0, len(VP_MIN) - 1)
pri = c[[f'{k}_prior' for k in cvc]].values
c['rv_clock'] = c.cumv_entry.values / pd.Series(pri[np.arange(len(c)), ck_idx]).replace(0, np.nan).values
c = c.drop(columns=[f'{k}_prior' for k in cvc])

# ---- anchor / wrapper / causal sibling cohort ----------------------------------------------
from trading.orb_asset_class import DEFAULT_CLASS_MAP, load_class_map, underlying_anchor  # noqa: E402
names = {}
for p in (f'{ROOT}/data/research/alpaca_assets_all_20260905.csv', DEFAULT_CLASS_MAP):
    try:
        for r in csv.DictReader(open(p, newline='')):
            if r.get('symbol') and r.get('name') and r['symbol'] not in names:
                names[r['symbol']] = r['name']
    except FileNotFoundError:
        pass
cmap = load_class_map()
anc = {s: underlying_anchor(s, names.get(s), cmap) for s in c.symbol.unique()}
c['anchor'] = c.symbol.map(anc)
c['is_wrapper'] = c.symbol.map(lambda s: int(cmap.get(s) == 'wrapper'))
c = c.sort_values(['day', 'anchor', 'entry_m']).reset_index(drop=True)
has = c.anchor.notna() & (c.anchor.astype(str) != '')
c['coh_by_t'] = 0
c.loc[has, 'coh_by_t'] = (c[has].groupby(['day', 'anchor']).entry_m.rank(method='min').astype(int) - 1).values

# ---- news (own-ticker premarket) ------------------------------------------------------------
nf = (f'{D}/news.csv' if os.path.exists(f'{D}/news.csv')
      else f'{ROOT}/data/research/orb_news_catalyst_nightly.csv')
print(f'news source: {nf}', flush=True)
nw = pd.read_csv(nf, **RD)[['symbol', 'day', 'n_articles']].drop_duplicates(['day', 'symbol'])
nw['has_news'] = (pd.to_numeric(nw.n_articles, errors='coerce').fillna(0) > 0).astype(int)
c = c.merge(nw[['day', 'symbol', 'has_news']].rename(columns={'has_news': '_news'}),
            on=['day', 'symbol'], how='left')
c['news_covered'] = c._news.notna().astype(int)               # per KEY, not per day (D1 rule)
c['has_news'] = c._news
c = c.drop(columns=['_news'])

# ---- COHORT (diagnostic only) ---------------------------------------------------------------
ck = pd.read_sql('select distinct bar_date as day, symbol from intraday_bars_1min '
                 "where bar_date>='2025-01-02'", con)
have = set(zip(ck.day.astype(str), ck.symbol.astype(str)))
c['cohort'] = ['cache' if (d, s) in have else 'pit' for d, s in zip(c.day, c.symbol)]
con.close()

c['split'] = np.where(c.day < '2026-01-01', 'TRAIN', np.where(c.day < '2026-06-01', 'VAL', 'TEST'))
c['half'] = np.where(c.day < '2025-07-01', 'H1', np.where(c.day < '2026-01-01', 'H2', ''))
c['wk'] = pd.to_datetime(c.day).dt.to_period('W-FRI').astype(str)
c.to_csv(f'{D}/features.csv', index=False)
print(f'features {len(c)} cols {len(c.columns)}', flush=True)
print(c.groupby(['split', 'cohort']).rr.agg(['size', 'mean']).round(3).to_string(), flush=True)
FEATS = ['gap_pct', 'prev_range_pct', 'dist_20d_high_pct', 'bar_vol_x', 'above_vwap', 'spy_5m_ret',
         'spy_range3', 'dist_open_pct', 'rv_clock', 'rv_profile', 'drive_min', 'n_prior',
         'is_wrapper', 'coh_by_t', 'entry_m', 'price', 'has_news']
print('\ncoverage per split (%):', flush=True)
print((c.groupby('split')[FEATS].apply(lambda d: d.notna().mean() * 100).round(1)).to_string(), flush=True)
print('DONE', flush=True)
