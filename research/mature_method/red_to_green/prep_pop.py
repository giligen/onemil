#!/usr/bin/env python3
"""Step 0 - the HONEST population for the red-to-green (F6-PDR) book.

Start from H/F6_reconcile/prev_table.csv (every research/bf_zero/universe.csv symbol-day 2025-01-03..2026-09-04
that has a prior row in the Databento EQUS daily panel; R4 proved universe.csv and the panel agree on prior-day
OHLC to 1e-6 on all 644,580 keys).  Apply the two STANDING membership rules from CLAUDE.md:

  * drop NASDAQ test tickers  ^Z[A-Z]ZZT   (research/scripts/pit_listings.is_test_ticker) - ZVZZT WAS the whole
    of implementation A's TEST "profit" (H/F6_reconcile/REPORT.md section 0.1);
  * drop any symbol absent from cache.db `daily_bars` - the live engine's universe comes from that table, so a
    name that is not in it could never have been traded.

Then attach the two LIVE universe fields the reconciliation never modelled, both CAUSAL (trailing 20 sessions
strictly BEFORE the day, from the same daily panel):
  * adv20     - the engine screens `adv >= min_adv20` (100,000) in _stream_the_universe;
  * prev_close- the engine screens `last close >= universe_min_prev_close` ($5) in the same place.
Live builds both from the LAST row of daily_bars (a static snapshot); this study builds them as of the day,
which is the causal version of the same screen.  The deviation is declared in PREREG section 5.
"""
import os, sys, sqlite3
import numpy as np, pandas as pd, pyarrow.parquet as pq

ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT); sys.path.insert(0, ROOT)
from research.scripts.pit_listings import is_test_ticker

OUT = 'research/mature_method/red_to_green'
PANEL = 'data/research/databento/equs_daily_2025_2026.parquet'
PREV = 'research/fuckup_audit/H/F6_reconcile/prev_table.csv'
log = lambda *a: (print(*a), sys.stdout.flush())

p = pd.read_csv(PREV, dtype={'symbol': str, 'day': str, 'prev_date': str}, keep_default_na=False, na_values=[''])
log('prev_table rows', len(p), 'symbols', p.symbol.nunique())

tt = p.symbol.map(is_test_ticker)
log('test-ticker rows dropped:', int(tt.sum()), sorted(p.symbol[tt].unique()))
p = p[~tt]

c = sqlite3.connect(f'file:{ROOT}/data/cache.db?mode=ro', uri=True, timeout=120)
db = {r[0] for r in c.execute('select distinct symbol from daily_bars')}
ind = p.symbol.isin(db)
log('non-daily_bars rows dropped:', int((~ind).sum()), 'symbols', p.symbol[~ind].nunique())
p = p[ind].reset_index(drop=True)
log('honest population:', len(p), 'symbol-days,', p.symbol.nunique(), 'symbols')

# ---- causal adv20 + prev_close from the panel (trailing 20 sessions strictly before the day)
syms = set(p.symbol.unique())
parts = []
pf = pq.ParquetFile(PANEL)
for rg in range(pf.metadata.num_row_groups):
    d = pf.read_row_group(rg, columns=['symbol', 'bar_date', 'close', 'volume']).to_pandas()
    d = d[d.symbol.isin(syms)]
    d['bar_date'] = d.bar_date.astype(str).str[:10]
    parts.append(d)
d = pd.concat(parts, ignore_index=True).drop_duplicates(['symbol', 'bar_date'])
del parts
d = d.sort_values(['symbol', 'bar_date']).reset_index(drop=True)
g = d.groupby('symbol', sort=False)
d['adv20'] = g.volume.transform(lambda s: s.rolling(20, min_periods=5).mean().shift(1))
log('panel rows for these symbols:', len(d))
m = d.set_index(['symbol', 'bar_date']).adv20
p['adv20'] = m.reindex(pd.MultiIndex.from_arrays([p.symbol.values, p.day.values])).to_numpy()
log('adv20 present:', int(p.adv20.notna().sum()), 'of', len(p))

keep = ['day', 'symbol', 'day_open', 'prev_date', 'prev_high_panel', 'prev_low_panel', 'prev_close_panel',
        'pdr_panel', 'adv20']
p = p[keep].rename(columns={'prev_high_panel': 'prev_high', 'prev_low_panel': 'prev_low',
                            'prev_close_panel': 'prev_close', 'pdr_panel': 'pdr'})
p.to_csv(f'{OUT}/pop.csv', index=False)
log('wrote pop.csv', len(p))
for th in (0, 6, 8, 10, 12):
    q = p[p.pdr >= th]
    log(f'  pdr>={th:2d}: {len(q):7d}   +adv20>=100K & prev_close>=5: '
        f'{int(((q.adv20 >= 1e5) & (q.prev_close >= 5)).sum()):7d}')
