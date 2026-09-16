#!/usr/bin/env python3
"""Daily point-in-time panel for the literature tests: every symbol-day 2025-2026 (delisted included) with the derived
fields the short-horizon hypotheses need. Output research/lit_review_2026/daily_panel.parquet (float32, sorted)."""
import os, sys
import numpy as np, pandas as pd, pyarrow.parquet as pq
ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT)
cols = ['symbol', 'bar_date', 'open', 'high', 'low', 'close', 'volume']
t = pq.read_table('data/research/databento/equs_daily_2025_2026.parquet', columns=cols)
d = t.to_pandas(); del t
d = d[d.symbol.notna() & (d.symbol.astype(str).str.strip() != '')]
d['bar_date'] = d.bar_date.astype(str).str[:10]
for k in ('open', 'high', 'low', 'close', 'volume'): d[k] = pd.to_numeric(d[k], errors='coerce').astype('float32')
d['symbol'] = d.symbol.astype('category')
d = d.sort_values(['symbol', 'bar_date']).reset_index(drop=True)
g = d.groupby('symbol', observed=True)
d['prev_close'] = g.close.shift(1); d['prev_volume'] = g.volume.shift(1)
d['adv20'] = g.volume.transform(lambda s: s.shift(1).rolling(20, min_periods=10).mean()).astype('float32')
d['dvol20'] = (g.apply(lambda x: (x.close * x.volume).shift(1).rolling(20, min_periods=10).mean()).reset_index(level=0, drop=True)).astype('float32')
d['ret_on'] = (d.open / d.prev_close - 1).astype('float32')            # overnight (close→open)
d['ret_id'] = (d.close / d.open - 1).astype('float32')                 # intraday (open→close)
d['ret_cc'] = (d.close / d.prev_close - 1).astype('float32')
d['ret_cc_next'] = g.ret_cc.shift(-1); d['ret_on_next'] = g.ret_on.shift(-1); d['ret_id_next'] = g.ret_id.shift(-1)
d['range_pct'] = ((d.high - d.low) / d.low).astype('float32')
d['vol_ratio'] = (d.volume / d.adv20).astype('float32')
d['ret5'] = g.close.transform(lambda s: s / s.shift(5) - 1).astype('float32'); d['ret20'] = g.close.transform(lambda s: s / s.shift(20) - 1).astype('float32')
d['high52'] = g.high.transform(lambda s: s.shift(1).rolling(250, min_periods=60).max()).astype('float32')
d['dow'] = pd.to_datetime(d.bar_date).dt.dayofweek.astype('int8')
d.to_parquet('research/lit_review_2026/daily_panel.parquet', index=False)
print('panel rows', len(d), 'symbols', d.symbol.nunique(), 'dates', d.bar_date.nunique(), d.bar_date.min(), d.bar_date.max(), flush=True)
