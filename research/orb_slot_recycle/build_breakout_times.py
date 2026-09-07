#!/usr/bin/env python3
"""Per-candidate BREAKOUT MINUTE for the slot-recycling study.

For every candidate symbol-day in the static-lock dump, the first 1-min bar
at/after 09:35 ET whose high >= the stop-limit trigger (range_high × 1.003;
range_high is entry_price for no-fill rows and entry_price/1.003 for entered
rows — the pipeline's own convention) -> minutes after 09:30. NaN = never.
Month-by-month bulk loads (one bulk job; pages released after each month).
Output: research/orb_slot_recycle/breakout_times.csv (symbol, date, breakout_min, range_high_est)
"""
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from persistence.database import Database

DUMP = 'research/orb_veto_study/candidates_static_lock_dump.csv'
OUT = 'research/orb_slot_recycle/breakout_times.csv'
BUFFER = 1.003

d = pd.read_csv(DUMP, low_memory=False)[['symbol', 'date', 'entry_price', 'entered']]
d['date'] = pd.to_datetime(d['date']).dt.strftime('%Y-%m-%d')
d['range_high_est'] = np.where(d.entered == 1, d.entry_price / BUFFER, d.entry_price)
d['trigger'] = d.range_high_est * BUFFER
db = Database(db_path='data/cache.db')
rows = []
for mo, g in d.groupby(d.date.str[:7]):
    t0 = time.time()
    pairs = list(zip(g.symbol, g.date))
    bars = db.get_intraday_bars_bulk(pairs)
    for r in g.itertuples():
        b = bars.get((r.symbol, r.date))
        bm = np.nan
        if b:
            for bar in b:
                ts = bar['timestamp']
                ts = pd.Timestamp(ts) if not isinstance(ts, pd.Timestamp) else ts
                if ts.tzinfo is not None:
                    ts = ts.tz_convert('US/Eastern')
                m = ts.hour * 60 + ts.minute - (9 * 60 + 30)
                if m < 5:
                    continue
                if float(bar['high']) >= r.trigger:
                    bm = m; break
        rows.append((r.symbol, r.date, bm, r.range_high_est))
    print(f"{mo}: {len(g)} candidates, bars for {len(bars)}, {time.time()-t0:.0f}s", flush=True)
db.close()
pd.DataFrame(rows, columns=['symbol', 'date', 'breakout_min', 'range_high_est']).to_csv(OUT, index=False)
print('DONE', OUT, flush=True)
