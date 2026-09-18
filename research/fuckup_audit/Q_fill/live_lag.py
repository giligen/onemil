#!/usr/bin/env python3
"""Stage Q step 4b — the LIVE trigger->fill lag, the ground truth for the
simulated quote walk.  For every filled live ORB order, find the MARKET breakout
bar (first 1-min bar after 09:35 whose high cleared range_high, from cache.db
read-only) and measure how long after that bar's OPEN the order actually filled
(`order_filled_at`).  Also the fill price against the cap.
"""
import json, os, sqlite3, sys
import numpy as np, pandas as pd
ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT)
Q = f'{ROOT}/research/fuckup_audit/Q_fill'
t = sqlite3.connect('file:data/trades.db?mode=ro', uri=True)
d = pd.read_sql("SELECT trade_date,symbol,order_status,entry_price,fill_price,"
                "order_submitted_at,order_filled_at,submit_to_fill_ms,pattern_data "
                "FROM trades WHERE strategy='orb' AND order_filled_at IS NOT NULL", t)
t.close()
def pget(js, k):
    try: p = json.loads(js) if js else {}
    except Exception: return np.nan
    v = p.get(k)
    try: return float(v)
    except Exception: return np.nan
d['range_high'] = [pget(x, 'range_high') for x in d.pattern_data]
b = sqlite3.connect('file:data/cache.db?mode=ro', uri=True)
rows = []
for r in d.itertuples():
    if not np.isfinite(r.range_high): continue
    bars = pd.DataFrame(b.execute("SELECT timestamp,high FROM intraday_bars_1min "
                                  "WHERE symbol=? AND bar_date=? ORDER BY timestamp",
                                  (r.symbol, r.trade_date)).fetchall(),
                        columns=['timestamp', 'high'])
    if bars.empty: continue
    ts = pd.to_datetime(bars.timestamp, utc=True, format='mixed')
    et = ts.dt.tz_convert('America/New_York'); m = et.dt.hour * 60 + et.dt.minute
    w = bars[(m >= 575) & (m < 635) & (bars.high > r.range_high)]
    if w.empty: continue
    brk = pd.to_datetime(w.timestamp.iloc[0], utc=True, format='mixed')
    fill = pd.to_datetime(r.order_filled_at, utc=True, format='mixed')
    rows.append(dict(trade_date=r.trade_date, symbol=r.symbol, cap=r.entry_price,
                     fill_price=r.fill_price, brk_ts=brk.isoformat(),
                     fill_ts=fill.isoformat(),
                     lag_s=(fill - brk).total_seconds(),
                     fill_vs_cap_bps=(r.fill_price / r.entry_price - 1) * 1e4))
b.close()
o = pd.DataFrame(rows); o.to_csv(f'{Q}/live_lag.csv', index=False)
lg = o.lag_s
print(f'live filled orders with a locatable breakout bar: {len(o)}')
print(f'lag from the breakout BAR OPEN to the live fill (s): '
      f'median {lg.median():.0f}  mean {lg.mean():.0f}  p25 {lg.quantile(.25):.0f} '
      f'p75 {lg.quantile(.75):.0f}  p90 {lg.quantile(.9):.0f}  max {lg.max():.0f}')
print('buckets:', pd.cut(lg, [-1e9, 60, 120, 300, 900, 1e9],
      labels=['<=1min (the breakout bar)', '1-2min', '2-5min', '5-15min', '>15min']
      ).value_counts().sort_index().to_dict())
print(f'share filling INSIDE the breakout minute: {(lg <= 60).mean()*100:.1f}%')
print(f'fill vs cap bps: median {o.fill_vs_cap_bps.median():.1f} '
      f'mean {o.fill_vs_cap_bps.mean():.1f}')
late = o[lg > 60]
print(f'\nlate fills (n={len(late)}):')
print(late[['trade_date','symbol','cap','fill_price','lag_s','fill_vs_cap_bps']]
      .to_string(index=False))
