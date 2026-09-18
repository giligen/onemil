#!/usr/bin/env python3
"""Stage Q step 4 — the LIVE cross-check.  The only ground truth in the study.

`data/trades.db` (read-only) holds every ORB order the engine ever sent.  For
each one we know the limit price it was sent with (`entry_price` = the cap =
range_high x 1.003, rounded to the cent), whether it filled, at what price, and
-- for the ones that did not -- whether the market ever ELECTED the stop
(a 1-min bar high above `range_high`, from `data/cache.db`, read-only).

Three questions:
  1. of the orders whose stop ELECTED, what share never filled?  (the live
     analogue of Stage P's "ask above the cap" flag turning into a no-fill)
  2. of the orders that filled, how did the fill price sit against the cap?
  3. for the elected-but-unfilled ones, how far above the cap did the market
     trade -- i.e. is the simulated mechanism the one that actually bit?

Output: Q_fill/live_orders.csv
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from datetime import timedelta

import numpy as np
import pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
Q = f'{ROOT}/research/fuckup_audit/Q_fill'
BARSQL = ("SELECT timestamp, high, low FROM intraday_bars_1min "
          "WHERE symbol=? AND bar_date=? ORDER BY timestamp")


def main() -> int:
    tcon = sqlite3.connect('file:data/trades.db?mode=ro', uri=True)
    d = pd.read_sql(
        "SELECT trade_date, symbol, order_status, entry_price, fill_price, "
        "shares, pnl, exit_reason, entry_quote_bid, entry_quote_ask, "
        "entry_fill_quote_bid, entry_fill_quote_ask, pattern_data, reject_reason "
        "FROM trades WHERE strategy='orb' ORDER BY trade_date, symbol", tcon)
    tcon.close()
    print(f'live ORB orders in trades.db: {len(d)}', flush=True)
    print(d.order_status.value_counts(dropna=False).to_dict(), flush=True)

    def pget(js, *keys):
        try:
            p = json.loads(js) if js else {}
        except Exception:
            return np.nan
        for k in keys:
            if k in p and p[k] is not None:
                try:
                    return float(p[k])
                except Exception:
                    return np.nan
        return np.nan

    d['range_high'] = [pget(x, 'range_high', 'orb_range_high') for x in d.pattern_data]
    d['range_low'] = [pget(x, 'range_low', 'orb_range_low') for x in d.pattern_data]

    bcon = sqlite3.connect('file:data/cache.db?mode=ro', uri=True)
    elected, maxhigh, minlow_after = [], [], []
    for r in d.itertuples():
        rh = r.range_high
        if not np.isfinite(rh):
            elected.append(np.nan); maxhigh.append(np.nan); minlow_after.append(np.nan)
            continue
        bars = pd.DataFrame(bcon.execute(BARSQL, (r.symbol, r.trade_date)).fetchall(),
                            columns=['timestamp', 'high', 'low'])
        if bars.empty:
            elected.append(np.nan); maxhigh.append(np.nan); minlow_after.append(np.nan)
            continue
        ts = pd.to_datetime(bars.timestamp, utc=True, format='mixed')
        et = ts.dt.tz_convert('America/New_York')
        m = et.dt.hour * 60 + et.dt.minute
        w = bars[(m >= 575) & (m < 635)]          # 09:35 -> the 10:35 time stop
        if w.empty:
            elected.append(np.nan); maxhigh.append(np.nan); minlow_after.append(np.nan)
            continue
        hi = float(w.high.max())
        elected.append(int(hi > rh))
        maxhigh.append(hi)
        minlow_after.append(float(w.low.min()))
    bcon.close()
    d['elected'] = elected
    d['window_high'] = maxhigh
    d['window_low'] = minlow_after
    d['cap'] = d.entry_price
    d['overshoot_bps'] = (d.window_high / d.cap - 1) * 10000
    d.drop(columns=['pattern_data']).to_csv(f'{Q}/live_orders.csv', index=False)

    sub = d[d.order_status.isin(['closed', 'filled', 'time_stop_canceled'])].copy()
    el = sub[sub.elected == 1]
    filled = el[el.order_status != 'time_stop_canceled']
    unf = el[el.order_status == 'time_stop_canceled']
    print('\n--- 1. elected orders (a bar high cleared range_high before 10:35) ---')
    print(f'submitted & not rejected: {len(sub)} | elected: {len(el)} '
          f'({len(el)/max(len(sub),1)*100:.1f}%) | of those FILLED {len(filled)} '
          f'({len(filled)/max(len(el),1)*100:.1f}%), NEVER filled {len(unf)} '
          f'({len(unf)/max(len(el),1)*100:.1f}%)')
    print(f'non-elected (never broke out) time-stopped: '
          f'{int((sub.elected == 0).sum())}')
    print('\n--- 2. fill price vs the cap ---')
    f = filled[filled.fill_price.notna() & (filled.fill_price > 0)]
    rel = (f.fill_price / f.cap - 1) * 10000
    print(f'n={len(f)} | fill <= cap {int((rel <= 1e-6).sum())} '
          f'({(rel <= 1e-6).mean()*100:.1f}%) | fill == cap (within 1bp) '
          f'{int((rel.abs() <= 1).sum())} | fill > cap {int((rel > 1e-6).sum())}')
    print(f'fill-vs-cap bps: median {rel.median():.1f} mean {rel.mean():.1f} '
          f'p10 {rel.quantile(.1):.1f} p90 {rel.quantile(.9):.1f}')
    ask = f[f.entry_quote_ask.notna() & (f.entry_quote_ask > 0)]
    if len(ask):
        above = (ask.entry_quote_ask > ask.cap)
        print(f'submit-time NBBO ask above the cap on {int(above.sum())}/{len(ask)} '
              f'({above.mean()*100:.1f}%) of the FILLED orders '
              f'(quote is at 09:35 submit, not at the trigger)')
    print('\n--- 3. the unfilled elected orders ---')
    print(f'n={len(unf)} | market traded above the cap on '
          f'{int((unf.window_high > unf.cap).sum())} of them; '
          f'median overshoot {unf.overshoot_bps.median():.0f} bps')
    print(unf[['trade_date', 'symbol', 'cap', 'window_high', 'overshoot_bps']]
          .to_string(index=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
