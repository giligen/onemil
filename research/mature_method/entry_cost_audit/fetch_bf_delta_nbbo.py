#!/usr/bin/env python3
"""Measure the real NBBO half-spread at the entry instant of the BF ADV-gate DELTA trades.

The delta = the 48 trades that `scanner.min_daily_volume: 200000 -> 0` ADDS to the
shipped P1 book (research/bf_frequency/runs/VOL_OFF.csv minus runs/P1.csv; no P1 trade
is displaced).  Those 48 trades are the entire evidence for the Monday BF config.

Question: the shipped Stage-2 charges a FLAT 50 bps entry slip (config
`trading.entry_slippage_pct: 0.005`).  research/mature_method/red_to_green/REPORT.md
measured the true entry leg at 1.00 x half-spread.  Is 50 bps >= 1.00 x half-spread on
THESE (thin, low-ADV) names?

Decision instant: the BF backtest fills at the ENTRY BAR'S OPEN
(`fill = max(bar_open, breakout_level) * (1 + entry_slippage_pct)`), so the quote that
prices that fill is the LAST NBBO at or before the entry bar's start.  The median NBBO
over the entry minute is reported alongside as a dispersion check.

Source: Alpaca SIP consolidated quotes -- the same feed and definition Stage P/Q used.
Read-only; writes only under research/mature_method/entry_cost_audit/.

Usage: python3 fetch_bf_delta_nbbo.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
load_dotenv(f'{ROOT}/.env')

from alpaca.data.historical import StockHistoricalDataClient   # noqa: E402
from alpaca.data.requests import StockQuotesRequest            # noqa: E402
from alpaca.data.enums import DataFeed                         # noqa: E402

D = f'{ROOT}/research/mature_method/entry_cost_audit'
SRC = f'{D}/delta_trades.csv'
OUT = f'{D}/delta_nbbo.csv'
QLIMIT = 10000


def main() -> int:
    d = pd.read_csv(SRC, keep_default_na=False, na_values=[''], dtype={'symbol': str})
    ts = pd.to_datetime(d.date + ' ' + d.entry_time_et).dt.tz_localize('America/New_York')
    d['bar_start'] = ts.dt.tz_convert('UTC')

    cl = StockHistoricalDataClient(os.getenv('ALPACA_API_KEY'),
                                   os.getenv('ALPACA_API_SECRET'))
    rows = []
    for i, r in d.iterrows():
        t0 = r.bar_start - pd.Timedelta(minutes=2)
        t1 = r.bar_start + pd.Timedelta(minutes=1)
        try:
            q = cl.get_stock_quotes(StockQuotesRequest(
                symbol_or_symbols=r.symbol, start=t0.to_pydatetime(),
                end=t1.to_pydatetime(), feed=DataFeed.SIP, limit=QLIMIT))
            df = pd.DataFrame([{'ts': x.timestamp, 'bid': x.bid_price,
                                'ask': x.ask_price} for x in q[r.symbol]])
        except Exception as e:                       # noqa: BLE001
            print(f'  {r.symbol} {r.date}: FETCH FAIL {e}', flush=True)
            rows.append({'symbol': r.symbol, 'date': r.date, 'err': str(e)[:80]})
            continue
        if df.empty:
            print(f'  {r.symbol} {r.date}: no quotes', flush=True)
            rows.append({'symbol': r.symbol, 'date': r.date, 'err': 'no_quotes'})
            continue
        df['ts'] = pd.to_datetime(df.ts, utc=True)
        df = df[(df.bid > 0) & (df.ask > df.bid)]
        # R1 causality: the decision quote is at or before the bar's start.
        pre = df[df.ts <= r.bar_start]
        inbar = df[(df.ts >= r.bar_start) & (df.ts < r.bar_start + pd.Timedelta(minutes=1))]
        o = {'symbol': r.symbol, 'date': r.date, 'entry_price': r.entry_price,
             'stop_loss': r.stop_loss, 'pnl': r.pnl,
             'avg_volume_20d': r.avg_volume_20d, 'n_q_pre': len(pre),
             'n_q_bar': len(inbar), 'err': ''}
        if len(pre):
            q0 = pre.iloc[-1]
            o['q_ts'] = q0.ts.isoformat()
            o['bid'] = float(q0.bid)
            o['ask'] = float(q0.ask)
            o['spread'] = float(q0.ask - q0.bid)
            mid = 0.5 * (q0.ask + q0.bid)
            o['spread_bps'] = 1e4 * o['spread'] / mid
        else:
            o['err'] = 'no_pre_quote'
        if len(inbar):
            sp = (inbar.ask - inbar.bid)
            mid = 0.5 * (inbar.ask + inbar.bid)
            o['bar_med_spread_bps'] = float(np.median(1e4 * sp / mid))
        rows.append(o)
        if (i + 1) % 10 == 0:
            print(f'  {i + 1}/{len(d)}', flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    ok = out[out.get('spread_bps').notna()] if 'spread_bps' in out else out.iloc[:0]
    print(f'\nrows {len(out)}  measured {len(ok)}')
    if len(ok):
        print('full spread bps  ', ok.spread_bps.describe()[
            ['mean', '50%', '75%', 'max']].round(1).to_dict())
        print('HALF spread bps  ', (ok.spread_bps / 2).describe()[
            ['mean', '50%', '75%', 'max']].round(1).to_dict())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
