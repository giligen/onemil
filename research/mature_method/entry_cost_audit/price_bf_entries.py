#!/usr/bin/env python3
"""Obtainability of the BF Stage-2 entry fill, measured against Alpaca SIP NBBO.

The BF backtest fills at `max(bar_open, breakout_level) * (1 + entry_slippage_pct)`
with `entry_slippage_pct = 0.005` (a flat 50 bps).  Live, the buy-stop-limit elects
when price trades through the level and then lifts the OFFER, so the obtainable
price is the NBBO **ask at the election instant** (CLAUDE.md rule 1b, and the
convention Stage P/Q used: the NBBO is the last quote at or before the fill print).

For every trade of a book this measures, at the entry bar:
  t_fill     : the first TRADE print in the entry minute (the bar's open print)
  ask_fill   : the last NBBO ask at or before t_fill      <- the obtainable price
  ask_pre    : the last NBBO ask at or before the bar's START (causal, may be stale)
  excess     : ask_fill - modelled entry_price            <- what the BT did not charge
  dR         : excess / (entry_price - stop_loss)         <- in R, at the BT's own risk

Usage: python3 price_bf_entries.py <runs_csv> <out_csv>
Read-only outside research/mature_method/entry_cost_audit/.
"""
from __future__ import annotations

import os
import sys

import pandas as pd
from dotenv import load_dotenv

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
load_dotenv(f'{ROOT}/.env')

from alpaca.data.historical import StockHistoricalDataClient   # noqa: E402
from alpaca.data.requests import StockQuotesRequest, StockTradesRequest  # noqa: E402
from alpaca.data.enums import DataFeed                         # noqa: E402

QLIMIT = 10000


def main() -> int:
    src, out = sys.argv[1], sys.argv[2]
    d = pd.read_csv(src, keep_default_na=False, na_values=[''], dtype={'symbol': str})
    ts = pd.to_datetime(d.date + ' ' + d.entry_time_et).dt.tz_localize('America/New_York')
    d['bar_start'] = ts.dt.tz_convert('UTC')

    cl = StockHistoricalDataClient(os.getenv('ALPACA_API_KEY'),
                                   os.getenv('ALPACA_API_SECRET'))
    rows = []
    for i, r in d.iterrows():
        o = {'symbol': r.symbol, 'date': r.date, 'entry_time_et': r.entry_time_et,
             'entry_price': r.entry_price, 'stop_loss': r.stop_loss, 'pnl': r.pnl,
             'avg_volume_20d': r.avg_volume_20d, 'err': ''}
        t0 = r.bar_start - pd.Timedelta(minutes=2)
        t1 = r.bar_start + pd.Timedelta(minutes=1)
        try:
            q = cl.get_stock_quotes(StockQuotesRequest(
                symbol_or_symbols=r.symbol, start=t0.to_pydatetime(),
                end=t1.to_pydatetime(), feed=DataFeed.SIP, limit=QLIMIT))
            qd = pd.DataFrame([{'ts': x.timestamp, 'bid': x.bid_price,
                                'ask': x.ask_price} for x in q[r.symbol]])
            tr = cl.get_stock_trades(StockTradesRequest(
                symbol_or_symbols=r.symbol, start=r.bar_start.to_pydatetime(),
                end=t1.to_pydatetime(), feed=DataFeed.SIP, limit=QLIMIT))
            td = pd.DataFrame([{'ts': x.timestamp, 'px': x.price, 'sz': x.size}
                               for x in tr[r.symbol]])
        except Exception as e:                                  # noqa: BLE001
            o['err'] = f'fetch:{str(e)[:60]}'
            rows.append(o)
            continue
        if qd.empty:
            o['err'] = 'no_quotes'
            rows.append(o)
            continue
        qd['ts'] = pd.to_datetime(qd.ts, utc=True)
        qd = qd[(qd.bid > 0) & (qd.ask > qd.bid)].sort_values('ts')
        pre = qd[qd.ts <= r.bar_start]
        if len(pre):
            o['ask_pre'] = float(pre.iloc[-1].ask)
            o['bid_pre'] = float(pre.iloc[-1].bid)
        if not td.empty:
            td['ts'] = pd.to_datetime(td.ts, utc=True)
            td = td.sort_values('ts')
            t_fill = td.iloc[0].ts
            o['t_fill'] = t_fill.isoformat()
            o['first_trade_px'] = float(td.iloc[0].px)
            o['bar_trade_hi'] = float(td.px.max())
            o['bar_n_trades'] = int(len(td))
            at = qd[qd.ts <= t_fill]
            if len(at):
                o['ask_fill'] = float(at.iloc[-1].ask)
                o['bid_fill'] = float(at.iloc[-1].bid)
        else:
            o['err'] = 'no_trades_in_bar'
        rows.append(o)
        if (i + 1) % 20 == 0:
            print(f'  {i + 1}/{len(d)}', flush=True)

    x = pd.DataFrame(rows)
    x.to_csv(out, index=False)
    print(f'wrote {out}  rows={len(x)}  with ask_fill={x.get("ask_fill").notna().sum()}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
