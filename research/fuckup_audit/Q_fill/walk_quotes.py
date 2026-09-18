#!/usr/bin/env python3
"""Stage Q step 1 — walk the NBBO forward over the LIFE of the ORB stop-limit
order for every fill Stage P flagged as "ask above the cap at the trigger
instant".

The live order (trading/orb_engine.py::_submit_entry_order):
    stop_price  = round(range_high, 2)                 -- the trigger
    limit_price = round(range_high x (1 + 30bps), 2)   -- the CAP
    submitted at range_end (09:35 ET), cancelled by `_cancel_stale_pending_orders`
    at submit + `entry.time_stop_minutes` (60) -> 10:35 ET, order_status
    'time_stop_canceled'.

So: when a trade prints above `stop_price` the stop-limit ELECTS and becomes a
plain limit BUY at the cap.  If the NBBO ask at that instant is at or below the
cap the order is marketable and lifts the ask (Stage P's `measured` entry).  If
the ask is ABOVE the cap the order is NOT marketable: it rests as a bid at the
cap until the offer comes down to it, or until the 10:35 time stop cancels it.

This script answers, per flagged row: did the NBBO ask ever reach the cap while
the order was still live, and when.

Fill convention for a resting bid: the order is a MAKER at the cap, so the fill
price is the CAP (a seller crosses into our bid).  We do not credit ourselves
the price improvement of the incoming offer.  Queue priority is not modelled —
see REPORT.md approximations.

Output (appended, resumable): Q_fill/walk_rows.csv
Usage: python3 walk_quotes.py [--limit N] [--sleep S]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import timedelta

import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
load_dotenv(f'{ROOT}/.env')

from config import Config                                    # noqa: E402
from alpaca.data.historical import StockHistoricalDataClient  # noqa: E402
from alpaca.data.requests import StockQuotesRequest           # noqa: E402
from alpaca.data.enums import DataFeed                        # noqa: E402

Q = f'{ROOT}/research/fuckup_audit/Q_fill'
SRC = f'{ROOT}/research/fuckup_audit/P_cost/spreads.parquet'
OUT = f'{Q}/walk_rows.csv'
PAGE = 10000
MAX_PAGES = 30
FLUSH = 50
EXPIRY_ET = '10:35'          # range_end 09:35 + entry.time_stop_minutes 60

FIELDS = ['symbol', 'date', 'entry_price', 'cap', 'range_high', 'range_low',
          'entry_fill_ts', 'entry_ask', 'expiry_ts', 'life_s',
          'filled_later', 'fill_ts', 'secs_to_fill', 'ask_at_fill',
          'min_ask', 'min_ask_ts', 'n_quotes', 'pages', 'truncated', 'err']


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--sleep', type=float, default=0.04)
    args = ap.parse_args()

    sp = pd.read_parquet(SRC)
    d = sp[sp.cov_entry & (sp.entry_ask > sp.entry_price * (1 + 1e-12))].copy()
    d['entry_fill_ts'] = pd.to_datetime(d.entry_fill_ts, utc=True)
    d = d.sort_values(['date', 'symbol']).reset_index(drop=True)
    if args.limit:
        d = d.head(args.limit)

    done = set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT, keep_default_na=False, na_values=[''],
                           dtype={'symbol': str, 'date': str})
        done = set(zip(prev.symbol, prev.date))
    todo = [r for r in d.itertuples() if (r.symbol, r.date) not in done]
    print(f'flagged {len(d)} | done {len(done)} | to walk {len(todo)}', flush=True)

    cfg = Config()
    cl = StockHistoricalDataClient(cfg.alpaca_api_key, cfg.alpaca_api_secret)
    rows, t0 = [], time.time()
    for i, r in enumerate(todo, 1):
        cap = float(r.entry_price)
        t_start = r.entry_fill_ts
        expiry = pd.Timestamp(f'{r.date} {EXPIRY_ET}', tz='America/New_York').tz_convert('UTC')
        rec = dict(symbol=r.symbol, date=r.date, entry_price=cap, cap=cap,
                   range_high=r.range_high, range_low=r.range_low,
                   entry_fill_ts=t_start.isoformat(), entry_ask=r.entry_ask,
                   expiry_ts=expiry.isoformat(),
                   life_s=round((expiry - t_start).total_seconds(), 1),
                   filled_later=0, fill_ts='', secs_to_fill=np.nan,
                   ask_at_fill=np.nan, min_ask=np.nan, min_ask_ts='',
                   n_quotes=0, pages=0, truncated=0, err='')
        if expiry <= t_start:
            rec['err'] = 'expired_at_trigger'
            rows.append(rec)
            continue
        cur = t_start
        n_q, pages, best, best_ts, hit = 0, 0, np.inf, None, None
        try:
            while pages < MAX_PAGES:
                q = cl.get_stock_quotes(StockQuotesRequest(
                    symbol_or_symbols=r.symbol, start=cur, end=expiry,
                    feed=DataFeed.SIP, limit=PAGE))
                qs = q.data.get(r.symbol, []) if hasattr(q, 'data') else q.get(r.symbol, [])
                pages += 1
                if not qs:
                    break
                n_q += len(qs)
                last_ts = None
                for x in qs:
                    b, a = float(x.bid_price), float(x.ask_price)
                    ts = pd.Timestamp(x.timestamp)
                    last_ts = ts
                    if ts <= t_start:
                        continue
                    if not (b > 0 and a > 0 and a >= b):
                        continue
                    if a < best:
                        best, best_ts = a, ts
                    if a <= cap * (1 + 1e-12):
                        hit = (ts, a)
                        break
                if hit is not None or len(qs) < PAGE:
                    break
                cur = last_ts + timedelta(microseconds=1)
            else:
                rec['truncated'] = 1
            rec['n_quotes'] = n_q
            rec['pages'] = pages
            if np.isfinite(best):
                rec['min_ask'] = best
                rec['min_ask_ts'] = best_ts.isoformat()
            if hit is not None:
                rec['filled_later'] = 1
                rec['fill_ts'] = hit[0].isoformat()
                rec['secs_to_fill'] = round((hit[0] - t_start).total_seconds(), 3)
                rec['ask_at_fill'] = hit[1]
        except Exception as ex:
            rec['err'] = type(ex).__name__
            if '429' in str(ex):
                time.sleep(10)
        rows.append(rec)
        if args.sleep:
            time.sleep(args.sleep)
        if len(rows) >= FLUSH or i == len(todo):
            pd.DataFrame(rows).reindex(columns=FIELDS).to_csv(
                OUT, mode='a', header=not os.path.exists(OUT), index=False)
            rows = []
            el = time.time() - t0
            print(f'  {i}/{len(todo)}  {el/60:.1f} min  '
                  f'({i/max(el,1)*60:.0f} rows/min)', flush=True)
    print('DONE', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
