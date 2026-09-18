#!/usr/bin/env python3
"""Stage P step 2 — the MEASURED NBBO spread at the entry instant and the exit
instant of every fill in the honest ORB book.

Source: Alpaca **SIP consolidated** quotes/trades — deliberately the SAME source
and the SAME definition the band table (`research/lit_review_2026/cost_curve.csv`,
`build_cost_curve.py`) was built from, so band-vs-measured is apples to apples.
Databento EQUS.MINI `tbbo` (already on disk, Stage N1) is a **venue subset** whose
`bid_px_00/ask_px_00` is a single publisher's BBO, not the NBBO — it is used only
as a cross-check (`crosscheck_dbn.py`), never as the measurement.

Honesty rail (asserted in `spreads.py`): the quote used for a decision is the last
quote at or before that decision's timestamp.

Instants
--------
entry : the fill instant = the first TRADE in the breakout minute with
        price > range_high (the stop-limit's trigger print).  NBBO = last quote
        at or before it.
exit  : stop / lock / scale_stop / scale_lock -> the first quote in the exit
        minute whose BID <= the stop level (the trigger instant).
        eod / scale_eod / tag_bb / tag_b1 -> a closed-bar decision, so the LAST
        quote at or before the bar's close.

Output (appended, resumable): P_cost/spread_rows.csv
Usage: python3 fetch_spreads.py [--limit N] [--sleep S]
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
from alpaca.data.requests import StockQuotesRequest, StockTradesRequest  # noqa: E402
from alpaca.data.enums import DataFeed                        # noqa: E402

P = f'{ROOT}/research/fuckup_audit/P_cost'
SRC = f'{P}/exit_times.csv'
OUT = f'{P}/spread_rows.csv'
QLIMIT = 6000
EXIT_SLIP = 10.0 / 10000.0
STOP_REASONS = {'stop', 'lock', 'scale_stop', 'scale_lock'}
FLUSH = 100
# fixed schema: batches are appended to one CSV, so the column set must not
# vary with which error branch fired (that would silently misalign the file).
FIELDS = ['symbol', 'date', 'entry_price', 'range_high', 'range_low',
          'exit_reason', 'exit_price', 'entry_ts', 'exit_ts',
          'entry_n_quotes', 'entry_n_trades', 'entry_med_spread',
          'entry_fill_ts', 'entry_q_ts', 'entry_bid', 'entry_ask',
          'entry_spread', 'entry_err',
          'exit_n_quotes', 'exit_med_spread', 'exit_instant_rule',
          'exit_q_ts', 'exit_bid', 'exit_ask', 'exit_spread', 'exit_err']


def _quotes(cl, sym, t0, t1):
    q = cl.get_stock_quotes(StockQuotesRequest(
        symbol_or_symbols=sym, start=t0, end=t1, feed=DataFeed.SIP, limit=QLIMIT))
    qs = q.data.get(sym, []) if hasattr(q, 'data') else q.get(sym, [])
    out = []
    for x in qs:
        b, a = float(x.bid_price), float(x.ask_price)
        if b > 0 and a > 0 and a >= b:
            out.append((pd.Timestamp(x.timestamp), b, a))
    return out


def _trades(cl, sym, t0, t1):
    t = cl.get_stock_trades(StockTradesRequest(
        symbol_or_symbols=sym, start=t0, end=t1, feed=DataFeed.SIP, limit=QLIMIT))
    ts = t.data.get(sym, []) if hasattr(t, 'data') else t.get(sym, [])
    return [(pd.Timestamp(x.timestamp), float(x.price)) for x in ts if float(x.price) > 0]


def _med(qs):
    return float(np.median([a - b for _, b, a in qs])) if qs else np.nan


def _at(qs, when):
    """Last quote at or before `when` (the honesty rail). None if none exists."""
    prev = None
    for t, b, a in qs:
        if t <= when:
            prev = (t, b, a)
        else:
            break
    return prev


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0, help='first N rows only (smoke test)')
    ap.add_argument('--sleep', type=float, default=0.06)
    args = ap.parse_args()

    d = pd.read_csv(SRC, keep_default_na=False, na_values=[''],
                    dtype={'symbol': str, 'date': str})
    d['entry_ts'] = pd.to_datetime(d.entry_ts, utc=True)
    d['exit_ts'] = pd.to_datetime(d.exit_ts, utc=True)
    if args.limit:
        d = d.head(args.limit)

    done = set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT, keep_default_na=False, na_values=[''],
                           dtype={'symbol': str, 'date': str})
        done = set(zip(prev.symbol, prev.date))
    todo = [r for r in d.itertuples() if (r.symbol, r.date) not in done]
    print(f'rows {len(d)} | already done {len(done)} | to fetch {len(todo)}', flush=True)

    cfg = Config()
    cl = StockHistoricalDataClient(cfg.alpaca_api_key, cfg.alpaca_api_secret)
    rows = []
    t0 = time.time()
    for i, r in enumerate(todo, 1):
        rec = dict(symbol=r.symbol, date=r.date, entry_price=r.entry_price,
                   range_high=r.range_high, range_low=r.range_low,
                   exit_reason=r.exit_reason, exit_price=r.exit_price,
                   entry_ts=r.entry_ts.isoformat(), exit_ts=r.exit_ts.isoformat())
        # ---- entry minute
        e0 = r.entry_ts.floor('min')
        e1 = e0 + timedelta(minutes=1)
        try:
            qs = _quotes(cl, r.symbol, e0, e1)
            tr = _trades(cl, r.symbol, e0, e1)
            fill = next((t for t, px in tr if px > r.range_high), None)
            rec['entry_n_quotes'] = len(qs)
            rec['entry_n_trades'] = len(tr)
            rec['entry_med_spread'] = _med(qs)
            rec['entry_fill_ts'] = fill.isoformat() if fill is not None else ''
            q = _at(qs, fill) if fill is not None else None
            if q is not None:
                rec['entry_q_ts'] = q[0].isoformat()
                rec['entry_bid'] = q[1]
                rec['entry_ask'] = q[2]
                rec['entry_spread'] = q[2] - q[1]
            else:
                rec['entry_q_ts'] = ''
                rec['entry_bid'] = rec['entry_ask'] = rec['entry_spread'] = np.nan
        except Exception as ex:
            rec['entry_n_quotes'] = -1
            rec['entry_err'] = f'{type(ex).__name__}'
            if '429' in str(ex):
                time.sleep(10)
        # ---- exit minute
        x0 = r.exit_ts.floor('min')
        x1 = x0 + timedelta(minutes=1)
        try:
            qs = _quotes(cl, r.symbol, x0, x1)
            rec['exit_n_quotes'] = len(qs)
            rec['exit_med_spread'] = _med(qs)
            q = None
            if r.exit_reason in STOP_REASONS:
                lvl = r.exit_price / (1 - EXIT_SLIP)
                q = next(((t, b, a) for t, b, a in qs if b <= lvl), None)
                if q is None and qs:
                    q = qs[-1]
                    rec['exit_instant_rule'] = 'last_quote_fallback'
                else:
                    rec['exit_instant_rule'] = 'first_bid_at_or_below_stop'
            else:
                q = qs[-1] if qs else None
                rec['exit_instant_rule'] = 'bar_close_last_quote'
            if q is not None:
                rec['exit_q_ts'] = q[0].isoformat()
                rec['exit_bid'] = q[1]
                rec['exit_ask'] = q[2]
                rec['exit_spread'] = q[2] - q[1]
            else:
                rec['exit_q_ts'] = ''
                rec['exit_bid'] = rec['exit_ask'] = rec['exit_spread'] = np.nan
        except Exception as ex:
            rec['exit_n_quotes'] = -1
            rec['exit_err'] = f'{type(ex).__name__}'
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
            print(f'  {i}/{len(todo)}  {el / 60:.1f} min  '
                  f'({i / max(el, 1) * 60:.0f} rows/min)', flush=True)
    print('DONE', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
