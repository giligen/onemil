#!/usr/bin/env python3
"""exec_cost entry cells 1,289 (post-then-cross) and 1,290 (spread-in-R gate).

Population: analysis_results/orb_bplus_book.csv, entered==1 (filled), TRAIN (2025) +
VAL (2026-01..05) only -- TEST (>=2026-06-01) is sealed, never touched here.

Trigger-instant NBBO reused from research/fuckup_audit/P_cost/spreads.parquet
(entry_fill_ts / entry_bid / entry_ask -- same honesty rail: last quote at or before the
first trade in the breakout minute with price > range_high).

Cell 1,290 needs nothing else. Cell 1,289 additionally fetches a fresh SIP trade+quote
tape for [entry_fill_ts, entry_fill_ts+30s] per fill via the same client path as
fetch_spreads.py, to test the post-then-cross rule.
"""
from __future__ import annotations
import os, sys, time
from datetime import timedelta
import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
load_dotenv(f'{ROOT}/.env')

from config import Config
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockQuotesRequest, StockTradesRequest
from alpaca.data.enums import DataFeed

OUT_DIR = f'{ROOT}/research/exec_cost'
PER_TRADE_OUT = f'{OUT_DIR}/per_trade.csv'
QLIMIT = 2000
WINDOW_S = 30

cfg = Config()
client = StockHistoricalDataClient(cfg.alpaca_api_key, cfg.alpaca_api_secret)


def load_book():
    book = pd.read_csv(f'{ROOT}/analysis_results/orb_bplus_book.csv',
                        usecols=['symbol', 'date', 'entry_price', 'pnl', 'pnl_pct',
                                 'win', 'entered', 'exit_reason'])
    book = book[book.entered == 1].copy()
    book['date'] = pd.to_datetime(book.date)
    sp = pd.read_parquet(f'{ROOT}/research/fuckup_audit/P_cost/spreads.parquet',
                          columns=['symbol', 'date', 'range_high', 'range_low',
                                   'entry_fill_ts', 'cov_entry', 'entry_ask', 'entry_bid',
                                   'entry_spread'])
    sp['date'] = pd.to_datetime(sp.date)
    m = book.merge(sp, on=['symbol', 'date'], suffixes=('_book', '_sp'))
    m = m[m.cov_entry].copy()
    bins = [pd.Timestamp('2000-01-01'), pd.Timestamp('2025-12-31'),
            pd.Timestamp('2026-05-31'), pd.Timestamp('2099-01-01')]
    m['split'] = pd.cut(m.date, bins=bins, labels=['TRAIN', 'VAL', 'TEST'])
    m = m[m.split.isin(['TRAIN', 'VAL'])].copy()
    m['entry_fill_ts'] = pd.to_datetime(m.entry_fill_ts, utc=True)
    m['R'] = m.range_high - m.range_low
    m['mid0'] = (m.entry_bid + m.entry_ask) / 2.0
    m['spread0'] = m.entry_spread
    return m.reset_index(drop=True)


def fetch_window(sym, t0):
    t1 = t0 + timedelta(seconds=WINDOW_S + 1)
    try:
        tr = client.get_stock_trades(StockTradesRequest(
            symbol_or_symbols=sym, start=t0, end=t1, feed=DataFeed.SIP, limit=QLIMIT))
        tdf = tr.df.reset_index() if len(tr.data.get(sym, [])) else pd.DataFrame()
    except Exception as e:
        tdf = None
        terr = str(e)
    else:
        terr = None
    try:
        qt = client.get_stock_quotes(StockQuotesRequest(
            symbol_or_symbols=sym, start=t0, end=t1, feed=DataFeed.SIP, limit=QLIMIT))
        qdf = qt.df.reset_index() if len(qt.data.get(sym, [])) else pd.DataFrame()
    except Exception as e:
        qdf = None
        qerr = str(e)
    else:
        qerr = None
    return tdf, terr, qdf, qerr


def simulate_row(row, tdf, qdf):
    t0 = row.entry_fill_ts
    mid0 = row.mid0
    tw = timedelta(seconds=WINDOW_S)
    passive_ts = None
    if tdf is not None and len(tdf):
        tdf = tdf.copy()
        tdf['ts'] = pd.to_datetime(tdf['timestamp'], utc=True)
        w = tdf[(tdf.ts > t0) & (tdf.ts <= t0 + tw) & (tdf.price <= mid0)]
        if len(w):
            passive_ts = w.ts.min()
    ask30 = bid30 = None
    if qdf is not None and len(qdf):
        qdf = qdf.copy()
        qdf['ts'] = pd.to_datetime(qdf['timestamp'], utc=True)
        target = t0 + tw
        cand = qdf[qdf.ts <= target]
        if len(cand):
            last = cand.iloc[-1]
            ask30, bid30 = last.ask_price, last.bid_price
        elif len(qdf):
            first = qdf.iloc[0]
            ask30, bid30 = first.ask_price, first.bid_price
    passive = passive_ts is not None
    if passive:
        new_entry = mid0
        cost_new = 0.0
    elif ask30 is not None:
        new_entry = ask30
        spread30 = ask30 - bid30 if bid30 is not None else np.nan
        cost_new = spread30 / 2.0 if not np.isnan(spread30) else np.nan
    else:
        new_entry = np.nan
        cost_new = np.nan
    return passive, new_entry, cost_new


def main():
    m = load_book()
    print(f'population: {len(m)} rows (TRAIN {sum(m.split=="TRAIN")}, VAL {sum(m.split=="VAL")})', flush=True)

    results = []
    done = set()
    if os.path.exists(PER_TRADE_OUT):
        prev = pd.read_csv(PER_TRADE_OUT)
        done = set(zip(prev.symbol, prev.date.astype(str)))
        results = prev.to_dict('records')
        print(f'resuming, {len(done)} already done', flush=True)

    for i, row in m.iterrows():
        key = (row.symbol, str(row.date.date()))
        if key in done:
            continue
        tdf, terr, qdf, qerr = fetch_window(row.symbol, row.entry_fill_ts)
        passive, new_entry, cost_new = simulate_row(row, tdf, qdf)
        results.append(dict(
            symbol=row.symbol, date=str(row.date.date()), split=row.split,
            entry_price_book=row.entry_price, pnl=row.pnl, pnl_pct=row.pnl_pct,
            range_high=row.range_high, range_low=row.range_low, R=row.R,
            entry_bid=row.entry_bid, entry_ask=row.entry_ask, mid0=row.mid0,
            spread0=row.spread0, passive=passive, new_entry=new_entry,
            cost_new=cost_new, terr=terr, qerr=qerr,
        ))
        if len(results) % 20 == 0:
            pd.DataFrame(results).to_csv(PER_TRADE_OUT, index=False)
            print(f'  ... {len(results)}/{len(m)}', flush=True)
        time.sleep(0.03)

    pd.DataFrame(results).to_csv(PER_TRADE_OUT, index=False)
    print(f'done: {len(results)} rows written to {PER_TRADE_OUT}', flush=True)


if __name__ == '__main__':
    main()
