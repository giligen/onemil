#!/usr/bin/env python3
"""S1-REVIVE step 1 — entry-instant NBBO for the LONG book's events only.

Identical conventions to `PASSIVE/fetch_entry_nbbo.py` (Alpaca SIP; honesty rail = the last quote at
or BEFORE the decision instant `entry_t`; mean NBBO spread over the resume minute as the cost
measure). The ONLY change is the admission rule: the LONG book is `reopen <= ref * 1.006`.

Events already present in PASSIVE/entry_nbbo.csv (the +-0.6% overlap) are NOT refetched.
Output appends to REVIVE/entry_nbbo_long.csv with the same schema.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
load_dotenv(f'{ROOT}/.env')

from alpaca.data.historical import StockHistoricalDataClient      # noqa: E402
from alpaca.data.requests import StockQuotesRequest               # noqa: E402
from alpaca.data.enums import DataFeed                            # noqa: E402

P = f'{ROOT}/research/fuckup_audit/O_halt/REVIVE'
PAS = f'{ROOT}/research/fuckup_audit/O_halt/PASSIVE'
SRC = f'{ROOT}/research/fuckup_audit/O_halt/trades.parquet'
OUT = f'{P}/entry_nbbo_long.csv'
CEIL = 1.006
FIELDS = ['symbol', 'day', 'halt_seq', 'entry_t', 'n_quotes',
          'q_ts', 'nbb', 'nbo', 'spread', 'mean_spread', 'err']


def main() -> int:
    t = pd.read_parquet(SRC)
    t = t[t['fill'] <= t['ref'] * CEIL].copy()
    t['entry_t'] = pd.to_datetime(t['entry_t'], utc=True)
    print(f'long candidates {len(t):,}', flush=True)

    done = set()
    for path in (f'{PAS}/entry_nbbo.csv', OUT):
        if os.path.exists(path):
            prev = pd.read_csv(path, keep_default_na=False, na_values=[''],
                               dtype={'symbol': str, 'day': str})
            done |= set(zip(prev.symbol, prev.day, prev.halt_seq))
    todo = [r for r in t.itertuples() if (r.symbol, r.day, r.halt_seq) not in done]
    print(f'todo {len(todo):,}', flush=True)
    if not todo:
        return 0

    cl = StockHistoricalDataClient(os.getenv('ALPACA_API_KEY'), os.getenv('ALPACA_API_SECRET'))
    buf, t0 = [], time.time()
    for i, r in enumerate(todo, 1):
        rec = {k: '' for k in FIELDS}
        rec.update(symbol=r.symbol, day=r.day, halt_seq=r.halt_seq, entry_t=r.entry_t.isoformat())
        try:
            q = cl.get_stock_quotes(StockQuotesRequest(
                symbol_or_symbols=r.symbol,
                start=(r.entry_t - pd.Timedelta(seconds=30)).to_pydatetime(),
                end=(r.entry_t + pd.Timedelta(seconds=60)).to_pydatetime(),
                feed=DataFeed.SIP))
            qs = q.data.get(r.symbol, []) if hasattr(q, 'data') else q.get(r.symbol, [])
            rows = [(pd.Timestamp(x.timestamp), float(x.bid_price), float(x.ask_price))
                    for x in qs if float(x.bid_price) > 0 and float(x.ask_price) >= float(x.bid_price)]
            rec['n_quotes'] = len(rows)
            minute = [(b, a) for ts, b, a in rows
                      if r.entry_t <= ts < r.entry_t + pd.Timedelta(seconds=60)]
            if minute:
                rec['mean_spread'] = float(np.mean([a - b for b, a in minute]))
            prev_q = None
            for ts, b, a in rows:
                if ts <= r.entry_t:
                    prev_q = (ts, b, a)
                else:
                    break
            if prev_q is None:
                rec['err'] = 'no_quote_at_or_before_entry'
            else:
                rec['q_ts'], rec['nbb'], rec['nbo'] = prev_q[0].isoformat(), prev_q[1], prev_q[2]
                rec['spread'] = prev_q[2] - prev_q[1]
        except Exception as exc:                                  # noqa: BLE001
            rec['err'] = f'{type(exc).__name__}: {exc}'[:120]
        buf.append(rec)
        if len(buf) >= 50 or i == len(todo):
            pd.DataFrame(buf, columns=FIELDS).to_csv(
                OUT, mode='a', header=not os.path.exists(OUT), index=False)
            buf = []
            print(f'{i}/{len(todo)}  {time.time()-t0:.0f}s', flush=True)
        time.sleep(0.05)
    return 0


if __name__ == '__main__':
    sys.exit(main())
