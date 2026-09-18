#!/usr/bin/env python3
"""S1-PASSIVE step 3b — the measured NBBO at every distinct buy-to-cover instant.

Same source and honesty rail as `fetch_entry_nbbo.py` (Alpaca SIP; last quote at or
before the instant). Input `cover_keys.csv` is emitted by `sim.py`.
Resumable: appends to cover_nbbo.csv.
"""
from __future__ import annotations

import os
import sys
import time

import pandas as pd
from dotenv import load_dotenv

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
load_dotenv(f'{ROOT}/.env')

from alpaca.data.historical import StockHistoricalDataClient      # noqa: E402
from alpaca.data.requests import StockQuotesRequest               # noqa: E402
from alpaca.data.enums import DataFeed                            # noqa: E402

P = f'{ROOT}/research/fuckup_audit/O_halt/PASSIVE'
OUT = f'{P}/cover_nbbo.csv'
FIELDS = ['symbol', 'day', 'ts', 'n_quotes', 'q_ts', 'nbb', 'nbo', 'spread', 'err']


def main() -> int:
    k = pd.read_csv(f'{P}/cover_keys.csv', keep_default_na=False, na_values=[''],
                    dtype={'symbol': str, 'day': str, 'ts': str})
    print(f'cover instants {len(k):,}', flush=True)
    done = set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT, keep_default_na=False, na_values=[''],
                           dtype={'symbol': str, 'day': str, 'ts': str})
        done = set(zip(prev.symbol, prev.day, prev.ts))
    todo = [r for r in k.itertuples() if (r.symbol, r.day, r.ts) not in done]
    print(f'todo {len(todo):,}', flush=True)

    cl = StockHistoricalDataClient(os.getenv('ALPACA_API_KEY'), os.getenv('ALPACA_API_SECRET'))
    buf, t0 = [], time.time()
    for i, r in enumerate(todo, 1):
        when = pd.Timestamp(r.ts).tz_convert('UTC')
        rec = {c: '' for c in FIELDS}
        rec.update(symbol=r.symbol, day=r.day, ts=r.ts)
        try:
            q = cl.get_stock_quotes(StockQuotesRequest(
                symbol_or_symbols=r.symbol,
                start=(when - pd.Timedelta(seconds=300)).to_pydatetime(),
                end=(when + pd.Timedelta(seconds=1)).to_pydatetime(),
                feed=DataFeed.SIP))
            qs = q.data.get(r.symbol, []) if hasattr(q, 'data') else q.get(r.symbol, [])
            rows = [(pd.Timestamp(x.timestamp), float(x.bid_price), float(x.ask_price))
                    for x in qs if float(x.bid_price) > 0 and float(x.ask_price) >= float(x.bid_price)]
            rec['n_quotes'] = len(rows)
            prev_q = None
            for ts, b, a in rows:
                if ts <= when:
                    prev_q = (ts, b, a)
                else:
                    break
            if prev_q is None:
                rec['err'] = 'no_quote_at_or_before_cover'
            else:
                rec['q_ts'], rec['nbb'], rec['nbo'] = prev_q[0].isoformat(), prev_q[1], prev_q[2]
                rec['spread'] = prev_q[2] - prev_q[1]
        except Exception as exc:                                   # noqa: BLE001
            rec['err'] = f'{type(exc).__name__}: {exc}'[:120]
        buf.append(rec)
        if len(buf) >= 100 or i == len(todo):
            pd.DataFrame(buf, columns=FIELDS).to_csv(
                OUT, mode='a', header=not os.path.exists(OUT), index=False)
            buf = []
            print(f'{i}/{len(todo)}  {time.time()-t0:.0f}s', flush=True)
        time.sleep(0.03)
    return 0


if __name__ == '__main__':
    sys.exit(main())
