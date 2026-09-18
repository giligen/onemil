#!/usr/bin/env python3
"""S1-REVIVE step 2b — measured NBBO at every CHARGED exit instant of the REVIVE sim.

Same source and honesty rail as PASSIVE/fetch_cover_nbbo.py (Alpaca SIP; last quote at or BEFORE the
instant, 300 s lookback). Target exits are passive limits and are charged 0, so they are not fetched.
Instants already present in PASSIVE/cover_nbbo.csv are NOT refetched.
Resumable: appends to REVIVE/exit_nbbo.csv.
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

P = f'{ROOT}/research/fuckup_audit/O_halt/REVIVE'
PAS = f'{ROOT}/research/fuckup_audit/O_halt/PASSIVE'
OUT = f'{P}/exit_nbbo.csv'
FIELDS = ['symbol', 'day', 'ts', 'n_quotes', 'q_ts', 'nbb', 'nbo', 'spread', 'err']


def main() -> int:
    s = pd.read_parquet(f'{P}/sim_rows.parquet')
    s = s[(s.filled == 1) & (s.why != 'target')].copy()
    s['ts'] = pd.to_datetime(s['exit_t']).dt.tz_convert('UTC').map(lambda x: x.isoformat())
    k = s[['symbol', 'day', 'ts']].drop_duplicates()
    done = set()
    for path in (f'{PAS}/cover_nbbo.csv', OUT):
        if os.path.exists(path):
            prev = pd.read_csv(path, keep_default_na=False, na_values=[''],
                               dtype={'symbol': str, 'day': str, 'ts': str})
            done |= set(zip(prev.symbol, prev.day, prev.ts))
    todo = [r for r in k.itertuples() if (r.symbol, r.day, r.ts) not in done]
    print(f'charged exit instants {len(k):,}   todo {len(todo):,}', flush=True)

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
                rec['err'] = 'no_quote_at_or_before_exit'
            else:
                rec['q_ts'], rec['nbb'], rec['nbo'] = prev_q[0].isoformat(), prev_q[1], prev_q[2]
                rec['spread'] = prev_q[2] - prev_q[1]
        except Exception as exc:                                   # noqa: BLE001
            rec['err'] = f'{type(exc).__name__}: {exc}'[:120]
        buf.append(rec)
        if len(buf) >= 200 or i == len(todo):
            pd.DataFrame(buf, columns=FIELDS).to_csv(
                OUT, mode='a', header=not os.path.exists(OUT), index=False)
            buf = []
            print(f'{i}/{len(todo)}  {time.time()-t0:.0f}s', flush=True)
        time.sleep(0.03)
    return 0


if __name__ == '__main__':
    sys.exit(main())
