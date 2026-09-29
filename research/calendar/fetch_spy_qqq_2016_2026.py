"""
Cell 1,672 follow-up: fetch full-span SPY/QQQ daily bars from Alpaca (SIP feed) so the
pre-holiday sleeve can be judged on the PREREG's stated 2016-2026 window instead of the
~2.25yr slice that happened to be sitting in data/cache.db.

Read-only market-data call only (no orders, no account writes). Output is a standalone CSV
under research/calendar/ -- data/cache.db is never opened for write.
"""
import os
import sys
from datetime import date

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data_sources.alpaca_client import AlpacaClient  # noqa: E402

OUT_CSV = 'research/calendar/spy_qqq_daily_2016_2026.csv'


def main():
    load_dotenv()
    api_key = os.getenv('ALPACA_API_KEY')
    api_secret = os.getenv('ALPACA_API_SECRET')
    if not api_key or not api_secret:
        print('ERROR: missing ALPACA_API_KEY/ALPACA_API_SECRET in environment (.env)')
        sys.exit(1)

    client = AlpacaClient(api_key=api_key, api_secret=api_secret)  # paper=True default; data reads only

    start = date(2016, 1, 1)
    end = date(2026, 9, 29)
    print(f'[fetch] SPY/QQQ daily bars {start}..{end} via get_daily_bars_range (SIP) ...')
    bars = client.get_daily_bars_range(['SPY', 'QQQ'], start, end)

    rows = []
    for sym, blist in bars.items():
        for b in blist:
            rows.append(dict(symbol=sym, bar_date=b['date'], open=b['open'], high=b['high'],
                              low=b['low'], close=b['close'], volume=b['volume']))
    df = pd.DataFrame(rows).sort_values(['symbol', 'bar_date'])
    df.to_csv(OUT_CSV, index=False)

    for sym in ['SPY', 'QQQ']:
        sub = df[df.symbol == sym]
        print(f'[fetch] {sym}: n={len(sub)} range={sub.bar_date.min()}..{sub.bar_date.max()}')
    print(f'[fetch] wrote {OUT_CSV} ({len(df)} rows)')


if __name__ == '__main__':
    main()
