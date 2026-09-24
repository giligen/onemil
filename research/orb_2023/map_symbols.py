"""Map XNAS.ITCH ohlcv-1d rows (instrument_id) to tickers, month by month via Databento symbology.resolve.

The ALL_SYMBOLS download carries no embedded symbology, and one resolve over 18 months times out (504), so this
resolves each calendar month (raw_symbol -> instrument_id intervals), joins every daily row on (instrument_id,
date in [d0, d1)), and writes data/research/databento/xnas_daily_2023_2024H1.parquet with a symbol column.

Usage: python3 research/orb_2023/map_symbols.py
"""
import os
import time
from pathlib import Path

import databento as db
import pandas as pd
from dotenv import load_dotenv

ROOT = Path('/home/ec2-user/onemil')
DBN = ROOT / 'data/research/databento/xnas_itch_ohlcv1d_ALL_20230101_20240630.dbn.zst'
OUT = ROOT / 'data/research/databento/xnas_daily_2023_2024H1.parquet'


def resolve_month(client, start: str, end: str) -> pd.DataFrame:
    """raw_symbol -> instrument_id intervals for [start, end), with retries."""
    for attempt in range(4):
        try:
            r = client.symbology.resolve(dataset='XNAS.ITCH', symbols='ALL_SYMBOLS', stype_in='raw_symbol',
                                         stype_out='instrument_id', start_date=start, end_date=end)
            rows = [(sym, iv['d0'], iv['d1'], int(iv['s'])) for sym, ivs in r['result'].items() for iv in ivs]
            return pd.DataFrame(rows, columns=['symbol', 'd0', 'd1', 'instrument_id'])
        except Exception as e:  # noqa: BLE001
            print(f'[WARNING] resolve {start}..{end} attempt {attempt + 1}: {str(e)[:120]}', flush=True)
            time.sleep(10 * (attempt + 1))
    raise SystemExit(f'[ERROR] symbology resolve failed for {start}..{end}')


def main():
    load_dotenv(ROOT / '.env')
    client = db.Historical(os.environ['DATABENTO_API_KEY'])
    df = db.DBNStore.from_file(DBN).to_df(map_symbols=False).reset_index()
    df['bar_date'] = pd.to_datetime(df['ts_event']).dt.strftime('%Y-%m-%d')
    months = pd.period_range('2023-01', '2024-06', freq='M')
    maps = []
    for p in months:
        a, b = p.start_time.strftime('%Y-%m-%d'), (p + 1).start_time.strftime('%Y-%m-%d')
        m = resolve_month(client, a, b)
        maps.append(m)
        print(f'{p}: {len(m):,} intervals, {m.symbol.nunique():,} symbols', flush=True)
    mp = pd.concat(maps, ignore_index=True).drop_duplicates()
    mp.to_parquet(ROOT / 'research/orb_2023/symbology_intervals.parquet', index=False)
    # Expand each [d0, d1) interval onto the trading days present in the data, then join exactly on
    # (bar_date, instrument_id). A join on instrument_id alone multiplies rows by every interval an id ever had.
    days = pd.Series(sorted(df.bar_date.unique()))
    mp['i0'] = days.searchsorted(mp.d0)
    mp['i1'] = days.searchsorted(mp.d1)
    mp = mp[mp.i1 > mp.i0]
    rep = mp.loc[mp.index.repeat(mp.i1 - mp.i0)].copy()
    rep['k'] = rep.groupby(level=0).cumcount()
    rep['bar_date'] = days.values[(rep.i0 + rep.k).values]
    key = rep[['bar_date', 'instrument_id', 'symbol']].drop_duplicates(['bar_date', 'instrument_id'])
    j = df.merge(key, on=['bar_date', 'instrument_id'], how='inner')
    out = j[['bar_date', 'symbol', 'instrument_id', 'open', 'high', 'low', 'close', 'volume']]
    out.to_parquet(OUT, index=False)
    print(f'rows {len(df):,} -> mapped {len(out):,} ({len(out) / len(df):.1%}), symbols {out.symbol.nunique():,}, '
          f'days {out.bar_date.nunique()}', flush=True)
    chk = out[out.symbol.isin(['AAPL', 'GME', 'SMCI'])].groupby('symbol').bar_date.agg(['min', 'max', 'count'])
    print(chk.to_string(), flush=True)


if __name__ == '__main__':
    main()
