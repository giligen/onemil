#!/usr/bin/env python3
"""What is inside EQUS.MINI ohlcv-1m: publisher ids on the records vs Databento's publisher table, and the spec's
ADV20 source (EQUS.SUMMARY daily parquet) vs Alpaca daily_bars volume for the same day — i.e. is the study's
`rv_profile` numerator (EQUS.MINI minute volume) on the same scale as its denominator (EQUS.SUMMARY ADV20)?
Read-only; one tiny Databento request (3 symbols, 1 day)."""
import os, sys, sqlite3
import pandas as pd
ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT)
from dotenv import load_dotenv; load_dotenv(f'{ROOT}/.env')
import databento as dbn
client = dbn.Historical(os.environ['DATABENTO_API_KEY'])
syms = ['A', 'AAPL', 'AAOI']; DAY = '2026-09-11'
st = client.timeseries.get_range(dataset='EQUS.MINI', schema='ohlcv-1m', symbols=syms, stype_in='raw_symbol', start=DAY, end='2026-09-12')
df = st.to_df().reset_index()
print('ohlcv-1m columns:', list(df.columns))
print('publisher_id values on EQUS.MINI ohlcv-1m records:', sorted(df.publisher_id.unique().tolist()))
pubs = pd.DataFrame(client.metadata.list_publishers())
print('publishers for EQUS.MINI:\n', pubs[pubs.dataset == 'EQUS.MINI'].to_string(index=False))
ts = pd.to_datetime(df.ts_event, utc=True).dt.tz_convert('America/New_York'); m = ts.dt.hour * 60 + ts.dt.minute
rth = df[(m >= 570) & (m < 960)]
mini_vol = rth.groupby('symbol').volume.sum()
daily = pd.read_parquet(f'{ROOT}/data/research/databento/equs_daily_2025_2026.parquet', columns=['symbol', 'bar_date', 'volume'])
daily['bar_date'] = daily.bar_date.astype(str).str[:10]
summ = daily[(daily.symbol.isin(syms)) & (daily.bar_date == DAY)].set_index('symbol').volume
con = sqlite3.connect(f'file:{ROOT}/data/cache.db?mode=ro', uri=True, timeout=5)
alp = dict(con.execute(f"select symbol, volume from daily_bars where bar_date='{DAY}' and symbol in ('A','AAPL','AAOI')").fetchall())
print(f'\n{DAY} volume: EQUS.MINI ohlcv-1m RTH sum | EQUS.SUMMARY daily (the spec ADV20 source) | Alpaca daily_bars (live ADV20 source)')
for s in syms:
    print(f'  {s}: {int(mini_vol.get(s, 0)):>12,} | {int(summ.get(s, 0)):>12,} | {int(alp.get(s, 0)):>12,}  -> MINI/SUMMARY = {mini_vol.get(s, 0) / max(1, summ.get(s, 0)):.1%}')
# spec book: rv by source (from the census output)
p = f'{ROOT}/research/bf_zero/parity_review/spec_book_sources.csv'
if os.path.exists(p):
    S = pd.read_csv(p)
    print('\nspec book rv_profile by source (median / p90) and mean R:\n', S.groupby('source').agg(n=('rr', 'size'), rv_med=('rv', 'median'), rv_p90=('rv', lambda x: x.quantile(.9)), meanR=('rr', 'mean')).round(2).to_string())
