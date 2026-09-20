#!/usr/bin/env python3
"""orb_inplay step 1 — fetch the 09:30-09:34 ET tape for the WHOLE ADV>=1M universe.

cache.db's intraday_bars_1min covers only ~10% of that universe (it is the gap-up/mover
seed) — using it would make the population a look-ahead. We fetch the open tape directly.
"""
import os, sys, time, sqlite3
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pandas as pd
ROOT='/home/ec2-user/onemil'; os.chdir(ROOT); sys.path.insert(0,ROOT)
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed, Adjustment
from config import Config
ET=ZoneInfo('America/New_York'); D=f'{ROOT}/research/orb_inplay'
START='2024-12-01'; END='2026-05-31'   # TEST (>=2026-06-01) never touched

def universe_days(conn):
    """Per day: symbols with ADV20>=1M and prev close>=5, from daily_bars only (causal)."""
    d=pd.read_sql("select symbol,bar_date,open,high,low,close,volume from daily_bars "
                  "where bar_date>='2024-10-01' and bar_date<=?",conn,params=(END,))
    d=d.sort_values(['symbol','bar_date'])
    g=d.groupby('symbol',sort=False)
    d['adv20']=g.volume.transform(lambda s:s.rolling(20).mean().shift(1))
    d['prev_close']=g.close.transform(lambda s:s.shift(1))
    pc=g.close.shift(1)
    tr=pd.concat([d.high-d.low,(d.high-pc).abs(),(d.low-pc).abs()],axis=1).max(axis=1)
    d['atr14']=tr.groupby(d.symbol,sort=False).transform(lambda s:s.rolling(14).mean().shift(1))
    d=d[(d.bar_date>=START)&(d.adv20>=1e6)&(d.prev_close>=5.0)]
    return d[['symbol','bar_date','open','prev_close','adv20','atr14','volume']]

def main():
    conn=sqlite3.connect(f'file:{ROOT}/data/cache.db?mode=ro',uri=True)
    u=universe_days(conn); conn.close()
    u.to_parquet(f'{D}/universe.parquet',index=False)
    days=sorted(u.bar_date.unique())
    print(f'[fetch] {len(days)} days, {u.symbol.nunique()} distinct symbols, '
          f'median univ/day {int(u.groupby("bar_date").size().median())}',flush=True)
    cl=StockHistoricalDataClient(Config().alpaca_api_key,Config().alpaca_api_secret)
    OUT=f'{D}/open5.parquet'
    done=set()
    if os.path.exists(OUT):
        done=set(pd.read_parquet(OUT,columns=['bar_date']).bar_date.unique())
    parts=[]
    if os.path.exists(OUT): parts.append(pd.read_parquet(OUT))
    t0=time.time()
    for k,day in enumerate(days):
        if day in done: continue
        syms=sorted(u[u.bar_date==day].symbol.unique())
        s=datetime.strptime(day,'%Y-%m-%d').replace(hour=9,minute=30,tzinfo=ET)
        rows=[]
        for i in range(0,len(syms),400):
            ch=syms[i:i+400]
            for att in range(3):
                try:
                    r=cl.get_stock_bars(StockBarsRequest(symbol_or_symbols=ch,
                        timeframe=TimeFrame.Minute,start=s,end=s+timedelta(minutes=5),
                        feed=DataFeed.SIP,adjustment=Adjustment.RAW,limit=20000))
                    df=r.df
                    if len(df): rows.append(df.reset_index())
                    break
                except Exception as e:
                    if att==2: print(f'  ERR {day} chunk{i}: {str(e)[:80]}',flush=True)
                    time.sleep(2)
        if rows:
            df=pd.concat(rows,ignore_index=True)
            df['ts']=pd.to_datetime(df.timestamp,utc=True).dt.tz_convert(ET)
            df['m']=df.ts.dt.hour*60+df.ts.dt.minute
            df=df[(df.m>=570)&(df.m<=574)]
            df['bar_date']=day
            parts.append(df[['bar_date','symbol','m','open','high','low','close','volume']])
        if k%20==0:
            pd.concat(parts,ignore_index=True).to_parquet(OUT,index=False)
            print(f'  {k+1}/{len(days)} {day} elapsed {time.time()-t0:.0f}s',flush=True)
    pd.concat(parts,ignore_index=True).to_parquet(OUT,index=False)
    print(f'[fetch] DONE {time.time()-t0:.0f}s',flush=True)

if __name__=='__main__': sys.exit(main())
