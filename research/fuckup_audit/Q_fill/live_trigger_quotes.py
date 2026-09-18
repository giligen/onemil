#!/usr/bin/env python3
"""Stage Q step 4c — the NBBO at the TRIGGER instant of every live ORB order,
the direct ground truth for the stop-limit cap question.

For each live ORB order with a locatable market breakout bar, pull the Alpaca
SIP trades/quotes of that minute, find the first print above range_high (the
instant the stop elects) and read the NBBO there.  Report the ask as bps over
range_high -- the quantity `entry.stop_limit_buffer_bps` (30) has to cover.
"""
import json, os, sqlite3, sys, time
from datetime import timedelta
import numpy as np, pandas as pd
from dotenv import load_dotenv
ROOT='/home/ec2-user/onemil'; os.chdir(ROOT); sys.path.insert(0,ROOT); load_dotenv(f'{ROOT}/.env')
from config import Config
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockQuotesRequest, StockTradesRequest
from alpaca.data.enums import DataFeed
Q=f'{ROOT}/research/fuckup_audit/Q_fill'
t=sqlite3.connect('file:data/trades.db?mode=ro',uri=True)
d=pd.read_sql("SELECT trade_date,symbol,order_status,entry_price,fill_price,"
              "order_filled_at,pattern_data FROM trades WHERE strategy='orb'",t); t.close()
def pget(js,k):
    try: p=json.loads(js) if js else {}
    except Exception: return np.nan
    try: return float(p.get(k))
    except Exception: return np.nan
d['range_high']=[pget(x,'range_high') for x in d.pattern_data]
b=sqlite3.connect('file:data/cache.db?mode=ro',uri=True)
cfg=Config(); cl=StockHistoricalDataClient(cfg.alpaca_api_key,cfg.alpaca_api_secret)
rows=[]
for r in d.itertuples():
    if not np.isfinite(r.range_high): continue
    bars=pd.DataFrame(b.execute("SELECT timestamp,high FROM intraday_bars_1min "
        "WHERE symbol=? AND bar_date=? ORDER BY timestamp",(r.symbol,r.trade_date)).fetchall(),
        columns=['timestamp','high'])
    if bars.empty: continue
    ts=pd.to_datetime(bars.timestamp,utc=True,format='mixed')
    et=ts.dt.tz_convert('America/New_York'); m=et.dt.hour*60+et.dt.minute
    sel=(m>=575)&(m<635)&(bars.high>r.range_high)
    if not sel.any(): continue
    brk=pd.to_datetime(bars.timestamp[sel].iloc[0],utc=True,format='mixed').floor('min')
    rec=dict(trade_date=r.trade_date,symbol=r.symbol,status=r.order_status,
             range_high=r.range_high,cap30=r.entry_price,fill_price=r.fill_price,
             ask=np.nan,bid=np.nan,ask_over_rh_bps=np.nan)
    try:
        tr=cl.get_stock_trades(StockTradesRequest(symbol_or_symbols=r.symbol,start=brk,
            end=brk+timedelta(minutes=1),feed=DataFeed.SIP,limit=6000))
        tl=tr.data.get(r.symbol,[]) if hasattr(tr,'data') else tr.get(r.symbol,[])
        fire=next((pd.Timestamp(x.timestamp) for x in tl if float(x.price)>r.range_high),None)
        if fire is not None:
            qq=cl.get_stock_quotes(StockQuotesRequest(symbol_or_symbols=r.symbol,start=brk,
                end=brk+timedelta(minutes=1),feed=DataFeed.SIP,limit=6000))
            ql=qq.data.get(r.symbol,[]) if hasattr(qq,'data') else qq.get(r.symbol,[])
            prev=None
            for x in ql:
                if pd.Timestamp(x.timestamp)<=fire and float(x.bid_price)>0 and float(x.ask_price)>0:
                    prev=(float(x.bid_price),float(x.ask_price))
                elif pd.Timestamp(x.timestamp)>fire: break
            if prev: rec.update(bid=prev[0],ask=prev[1],
                                ask_over_rh_bps=(prev[1]/r.range_high-1)*1e4)
    except Exception as ex: rec['err']=type(ex).__name__
    rows.append(rec); time.sleep(0.05)
b.close()
o=pd.DataFrame(rows); o.to_csv(f'{Q}/live_trigger_quotes.csv',index=False)
v=o.ask_over_rh_bps.dropna()
print(f'live orders with a trigger-instant NBBO: {len(v)} of {len(o)}')
print(f'ask over range_high (bps): median {v.median():.1f} mean {v.mean():.1f} '
      f'p75 {v.quantile(.75):.0f} p90 {v.quantile(.9):.0f}')
for c in (30,50,100,300):
    print(f'  ask within {c} bps of range_high: {(v<=c).mean()*100:.1f}%')
fil=o[o.status.isin(['closed','filled'])].ask_over_rh_bps.dropna()
unf=o[o.status=='time_stop_canceled'].ask_over_rh_bps.dropna()
print(f'FILLED orders: n={len(fil)} within 30bps {(fil<=30).mean()*100:.1f}% | '
      f'within 50 {(fil<=50).mean()*100:.1f}%')
print(f'TIME-STOPPED orders with a trigger print: n={len(unf)}; '
      f'{unf.round(0).tolist()}')
