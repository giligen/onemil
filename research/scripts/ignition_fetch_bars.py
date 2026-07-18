"""Fetch full-day 1-min bars for a sample of flat-open market monsters."""
from dotenv import load_dotenv; load_dotenv('/home/ec2-user/onemil/.env')
import os, sys, time, requests, pandas as pd
sys.stdout.reconfigure(line_buffering=True)
H={'APCA-API-KEY-ID':os.environ['ALPACA_API_KEY'],'APCA-API-SECRET-KEY':os.environ['ALPACA_API_SECRET']}
F=pd.read_csv('/tmp/flat_open_monsters.csv')
# stratified sample: spread over time, cap 320
F=F.sort_values('bar_date')
samp=F.groupby(pd.to_datetime(F['bar_date']).dt.strftime('%Y-%m')).apply(
    lambda g: g.sample(min(len(g),17),random_state=7)).reset_index(drop=True)
print(f"sample: {len(samp)} monster-days",flush=True)
rows=[]
by_day=samp.groupby('bar_date')['symbol'].apply(list)
for i,(day,syms) in enumerate(by_day.items()):
    st=pd.Timestamp(day).tz_localize('America/New_York')+pd.Timedelta(hours=9,minutes=30)
    en=pd.Timestamp(day).tz_localize('America/New_York')+pd.Timedelta(hours=16)
    url='https://data.alpaca.markets/v2/stocks/bars'
    token=None
    for _ in range(20):
        p={'symbols':','.join(syms),'timeframe':'1Min','feed':'sip','limit':10000,
           'start':st.tz_convert('UTC').isoformat(),'end':en.tz_convert('UTC').isoformat()}
        if token: p['page_token']=token
        for att in range(3):
            try:
                r=requests.get(url,params=p,headers=H,timeout=(5,30)); r.raise_for_status(); break
            except Exception as e:
                if att==2: r=None; print(f"FAIL {day}: {e}",flush=True)
                else: time.sleep(2)
        if r is None: break
        j=r.json()
        for sym,bars in (j.get('bars') or {}).items():
            for b in bars:
                rows.append({'symbol':sym,'day':day,'t':b['t'],'o':b['o'],'h':b['h'],'l':b['l'],'c':b['c'],'v':b['v']})
        token=j.get('next_page_token')
        if not token: break
    if (i+1)%25==0: print(f"  {i+1}/{len(by_day)} days, {len(rows)} bars",flush=True)
pd.DataFrame(rows).to_csv('/tmp/ignition_bars.csv',index=False)
print(f"DONE {len(rows)} bars",flush=True)
