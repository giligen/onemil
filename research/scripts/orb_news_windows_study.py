"""Per-window news backfill: A=prev session (prev 04:00-15:00 ET),
B=overnight (prev 15:00 -> today 04:00), C=fresh premarket (today 04:00-09:35).
For all qualified candidates (own ticker) AND wrapper underlyings."""
from dotenv import load_dotenv; load_dotenv('/home/ec2-user/onemil/.env')
import os, sys, time, requests, pandas as pd
H={'APCA-API-KEY-ID':os.environ['ALPACA_API_KEY'],'APCA-API-SECRET-KEY':os.environ['ALPACA_API_SECRET']}
URL='https://data.alpaca.markets/v1beta1/news'

df=pd.read_csv('/tmp/univ_classified.csv')
un=pd.read_csv('/tmp/und_news.csv')[['symbol','day','und']]
df=df.merge(un,on=['symbol','day'],how='left',suffixes=('','_u'))
# query set per day: own tickers + underlyings
tasks={}
for _,r in df.iterrows():
    tasks.setdefault(r['day'],set()).add(r['symbol'])
    if isinstance(r.get('und'),str): tasks[r['day']].add(r['und'])
print(f"{len(df)} candidates, {len(tasks)} days",flush=True)

def et(ts): return pd.Timestamp(ts).tz_convert('America/New_York')
rows=[]
for i,(day,syms) in enumerate(sorted(tasks.items())):
    d=pd.Timestamp(day).tz_localize('America/New_York')
    prev=d-pd.Timedelta(days=1)
    wA0,wA1=prev+pd.Timedelta(hours=4),prev+pd.Timedelta(hours=15)
    wB1=d+pd.Timedelta(hours=4)
    wC1=d+pd.Timedelta(hours=9,minutes=35)
    arts=[]; token=None
    for _p in range(10):
        p={'symbols':','.join(sorted(syms)),'start':wA0.tz_convert('UTC').isoformat(),
           'end':wC1.tz_convert('UTC').isoformat(),'limit':50,'sort':'desc'}
        if token: p['page_token']=token
        for att in range(3):
            try:
                r=requests.get(URL,params=p,headers=H,timeout=(5,30)); r.raise_for_status(); break
            except Exception as e:
                if att==2: print(f"FAIL {day}: {e}",flush=True); r=None
                else: time.sleep(2)
        if r is None: break
        j=r.json(); arts+=j.get('news',[]); token=j.get('next_page_token')
        if not token: break
    per={s:[0,0,0] for s in syms}   # A,B,C counts
    for a in arts:
        t=et(a['created_at'])
        b=0 if t<=wA1 else (1 if t<=wB1 else 2)
        for s in a.get('symbols',[]):
            if s in per: per[s][b]+=1
    for s in syms:
        rows.append({'q':s,'day':day,'nA':per[s][0],'nB':per[s][1],'nC':per[s][2]})
    if (i+1)%25==0: print(f"  {i+1}/{len(tasks)} days",flush=True)
pd.DataFrame(rows).to_csv('/tmp/news_windows.csv',index=False)
print("DONE",flush=True)
