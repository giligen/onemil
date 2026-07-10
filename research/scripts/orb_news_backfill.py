"""Backfill pre-market news for every qualified ORB candidate (18mo book).
Window: prev calendar day 15:00 ET -> trade day 09:35 ET. One API call per
trading day (all symbols batched), paginated."""
from dotenv import load_dotenv; load_dotenv('/home/ec2-user/onemil/.env')
import os, sys, requests, pandas as pd, time

k=os.environ['ALPACA_API_KEY']; s=os.environ['ALPACA_API_SECRET']
H={'APCA-API-KEY-ID':k,'APCA-API-SECRET-KEY':s}
URL='https://data.alpaca.markets/v1beta1/news'

u=pd.read_csv('/tmp/orb_qualified_universe.csv')
days=sorted(u['day'].unique())
print(f"{len(u)} candidates over {len(days)} days", flush=True)

rows=[]
for i,day in enumerate(days):
    syms=sorted(u[u['day']==day]['symbol'].unique())
    d=pd.Timestamp(day)
    st=((d-pd.Timedelta(days=1)).tz_localize('America/New_York')+pd.Timedelta(hours=15)).tz_convert('UTC').isoformat()
    en=(d.tz_localize('America/New_York')+pd.Timedelta(hours=9,minutes=35)).tz_convert('UTC').isoformat()
    arts=[]; token=None
    for _page in range(6):
        p={'symbols':','.join(syms),'start':st,'end':en,'limit':50,'sort':'desc'}
        if token: p['page_token']=token
        for att in (1,2,3):
            try:
                r=requests.get(URL,params=p,headers=H,timeout=(5,30)); r.raise_for_status()
                break
            except Exception as e:
                if att==3: print(f"FAIL {day}: {e}",flush=True); r=None
                else: time.sleep(2)
        if r is None: break
        j=r.json(); arts+=j.get('news',[]); token=j.get('next_page_token')
        if not token: break
    per={sym:[] for sym in syms}
    for a in arts:
        for sym in a.get('symbols',[]):
            if sym in per: per[sym].append(a)
    for sym in syms:
        aa=per[sym]
        rows.append({'symbol':sym,'day':day,'n_articles':len(aa),
            'earliest':min((a['created_at'] for a in aa),default=''),
            'latest':max((a['created_at'] for a in aa),default=''),
            'headlines':' || '.join((a['headline'] or '')[:110] for a in aa[:4])})
    if (i+1)%25==0: print(f"  {i+1}/{len(days)} days",flush=True)

out=pd.DataFrame(rows)
out.to_csv('data/research/orb_news_catalyst_20260710.csv',index=False)
print(f"DONE: {len(out)} rows, has_news rate {(out['n_articles']>0).mean()*100:.1f}%",flush=True)
