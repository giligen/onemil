from dotenv import load_dotenv; load_dotenv('/home/ec2-user/onemil/.env')
import os, sys, time, requests, pandas as pd
sys.stdout.reconfigure(line_buffering=True)
H={'APCA-API-KEY-ID':os.environ['ALPACA_API_KEY'],'APCA-API-SECRET-KEY':os.environ['ALPACA_API_SECRET']}
t=pd.read_csv('/tmp/ignition_g2.csv')
t=t[(t['trig_min']<=630)&(t['R_pct']>=5)]      # the G3 book candidates
pairs=t[['symbol','day']].drop_duplicates()
print(f"{len(pairs)} pairs",flush=True)
rows=[]
for i,(day,g) in enumerate(pairs.groupby('day')):
    syms=sorted(g['symbol'].unique())
    d=pd.Timestamp(day)
    st=((d-pd.Timedelta(days=1)).tz_localize('America/New_York')+pd.Timedelta(hours=15)).tz_convert('UTC').isoformat()
    en=(d.tz_localize('America/New_York')+pd.Timedelta(hours=10,minutes=30)).tz_convert('UTC').isoformat()
    arts=[]; token=None
    for _ in range(6):
        p={'symbols':','.join(syms),'start':st,'end':en,'limit':50,'sort':'desc'}
        if token: p['page_token']=token
        for att in range(3):
            try:
                r=requests.get('https://data.alpaca.markets/v1beta1/news',params=p,headers=H,timeout=(5,30)); r.raise_for_status(); break
            except Exception as e:
                if att==2: r=None
                else: time.sleep(2)
        if r is None: break
        j=r.json(); arts+=j.get('news',[]); token=j.get('next_page_token')
        if not token: break
    per={s:0 for s in syms}
    for a in arts:
        for s in a.get('symbols',[]):
            if s in per: per[s]+=1
    for s in syms: rows.append({'symbol':s,'day':day,'n_articles':per[s]})
    if (i+1)%60==0: print(f"  {i+1} days",flush=True)
pd.DataFrame(rows).to_csv('/tmp/ignition_news.csv',index=False)
print("DONE",flush=True)
