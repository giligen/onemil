#!/usr/bin/env python3
"""Historical pre-trigger NEWS for every 2026 ignition BT trade (owner 9/13:
"rising stocks with a news accelerator should continue — there's gold you're
dismissing"). The BT never had the news leg. Alpaca news API, one batched
call per trading day, window prev-day 15:00 ET -> trade-day 10:35 ET; per
trade has_news_pre = any article on the symbol created BEFORE its trigger
minute. Resumable; output research/ignition_news/news_2026.csv"""
from dotenv import load_dotenv; load_dotenv('/home/ec2-user/onemil/.env')
import os, sys, json, time, requests, pandas as pd
ROOT='/home/ec2-user/onemil'; OUT=f'{ROOT}/research/ignition_news'
H={'APCA-API-KEY-ID':os.environ['ALPACA_API_KEY'],'APCA-API-SECRET-KEY':os.environ['ALPACA_API_SECRET']}
URL='https://data.alpaca.markets/v1beta1/news'
t=pd.read_csv(f'{ROOT}/research/ignition_capcheck/trades_NODOLLAR2026.csv')
days=sorted(t.day.unique()); state=f'{OUT}/state_2026.json'
done=json.load(open(state)) if os.path.exists(state) else {}
arts=[]
if os.path.exists(f'{OUT}/articles_2026.csv'):
    arts=pd.read_csv(f'{OUT}/articles_2026.csv').to_dict('records')
print(f'{len(t)} trades over {len(days)} days; done {len(done)}', flush=True)
for i,day in enumerate(days):
    if day in done: continue
    syms=sorted(t[t.day==day].symbol.unique()); d=pd.Timestamp(day)
    st=((d-pd.Timedelta(days=1)).tz_localize('America/New_York')+pd.Timedelta(hours=15)).tz_convert('UTC').isoformat()
    en=(d.tz_localize('America/New_York')+pd.Timedelta(hours=10,minutes=35)).tz_convert('UTC').isoformat()
    token=None; got=0
    for _ in range(8):
        p={'symbols':','.join(syms),'start':st,'end':en,'limit':50,'sort':'desc'}
        if token: p['page_token']=token
        r=None
        for att in range(3):
            try:
                r=requests.get(URL, headers=H, params=p, timeout=20)
                if r.status_code==429: time.sleep(3); continue
                r.raise_for_status(); break
            except Exception as e:
                time.sleep(2); r=None
        if r is None: break
        j=r.json()
        for a in j.get('news',[]):
            for s in a.get('symbols',[]):
                if s in syms: arts.append({'day':day,'symbol':s,'created_utc':a.get('created_at'),'headline':(a.get('headline') or '')[:120]}); got+=1
        token=j.get('next_page_token')
        if not token: break
    done[day]=got; json.dump(done, open(state,'w'))
    pd.DataFrame(arts).to_csv(f'{OUT}/articles_2026.csv', index=False)
    if i%10==0: print(f'{i+1}/{len(days)} {day}: {len(syms)} syms, {got} article-tags', flush=True)
    time.sleep(0.25)
A=pd.DataFrame(arts)
if len(A):
    A['created_utc']=pd.to_datetime(A.created_utc, utc=True)
    t['trig_utc']=[ (pd.Timestamp(d).tz_localize('America/New_York')+pd.Timedelta(minutes=int(m))).tz_convert('UTC') for d,m in zip(t.day,t.trig_m)]
    m=t.merge(A, on=['day','symbol'], how='left')
    pre=m[m.created_utc.notna() & (m.created_utc<=m.trig_utc)].groupby(['day','symbol']).size().rename('n_pre')
    out=t.merge(pre, left_on=['day','symbol'], right_index=True, how='left'); out['n_pre']=out.n_pre.fillna(0).astype(int); out['has_news_pre']=out.n_pre>0
    out.to_csv(f'{OUT}/news_2026.csv', index=False)
    print(f'DONE: {len(out)} trades, has_news_pre rate {out.has_news_pre.mean()*100:.1f}%', flush=True)
