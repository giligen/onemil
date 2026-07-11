"""Backfill news for ETF UNDERLYINGS (mapping study)."""
from dotenv import load_dotenv; load_dotenv('/home/ec2-user/onemil/.env')
import os, json, re, requests, pandas as pd
H={'APCA-API-KEY-ID':os.environ['ALPACA_API_KEY'],'APCA-API-SECRET-KEY':os.environ['ALPACA_API_SECRET']}
assets=json.load(open('/tmp/asset_names.json'))
df=pd.read_csv('/tmp/univ_classified.csv')

# refined extraction (REX/FANG etc. are fund-brand or basket tokens)
ETF_TOKENS=re.compile(r'\bETF\b|ProShares|Direxion|GraniteShares|T-?REX|Tradr|Defiance|Volatility Shares|Leverage Shares|Daily Target|\b2X\b|\b1\.5X\b|\bInverse\b|UltraShort|UltraPro|\bUltra\b|\bBull\b|\bBear\b',re.I)
STOP={'ETF','ETN','TRUST','DAILY','TARGET','SHORT','LONG','BULL','BEAR','INVERSE',
      'ULTRA','ULTRASHORT','ULTRAPRO','SHARES','FUND','II','III','2X','X','REX',
      'FANG','SPDR','NYSE','AMEX','VS','PLUS','INDEX','CAP'}
def und(sym):
    name=assets.get(sym,'')
    if not ETF_TOKENS.search(name): return None
    for tok in re.findall(r'\b[A-Z]{2,5}\b',name):
        if tok in STOP or tok==sym: continue
        if tok in assets and not ETF_TOKENS.search(assets.get(tok,'')):
            return tok
    return None
df['und']=df['symbol'].map(und)
etf=df[(df['cls']=='ETF')&df['und'].notna()][['symbol','und','day']].drop_duplicates()
print(f"{len(etf)} ETF candidate-days with underlying, {etf['und'].nunique()} distinct underlyings",flush=True)

rows=[]; days=sorted(etf['day'].unique())
for i,day in enumerate(days):
    g=etf[etf['day']==day]; syms=sorted(g['und'].unique())
    d=pd.Timestamp(day)
    st=((d-pd.Timedelta(days=1)).tz_localize('America/New_York')+pd.Timedelta(hours=15)).tz_convert('UTC').isoformat()
    en=(d.tz_localize('America/New_York')+pd.Timedelta(hours=9,minutes=35)).tz_convert('UTC').isoformat()
    arts=[]; token=None
    for _ in range(6):
        p={'symbols':','.join(syms),'start':st,'end':en,'limit':50,'sort':'desc'}
        if token: p['page_token']=token
        for att in range(3):
            try:
                r=requests.get('https://data.alpaca.markets/v1beta1/news',params=p,headers=H,timeout=(5,30)); r.raise_for_status(); break
            except Exception as e:
                if att==2: print(f"FAIL {day}: {e}",flush=True); r=None
                else: __import__('time').sleep(2)
        if r is None: break
        j=r.json(); arts+=j.get('news',[]); token=j.get('next_page_token')
        if not token: break
    per={s:0 for s in syms}
    for a in arts:
        for s in a.get('symbols',[]):
            if s in per: per[s]+=1
    for _,rr in g.iterrows():
        rows.append({'symbol':rr['symbol'],'day':day,'und':rr['und'],'und_articles':per.get(rr['und'],0)})
    if (i+1)%50==0: print(f"  {i+1}/{len(days)}",flush=True)
out=pd.DataFrame(rows)
out.to_csv('/tmp/und_news.csv',index=False)
print(f"DONE {len(out)} rows, und news rate {(out['und_articles']>0).mean()*100:.1f}%",flush=True)
