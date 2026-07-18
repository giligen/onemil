"""Theme/underlying co-movement features (2026-07-18 oracle hunt).
For each candidate-day: how many OTHER candidates share its underlying /
direction / theme-sector? Point-in-time (built purely from the day's
cohort). Test: monster separation + era consistency."""
import pandas as pd, numpy as np, json, re, sys
sys.path.insert(0,'/home/ec2-user/onemil')
names=json.load(open('/tmp/asset_names.json'))
from trading.orb_asset_class import WRAPPER_RE

STOP={'ETF','ETN','TRUST','DAILY','TARGET','SHORT','LONG','BULL','BEAR','INVERSE',
      'ULTRA','ULTRASHORT','ULTRAPRO','SHARES','FUND','II','III','2X','X','REX',
      'FANG','SPDR','NYSE','AMEX','VS','PLUS','INDEX','CAP'}
def und(sym):
    nm=names.get(sym,'') or ''
    if not WRAPPER_RE.search(nm): return None
    for tok in re.findall(r'\b[A-Z]{2,5}\b',nm):
        if tok in STOP or tok==sym: continue
        if tok in names and not WRAPPER_RE.search(names.get(tok,'') or ''):
            return tok
    return None
def direction(sym):
    nm=(names.get(sym,'') or '').lower()
    if any(w in nm for w in ('short','inverse','bear')): return 'short'
    if WRAPPER_RE.search(names.get(sym,'') or ''): return 'long'
    return 'stock'

# curated theme map for the wrapper-complex underlyings (deliberate, name-level)
THEME={
 'quantum':['RGTI','QUBT','IONQ','QBTS','IQM','ARQQ','QSI'],
 'space':['RKLB','ASTS','LUNR','RDW','PL'],
 'ai_infra':['NBIS','CRWV','IREN','APLD','WULF','CORZ','CIFR','HUT','BTDR','AAOI','CLSK'],
 'crypto':['MSTR','COIN','MARA','RIOT','BMNR','HOOD','BTBT','GLXY','CRCL','BITF','SBET'],
 'nuclear':['OKLO','SMR','NNE','LEU','LTBR'],
 'ev_solar':['TSLA','ENPH','PLUG','FCEL','BE','EOSE','SLDP'],
 'meme_tech':['PLTR','META','AMD','NVDA','SMCI','DJT','GME','AMC','UBER','AVGO','QCOM'],
}
U2T={u:t for t,us in THEME.items() for u in us}

df=pd.read_csv('/tmp/orb_cands_liveparity.csv')
df['date']=pd.to_datetime(df['date']); df['day']=df['date'].dt.strftime('%Y-%m-%d')
df['und']=df['symbol'].map(und)
df['dir']=df['symbol'].map(direction)
df['theme']=df['und'].map(U2T)
df.loc[df['dir']=='stock','theme']=df.loc[df['dir']=='stock','symbol'].map(U2T)

# cohort features (point-in-time: the day's own candidate list)
def cohort_feats(g):
    g=g.copy()
    uc=g['und'].value_counts()
    g['und_cohort']=g['und'].map(uc).fillna(1)          # same-underlying count (incl self)
    td=g.groupby(['theme','dir']).size()
    g['themedir_cohort']=[td.get((t,d),1) if pd.notna(t) else 1
                          for t,d in zip(g['theme'],g['dir'])]
    tc=g['theme'].value_counts()
    g['theme_cohort']=g['theme'].map(tc).fillna(1)
    g['short_frac_day']=(g['dir']=='short').mean()
    return g
df=df.groupby('day',group_keys=False).apply(cohort_feats)
df['era3']=np.where(df['day']<'2025-07-01','25H1',np.where(df['day']<'2026-01-01','25H2','2026'))
df['monster']=df['_rp_pnl']>=2000

print("=== und_cohort (same-underlying wrappers qualifying together) ===")
df['uc_b']=pd.cut(df['und_cohort'],[0,1,2,10],labels=['1','2','3+'])
for era,g in df.groupby('era3'):
    r=g.groupby('uc_b',observed=True)['_rp_pnl'].agg(['size','mean'])
    m=g.groupby('uc_b',observed=True)['monster'].mean()*100
    print(f"  {era}: "+"  ".join(f"{i}: n={int(r['size'][i])} ${r['mean'][i]:+.0f} M{m[i]:.0f}%" for i in r.index))
print("\n=== theme_cohort (same THEME candidates, any direction/form) ===")
df['tc_b']=pd.cut(df['theme_cohort'],[0,1,3,30],labels=['1','2-3','4+'])
for era,g in df.groupby('era3'):
    r=g.groupby('tc_b',observed=True)['_rp_pnl'].agg(['size','mean'])
    m=g.groupby('tc_b',observed=True)['monster'].mean()*100
    print(f"  {era}: "+"  ".join(f"{i}: n={int(r['size'][i])} ${r['mean'][i]:+.0f} M{m[i]:.0f}%" for i in r.index))
print("\n=== themedir_cohort (same theme AND direction — the 7/16 pattern) ===")
df['tdc_b']=pd.cut(df['themedir_cohort'],[0,1,2,30],labels=['1','2','3+'])
for era,g in df.groupby('era3'):
    r=g.groupby('tdc_b',observed=True)['_rp_pnl'].agg(['size','mean'])
    m=g.groupby('tdc_b',observed=True)['monster'].mean()*100
    print(f"  {era}: "+"  ".join(f"{i}: n={int(r['size'][i])} ${r['mean'][i]:+.0f} M{m[i]:.0f}%" for i in r.index))
df.to_csv('/tmp/theme_feats.csv',index=False)
print("\nJuly winners' themedir_cohort:", df[(df['day']>='2026-07-01')&(df['_rp_pnl']>0)]['themedir_cohort'].value_counts().to_dict())
print("July losers' themedir_cohort:", df[(df['day']>='2026-07-01')&(df['_rp_pnl']<=0)]['themedir_cohort'].value_counts().to_dict())
