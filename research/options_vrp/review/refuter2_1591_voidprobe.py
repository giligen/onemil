"""Refuter 2: does the 69 % VOID set select outcomes? Settle EVERY Monday's 30-delta (and 20-delta) $10 spread at
intrinsic (Management B) using the rebuild's strikes for all Mondays and SPY's close on expiry; credit imputed at
the builder's median credit for that delta (report-only probe, not a P&L claim)."""
import pandas as pd, numpy as np, math
D='/home/ec2-user/onemil/research/options_vrp/'
spy=pd.read_parquet(D+'opt_cache/spy_daily.parquet'); print(spy.columns.tolist()[:8])
spy['day']=pd.to_datetime(spy['day']).dt.tz_localize(None).dt.normalize() if pd.to_datetime(spy['day']).dt.tz is not None else pd.to_datetime(spy['day']).dt.normalize()
closes=spy.set_index('day')['c']
rb=pd.read_csv(D+'rebuild_1591_cycles.csv',parse_dates=['monday','expiry'])
bl=pd.read_csv(D+'cell_1591_cycles.csv',parse_dates=['entry_date','expiry'])
def strike(sym): return int(sym[-8:])/1000.0
for cell,delta in [(1597,0.3),(1593,0.2)]:
    r=rb[rb.cell==cell].dropna(subset=['short_symbol']).copy()
    r['ks']=r.short_symbol.map(strike); r['kl']=r.long_symbol.map(strike)
    med=bl[bl.cell==cell].credit.median()
    def settle(row):
        px=closes.get(row.expiry.normalize(), np.nan)
        if np.isnan(px):
            prior=closes[closes.index<=row.expiry.normalize()]; px=prior.iloc[-1] if len(prior) else np.nan
        loss=min(max(row.ks-px,0),row.ks-row.kl); return (med-loss)*100-0.06
    r['imp_pnl']=r.apply(settle,axis=1)
    traded=set(bl[bl.cell==cell].entry_date)
    r['builder_traded']=r.monday.isin(traded)
    r['split']=np.where(r.monday<=pd.Timestamp('2025-06-30'),'TRAIN','VAL')
    r=r[r.monday<=pd.Timestamp('2026-08-17')]
    g=r.groupby(['split','builder_traded']).imp_pnl.agg(['count','mean','min',lambda s:(s<0).mean()])
    print(f'\ncell {cell} (median credit {med:.2f}) imputed intrinsic P&L per cycle, traded vs NOT traded by builder:'); print(g.round(2))
    losers=r[r.imp_pnl<0][['monday','expiry','ks','imp_pnl','builder_traded','split']]
    print('VAL losers:'); print(losers[losers.split=='VAL'].to_string())

# ---- full-ladder (VOID-free) imputed monthly book, 1597, and SPY per-unit-of-DD comparison
B=6500.0
r=rb[rb.cell==1597].dropna(subset=['short_symbol']).copy()
r['ks']=r.short_symbol.map(strike); r['kl']=r.long_symbol.map(strike); med=bl[bl.cell==1597].credit.median()
def px_at(d):
    p=closes[closes.index<=d.normalize()]; return p.iloc[-1]
r['imp_pnl']=[(med-min(max(k-px_at(e),0),k-l))*100-0.06 for k,l,e in zip(r.ks,r.kl,r.expiry)]
r=r[r.monday<=pd.Timestamp('2026-08-17')]
# budget check on the full ladder: open worst cases at each Monday
wc=(10-med)*100
mx=max(((r.monday<=m)&(r.expiry>m)).sum() for m in r.monday); print(f'\nfull ladder max open {mx} -> worst case ${mx*wc:,.0f} vs B {B:,.0f}')
for s,(a,b) in {'TRAIN':('2024-02-05','2025-06-30'),'VAL':('2025-07-07','2026-08-17'),'ALL':('2024-02-05','2026-08-17')}.items():
    d=r[(r.monday>=a)&(r.monday<=b)]
    mo=d.groupby(d.expiry.dt.to_period('M')).imp_pnl.sum()
    mo=mo.reindex(pd.period_range(mo.index.min(),mo.index.max(),freq='M'),fill_value=0.0)
    cum=mo.cumsum(); dd=-(cum-cum.cummax()).min()
    print(f'{s}: n {len(d)} months {len(mo)} mean ${mo.mean():,.0f}/mo ({mo.mean()/B:.2%} of B) sharpe {mo.mean()/mo.std()*math.sqrt(12):.2f} '
          f'green {(mo>0).mean():.0%} worst month ${mo.min():,.0f} maxDD ${dd:,.0f} worst2 {mo.nsmallest(2).round(0).to_dict()}')
    if s=='ALL':
        print(mo.round(0).to_string())
for a,b in [('2025-07-07','2026-09-18'),('2024-02-05','2026-09-18')]:
    p=closes[(closes.index>=a)&(closes.index<=b)]; ddp=(p/p.cummax()-1).min()
    print(f'SPY {a}..{b}: ret {p.iloc[-1]/p.iloc[0]-1:.2%} maxDD {ddp:.2%} ret/DD {(p.iloc[-1]/p.iloc[0]-1)/-ddp:.2f}  $ on B {B*(p.iloc[-1]/p.iloc[0]-1):,.0f}')
