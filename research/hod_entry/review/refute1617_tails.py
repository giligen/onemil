import pandas as pd, numpy as np
d=pd.read_csv('/home/ec2-user/onemil/research/hod_entry/cell_1617_nights.csv')
d=d[~d['flags'].str.contains('excluded')].copy()
d['net']=d.ret_bps-10
def dct(y,day):
    g=pd.DataFrame({'y':y-y.mean(),'d':day}).groupby('d').y.sum()
    se=np.sqrt((g**2).sum())/len(y)*np.sqrt(len(g)/(len(g)-1))
    return y.mean()/se
for cell in ['1617','1618_failed','1618_universe']:
  for sp in ['TRAIN','VAL']:
    s=d[(d.cell.astype(str)==cell)&(d.split==sp)]
    y=s.net.values
    q95,q99,q01,q05=np.percentile(y,[95,99,1,5])
    dm=s.groupby('date').net.mean()
    top2=dm.nlargest(2).index; bot2=dm.nsmallest(2).index
    print(cell,sp,'n',len(y),'mean %.2f t %.2f median %.2f'%(y.mean(),dct(s.net,s.date),np.median(y)),
      'ex-top5 %.2f ex-bot5 %.2f wins5/95 %.2f clip10%% %.2f'%(y[y<=q95].mean(),y[y>=q05].mean(),np.clip(y,q05,q95).mean(),np.clip(y,-1000,1000).mean()),
      'drop-best2nights %.2f drop-worst2 %.2f'%(s[~s.date.isin(top2)].net.mean(),s[~s.date.isin(bot2)].net.mean()),
      'daymean-mean %.2f'%dm.mean())
s=d[(d.cell.astype(str)=='1617')&(d.split=='VAL')]
print('VAL worst nights'); print(s.groupby('date').net.agg(['mean','count']).sort_values('mean').head(5))
print('VAL best nights'); print(s.groupby('date').net.agg(['mean','count']).sort_values('mean').tail(5))
print('VAL 20 worst rows'); print(s.nsmallest(12,'net')[['date','symbol','close','next_open','net']])
print('VAL 12 best rows'); print(s.nlargest(12,'net')[['date','symbol','close','next_open','net']])
