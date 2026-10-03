import pandas as pd
o = pd.read_parquet('research/known_strategies/1703g_btc_ohlc.parquet')
if 'date' in o.columns: o = o.set_index('date')
o.index = pd.to_datetime(o.index).tz_localize(None); px = o.close.sort_index()
r = px.pct_change()
def rule(cond):
    pos = cond.shift(1).astype(float); return pos * r - 0.001 * pos.diff().abs().fillna(0)
m20 = rule(px > px.shift(20)); s100 = rule(px > px.rolling(100).mean()); hold = r
w = pd.read_csv('research/known_strategies/1703i_worst_weeks.csv', parse_dates=['date'])
rows = []
for d, sr in zip(w.date, w.sleeve_ret):
    a = d - pd.Timedelta(days=6)
    rows.append(dict(week=d.date(), sleeve=round(100*sr,1), mom20=round(100*((1+m20[a:d]).prod()-1),1), sma100=round(100*((1+s100[a:d]).prod()-1),1), btc_hold=round(100*((1+hold[a:d]).prod()-1),1)))
t = pd.DataFrame(rows); print(t.to_string(index=False)); print('mean', t[['mom20','sma100','btc_hold']].mean().round(2).to_dict())
