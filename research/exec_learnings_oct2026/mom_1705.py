"""Cell 1,705: MOM sleeve entry timing. Read-only. Writes mom_1705.csv and prints per-rotation summary."""
import os, sys, glob
sys.path.insert(0, '/home/ec2-user/onemil')
import pandas as pd, numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
load_dotenv('/home/ec2-user/onemil/.env')
from data_sources.alpaca_client import AlpacaClient
ET = ZoneInfo('America/New_York')
paper = pd.read_csv('logs/momentum_sleeve_ledger.csv'); paper['book'] = 'paper'
live = pd.read_csv('logs/momentum_sleeve_live_ledger.csv'); live['book'] = 'live'
led = pd.concat([paper, live], ignore_index=True)
panel = pd.read_parquet(sorted(glob.glob('data/momentum_sleeve/daily_*.parquet'))[-1])
panel['bar_date'] = pd.to_datetime(panel['bar_date']).dt.tz_localize(None).dt.normalize()
def prior_close(sym, day):
    s = panel[(panel.symbol == sym) & (panel.bar_date < pd.Timestamp(day))]
    return (float(s.iloc[-1].close), s.iloc[-1].bar_date.date()) if len(s) else (np.nan, None)
led[['prior_close', 'prior_date']] = led.apply(lambda r: pd.Series(prior_close(r.symbol, r.date)), axis=1)
led['gap_bps'] = (led.avg_price - led.prior_close) / led.prior_close * 1e4
# minute bars for buys 10/5.. (not 10/2: the 51-name initial build)
key, sec = os.environ['ALPACA_MOM_API_KEY'], os.environ['ALPACA_MOM_API_SECRET']
cl = AlpacaClient(key, sec, paper=True)
bar_open = {}
for day, g in led[(led.side == 'buy') & (led.date >= '2026-10-05')].groupby('date'):
    syms = sorted(g.symbol.unique())
    d = pd.Timestamp(day)
    st = datetime(d.year, d.month, d.day, 9, 44, tzinfo=ET); en = datetime(d.year, d.month, d.day, 9, 52, tzinfo=ET)
    res = cl.get_1min_bars_range_multi(syms, st, en)
    for s, df in res.items():
        if len(df) == 0: print('NO BARS', day, s); continue
        df['t'] = pd.to_datetime(df.timestamp, utc=True).dt.tz_convert(ET).dt.strftime('%H:%M')
        for hhmm in ('09:45', '09:50'):
            r = df[df.t == hhmm]
            bar_open[(day, s, hhmm)] = float(r.iloc[0].open) if len(r) else np.nan
def ref(r):
    hh = '09:50' if r.book == 'live' else '09:45'
    return bar_open.get((r.date, r.symbol, hh), np.nan)
led['bar_open'] = led.apply(ref, axis=1)
led['slip_vs_bar_bps'] = (led.avg_price - led.bar_open) / led.bar_open * 1e4
led.to_csv('research/exec_learnings_oct2026/mom_1705.csv', index=False)
b = led[led.side == 'buy']
def agg(g):
    w = g.notional
    ok = g.dropna(subset=['gap_bps']); ok2 = g.dropna(subset=['slip_vs_bar_bps']); ok3 = g.dropna(subset=['slip_bps_vs_open'])
    return pd.Series({'n_buys': len(g), 'buy_$': w.sum(),
        'gap_bps_w': np.average(ok.gap_bps, weights=ok.notional), 'gap_bps_med': ok.gap_bps.median(), 'gap_abs_w': np.average(ok.gap_bps.abs(), weights=ok.notional),
        'slip_bar_w': np.average(ok2.slip_vs_bar_bps, weights=ok2.notional) if len(ok2) else np.nan, 'n_bar': len(ok2),
        'slip_dailyopen_w': np.average(ok3.slip_bps_vs_open, weights=ok3.notional) if len(ok3) else np.nan})
s = b.groupby(['date', 'book']).apply(agg)
s['turnover_$'] = led.groupby(['date', 'book']).notional.sum()
pd.set_option('display.width', 250); print(s.round(1).to_string())
print('nan prior', led.prior_close.isna().sum(), 'max prior_date gap check:'); print(led.groupby('date').prior_date.agg(['min', 'max']))
# paper vs live 10/19 same names
p = b[(b.date == '2026-10-19') & (b.book == 'paper')].set_index('symbol'); l = b[(b.date == '2026-10-19') & (b.book == 'live')].set_index('symbol')
c = p.index.intersection(l.index); print('common', list(c))
if len(c):
    d = pd.DataFrame({'paper_fill': p.loc[c].avg_price, 'live_fill': l.loc[c].avg_price, 'p_bar45': p.loc[c].bar_open, 'l_bar50': l.loc[c].bar_open})
    d['live_minus_paper_bps'] = (d.live_fill - d.paper_fill) / d.paper_fill * 1e4
    print(d.round(3).to_string()); print('mean', d.live_minus_paper_bps.mean(), 'median', d.live_minus_paper_bps.median())
# tail: top names per rotation
print(b.sort_values('gap_bps', key=abs, ascending=False).groupby('date').head(3)[['date','book','symbol','avg_price','prior_close','gap_bps','slip_vs_bar_bps']].round(1).to_string())
print(b.groupby(['date','book']).apply(lambda g: g.gap_bps.abs().max()))
