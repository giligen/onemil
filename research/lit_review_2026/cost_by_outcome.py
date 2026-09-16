#!/usr/bin/env python3
"""STAGE 2b — cost by OUTCOME. Correcting an error the owner caught in cost_curve.py.

cost_curve.py charged every trade one full spread, measured at the SIGNAL minute. That is wrong twice over:
  * a winner that exits on a RESTING LIMIT at the target pays NO exit spread - we are the passive side there;
  * a loser exits at market into a falling tape, where the spread is WIDEST, and the signal-minute spread understates it.
So the honest expected cost is
      E[cost in R] = 0.5 * s_entry/R  +  P(market exit) * 0.5 * s_exit|market / R
with s_exit measured AT THE EXIT MINUTE, separately for stops and for 15:55 closes.

This samples real trades from the corrected pool, pulls Alpaca SIP NBBO in the entry minute AND in the exit minute, and
reports the spread by outcome, so min_edge_R becomes a function of a strategy's exit mix instead of a constant.
Usage: python3 cost_by_outcome.py [N=900]"""
import os, sys, time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd
ROOT = '/home/ec2-user/onemil'; os.chdir(ROOT); sys.path.insert(0, ROOT)
from config import Config
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockQuotesRequest
from alpaca.data.enums import DataFeed
ET = ZoneInfo('America/New_York'); D = 'research/lit_review_2026'; OUT = f'{D}/cost_by_outcome.csv'
N = int(sys.argv[1]) if len(sys.argv) > 1 else 900
c = pd.read_csv('research/bf_zero2/candidates3.csv',
                usecols=['day', 'symbol', 'entry_m', 'price', 'r_pct', 'rr_2r', 'why_2r', 'exit_m_2r'],
                dtype={'symbol': str, 'day': str, 'why_2r': str}, keep_default_na=False, na_values=[''], low_memory=True)
c = c[(c.price >= 5) & (c.r_pct >= 1) & c.exit_m_2r.notna() & c.why_2r.isin(['stop', 'target', 'eod'])].copy()
c['pb'] = pd.cut(c.price, [5, 10, 20, 50, 1e9], labels=['$5-10', '$10-20', '$20-50', '$50+'])
samp = c.groupby(['why_2r', 'pb'], observed=True, group_keys=False).apply(lambda g: g.sample(min(len(g), N // 12), random_state=5))
print(f'pool {len(c):,} | sampling {len(samp)} | mix {c.why_2r.value_counts(normalize=True).round(3).to_dict()}', flush=True)
done = set()
if os.path.exists(OUT):
    p = pd.read_csv(OUT, dtype={'symbol': str, 'day': str}); done = set(zip(p.day, p.symbol, p.entry_m))
cfg = Config(); cl = StockHistoricalDataClient(cfg.alpaca_api_key, cfg.alpaca_api_secret)


def spread_at(sym, day, minute):
    st = datetime.strptime(day, '%Y-%m-%d').replace(tzinfo=ET) + timedelta(minutes=int(minute))
    try:
        q = cl.get_stock_quotes(StockQuotesRequest(symbol_or_symbols=sym, start=st.astimezone(timezone.utc),
                                                   end=(st + timedelta(minutes=1)).astimezone(timezone.utc), feed=DataFeed.SIP, limit=600))
        qs = q.data.get(sym, []) if hasattr(q, 'data') else q.get(sym, [])
        sp = [(float(x.ask_price) - float(x.bid_price)) for x in qs if float(x.ask_price) > 0 and float(x.bid_price) > 0 and float(x.ask_price) >= float(x.bid_price)]
        return float(np.median(sp)) if sp else np.nan
    except Exception as e:
        if '429' in str(e) or 'too many' in str(e).lower(): time.sleep(15)
        return np.nan


rows = []; t0 = time.time()
for i, r in enumerate(samp.itertuples()):
    if (r.day, r.symbol, r.entry_m) in done: continue
    se = spread_at(r.symbol, r.day, r.entry_m); time.sleep(0.32)
    sx = spread_at(r.symbol, r.day, r.exit_m_2r); time.sleep(0.32)
    rows.append(dict(day=r.day, symbol=r.symbol, entry_m=r.entry_m, exit_m=r.exit_m_2r, why=r.why_2r, pb=str(r.pb),
                     price=r.price, r_pct=r.r_pct, s_entry=se, s_exit=sx))
    if len(rows) >= 60:
        pd.DataFrame(rows).to_csv(OUT, mode='a', header=not os.path.exists(OUT), index=False); rows = []
        print(f'{i+1}/{len(samp)} | {(time.time()-t0)/60:.1f} min', flush=True)
if rows: pd.DataFrame(rows).to_csv(OUT, mode='a', header=not os.path.exists(OUT), index=False)
d = pd.read_csv(OUT, dtype={'symbol': str, 'day': str})
d = d[(d.s_entry > 0)].copy()
d['R_abs'] = d.price * d.r_pct / 100
d['entry_cost_R'] = 0.5 * d.s_entry / d.R_abs
d['exit_cost_R'] = np.where(d.why == 'target', 0.0, 0.5 * d.s_exit / d.R_abs)
d['tot_R'] = d.entry_cost_R + d.exit_cost_R
L = ['# Cost by OUTCOME — correcting the flat round-trip assumption', '',
     'A target exit rests on a limit and pays NO exit spread. A stop or a 15:55 close crosses the spread at the EXIT minute.',
     'Costs are in R, i.e. as a fraction of the trade\'s own risk.', '']
g = d.groupby('why', observed=True).agg(n=('tot_R', 'size'), s_entry_bps=('s_entry', lambda s: round(np.median(s / d.loc[s.index, 'price'] * 1e4), 1)),
                                        s_exit_bps=('s_exit', lambda s: round(np.nanmedian(s / d.loc[s.index, 'price'] * 1e4), 1)),
                                        entry_R=('entry_cost_R', 'median'), exit_R=('exit_cost_R', 'median'), total_R=('tot_R', 'median')).round(3)
L += ['## by exit type', g.to_string(), '']
L += ['## spread widening from entry to exit (median ratio s_exit / s_entry)',
      d.groupby('why', observed=True).apply(lambda x: round(float(np.nanmedian(x.s_exit / x.s_entry)), 3)).to_string(), '']
L += ['## by price band and exit type (median total cost in R)',
      d.pivot_table(index='pb', columns='why', values='tot_R', aggfunc='median', observed=True).round(3).to_string(), '']
mix = d.why.value_counts(normalize=True)
blend = (d.groupby('why', observed=True).tot_R.median() * mix).sum()
flat = float(np.median(d.s_entry / d.R_abs))
L += [f'## the number that replaces the old one',
      f'observed exit mix: {mix.round(3).to_dict()}',
      f'BLENDED expected cost = {blend:.3f} R per trade (weighting each exit type by how often it happens)',
      f'the OLD flat assumption (one full entry-minute spread on every trade) = {flat:.3f} R',
      f'so the flat model was {"OVER" if flat > blend else "UNDER"}stating the true cost by {abs(flat - blend):.3f} R per trade.']
open(f'{D}/cost_by_outcome.md', 'w').write('\n'.join(L)); print('\n'.join(L)); print('DONE', flush=True)
