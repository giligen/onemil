"""Exit-escalation re-price study (2026-07-10).

For each stop_loss_market_fallback event: reconstruct the NBBO tape from
stop-trigger to fill+60s via historical quotes, then simulate the
proposed ladder — at escalation time (T+10s), instead of MARKET, place a
sell limit at (current_bid - 0.3*spread) with 6s patience; if unfilled,
market at T+16s priced at the then-current bid minus this event's OWN
observed market-tax (conservative: reuse each event's realized tax).
Fill rule for the re-price: first subsequent quote with bid >= limit.
Outputs per event: actual cost vs ladder cost vs immediate-market
baseline, and the tail check (did waiting ever make it WORSE?).
"""
import os, sys
from datetime import datetime, timedelta
import pandas as pd
sys.path.insert(0,'/home/ec2-user/onemil')
from dotenv import load_dotenv; load_dotenv('/home/ec2-user/onemil/.env')
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockQuotesRequest
import sqlite3

c=sqlite3.connect('/home/ec2-user/onemil/data/trades.db')
ev=pd.read_sql("""SELECT symbol,trade_date,shares,exit_price,exit_quote_bid,exit_quote_ask,
 exit_slippage,exit_submitted_at,exited_at FROM trades
 WHERE exit_reason='stop_loss_market_fallback' ORDER BY trade_date""", c)
c.close()
qc=StockHistoricalDataClient(os.getenv('ALPACA_API_KEY'), os.getenv('ALPACA_API_SECRET'))

def quotes(sym, t0, t1):
    try:
        r=qc.get_stock_quotes(StockQuotesRequest(symbol_or_symbols=sym,
            start=t0, end=t1, limit=8000)).data.get(sym, [])
        return [(q.timestamp, float(q.bid_price or 0), float(q.ask_price or 0))
                for q in r if q.bid_price]
    except Exception as e:
        print(f"  {sym}: quote fetch failed {str(e)[:60]}")
        return []

print(f"{'sym':>6} {'actual_fill':>11} {'ladder_fill':>11} {'imm_mkt':>9} {'saved/sh':>9} {'saved_$':>8}  path")
tot_saved=0.0; worse=0
for _,r in ev.iterrows():
    sym=r['symbol']
    # anchor: exit submitted (limit placed). escalation at +10s.
    if pd.notna(r['exit_submitted_at']):
        t_sub=pd.Timestamp(r['exit_submitted_at']).to_pydatetime()
    else:
        # fall back: exited_at minus ~12s (limit 10s + cancel+mkt ~2s)
        if pd.isna(r['exited_at']): print(f"{sym:>6}  no timestamps — skip"); continue
        t_sub=pd.Timestamp(r['exited_at']).to_pydatetime()-timedelta(seconds=12)
    tape=quotes(sym, t_sub, t_sub+timedelta(seconds=70))
    if len(tape)<5:
        print(f"{sym:>6}  thin tape ({len(tape)} quotes) — skip"); continue
    def bid_at(dt_s):
        target=t_sub+timedelta(seconds=dt_s)
        last=tape[0]
        for q in tape:
            if q[0]<=target: last=q
            else: break
        return last
    esc=bid_at(10.0)                      # NBBO at escalation moment
    esc_bid, esc_ask = esc[1], esc[2]
    spread=max(esc_ask-esc_bid, 0.01)
    reprice=esc_bid-0.3*spread
    # observed market tax for THIS event: actual fill vs bid at escalation
    actual_fill=r['exit_price']
    mkt_tax=esc_bid-actual_fill           # >=0: how far below esc-bid the market filled
    # ladder sim: does any quote in (10s,16s] show bid >= reprice?
    filled=None
    for q in tape:
        _t0=pd.Timestamp(t_sub)
        _t0=_t0.tz_localize(q[0].tzinfo) if _t0.tzinfo is None else _t0.tz_convert(q[0].tzinfo)
        dt=(q[0]-_t0).total_seconds()
        if 10.0<dt<=16.0 and q[1]>=reprice:
            filled=reprice; path='REPRICE-filled'; break
    if filled is None:
        late=bid_at(16.0)
        filled=late[1]-max(mkt_tax,0)     # market at T+16 with same tax
        path='mkt@T+16'
    imm=actual_fill                       # immediate market = what actually happened
    saved_sh=filled-imm
    saved=saved_sh*r['shares']
    tot_saved+=saved
    if saved<-1: worse+=1
    print(f"{sym:>6} {imm:>11.3f} {filled:>11.3f} {imm:>9.3f} {saved_sh:>+9.3f} {saved:>+8.0f}  {path}")
print(f"\nTOTAL simulated savings on {len(ev)} events: ${tot_saved:+,.0f}  |  events made WORSE: {worse}")
