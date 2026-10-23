"""Fetch broker created_at/submitted_at/filled_at for every filled ORB entry order (read-only, orders by id only)."""
import os, sqlite3, csv, sys
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient

OUT = '/home/ec2-user/onemil/research/exec_learnings_oct2026/broker_ts.csv'
load_dotenv('/home/ec2-user/onemil/.env')
con = sqlite3.connect('file:/home/ec2-user/onemil/data/trades.db?mode=ro', uri=True)
rows = con.execute("select trade_date,symbol,order_id,coalesce(account,'') from trades where strategy='orb' "
                   "and fill_price is not null and order_id is not null order by id").fetchall()
live = TradingClient(os.environ['ALPACA_API_KEY'], os.environ['ALPACA_API_SECRET'], paper=False)
paper = TradingClient(os.environ['ALPACA_ORB_API_KEY'], os.environ['ALPACA_ORB_API_SECRET'], paper=True)
lost = 0
with open(OUT, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['date', 'symbol', 'order_id', 'account', 'brk_created', 'brk_submitted', 'brk_filled', 'brk_status', 'brk_avg'])
    for d, s, oid, acct in rows:
        cl = paper if acct == 'paper' else live
        try:
            o = cl.get_order_by_id(oid)
            w.writerow([d, s, oid, acct or 'live', o.created_at, o.submitted_at, o.filled_at, str(o.status), o.filled_avg_price])
        except Exception as e:
            lost += 1
            print('LOST', d, s, acct, type(e).__name__, str(e)[:80], flush=True)
print('rows', len(rows), 'lost', lost, flush=True)
