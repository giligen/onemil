"""Attribute MTD account P&L by symbol via filled orders (no activities API).

Net cashflow per symbol (sell - buy notional) = realized P&L for symbols flat at
both month boundaries. Buckets symbols into OneMil (trades.db) vs divergence shorts.
"""
import os
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus

client = TradingClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_API_SECRET"], paper=False)

today = datetime.now(timezone.utc).date()
month_start = today.replace(day=1)
start_dt = datetime(month_start.year, month_start.month, month_start.day, tzinfo=timezone.utc)

db = sqlite3.connect(os.path.join(os.path.dirname(__file__), "..", "data", "trades.db"))
onemil_syms = {r[0] for r in db.execute(
    "SELECT DISTINCT symbol FROM trades WHERE trade_date >= ?", (month_start.isoformat(),)
).fetchall()}

# Paginate closed orders since month start (Alpaca caps limit at 500)
orders = []
until = None
while True:
    req = GetOrdersRequest(status=QueryOrderStatus.CLOSED, after=start_dt, until=until,
                           limit=500, direction="desc", nested=False)
    batch = client.get_orders(req)
    if not batch:
        break
    orders.extend(batch)
    if len(batch) < 500:
        break
    until = batch[-1].submitted_at

cash = defaultdict(float)
qty = defaultdict(float)
nfill = defaultdict(int)
for o in orders:
    if not o.filled_qty or float(o.filled_qty) == 0 or o.filled_avg_price is None:
        continue
    sym = o.symbol
    q = float(o.filled_qty)
    px = float(o.filled_avg_price)
    side = str(o.side).lower()
    notional = px * q
    if "sell" in side:
        cash[sym] += notional
        qty[sym] -= q
    else:
        cash[sym] -= notional
        qty[sym] += q
    nfill[sym] += 1

onemil_total = div_total = 0.0
print(f"=== Symbol attribution via filled orders (since {month_start}, {len(orders)} orders) ===\n")
print(f"{'SYMBOL':<8}{'bucket':<10}{'net_cash':>12}{'net_qty':>9}{'fills':>7}")
for sym in sorted(cash, key=lambda s: cash[s]):
    bucket = "onemil" if sym in onemil_syms else "DIVERG"
    flag = "" if abs(qty[sym]) < 1e-6 else "  <OPEN>"
    if bucket == "onemil":
        onemil_total += cash[sym]
    else:
        div_total += cash[sym]
    print(f"{sym:<8}{bucket:<10}{cash[sym]:>12,.2f}{qty[sym]:>9.0f}{nfill[sym]:>7}{flag}")

print(f"\n  OneMil symbols net cashflow    : ${onemil_total:,.2f}")
print(f"  Divergence symbols net cashflow: ${div_total:,.2f}")
print(f"  TOTAL net cashflow             : ${onemil_total + div_total:,.2f}")
print("\n  (<OPEN> = nonzero residual qty -> cashflow != realized for that symbol)")
