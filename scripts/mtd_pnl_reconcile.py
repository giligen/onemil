"""Month-to-date LIVE P&L reconciliation: Alpaca broker vs trades.db.

Pulls account-level P&L from Alpaca portfolio history (blends ALL systems on the
shared live account: bull_flag + ORB + stupid-money divergence shorts) and
compares against trades.db realized P&L (bull_flag + orb only).
"""
import os
import sqlite3
from datetime import date, datetime, timezone

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetPortfolioHistoryRequest

KEY = os.environ["ALPACA_API_KEY"]
SEC = os.environ["ALPACA_API_SECRET"]
client = TradingClient(KEY, SEC, paper=False)

today = datetime.now(timezone.utc).date()
month_start = today.replace(day=1)

print(f"=== LIVE MTD P&L — {month_start} .. {today} ===\n")

# --- Account snapshot ---
acct = client.get_account()
print(f"Account equity now : ${float(acct.equity):,.2f}")
print(f"Last equity (prev) : ${float(acct.last_equity):,.2f}")
print(f"Cash               : ${float(acct.cash):,.2f}\n")

# --- Portfolio history (account-level, all systems) ---
req = GetPortfolioHistoryRequest(
    start=datetime(month_start.year, month_start.month, month_start.day, tzinfo=timezone.utc),
    end=datetime(today.year, today.month, today.day, 23, 59, tzinfo=timezone.utc),
    timeframe="1D",
)
ph = client.get_portfolio_history(req)
ts = ph.timestamp or []
pl = ph.profit_loss or []
eq = ph.equity or []

print("Per-day account P&L (portfolio history; label = UTC point):")
acct_mtd = 0.0
for t, p, e in zip(ts, pl, eq):
    d = datetime.fromtimestamp(t, tz=timezone.utc).date()
    pv = float(p) if p is not None else 0.0
    acct_mtd += pv
    print(f"  {d}  P&L ${pv:>10,.2f}   equity ${float(e):>12,.2f}")
print(f"\n  ACCOUNT MTD P&L (all systems): ${acct_mtd:,.2f}\n")

# --- trades.db realized P&L (bull_flag + orb) ---
db = sqlite3.connect(os.path.join(os.path.dirname(__file__), "..", "data", "trades.db"))
db.row_factory = sqlite3.Row
rows = db.execute(
    "SELECT strategy, COUNT(*) n, "
    "SUM(CASE WHEN pnl IS NOT NULL THEN 1 ELSE 0 END) closed, "
    "COALESCE(SUM(pnl),0) pnl "
    "FROM trades WHERE trade_date >= ? AND trade_date <= ? "
    "GROUP BY strategy ORDER BY strategy",
    (month_start.isoformat(), today.isoformat()),
).fetchall()

print("trades.db realized P&L (MTD, by strategy):")
db_total = 0.0
for r in rows:
    db_total += float(r["pnl"])
    print(f"  {r['strategy']:<12} trades={r['n']:>3}  closed={r['closed']:>3}  pnl=${float(r['pnl']):>10,.2f}")
print(f"\n  DB MTD realized (bull_flag+orb): ${db_total:,.2f}")

# open positions (unrealized, in DB scope)
opn = db.execute(
    "SELECT strategy, symbol, shares, entry_price FROM trades "
    "WHERE trade_date >= ? AND exit_price IS NULL "
    "AND order_status IN ('filled','partially_filled') ORDER BY strategy",
    (month_start.isoformat(),),
).fetchall()
if opn:
    print("\n  Open DB positions (unrealized, not in DB realized total):")
    for o in opn:
        print(f"    {o['strategy']:<10} {o['symbol']:<6} {o['shares']} @ ${o['entry_price']}")

print("\n=== RECONCILIATION ===")
gap = acct_mtd - db_total
print(f"  Account MTD (all)        : ${acct_mtd:,.2f}")
print(f"  DB realized (bf+orb)     : ${db_total:,.2f}")
print(f"  GAP (divergence + unrl + : ${gap:,.2f}")
print(f"   accounting differences)")
