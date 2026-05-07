"""One-shot back-fill: PN trade 106 lost its exit price to the race-condition
bug. Query Alpaca for the actual SL leg fill, recompute P&L, update DB row."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()
import sqlite3
from datetime import datetime, timezone
from data_sources.alpaca_client import AlpacaClient

SL_LEG_ID = '308849ce-3751-4b2b-b135-322be6f0d3cb'
TRADE_ROW_ID = 106

client = AlpacaClient(
    api_key=os.environ['ALPACA_API_KEY'],
    api_secret=os.environ['ALPACA_API_SECRET'],
    paper=False,
)

order = client.get_order(SL_LEG_ID)
status = order.get('status', '')
fill_price = order.get('filled_avg_price')
filled_qty = order.get('filled_qty')
filled_at = order.get('filled_at')

print(f"SL leg {SL_LEG_ID}:")
print(f"  status: {status}")
print(f"  filled_avg_price: {fill_price}")
print(f"  filled_qty: {filled_qty}")
print(f"  filled_at: {filled_at}")

if status.lower() != 'filled' or fill_price is None:
    print("REFUSING TO BACKFILL — leg not in expected state.")
    raise SystemExit(1)

fill_price_f = float(fill_price)

conn = sqlite3.connect('/home/ec2-user/onemil/data/trades.db')
cur = conn.cursor()
cur.execute(
    "SELECT fill_price, shares FROM trades WHERE id=?", (TRADE_ROW_ID,)
)
entry_fill, shares = cur.fetchone()
pnl = (fill_price_f - entry_fill) * shares
pnl_pct = (pnl / (entry_fill * shares)) * 100

print(f"\nDB row {TRADE_ROW_ID}:")
print(f"  entry fill: ${entry_fill}, shares: {shares}")
print(f"  recomputed exit: ${fill_price_f}")
print(f"  recomputed P&L: ${pnl:+.2f} ({pnl_pct:+.2f}%)")

print("\n(dry run — printed only; will not write to DB unless you re-run with --write)")

import sys
if len(sys.argv) > 1 and sys.argv[1] == '--write':
    cur.execute(
        "UPDATE trades SET exit_price=?, exit_reason=?, pnl=?, pnl_pct=?, "
        "updated_at=? WHERE id=?",
        (fill_price_f, 'stop_loss_bracket_sl_race', pnl, pnl_pct,
         datetime.now(timezone.utc).isoformat(), TRADE_ROW_ID)
    )
    conn.commit()
    print(f"\nDB row {TRADE_ROW_ID} updated.")
conn.close()
