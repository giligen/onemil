#!/usr/bin/env python3
"""Harmless live probe of Alpaca's stop-limit order lifecycle (owner-run, 2026-09-25).

Places ONE buy stop-limit on AAPL for 1 share with stop == limit set 15 % ABOVE the current ask (it cannot fill),
replaces it once (+$0.10), cancels it and polls the status transitions; repeats with a 7-share odd lot; finally
asserts that no order carrying the 'onemil-probe-' client_order_id prefix remains open. Never touches any other
order (the owner trades manually on this account). Prints every raw response and timing, and appends the same to
docs/alpaca_stop_limit_probe_20260925.md.

Run:  python3 scripts/alpaca_stop_limit_probe.py          (live keys from .env; ~2 minutes)
"""
import json
import os
import sys
import time
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from dotenv import load_dotenv  # noqa: E402

load_dotenv('.env')
from config import Config  # noqa: E402
from alpaca.trading.client import TradingClient  # noqa: E402
from alpaca.trading.requests import StopLimitOrderRequest, ReplaceOrderRequest, GetOrdersRequest  # noqa: E402
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus  # noqa: E402
from alpaca.data.historical import StockHistoricalDataClient  # noqa: E402
from alpaca.data.requests import StockLatestQuoteRequest  # noqa: E402

PREFIX = 'onemil-probe-'
SYMBOL = 'AAPL'
OUT = 'docs/alpaca_stop_limit_probe_20260925.md'
LOG = []


def say(msg):
    """Print and buffer one line for the markdown log."""
    line = f"[{datetime.now(timezone.utc).strftime('%H:%M:%S.%f')[:-3]} UTC] {msg}"
    print(line, flush=True)
    LOG.append(line)


def raw(order):
    """Compact JSON of the fields that matter."""
    keys = ('id', 'client_order_id', 'status', 'type', 'order_type', 'side', 'qty', 'stop_price', 'limit_price',
            'time_in_force', 'submitted_at', 'updated_at', 'canceled_at', 'replaced_by', 'replaces', 'filled_qty')
    d = {k: str(getattr(order, k)) for k in keys if getattr(order, k, None) is not None}
    return json.dumps(d)


def poll(client, order_id, want, timeout=20.0):
    """Poll until status == want (or timeout); print each transition with timing."""
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout:
        o = client.get_order_by_id(order_id)
        st = str(o.status)
        if st != last:
            say(f"  status {st} after {time.time() - t0:.2f}s")
            last = st
        if want in st:
            return o
        time.sleep(0.5)
    return client.get_order_by_id(order_id)


def one_probe(client, qty, ask):
    """Place → replace → cancel one probe order; return nothing, log everything."""
    px = round(ask * 1.15, 2)
    coid = f"{PREFIX}{int(time.time() * 1000)}-q{qty}"
    say(f"PLACE {SYMBOL} qty {qty} buy stop-limit stop=limit=${px:.2f} (ask ${ask:.2f}) client_order_id {coid}")
    t0 = time.time()
    o = client.submit_order(StopLimitOrderRequest(symbol=SYMBOL, qty=qty, side=OrderSide.BUY,
                                                  time_in_force=TimeInForce.DAY, stop_price=px, limit_price=px,
                                                  client_order_id=coid))
    say(f"  accepted in {time.time() - t0:.2f}s: {raw(o)}")
    o = poll(client, o.id, 'accepted', timeout=10)
    say(f"REPLACE stop/limit -> ${px + 0.10:.2f}")
    t0 = time.time()
    try:
        r = client.replace_order_by_id(o.id, ReplaceOrderRequest(stop_price=round(px + 0.10, 2),
                                                                 limit_price=round(px + 0.10, 2)))
        say(f"  replace returned in {time.time() - t0:.2f}s: {raw(r)} (id changed: {r.id != o.id})")
        current = r
    except Exception as e:  # noqa: BLE001
        say(f"  replace FAILED in {time.time() - t0:.2f}s: {type(e).__name__}: {e}")
        current = o
    say("CANCEL")
    t0 = time.time()
    client.cancel_order_by_id(current.id)
    c = poll(client, current.id, 'canceled', timeout=20)
    say(f"  cancel confirmed in {time.time() - t0:.2f}s: {raw(c)}")
    if 'canceled' not in str(c.status):
        say("  WARNING: order not confirmed canceled — check the dashboard for prefix onemil-probe-")


def main():
    cfg = Config()
    client = TradingClient(cfg.alpaca_api_key, cfg.alpaca_api_secret, paper=False)
    data = StockHistoricalDataClient(cfg.alpaca_api_key, cfg.alpaca_api_secret)
    q = data.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=SYMBOL))[SYMBOL]
    ask = float(q.ask_price)
    say(f"latest {SYMBOL} quote bid {q.bid_price} ask {ask} (probe prices 15 % above the ask cannot fill)")
    for qty in (1, 7):
        one_probe(client, qty, ask)
        time.sleep(1.0)
    opens = client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))
    left = [o for o in opens if str(o.client_order_id).startswith(PREFIX)]
    say(f"open orders with prefix {PREFIX}: {len(left)}")
    for o in left:
        say(f"  CANCELLING leftover {o.id}")
        client.cancel_order_by_id(o.id)
    say("DONE" if not left else "DONE (leftovers cancelled — verify in the dashboard)")
    with open(OUT, 'a') as fh:
        fh.write(f"\n## Probe run {datetime.now(timezone.utc).isoformat()}\n\n```\n" + '\n'.join(LOG) + "\n```\n")


if __name__ == '__main__':
    main()
