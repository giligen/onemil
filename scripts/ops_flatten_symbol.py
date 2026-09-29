#!/usr/bin/env python3
"""Flatten ONE symbol's position on the LIVE account with a marketable limit (ops tool, run by the owner).

Why this exists (2026-09-29): the HOD-break force-close sold CDNA a second time on 9/25 after the stop had already
closed it, leaving the live account SHORT 57 shares for two sessions; the orphan reconciler had labelled our own
over-exit "owner's manual trade". The main session cannot submit live orders from the harness, so the owner runs:

    python3 scripts/ops_flatten_symbol.py CDNA          # show the position, then flatten it (asks y/N)
    python3 scripts/ops_flatten_symbol.py CDNA --yes    # no prompt

Behaviour: reads the live position for the symbol; a SHORT is covered with a BUY, a LONG is sold; the order is a DAY
limit at the far touch ± 0.3 % (marketable, never a naked market order); waits up to 30 s for the fill and prints the
position afterwards. Refuses to act when the account is not the live one or the symbol has no position. Every step is
printed; nothing else on the account is touched.
"""
import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(str(ROOT / '.env'))

from alpaca.data.historical import StockHistoricalDataClient  # noqa: E402
from alpaca.data.requests import StockLatestQuoteRequest  # noqa: E402
from alpaca.trading.client import TradingClient  # noqa: E402
from alpaca.trading.enums import OrderSide, TimeInForce  # noqa: E402
from alpaca.trading.requests import LimitOrderRequest  # noqa: E402

MARKETABLE_PAD = 0.003   # 0.3 % beyond the far touch so a moving quote still fills
FILL_WAIT_S = 30


def live_clients():
    """Trading + data clients for the LIVE account (ALPACA_API_KEY / ALPACA_API_SECRET), never paper."""
    key, secret = os.environ.get('ALPACA_API_KEY'), os.environ.get('ALPACA_API_SECRET')
    if not key or not secret:
        sys.exit('ERROR: ALPACA_API_KEY / ALPACA_API_SECRET missing in .env')
    return TradingClient(key, secret, paper=False), StockHistoricalDataClient(key, secret)


def find_position(tc, symbol):
    """The live position object for `symbol`, or None."""
    for p in tc.get_all_positions():
        if p.symbol == symbol:
            return p
    return None


def flatten(symbol: str, assume_yes: bool) -> int:
    """Print the position, ask, submit the marketable limit on the opposite side, report the result."""
    tc, dc = live_clients()
    acct = tc.get_account()
    print(f"account {acct.account_number} status={acct.status} equity=${float(acct.equity):,.0f}")
    pos = find_position(tc, symbol)
    if pos is None:
        print(f"{symbol}: no open position on the live account — nothing to do")
        return 0
    qty = int(abs(float(pos.qty)))
    short = float(pos.qty) < 0
    quote = dc.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=symbol))[symbol]
    bid, ask = float(quote.bid_price), float(quote.ask_price)
    side = OrderSide.BUY if short else OrderSide.SELL
    limit = round(ask * (1 + MARKETABLE_PAD), 2) if short else round(bid * (1 - MARKETABLE_PAD), 2)
    print(f"{symbol}: {'SHORT' if short else 'LONG'} {qty} sh, avg ${float(pos.avg_entry_price):.4f}, "
          f"mark ${float(pos.current_price):.2f}, unrealized ${float(pos.unrealized_pl):+.2f}")
    print(f"quote bid ${bid:.2f} / ask ${ask:.2f} -> {side.value.upper()} {qty} DAY limit @ ${limit:.2f}")
    if not assume_yes:
        if input("submit? [y/N] ").strip().lower() != 'y':
            print("aborted"); return 1
    order = tc.submit_order(LimitOrderRequest(symbol=symbol, qty=qty, side=side, time_in_force=TimeInForce.DAY,
                                              limit_price=limit,
                                              client_order_id=f"ops-flatten-{symbol}-{int(time.time())}"))
    print(f"submitted {order.id}")
    status = None
    for _ in range(FILL_WAIT_S):
        time.sleep(1)
        o = tc.get_order_by_id(order.id)
        status = o.status.value
        if status in ('filled', 'canceled', 'rejected', 'expired'):
            break
    print(f"order {status}: filled {o.filled_qty} @ {o.filled_avg_price}")
    after = find_position(tc, symbol)
    print(f"position now: {'FLAT' if after is None else f'{after.qty} sh'}")
    return 0 if after is None else 2


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('symbol')
    ap.add_argument('--yes', action='store_true', help='skip the confirmation prompt')
    a = ap.parse_args()
    sys.exit(flatten(a.symbol.upper(), a.yes))
