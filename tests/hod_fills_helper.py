"""Shared test helper (review B2, 2026-10-03): fake the orders API listing THIS book's buy fills per symbol."""
from types import SimpleNamespace
from unittest.mock import MagicMock


def set_our_fills(alpaca, fills_by_symbol, foreign_buy=0):
    """Make `alpaca.trading_client.get_orders` list this book's buy fills per symbol (prefix `hod-rest-`), plus an
    optional foreign (non-prefixed) buy and a prefixed SELL that must never count toward our bought quantity."""
    alpaca.trading_client = MagicMock()

    def get_orders(filter=None):
        sym = filter.symbols[0]
        out = []
        q = fills_by_symbol.get(sym)
        if q is not None:
            out.append(SimpleNamespace(client_order_id=f'hod-rest-{sym}-10-03-ab12cd34', side='buy', filled_qty=str(q), symbol=sym))
            out.append(SimpleNamespace(client_order_id=f'hod-fc-{sym}-10-03-aaaaaa', side='sell', filled_qty='5', symbol=sym))
        if foreign_buy:
            out.append(SimpleNamespace(client_order_id='owner-manual-1', side='buy', filled_qty=str(foreign_buy), symbol=sym))
        return out
    alpaca.trading_client.get_orders.side_effect = get_orders
