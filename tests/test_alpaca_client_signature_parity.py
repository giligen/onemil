"""Signature parity between the ORB engine's order calls and AlpacaClient (2026-10-05 incident).

Commit 160b70f passed `client_order_id=` to `submit_stop_bracket_order`, which had no such parameter; every ORB
stop-limit entry raised TypeError for three sessions while the mocked engine tests stayed green (a `MagicMock(spec=)`
does not check call signatures). These tests bind the engine's real keyword sets to the real method signatures.
"""
import inspect
from unittest.mock import MagicMock, patch

from data_sources.alpaca_client import AlpacaClient

ENGINE_STOP_BRACKET_KWARGS = {
    'symbol', 'qty', 'side', 'stop_price', 'limit_price', 'tp_price', 'sl_price', 'client_order_id',
}


def test_submit_stop_bracket_order_accepts_engine_kwargs():
    """Every keyword the ORB engine passes (trading/orb_engine.py submit_entry) must bind to the real signature."""
    sig = inspect.signature(AlpacaClient.submit_stop_bracket_order)
    params = set(sig.parameters) - {'self'}
    missing = ENGINE_STOP_BRACKET_KWARGS - params
    assert not missing, f"engine passes kwargs the client does not accept: {missing}"
    sig.bind(None, symbol='JAGX', qty=10, side='buy', stop_price=5.0, limit_price=5.05,
             tp_price=6.0, sl_price=4.5, client_order_id='orb-20261005-JAGX')


def test_engine_source_kwargs_are_subset_of_client_signature():
    """Parse the engine source for `submit_stop_bracket_order(` call sites and check each keyword exists."""
    import re
    import pathlib
    src = pathlib.Path('trading/orb_engine.py').read_text()
    sig_params = set(inspect.signature(AlpacaClient.submit_stop_bracket_order).parameters)
    for m in re.finditer(r'submit_stop_bracket_order\((.*?)\)\n', src, re.S):
        kws = set(re.findall(r'(\w+)=', m.group(1)))
        assert kws <= sig_params, f"call site passes unknown kwargs: {kws - sig_params}"


def test_client_order_id_forwarded_to_request():
    """The id reaches the Alpaca request object; omitted -> not set (Alpaca then generates one)."""
    client = AlpacaClient.__new__(AlpacaClient)
    client.trading_client = MagicMock()
    client._call_with_timeout = lambda fn, _label: fn()
    seen = {}

    class _Req:
        def __init__(self, **kw):
            seen.update(kw)

    with patch('alpaca.trading.requests.StopLimitOrderRequest', _Req):
        order = MagicMock(); order.id = 'o1'; order.legs = []
        client.trading_client.submit_order.return_value = order
        client.submit_stop_bracket_order('JAGX', 10, 'buy', 5.0, 5.05, 6.0, 4.5,
                                         client_order_id='orb-20261005-JAGX')
        assert seen['client_order_id'] == 'orb-20261005-JAGX'
        seen.clear()
        client.submit_stop_bracket_order('JAGX', 10, 'buy', 5.0, 5.05, 6.0, 4.5)
        assert 'client_order_id' not in seen
