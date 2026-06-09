"""Pin that AlpacaClient.submit_bracket_order returns the bracket leg
IDs in the response. Without these, ORB's StopMonitor watch had empty
tp_leg_id / sl_leg_id, which silently disabled the SL-leg-race recovery
path in stop_monitor.py. FABC 2026-06-09 was bitten by this.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from data_sources.alpaca_client import AlpacaClient


def _mk_leg(leg_id, side, type_, stop_price=None, limit_price=None):
    leg = MagicMock()
    leg.id = leg_id
    side_m = MagicMock(); side_m.value = side
    type_m = MagicMock(); type_m.value = type_
    leg.side = side_m
    leg.type = type_m
    leg.stop_price = stop_price
    leg.limit_price = limit_price
    return leg


def _mk_order(order_id='parent-1', status='accepted', legs=None):
    o = MagicMock()
    o.id = order_id
    sm = MagicMock(); sm.value = status
    o.status = sm
    o.legs = legs or []
    return o


def _client_with_mocked_sdk():
    c = AlpacaClient.__new__(AlpacaClient)
    c.trading_client = MagicMock()
    c._call_with_timeout = lambda fn, label: fn()  # bypass timeout wrapper
    return c


class TestSubmitBracketReturnsLegs:

    def test_legs_extracted_with_correct_ids(self):
        tp = _mk_leg('tp-id-1', 'sell', 'limit', limit_price=5.00)
        sl = _mk_leg('sl-id-1', 'sell', 'stop', stop_price=3.99)
        c = _client_with_mocked_sdk()
        c.trading_client.submit_order = MagicMock(
            return_value=_mk_order(legs=[tp, sl])
        )

        result = c.submit_bracket_order(
            symbol='FABC', qty=10627, side='buy',
            limit_price=4.30, tp_price=5.00, sl_price=3.99,
        )

        assert 'legs' in result
        assert len(result['legs']) == 2
        ids = {leg['id'] for leg in result['legs']}
        assert ids == {'tp-id-1', 'sl-id-1'}

    def test_each_leg_has_full_shape(self):
        tp = _mk_leg('tp-1', 'sell', 'limit', limit_price=5.00)
        sl = _mk_leg('sl-1', 'sell', 'stop', stop_price=3.99)
        c = _client_with_mocked_sdk()
        c.trading_client.submit_order = MagicMock(
            return_value=_mk_order(legs=[tp, sl])
        )
        result = c.submit_bracket_order(
            symbol='X', qty=100, side='buy',
            limit_price=10, tp_price=12, sl_price=9,
        )
        by_id = {leg['id']: leg for leg in result['legs']}

        # TP leg
        tp_dict = by_id['tp-1']
        assert tp_dict['side'] == 'sell'
        assert tp_dict['type'] == 'limit'
        assert tp_dict['limit_price'] == 5.00
        assert tp_dict['stop_price'] is None

        # SL leg
        sl_dict = by_id['sl-1']
        assert sl_dict['side'] == 'sell'
        assert sl_dict['type'] == 'stop'
        assert sl_dict['stop_price'] == 3.99
        assert sl_dict['limit_price'] is None

    def test_no_legs_attribute_returns_empty_list(self):
        c = _client_with_mocked_sdk()
        # Order with no legs attribute (degenerate but possible)
        o = MagicMock()
        o.id = 'parent-1'
        sm = MagicMock(); sm.value = 'accepted'
        o.status = sm
        # Configure legs to be None via attribute access — MagicMock
        # default returns truthy by default, so override
        o.legs = None
        c.trading_client.submit_order = MagicMock(return_value=o)
        result = c.submit_bracket_order(
            symbol='X', qty=100, side='buy',
            limit_price=10, tp_price=12, sl_price=9,
        )
        assert result['legs'] == []
