"""Unit tests for cell_1442.py on synthetic ticks (no live tape/cache reads)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cell_1442 as c1442  # noqa: E402
import sip_rebuild as sr   # noqa: E402


def ticks(trades, quotes):
    """trades: list of (ts_ns, price, size); quotes: list of (ts_ns, bid, ask)."""
    t = pd.DataFrame(trades, columns=['ts', 'price', 'size'])
    q = pd.DataFrame(quotes, columns=['ts', 'bid', 'ask'])
    return t, q


S = 1_000_000_000_000   # arbitrary S_ns anchor
LEVEL = 10.0
TRIGGER = round(LEVEL + sr.TICK, 6)   # 10.01
LIMIT = LEVEL * (1.0 + sr.LIMIT_BPS)  # 10.0150


# --------------------------------------------------------------------------------------------
# find_broker: round-lot detection
# --------------------------------------------------------------------------------------------

def test_broker_skips_odd_lot_takes_first_round_lot():
    trades, quotes = ticks(
        [(S - 5_000_000_000, TRIGGER, 50), (S - 4_000_000_000, TRIGGER, 200)],
        [(S - 5_100_000_000, 10.00, 10.01), (S - 4_100_000_000, 10.00, 10.012)],
    )
    res = c1442.find_broker(trades, quotes, S, TRIGGER, LIMIT)
    assert res is not None
    assert res['t_hit'] == S - 4_000_000_000     # the round-lot print, not the odd-lot one
    assert res['ask'] == pytest.approx(10.012)


def test_broker_no_round_lot_print_is_no_fill():
    trades, quotes = ticks(
        [(S - 5_000_000_000, TRIGGER, 50)],
        [(S - 5_100_000_000, 10.00, 10.01)],
    )
    assert c1442.find_broker(trades, quotes, S, TRIGGER, LIMIT) is None


def test_broker_odd_lot_alone_never_arms_even_if_round_lot_fill_would():
    """An odd-lot print at trigger with no later round-lot print -> no fill under (a), matching
    Alpaca's NBBO-filtered (round-lot) arming rule."""
    trades, quotes = ticks(
        [(S - 5_000_000_000, TRIGGER, 99)],
        [(S - 5_100_000_000, 10.00, 10.011)],
    )
    assert c1442.find_broker(trades, quotes, S, TRIGGER, LIMIT) is None


# --------------------------------------------------------------------------------------------
# find_override: +300ms ask lookup
# --------------------------------------------------------------------------------------------

def test_override_uses_ask_300ms_after_any_size_print():
    trigger_ts = S - 5_000_000_000
    trades, quotes = ticks(
        [(trigger_ts, TRIGGER, 10)],   # odd lot, still arms override
        [(trigger_ts - 100_000_000, 10.00, 10.011),          # quote before trigger
         (trigger_ts + c1442.DELAY_NS - 1_000_000, 10.00, 10.012),   # just before t+300ms
         (trigger_ts + c1442.DELAY_NS + 5_000_000, 10.00, 10.013)],  # after t+300ms, must NOT be used
    )
    res = c1442.find_override(trades, quotes, S, TRIGGER, LIMIT)
    assert res is not None
    assert res['t_hit'] == trigger_ts
    assert res['ask'] == pytest.approx(10.012)   # prevailing at t+300ms, not the later quote


def test_override_no_quote_at_300ms_is_no_fill():
    trigger_ts = S - 5_000_000_000
    trades, quotes = ticks([(trigger_ts, TRIGGER, 10)], [])
    assert c1442.find_override(trades, quotes, S, TRIGGER, LIMIT) is None


# --------------------------------------------------------------------------------------------
# cap: both rules refuse a fill above limit
# --------------------------------------------------------------------------------------------

def test_broker_capped_at_limit():
    trades, quotes = ticks(
        [(S - 5_000_000_000, TRIGGER, 200)],
        [(S - 5_100_000_000, 10.00, LIMIT + 0.001)],
    )
    assert c1442.find_broker(trades, quotes, S, TRIGGER, LIMIT) is None


def test_override_capped_at_limit():
    trigger_ts = S - 5_000_000_000
    trades, quotes = ticks(
        [(trigger_ts, TRIGGER, 10)],
        [(trigger_ts + c1442.DELAY_NS - 1_000_000, 10.00, LIMIT + 0.001)],
    )
    assert c1442.find_override(trades, quotes, S, TRIGGER, LIMIT) is None


def test_broker_at_exactly_limit_fills():
    trades, quotes = ticks(
        [(S - 5_000_000_000, TRIGGER, 200)],
        [(S - 5_100_000_000, 10.00, LIMIT)],
    )
    res = c1442.find_broker(trades, quotes, S, TRIGGER, LIMIT)
    assert res is not None
    assert res['ask'] == pytest.approx(LIMIT)


# --------------------------------------------------------------------------------------------
# size-class cross helper: round/odd classification feeds the report, not the fill decision
# --------------------------------------------------------------------------------------------

def test_size_class_cross_empty_when_no_base_fills():
    out = c1442._size_class_cross(pd.DataFrame(columns=['day', 'symbol', 'entry_m', 'level',
                                                          'base_net_R', 'net_R_slip_a', 'net_R_slip_b']), {})
    assert len(out) == 0
