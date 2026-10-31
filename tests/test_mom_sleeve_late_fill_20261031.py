"""Late-fill booking for the MOM sleeve (docs/mom_sleeve_late_fill_booking_spec_20261031.md).

Unit (`reconcile_late_fills` on a state dict + temp files): an unbooked filled order is booked once (cash, positions,
ledger row, booked_coids), a second call books nothing, a cancelled / expired order books nothing, a partial fill
books its filled quantity and later only the increment, paper books the ledger row but leaves cash/positions to the
broker sync. Integration (`run` on real tmp state files, MagicMock(spec=AlpacaClient) over a fake broker): a run whose
poll times out on one order, then a second run that books it; state equals the broker's view at each boundary.
"""
import csv
import json
import logging
import os
import sys
from datetime import date
from types import SimpleNamespace

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'tests'))

from trading import momentum_live as ml  # noqa: E402
import test_momentum_sleeve_live as tls  # noqa: E402
from test_momentum_sleeve_live import world  # noqa: E402,F401  (pytest fixture)

R = tls.runner
TODAY = date(2026, 10, 5)
COID = 'moml-20261005-AAA-b'


def order(status='filled', qty=2.0, avg=50.0, coid=COID):
    """A broker order double."""
    return SimpleNamespace(client_order_id=coid, status=status, filled_qty=qty, filled_avg_price=avg)


def live_state(**extra):
    """A LIVE sleeve state with one pending buy of AAA."""
    st = {'cash': 1000.0, 'positions': {}, 'peak_equity': 1000.0, 'last_rebalance': None,
          'pending_coids': {COID: {'symbol': 'AAA', 'side': 'buy', 'date': str(TODAY)}}}
    st.update(extra)
    return st


def client_with(*orders):
    """Fake broker holding the given orders."""
    c, b = tls.live_client()
    for o in orders:
        b.orders[o.client_order_id] = o
    return c, b


def ledger_rows(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


# ------------------------------------------------------------------ unit
def test_unbooked_filled_order_is_booked_once(world, caplog):
    c, b = client_with(order('filled', 2.0, 50.0))
    st = live_state()
    with caplog.at_level(logging.INFO):
        assert R.reconcile_late_fills(c, st, TODAY, live=True) == 1
    assert st['positions'] == {'AAA': 2.0} and st['cash'] == pytest.approx(900.0)
    assert st['booked_coids'] == [COID] and 'pending_coids' not in st
    rows = ledger_rows(ml.LIVE_LEDGER_PATH)
    assert len(rows) == 1 and rows[0]['client_order_id'] == COID and float(rows[0]['qty']) == 2.0
    assert f'late fill booked {COID} qty 2.0 @ 50.0' in caplog.text
    assert json.load(open(ml.LIVE_STATE_PATH))['positions'] == {'AAA': 2.0}      # saved immediately


def test_second_call_books_nothing(world):
    c, b = client_with(order())
    st = live_state()
    assert R.reconcile_late_fills(c, st, TODAY, live=True) == 1
    assert R.reconcile_late_fills(c, st, TODAY, live=True) == 0
    # even a stale pending entry for an already-booked id is not booked again
    st['pending_coids'] = {COID: {'symbol': 'AAA', 'side': 'buy', 'date': str(TODAY)}}
    assert R.reconcile_late_fills(c, st, TODAY, live=True) == 0
    assert st['positions'] == {'AAA': 2.0} and len(ledger_rows(ml.LIVE_LEDGER_PATH)) == 1


@pytest.mark.parametrize('status', ['canceled', 'expired'])
def test_cancelled_or_expired_order_books_nothing(world, status):
    c, b = client_with(order(status, 0.0, 0.0))
    st = live_state()
    assert R.reconcile_late_fills(c, st, TODAY, live=True) == 0
    assert st['positions'] == {} and st['cash'] == 1000.0 and 'pending_coids' not in st
    assert not os.path.exists(ml.LIVE_LEDGER_PATH)


def test_still_open_order_stays_pending(world):
    c, b = client_with(order('new', 0.0, 0.0))
    st = live_state()
    assert R.reconcile_late_fills(c, st, TODAY, live=True) == 0
    assert COID in st['pending_coids'] and st['positions'] == {}


def test_partial_fill_books_filled_qty_then_only_the_increment(world):
    c, b = client_with(order('partially_filled', 1.0, 50.0))
    st = live_state()
    assert R.reconcile_late_fills(c, st, TODAY, live=True) == 1
    assert st['positions'] == {'AAA': 1.0} and st['cash'] == pytest.approx(950.0)
    assert COID in st['pending_coids'] and COID not in st.get('booked_coids', [])    # may still fill more
    b.orders[COID] = order('filled', 3.0, 52.0)                                       # total 3 @ avg 52 = 156
    assert R.reconcile_late_fills(c, st, TODAY, live=True) == 1
    assert st['positions'] == {'AAA': 3.0} and st['cash'] == pytest.approx(1000.0 - 156.0)
    assert st['booked_coids'] == [COID] and 'pending_coids' not in st


def test_sell_late_fill_reduces_position_and_adds_cash(world):
    coid = 'moml-20261005-AAA-s'
    c, b = client_with(order('filled', 2.0, 55.0, coid))
    st = live_state(positions={'AAA': 2.0}, pending_coids={coid: {'symbol': 'AAA', 'side': 'sell',
                                                                  'date': str(TODAY)}})
    assert R.reconcile_late_fills(c, st, TODAY, live=True) == 1
    assert st['positions'] == {} and st['cash'] == pytest.approx(1110.0)


def test_lookup_failure_keeps_pending_with_warning(world, caplog):
    c, b = client_with()
    b.get_order_by_client_id = lambda coid: (_ for _ in ()).throw(RuntimeError('503 upstream'))
    st = live_state()
    with caplog.at_level(logging.WARNING):
        assert R.reconcile_late_fills(c, st, TODAY, live=True) == 0
    assert COID in st['pending_coids'] and 'lookup' in caplog.text


def test_nothing_pending_is_a_noop_without_writes(world):
    c, b = client_with()
    st = {'cash': 1.0, 'positions': {}}
    assert R.reconcile_late_fills(c, st, TODAY, live=True) == 0
    assert not os.path.exists(ml.LIVE_STATE_PATH)


def test_paper_books_ledger_and_marker_but_not_cash_or_positions(monkeypatch, tmp_path):
    monkeypatch.setattr(R, 'STATE_PATH', str(tmp_path / 'state.json'))
    monkeypatch.setattr(R, 'LEDGER_PATH', str(tmp_path / 'ledger.csv'))
    coid = 'mom-20261005-AAA-b'
    c, b = client_with(order('filled', 2.0, 50.0, coid))
    st = live_state(pending_coids={coid: {'symbol': 'AAA', 'side': 'buy', 'date': str(TODAY)}})
    assert R.reconcile_late_fills(c, st, TODAY, live=False) == 1
    assert st['positions'] == {} and st['cash'] == 1000.0                  # the broker sync owns these on paper
    assert st['booked_coids'] == [coid]
    assert len(ledger_rows(str(tmp_path / 'ledger.csv'))) == 1


# ------------------------------------------------------------------ integration
def test_poll_timeout_then_next_run_books_the_late_fill(world, monkeypatch, caplog):
    args, asof, tmp = world
    c, b = tls.live_client()
    real_submit = b.submit_order
    held = []

    def submit(req):
        """The first buy stays open (status 'new', nothing filled); everything else fills."""
        o = real_submit(req)
        if not held:
            held.append(req.symbol)
            o.status, o.filled_qty, o.filled_avg_price = 'new', 0.0, 0.0
        return o
    b.submit_order = submit
    real_poll = R.poll_fills
    monkeypatch.setattr(R, 'poll_fills', lambda client, coids, **k: real_poll(client, coids, timeout=0.0, interval=0.0))

    with caplog.at_level(logging.WARNING):
        assert R.run(args(submit=True, force=False), c, None, tls.NOW) == 0
    assert '1 order(s) still open after' in caplog.text and 'will be booked at the next run' in caplog.text
    assert not any(r.levelno >= logging.ERROR for r in caplog.records if 'still open' in r.getMessage())
    st1 = json.load(open(ml.LIVE_STATE_PATH))
    filled1 = {o.symbol: o.filled_qty for o in b.orders.values() if o.filled_qty}
    assert held[0] not in st1['positions'] and st1['positions'] == pytest.approx(filled1)       # boundary 1
    assert list(st1['pending_coids']) == [o.client_order_id for o in b.orders.values() if o.symbol == held[0]]
    assert len(ledger_rows(ml.LIVE_LEDGER_PATH)) == 19

    late = next(o for o in b.orders.values() if o.symbol == held[0])                  # fills after the timeout
    late.status, late.filled_qty, late.filled_avg_price = 'filled', 4.0, 50.0
    assert R.run(args(submit=False), c, None, tls.NOW) == 0                         # next run (plan) books it first
    st2 = json.load(open(ml.LIVE_STATE_PATH))
    filled2 = {o.symbol: o.filled_qty for o in b.orders.values() if o.filled_qty}
    assert st2['positions'] == pytest.approx(filled2) and st2['positions'][held[0]] == 4.0     # boundary 2
    assert 'pending_coids' not in st2 and late.client_order_id in st2['booked_coids']
    assert st2['cash'] == pytest.approx(st1['cash'] - 4.0 * 50.0)
    rows = ledger_rows(ml.LIVE_LEDGER_PATH)
    assert len(rows) == 20 and sum(r['client_order_id'] == late.client_order_id for r in rows) == 1
    assert R.run(args(submit=False), c, None, tls.NOW) == 0                         # a third run books nothing
    assert len(ledger_rows(ml.LIVE_LEDGER_PATH)) == 20
    assert json.load(open(ml.LIVE_STATE_PATH))['positions'] == st2['positions']


def test_fill_landing_during_the_immediate_recheck_is_booked_in_the_same_run(world, monkeypatch):
    args, asof, tmp = world
    c, b = tls.live_client()
    real_submit = b.submit_order
    pending_orders = []

    def submit(req):
        o = real_submit(req)
        if not pending_orders:
            o.status, o.filled_qty, o.filled_avg_price = 'new', 0.0, 0.0
            pending_orders.append(o)
        return o
    b.submit_order = submit

    real_poll = R.poll_fills

    def poll_then_fill(client, coids, **k):
        res = real_poll(client, coids, timeout=0.0, interval=0.0)
        for o in pending_orders:
            o.status, o.filled_qty, o.filled_avg_price = 'filled', 4.0, 50.0
        return res
    monkeypatch.setattr(R, 'poll_fills', poll_then_fill)
    assert R.run(args(submit=True, force=False), c, None, tls.NOW) == 0
    st = json.load(open(ml.LIVE_STATE_PATH))
    assert st['positions'][pending_orders[0].symbol] == 4.0 and len(st['positions']) == 20 and 'pending_coids' not in st
    assert len(st['booked_coids']) == 20 and len(ledger_rows(ml.LIVE_LEDGER_PATH)) == 20
