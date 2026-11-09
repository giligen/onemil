"""ORB add-on fix 2026-11-09 (BEX incident, docs/orb_add_on_fix_spec_20261109.md):
(1) the 15:00 ET cutoff used UTC+offset instead of the ET clock; (2) the add was booked from the
submit response (paper OMS: 'accepted', filled_qty 0) instead of the order's RESULT."""
from __future__ import annotations

import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database
from trading import orb_add_on, orb_engine
from trading.orb_engine import ORBEngine, OpenPosition
from trading.stop_monitor import StopMonitor, WatchEntry
from tests.test_orb_add_on import orb_cfg, mock_db  # noqa: F401  (fixture re-export)

ENGINE_SRC = Path(orb_engine.__file__)


def _order(status, filled, px=None, qty=87):
    return {'id': 'add-1', 'status': status, 'qty': qty, 'filled_qty': filled, 'filled_avg_price': px}


class _Leg:
    """Mutable broker-side leg qty table so get_order(leg) and replace_order_qty agree."""
    def __init__(self, qty):
        self.qty = {'sl-1': qty, 'tp-1': qty}


def _build(orb_cfg, mock_db, *, add_orders, broker_qty, leg_qty=87, monkeypatch=None):
    """Real engine + real OpenPosition + real StopMonitor watch; Alpaca mocked with spec=."""
    alpaca = MagicMock(spec=AlpacaClient)
    alpaca.get_account_info.return_value = {'buying_power': 1e6}
    alpaca.submit_limit_buy_order.return_value = {'id': 'add-1', 'status': 'accepted', 'filled_qty': '0'}
    legs = _Leg(leg_qty)
    seq = list(add_orders)

    def _get(oid):
        if oid == 'add-1':
            return seq.pop(0) if len(seq) > 1 else seq[0]
        return {'id': oid, 'status': 'new', 'qty': legs.qty[oid], 'filled_qty': 0}

    def _replace(oid, q):
        legs.qty[oid] = q
        return {'id': oid}

    alpaca.get_order.side_effect = _get
    alpaca.replace_order_qty.side_effect = _replace
    alpaca.get_open_positions.return_value = [{'symbol': 'BEX', 'qty': broker_qty}]
    sm = StopMonitor.__new__(StopMonitor)
    sm._watch_lock = threading.Lock()
    sm._watches = {'BEX': WatchEntry(symbol='BEX', stop_price=13.0, shares=87,
                                      tp_leg_id='tp-1', sl_leg_id='sl-1')}
    orb_cfg['strategy']['enabled'] = True
    eng = ORBEngine(alpaca_client=alpaca, db=mock_db, stop_monitor=sm, config=orb_cfg)
    eng.add_on_cfg = {**orb_add_on.DEFAULT_ADD_ON, 'enabled': True, 'at_r': 1.0}
    monkeypatch.setattr(orb_engine.time, 'sleep', lambda s: None)
    monkeypatch.setattr(eng, '_et_now', lambda: datetime(2026, 11, 9, 14, 0, tzinfo=timezone.utc))
    pos = OpenPosition(
        symbol='BEX', entry_price=14.22, stop_price=13.0, shares=87, trade_id=100, order_id='',
        entry_time=datetime.now(timezone.utc), range_high=14.5, range_low=13.5, lock_arm_at_r=1.75,
        lock_stop_r=0.5, composite_score=1.0, quintile='Q3', tp_leg_id='tp-1', sl_leg_id='sl-1')
    return eng, alpaca, sm, pos, legs


BARS = pd.DataFrame({'close': [15.30]})


class TestEtClock:
    def test_cutoff_boundaries(self):
        assert not orb_add_on.is_past_add_cutoff(datetime(2026, 11, 9, 14, 0))
        assert not orb_add_on.is_past_add_cutoff(datetime(2026, 11, 9, 14, 59, 59))
        assert orb_add_on.is_past_add_cutoff(datetime(2026, 11, 9, 15, 0, 0))

    @pytest.mark.parametrize("utc_hm, blocked", [((19, 0), False), ((14, 59), False), ((20, 1), True)])
    def test_engine_passes_et_clock_to_cutoff(self, orb_cfg, mock_db, monkeypatch, utc_hm, blocked):
        """19:00 UTC on 2026-11-09 is 14:00 EST: open. 20:01 UTC is 15:01 EST: blocked."""
        eng, alpaca, _, pos, _ = _build(orb_cfg, mock_db, add_orders=[_order('filled', 87, 15.32)],
                                        broker_qty=174, monkeypatch=monkeypatch)
        fixed = datetime(2026, 11, 9, utc_hm[0], utc_hm[1], 1, tzinfo=timezone.utc)

        class _FakeDT(datetime):
            @classmethod
            def now(cls, tz=None):
                return fixed.astimezone(tz) if tz else fixed

        monkeypatch.setattr(orb_engine, 'datetime', _FakeDT)
        # use the REAL _et_now (patched datetime.now -> ZoneInfo conversion)
        monkeypatch.setattr(eng, '_et_now', ORBEngine._et_now.__get__(eng))
        eng._execute_add_on(pos, 15.30)
        assert (alpaca.submit_limit_buy_order.call_count == 0) is blocked

    def test_source_guard_no_plus_offset_clock(self):
        src = ENGINE_SRC.read_text()
        assert not re.search(r"\+\s*timedelta\(hours=_et_offset_hours", src)


class TestAddSettlesOnResult:
    def test_delayed_paper_fill_books_combined_and_resizes(self, orb_cfg, mock_db, monkeypatch):
        eng, alpaca, sm, pos, legs = _build(
            orb_cfg, mock_db, broker_qty=174, monkeypatch=monkeypatch,
            add_orders=[_order('accepted', 0), _order('accepted', 0), _order('filled', 87, 15.32)])
        eng._maybe_fire_add_on(pos, BARS)
        assert pos.shares == 174
        assert pos.add_count == 1
        assert pos.adds[-1]['px'] == 15.32 and pos.adds[-1]['qty'] == 87
        assert pos.combined_avg_price == pytest.approx((87 * 14.22 + 87 * 15.32) / 174)
        assert legs.qty == {'sl-1': 174, 'tp-1': 174}
        assert sm._watches['BEX'].shares == 174
        alpaca.cancel_order.assert_not_called()
        assert mock_db.update_trade.call_args.args[1]['shares'] == 174

    def test_never_fills_cancels_and_changes_nothing(self, orb_cfg, mock_db, monkeypatch):
        eng, alpaca, sm, pos, legs = _build(
            orb_cfg, mock_db, broker_qty=87, monkeypatch=monkeypatch,
            add_orders=[_order('accepted', 0)] * 25 + [_order('canceled', 0)])
        eng._maybe_fire_add_on(pos, BARS)
        alpaca.cancel_order.assert_called_once_with('add-1')
        assert pos.shares == 87 and pos.add_count == 0 and pos.adds == []
        assert pos.combined_avg_price is None
        assert legs.qty == {'sl-1': 87, 'tp-1': 87}
        assert sm._watches['BEX'].shares == 87
        alpaca.replace_order_qty.assert_not_called()
        assert pos.add_pending_order_id == ''

    def test_partial_fill_books_filled_qty_only(self, orb_cfg, mock_db, monkeypatch):
        eng, alpaca, sm, pos, legs = _build(
            orb_cfg, mock_db, broker_qty=127, monkeypatch=monkeypatch,
            add_orders=[_order('partially_filled', 40, 15.31)] * 25 + [_order('canceled', 40, 15.31)])
        eng._maybe_fire_add_on(pos, BARS)
        alpaca.cancel_order.assert_called_once_with('add-1')
        assert pos.shares == 127 and pos.add_count == 1
        assert pos.adds[-1]['qty'] == 40 and pos.adds[-1]['px'] == 15.31
        assert legs.qty == {'sl-1': 127, 'tp-1': 127}
        assert sm._watches['BEX'].shares == 127

    def test_legs_already_at_target_skip_replace_without_error(
            self, orb_cfg, mock_db, monkeypatch, caplog):
        eng, alpaca, sm, pos, legs = _build(
            orb_cfg, mock_db, broker_qty=174, leg_qty=174, monkeypatch=monkeypatch,
            add_orders=[_order('filled', 87, 15.32)])
        with caplog.at_level('INFO', logger='trading.orb_engine'):
            eng._maybe_fire_add_on(pos, BARS)
        alpaca.replace_order_qty.assert_not_called()
        assert not [r for r in caplog.records if r.levelname == 'ERROR']
        assert any('replace skipped' in r.getMessage() for r in caplog.records)
        assert pos.shares == 174 and sm._watches['BEX'].shares == 174

    def test_broker_short_of_target_is_an_error(self, orb_cfg, mock_db, monkeypatch, caplog):
        eng, alpaca, sm, pos, legs = _build(
            orb_cfg, mock_db, broker_qty=87, monkeypatch=monkeypatch,
            add_orders=[_order('filled', 87, 15.32)])
        with caplog.at_level('WARNING', logger='trading.orb_engine'):
            eng._maybe_fire_add_on(pos, BARS)
        assert any(r.levelname == 'ERROR' and 'broker shows only 87sh' in r.getMessage()
                   for r in caplog.records)
