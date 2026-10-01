"""eod_exit_mode — shared EOD exit pricing (trading/eod_exit.py) + engine wiring.

Deliverable A (docs/eod_exit_modes_20260928.md): config key `eod_exit_mode`,
values 'market' (default) / 'limit_then_market' / 'moc', wired into both the
ORB and HOD engines' EOD paths. flag off (key absent, as in the real
orb.yaml / hod_break cfg today) MUST be byte-identical to pre-flag behaviour.

Deliverable B: the HOD engine's dry/paper exits now carry the same
exit_pricing_method / exit_quote_bid / exit_quote_ask / exit_fill_latency_ms /
exit_slippage telemetry the ORB StopMonitor path already writes (previously
discarded in _drain_stop_monitor_exits, and never computed at all for
bracket target/stop legs or the EOD leg).
"""
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, ANY
from zoneinfo import ZoneInfo

import pytest

from trading import eod_exit as eod
from trading.hod_break_engine import HodBreakEngine, Position
from trading.orb_engine import ORBEngine, OpenPosition
from trading.stop_monitor import StopExitEvent
from data_sources.alpaca_client import AlpacaClient

from tests.test_hod_break_engine import (
    cfg as hod_cfg, mock_alpaca as hod_mock_alpaca, mock_db as hod_mock_db,
    mock_sm as hod_mock_sm, trades_db,
)
from tests.test_orb_engine import (
    orb_cfg, mock_alpaca as orb_mock_alpaca, mock_db as orb_mock_db,
    mock_stop_monitor as orb_mock_sm,
)


# =========================================================================
# Pure logic — trading/eod_exit.py (no engine, no clock dependency)
# =========================================================================

ET = ZoneInfo('America/New_York')


class TestResolveMode:
    def test_default_market_no_warning(self):
        r = eod.resolve_mode('market', datetime(2026, 9, 28, 14, 0, tzinfo=ET))
        assert r.mode == 'market' and r.warning is None and not r.moc_cutoff_fallback

    def test_unknown_mode_falls_back_to_market_with_warning(self):
        r = eod.resolve_mode('bogus', datetime(2026, 9, 28, 14, 0, tzinfo=ET))
        assert r.mode == 'market'
        assert r.warning is not None and 'bogus' in r.warning

    def test_moc_before_cutoff_stays_moc(self):
        r = eod.resolve_mode('moc', datetime(2026, 9, 28, 15, 49, tzinfo=ET))
        assert r.mode == 'moc' and r.warning is None and not r.moc_cutoff_fallback

    def test_moc_at_cutoff_falls_back_with_warning(self):
        r = eod.resolve_mode('moc', datetime(2026, 9, 28, 15, 50, tzinfo=ET))
        assert r.mode == 'limit_then_market' and r.moc_cutoff_fallback
        assert r.warning is not None and '15:50' in r.warning

    def test_moc_after_cutoff_falls_back(self):
        r = eod.resolve_mode('moc', datetime(2026, 9, 28, 15, 55, tzinfo=ET))
        assert r.mode == 'limit_then_market' and r.moc_cutoff_fallback

    def test_limit_then_market_unaffected_by_clock(self):
        r = eod.resolve_mode('limit_then_market', datetime(2026, 9, 28, 15, 59, tzinfo=ET))
        assert r.mode == 'limit_then_market' and not r.moc_cutoff_fallback and r.warning is None


class TestQuoteMid:
    def test_normal(self):
        assert eod.quote_mid(10.0, 10.10) == pytest.approx(10.05)

    def test_missing_side_returns_none(self):
        assert eod.quote_mid(0.0, 10.10) is None
        assert eod.quote_mid(10.0, 0.0) is None

    def test_crossed_quote_returns_none(self):
        assert eod.quote_mid(10.10, 10.0) is None


class TestPricingMethod:
    def test_market(self):
        r = eod.resolve_mode('market', datetime(2026, 9, 28, 14, 0, tzinfo=ET))
        assert eod.pricing_method(r, 'primary') == eod.PM_EOD_MARKET

    def test_moc(self):
        r = eod.resolve_mode('moc', datetime(2026, 9, 28, 14, 0, tzinfo=ET))
        assert eod.pricing_method(r, 'primary') == eod.PM_EOD_MOC

    def test_limit_then_market_primary_and_fallback(self):
        r = eod.resolve_mode('limit_then_market', datetime(2026, 9, 28, 14, 0, tzinfo=ET))
        assert eod.pricing_method(r, 'primary') == eod.PM_EOD_LIMIT
        assert eod.pricing_method(r, 'fallback') == eod.PM_EOD_LIMIT_MKT_FALLBACK

    def test_moc_cutoff_fallback_tags_distinctly_from_direct_limit_then_market(self):
        r = eod.resolve_mode('moc', datetime(2026, 9, 28, 15, 55, tzinfo=ET))
        assert eod.pricing_method(r, 'primary') == eod.PM_EOD_MOC_CUTOFF_LIMIT
        assert eod.pricing_method(r, 'fallback') == eod.PM_EOD_MOC_CUTOFF_MKT_FB


class TestBuildTelemetry:
    def test_full(self):
        out = eod.build_eod_exit_telemetry(
            method='eod_limit_mid', quote_bid=10.0, quote_ask=10.10, exit_price=10.03,
            submitted_at_epoch=100.0, filled_at_epoch=100.5, reference_price=10.05,
        )
        assert out['exit_pricing_method'] == 'eod_limit_mid'
        assert out['exit_quote_bid'] == 10.0 and out['exit_quote_ask'] == 10.10
        assert out['exit_price'] == 10.03
        assert out['exit_fill_latency_ms'] == pytest.approx(500.0)
        assert out['exit_slippage'] == pytest.approx(0.02)

    def test_zero_quote_becomes_none_and_missing_fields_omitted(self):
        out = eod.build_eod_exit_telemetry(method='eod_market')
        assert out['exit_quote_bid'] is None and out['exit_quote_ask'] is None
        assert 'exit_price' not in out and 'exit_fill_latency_ms' not in out and 'exit_slippage' not in out


class TestAlpacaClientEodOrderTypes:
    """order type / time-in-force per mode, at the SDK boundary."""

    def _client(self):
        c = AlpacaClient.__new__(AlpacaClient)  # bypass __init__ (no real Alpaca creds)
        c.trading_client = MagicMock()
        c._call_with_timeout = lambda fn, label: fn()
        return c

    def test_market_sell_uses_time_in_force_day(self):
        from alpaca.trading.enums import TimeInForce, OrderSide
        c = self._client()
        c.trading_client.submit_order.side_effect = lambda req: MagicMock(id='o1', status=MagicMock(value='accepted'))
        c.submit_market_sell_order('ABC', 50)
        req = c.trading_client.submit_order.call_args[0][0]
        assert req.time_in_force == TimeInForce.DAY
        assert req.side == OrderSide.SELL
        assert req.qty == 50

    def test_moc_sell_uses_time_in_force_cls(self):
        from alpaca.trading.enums import TimeInForce, OrderSide
        c = self._client()
        c.trading_client.submit_order.side_effect = lambda req: MagicMock(id='o2', status=MagicMock(value='accepted'))
        c.submit_moc_sell_order('ABC', 50)
        req = c.trading_client.submit_order.call_args[0][0]
        assert req.time_in_force == TimeInForce.CLS
        assert req.side == OrderSide.SELL
        assert req.qty == 50


# =========================================================================
# ORB engine wiring
# =========================================================================

def _open_orb_position(engine, sym='TSLA', shares=100, trade_id=1):
    engine.open_positions[sym] = OpenPosition(
        symbol=sym, entry_price=10.5, stop_price=10.0, shares=shares,
        trade_id=trade_id, order_id='', entry_time=datetime.now(timezone.utc),
        range_high=10.5, range_low=10.0, lock_arm_at_r=1.5, lock_stop_r=1.0,
        composite_score=0.5, quintile='Q4',
    )


def _make_orb_engine(orb_cfg, orb_mock_alpaca, orb_mock_db, orb_mock_sm):
    orb_cfg['strategy']['enabled'] = True
    return ORBEngine(alpaca_client=orb_mock_alpaca, db=orb_mock_db, stop_monitor=orb_mock_sm, config=orb_cfg)


class TestOrbEodExitFlagOff:
    def test_default_mode_is_market(self, orb_cfg, orb_mock_alpaca, orb_mock_db, orb_mock_sm):
        e = _make_orb_engine(orb_cfg, orb_mock_alpaca, orb_mock_db, orb_mock_sm)
        assert e.eod_exit_mode == 'market'
        assert e.eod_limit_timeout_s == 20.0

    def test_market_mode_calls_close_position_only(self, orb_cfg, orb_mock_alpaca, orb_mock_db, orb_mock_sm):
        e = _make_orb_engine(orb_cfg, orb_mock_alpaca, orb_mock_db, orb_mock_sm)
        _open_orb_position(e)
        result = e._close_position_with_held_qty_retry('TSLA')
        orb_mock_alpaca.close_position.assert_called_once_with('TSLA')
        orb_mock_alpaca.submit_limit_sell_order.assert_not_called()
        orb_mock_alpaca.submit_market_sell_order.assert_not_called()
        orb_mock_alpaca.submit_moc_sell_order.assert_not_called()
        assert result == {'id': 'close-1', 'status': 'accepted'}


class TestOrbEodExitModes:
    def test_moc_mode_submits_cls_order_and_records_telemetry(self, orb_cfg, orb_mock_alpaca, orb_mock_db, orb_mock_sm):
        orb_cfg['strategy']['enabled'] = True
        orb_cfg.setdefault('exit', {})['eod_exit_mode'] = 'moc'
        orb_mock_alpaca.submit_moc_sell_order.return_value = {'id': 'moc-1', 'status': 'accepted'}
        e = ORBEngine(alpaca_client=orb_mock_alpaca, db=orb_mock_db, stop_monitor=orb_mock_sm, config=orb_cfg)
        _open_orb_position(e)
        morning = datetime(2026, 9, 29, 14, 0, tzinfo=ET)
        with patch('trading.orb_engine.datetime') as mock_dt:
            mock_dt.now.return_value = morning
            result = e._close_position_with_held_qty_retry('TSLA')
        orb_mock_alpaca.submit_moc_sell_order.assert_called_once_with('TSLA', 100)
        orb_mock_alpaca.close_position.assert_not_called()
        assert result == {'id': 'moc-1', 'status': 'accepted'}
        found = [c for c in orb_mock_db.update_trade.call_args_list if c.args[1].get('exit_pricing_method') == 'eod_moc']
        assert found, orb_mock_db.update_trade.call_args_list

    def test_limit_then_market_timeout_escalates_to_market(self, orb_cfg, orb_mock_alpaca, orb_mock_db, orb_mock_sm):
        orb_cfg['strategy']['enabled'] = True
        orb_cfg.setdefault('exit', {})['eod_exit_mode'] = 'limit_then_market'
        orb_cfg['exit']['eod_limit_timeout_s'] = 0.01
        orb_mock_alpaca.submit_limit_sell_order.return_value = {'id': 'lim-1', 'status': 'accepted'}
        orb_mock_alpaca.get_order.return_value = {'status': 'accepted', 'filled_qty': 0}
        orb_mock_alpaca.submit_market_sell_order.return_value = {'id': 'mkt-1', 'status': 'accepted'}
        e = ORBEngine(alpaca_client=orb_mock_alpaca, db=orb_mock_db, stop_monitor=orb_mock_sm, config=orb_cfg)
        _open_orb_position(e)
        result = e._close_position_with_held_qty_retry('TSLA')
        orb_mock_alpaca.submit_limit_sell_order.assert_called_once_with('TSLA', 100, pytest.approx(9.975))
        orb_mock_alpaca.cancel_order.assert_called_once_with('lim-1')
        orb_mock_alpaca.submit_market_sell_order.assert_called_once_with('TSLA', 100)
        assert result == {'id': 'mkt-1', 'status': 'accepted'}


# =========================================================================
# HOD engine wiring (deliverable A: force_close_all) + telemetry (deliverable B)
# =========================================================================

def _hod_position(sym='ABC', trade_id=7, shares=100, stop=9.80, target=11.0, close_submitted_at=None):
    return Position(
        symbol=sym, trade_id=trade_id, order_id='', shares=shares, limit_price=10.50,
        stop=stop, target=target, level=10.60, submitted_at=datetime.now(timezone.utc),
        fill_price=10.50, status='open', close_submitted_at=close_submitted_at,
    )


class TestHodEodExitFlagOff:
    def test_default_mode_is_market(self, hod_mock_alpaca, hod_mock_db, hod_mock_sm):
        e = HodBreakEngine(hod_mock_alpaca, hod_mock_db, hod_mock_sm, cfg=hod_cfg())
        assert e.eod_exit_mode == 'market'
        assert e.eod_limit_timeout_s == 20.0

    def test_market_mode_force_close_matches_pre_flag_formula(self, hod_mock_alpaca, hod_mock_db, hod_mock_sm):
        e = HodBreakEngine(hod_mock_alpaca, hod_mock_db, hod_mock_sm, cfg=hod_cfg())
        e._roll_session()
        pos = _hod_position()
        e.positions['ABC'] = pos
        hod_mock_alpaca.get_open_positions.return_value = [{'symbol': 'ABC', 'qty': pos.shares}]   # broker-truth guard: broker holds it
        hod_mock_alpaca.submit_limit_sell_order.return_value = {'id': 'fc-1', 'status': 'accepted'}
        with patch.object(HodBreakEngine, '_minute_of_day', return_value=955):
            e.force_close_all()
        hod_mock_alpaca.submit_limit_sell_order.assert_called_once()
        args, kwargs = hod_mock_alpaca.submit_limit_sell_order.call_args
        assert args[0] == 'ABC' and args[1] == 100
        assert args[2] == pytest.approx(round(11.00 * 0.99, 2))  # mock quote bid=11.00 -> _close_reference_price uses q[0]=bid
        hod_mock_alpaca.submit_market_sell_order.assert_not_called()
        hod_mock_alpaca.submit_moc_sell_order.assert_not_called()


class TestHodEodExitModes:
    def test_moc_mode_submits_cls_order(self, hod_mock_alpaca, hod_mock_db, hod_mock_sm):
        c = hod_cfg(eod_exit_mode='moc')
        e = HodBreakEngine(hod_mock_alpaca, hod_mock_db, hod_mock_sm, cfg=c)
        e._roll_session()
        e.positions['ABC'] = _hod_position()
        hod_mock_alpaca.get_open_positions.return_value = [{'symbol': 'ABC', 'qty': 100}]   # broker-truth guard: broker holds it
        hod_mock_alpaca.submit_moc_sell_order.return_value = {'id': 'moc-1', 'status': 'accepted'}
        morning = datetime(2026, 9, 29, 14, 0, tzinfo=ET)
        with patch('trading.hod_break_engine.datetime') as mock_dt, \
                patch.object(HodBreakEngine, '_minute_of_day', return_value=955):
            mock_dt.now.side_effect = lambda tz=None: morning if tz is ET else datetime.now(tz)
            e.force_close_all()
        hod_mock_alpaca.submit_moc_sell_order.assert_called_once_with('ABC', 100)
        assert e.positions['ABC'].eod_pricing_method == eod.PM_EOD_MOC


class TestHodExitTelemetry:
    """Deliverable B: HOD exits now record the ORB-parity telemetry columns."""

    def test_stop_monitor_routed_exit_records_telemetry(self, hod_mock_alpaca, hod_mock_db, hod_mock_sm):
        e = HodBreakEngine(hod_mock_alpaca, hod_mock_db, hod_mock_sm, cfg=hod_cfg())
        e._roll_session()
        pos = _hod_position()
        e.positions['ABC'] = pos
        ev = StopExitEvent(
            symbol='ABC', stop_price=9.80, exit_price=9.78, shares=100, order_id='sm-1',
            exit_reason='stop_loss', pricing_method='quote_tight', exit_quote_bid=9.77,
            exit_quote_ask=9.80, exit_limit_price=9.79, submitted_at=time.time() - 0.25,
        )
        hod_mock_sm.drain_exit_events.return_value = [ev]
        e._drain_stop_monitor_exits()
        hod_mock_db.update_trade.assert_called_once()
        _, upd = hod_mock_db.update_trade.call_args[0]
        assert upd['exit_pricing_method'] == 'quote_tight'
        assert upd['exit_quote_bid'] == 9.77 and upd['exit_quote_ask'] == 9.80
        assert upd['exit_fill_latency_ms'] > 0
        assert upd['exit_slippage'] == pytest.approx(9.79 - 9.78, abs=1e-6)
        assert 'ABC' not in e.positions

    def test_bracket_stop_leg_records_pricing_method_and_slippage(self, hod_mock_alpaca, hod_mock_db, hod_mock_sm):
        e = HodBreakEngine(hod_mock_alpaca, hod_mock_db, hod_mock_sm, cfg=hod_cfg())
        e._roll_session()
        pos = _hod_position(stop=9.80)
        pos.sl_leg_id = 'sl1'
        e.positions['ABC'] = pos
        e._book_leg_fill(pos, 'sl1', {'filled_qty': 100, 'filled_avg_price': 9.75}, 'stop')
        hod_mock_db.update_trade.assert_called_once()
        _, upd = hod_mock_db.update_trade.call_args[0]
        assert upd['exit_pricing_method'] == 'bracket_stop_limit'
        assert upd['exit_slippage'] == pytest.approx(9.80 - 9.75)

    def test_eod_leg_records_pricing_method_from_force_close_submission(self, hod_mock_alpaca, hod_mock_db, hod_mock_sm):
        e = HodBreakEngine(hod_mock_alpaca, hod_mock_db, hod_mock_sm, cfg=hod_cfg())
        e._roll_session()
        pos = _hod_position(close_submitted_at=datetime.now(timezone.utc))
        pos.close_order_id = 'fc1'
        pos.eod_pricing_method = eod.PM_EOD_MARKET
        pos.eod_quote_bid, pos.eod_quote_ask = 10.95, 11.00
        e.positions['ABC'] = pos
        e._book_leg_fill(pos, 'fc1', {'filled_qty': 100, 'filled_avg_price': 10.94}, 'eod')
        calls = [c for c in hod_mock_db.update_trade.call_args_list if c.args[1].get('exit_pricing_method') == eod.PM_EOD_MARKET]
        assert calls, hod_mock_db.update_trade.call_args_list
        upd = calls[-1].args[1]
        assert upd['exit_quote_bid'] == 10.95 and upd['exit_quote_ask'] == 11.00
        assert 'exit_fill_latency_ms' in upd
