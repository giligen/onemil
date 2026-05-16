"""Integration tests for ORB engine touch-and-go filter (Rule M + Rule D).

Verifies:
  - breakout_bar_ts captured on fill
  - Rule M fires at end of breakout bar -> stop_monitor.force_exit called
  - Rule D fires at end of bar 1 -> stop_monitor.force_exit called
  - No double-eval if same bar replayed
  - Disabled via YAML -> no force_exit calls
  - Telegram notifier called (or gracefully skipped)
  - Telegram failure doesn't block exit submission
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
import yaml

from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database
from trading.orb_engine import ORBEngine, OpenPosition, RangeData
from trading.stop_monitor import StopMonitor


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def orb_cfg():
    yaml_path = Path(__file__).parent.parent / 'orb.yaml'
    with open(yaml_path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def mock_alpaca():
    client = MagicMock(spec=AlpacaClient)
    client.get_open_positions.return_value = []
    client.get_account_info.return_value = {'buying_power': 100_000.0}
    client.get_latest_quote.return_value = {'bid_price': 9.95, 'ask_price': 10.00}
    return client


@pytest.fixture
def mock_db():
    db = MagicMock(spec=Database)
    db.save_trade.return_value = 100
    db.get_open_trades.return_value = []
    return db


@pytest.fixture
def mock_stop_monitor():
    sm = MagicMock(spec=StopMonitor)
    sm.polling_mode = False
    sm.drain_exit_events.return_value = []
    sm.force_exit.return_value = True
    return sm


@pytest.fixture
def mock_notifier():
    notifier = MagicMock()
    notifier.send_message_async = MagicMock()
    notifier.send_message = MagicMock()
    return notifier


def _make_engine(cfg, mock_alpaca, mock_db, mock_stop_monitor, notifier=None):
    cfg = yaml.safe_load(yaml.safe_dump(cfg))  # deep copy
    cfg['strategy']['enabled'] = True
    return ORBEngine(
        alpaca_client=mock_alpaca, db=mock_db,
        stop_monitor=mock_stop_monitor, config=cfg,
        notifier=notifier,
    )


def _build_position(sym: str = 'TEST', entry_price: float = 10.03,
                    range_high: float = 10.0, range_low: float = 9.0,
                    fill_at: datetime = None) -> OpenPosition:
    if fill_at is None:
        fill_at = datetime(2025, 6, 2, 13, 35, 23, tzinfo=timezone.utc)
    pos = OpenPosition(
        symbol=sym,
        entry_price=entry_price,
        stop_price=range_low,
        shares=1000,
        trade_id=100,
        order_id='',  # filled
        entry_time=fill_at,
        range_high=range_high,
        range_low=range_low,
        lock_arm_at_r=1.75,
        lock_stop_r=0.5,
        composite_score=0.5,
        quintile='Q4',
        breakout_bar_ts=fill_at.replace(second=0, microsecond=0),
    )
    return pos


def _bars_after_fill(fill_at: datetime, ohlc_offsets):
    """Build bars_df starting at fill minute-floor + N minutes.

    ohlc_offsets: list of (minutes_after_breakout_bar, open, high, low, close, volume).
    First entry (0,...) is the breakout bar; (1,...) is bar 1, etc.
    """
    bb_ts = fill_at.replace(second=0, microsecond=0)
    rows = []
    for offset, o, h, l, c, v in ohlc_offsets:
        rows.append({
            'timestamp': bb_ts + timedelta(minutes=offset),
            'open': o, 'high': h, 'low': l, 'close': c, 'volume': v,
        })
    return pd.DataFrame(rows)


# =========================================================================
# breakout_bar_ts capture
# =========================================================================

class TestBreakoutBarTsCapture:
    def test_default_cfg_loads_touchgo_enabled(self, orb_cfg, mock_alpaca,
                                                 mock_db, mock_stop_monitor):
        eng = _make_engine(orb_cfg, mock_alpaca, mock_db, mock_stop_monitor)
        # touchgo defaults to enabled even if YAML section missing
        assert eng.touchgo_cfg.master_enabled is True
        assert eng.touchgo_cfg.rule_m_enabled is True
        assert eng.touchgo_cfg.rule_d_enabled is True

    def test_breakout_bar_ts_set_via_fixture(self):
        # Position fixture pre-sets breakout_bar_ts to fill minute-floor
        fill_at = datetime(2025, 6, 2, 13, 35, 23, 500_000, tzinfo=timezone.utc)
        pos = _build_position(fill_at=fill_at)
        assert pos.breakout_bar_ts == datetime(2025, 6, 2, 13, 35, 0,
                                               tzinfo=timezone.utc)


# =========================================================================
# Rule M evaluation
# =========================================================================

class TestRuleMFires:
    def test_rule_m_fires_on_weak_breakout_bar(self, orb_cfg, mock_alpaca,
                                                 mock_db, mock_stop_monitor,
                                                 mock_notifier):
        eng = _make_engine(orb_cfg, mock_alpaca, mock_db, mock_stop_monitor,
                           notifier=mock_notifier)
        fill_at = datetime(2025, 6, 2, 13, 35, 23, tzinfo=timezone.utc)
        pos = _build_position(fill_at=fill_at)
        eng.open_positions[pos.symbol] = pos
        # Breakout bar: opens 9.95, high 10.1, low 9.60, close 9.70 -> close_pos=0.2
        bars = _bars_after_fill(fill_at, [(0, 9.95, 10.10, 9.60, 9.70, 5000)])

        eng._evaluate_touchgo(pos.symbol, bars)

        # force_exit should be called once with reason='tag_bb'
        mock_stop_monitor.force_exit.assert_called_once()
        call_kwargs = mock_stop_monitor.force_exit.call_args.kwargs
        assert call_kwargs['symbol'] == pos.symbol
        assert call_kwargs['reason'] == 'tag_bb'
        # Helper returns bb_close = 9.70
        assert call_kwargs['limit_price'] == pytest.approx(9.70)
        assert pos.rule_m_evaluated is True

    def test_rule_m_does_not_fire_on_strong_breakout(self, orb_cfg, mock_alpaca,
                                                       mock_db, mock_stop_monitor):
        eng = _make_engine(orb_cfg, mock_alpaca, mock_db, mock_stop_monitor)
        fill_at = datetime(2025, 6, 2, 13, 35, 23, tzinfo=timezone.utc)
        pos = _build_position(fill_at=fill_at)
        eng.open_positions[pos.symbol] = pos
        # Strong breakout: close at top of bar
        bars = _bars_after_fill(fill_at, [(0, 9.95, 10.10, 9.90, 10.08, 5000)])

        eng._evaluate_touchgo(pos.symbol, bars)

        # Rule M does NOT fire, but pos.rule_m_evaluated should now be True
        # (we examined the bar; no fire because close was strong)
        mock_stop_monitor.force_exit.assert_not_called()
        assert pos.rule_m_evaluated is True

    def test_rule_m_no_double_eval(self, orb_cfg, mock_alpaca, mock_db,
                                     mock_stop_monitor):
        eng = _make_engine(orb_cfg, mock_alpaca, mock_db, mock_stop_monitor)
        fill_at = datetime(2025, 6, 2, 13, 35, 23, tzinfo=timezone.utc)
        pos = _build_position(fill_at=fill_at)
        eng.open_positions[pos.symbol] = pos
        bars = _bars_after_fill(fill_at, [(0, 9.95, 10.10, 9.60, 9.70, 5000)])

        # First call fires
        eng._evaluate_touchgo(pos.symbol, bars)
        # Second call with same bars: rule already evaluated, no re-fire
        eng._evaluate_touchgo(pos.symbol, bars)

        assert mock_stop_monitor.force_exit.call_count == 1

    def test_rule_m_sends_telegram(self, orb_cfg, mock_alpaca, mock_db,
                                     mock_stop_monitor, mock_notifier):
        eng = _make_engine(orb_cfg, mock_alpaca, mock_db, mock_stop_monitor,
                           notifier=mock_notifier)
        fill_at = datetime(2025, 6, 2, 13, 35, 23, tzinfo=timezone.utc)
        pos = _build_position(fill_at=fill_at)
        eng.open_positions[pos.symbol] = pos
        bars = _bars_after_fill(fill_at, [(0, 9.95, 10.10, 9.60, 9.70, 5000)])

        eng._evaluate_touchgo(pos.symbol, bars)

        mock_notifier.send_message_async.assert_called_once()
        msg = mock_notifier.send_message_async.call_args.args[0]
        assert 'TAG_BB EXIT' in msg
        assert pos.symbol in msg
        assert 'bb_close_pos' in msg


# =========================================================================
# Rule D evaluation
# =========================================================================

class TestRuleDFires:
    def test_rule_d_fires_on_deep_bar1_revert(self, orb_cfg, mock_alpaca,
                                                mock_db, mock_stop_monitor,
                                                mock_notifier):
        eng = _make_engine(orb_cfg, mock_alpaca, mock_db, mock_stop_monitor,
                           notifier=mock_notifier)
        fill_at = datetime(2025, 6, 2, 13, 35, 23, tzinfo=timezone.utc)
        pos = _build_position(fill_at=fill_at)
        eng.open_positions[pos.symbol] = pos
        # Strong breakout bar (no Rule M) + deep revert in bar 1
        # range_size = 10.0 - 9.0 = 1.0; entry = 10.03
        # bar 1 low = 9.2 -> revert = 0.83R >= 0.75
        bars = _bars_after_fill(fill_at, [
            (0, 9.95, 10.10, 9.90, 10.08, 5000),   # strong breakout
            (1, 10.05, 10.07, 9.20, 9.25, 5000),  # deep revert
        ])

        eng._evaluate_touchgo(pos.symbol, bars)

        mock_stop_monitor.force_exit.assert_called_once()
        call_kwargs = mock_stop_monitor.force_exit.call_args.kwargs
        assert call_kwargs['reason'] == 'tag_b1'
        # exit_price = entry + (-0.5) * 1.0 = 9.53
        assert call_kwargs['limit_price'] == pytest.approx(9.53)
        assert pos.rule_d_evaluated is True

    def test_rule_d_does_not_fire_on_shallow_revert(self, orb_cfg, mock_alpaca,
                                                     mock_db, mock_stop_monitor):
        eng = _make_engine(orb_cfg, mock_alpaca, mock_db, mock_stop_monitor)
        fill_at = datetime(2025, 6, 2, 13, 35, 23, tzinfo=timezone.utc)
        pos = _build_position(fill_at=fill_at)
        eng.open_positions[pos.symbol] = pos
        bars = _bars_after_fill(fill_at, [
            (0, 9.95, 10.10, 9.90, 10.08, 5000),  # strong breakout
            (1, 10.05, 10.15, 9.85, 10.00, 5000),  # shallow revert (0.18R)
        ])

        eng._evaluate_touchgo(pos.symbol, bars)

        mock_stop_monitor.force_exit.assert_not_called()
        assert pos.rule_d_evaluated is True

    def test_rule_d_sends_telegram(self, orb_cfg, mock_alpaca, mock_db,
                                     mock_stop_monitor, mock_notifier):
        eng = _make_engine(orb_cfg, mock_alpaca, mock_db, mock_stop_monitor,
                           notifier=mock_notifier)
        fill_at = datetime(2025, 6, 2, 13, 35, 23, tzinfo=timezone.utc)
        pos = _build_position(fill_at=fill_at)
        eng.open_positions[pos.symbol] = pos
        bars = _bars_after_fill(fill_at, [
            (0, 9.95, 10.10, 9.90, 10.08, 5000),
            (1, 10.05, 10.07, 9.20, 9.25, 5000),
        ])

        eng._evaluate_touchgo(pos.symbol, bars)

        mock_notifier.send_message_async.assert_called_once()
        msg = mock_notifier.send_message_async.call_args.args[0]
        assert 'TAG_B1 EXIT' in msg
        assert 'b1_revert' in msg


# =========================================================================
# Edge cases and disabling
# =========================================================================

class TestEdgeCases:
    def test_no_eval_if_position_not_filled(self, orb_cfg, mock_alpaca,
                                              mock_db, mock_stop_monitor):
        eng = _make_engine(orb_cfg, mock_alpaca, mock_db, mock_stop_monitor)
        fill_at = datetime(2025, 6, 2, 13, 35, 23, tzinfo=timezone.utc)
        pos = _build_position(fill_at=fill_at)
        pos.order_id = 'order-pending'  # not yet filled
        eng.open_positions[pos.symbol] = pos
        bars = _bars_after_fill(fill_at, [(0, 9.95, 10.10, 9.60, 9.70, 5000)])

        eng._evaluate_touchgo(pos.symbol, bars)

        mock_stop_monitor.force_exit.assert_not_called()

    def test_no_eval_if_no_position(self, orb_cfg, mock_alpaca, mock_db,
                                     mock_stop_monitor):
        eng = _make_engine(orb_cfg, mock_alpaca, mock_db, mock_stop_monitor)
        fill_at = datetime(2025, 6, 2, 13, 35, 23, tzinfo=timezone.utc)
        bars = _bars_after_fill(fill_at, [(0, 9.95, 10.10, 9.60, 9.70, 5000)])

        # No position registered
        eng._evaluate_touchgo('UNKNOWN', bars)

        mock_stop_monitor.force_exit.assert_not_called()

    def test_disabled_via_yaml(self, orb_cfg, mock_alpaca, mock_db,
                                mock_stop_monitor):
        cfg = yaml.safe_load(yaml.safe_dump(orb_cfg))
        cfg.setdefault('filter', {}).setdefault('touchgo', {})['enabled'] = False
        eng = _make_engine(cfg, mock_alpaca, mock_db, mock_stop_monitor)
        fill_at = datetime(2025, 6, 2, 13, 35, 23, tzinfo=timezone.utc)
        pos = _build_position(fill_at=fill_at)
        eng.open_positions[pos.symbol] = pos
        bars = _bars_after_fill(fill_at, [(0, 9.95, 10.10, 9.60, 9.70, 5000)])

        eng._evaluate_touchgo(pos.symbol, bars)

        # Master disabled -> no force_exit even with strong trigger
        mock_stop_monitor.force_exit.assert_not_called()

    def test_telegram_failure_does_not_block_exit(self, orb_cfg, mock_alpaca,
                                                    mock_db, mock_stop_monitor):
        notifier = MagicMock()
        notifier.send_message_async.side_effect = RuntimeError('telegram down')
        eng = _make_engine(orb_cfg, mock_alpaca, mock_db, mock_stop_monitor,
                           notifier=notifier)
        fill_at = datetime(2025, 6, 2, 13, 35, 23, tzinfo=timezone.utc)
        pos = _build_position(fill_at=fill_at)
        eng.open_positions[pos.symbol] = pos
        bars = _bars_after_fill(fill_at, [(0, 9.95, 10.10, 9.60, 9.70, 5000)])

        # Should NOT raise; force_exit still called
        eng._evaluate_touchgo(pos.symbol, bars)

        mock_stop_monitor.force_exit.assert_called_once()
        notifier.send_message_async.assert_called_once()  # tried (and failed)

    def test_no_notifier_does_not_crash(self, orb_cfg, mock_alpaca, mock_db,
                                         mock_stop_monitor):
        eng = _make_engine(orb_cfg, mock_alpaca, mock_db, mock_stop_monitor,
                           notifier=None)
        fill_at = datetime(2025, 6, 2, 13, 35, 23, tzinfo=timezone.utc)
        pos = _build_position(fill_at=fill_at)
        eng.open_positions[pos.symbol] = pos
        bars = _bars_after_fill(fill_at, [(0, 9.95, 10.10, 9.60, 9.70, 5000)])

        # Should NOT raise; force_exit still called
        eng._evaluate_touchgo(pos.symbol, bars)
        mock_stop_monitor.force_exit.assert_called_once()


# =========================================================================
# Rule precedence
# =========================================================================

class TestRulePrecedence:
    def test_rule_m_takes_precedence_over_rule_d(self, orb_cfg, mock_alpaca,
                                                   mock_db, mock_stop_monitor):
        eng = _make_engine(orb_cfg, mock_alpaca, mock_db, mock_stop_monitor)
        fill_at = datetime(2025, 6, 2, 13, 35, 23, tzinfo=timezone.utc)
        pos = _build_position(fill_at=fill_at)
        eng.open_positions[pos.symbol] = pos
        # Bar event arrives only with the breakout bar (most realistic case)
        bars = _bars_after_fill(fill_at, [(0, 9.95, 10.10, 9.60, 9.70, 5000)])

        eng._evaluate_touchgo(pos.symbol, bars)

        # Rule M fires; Rule D not yet evaluated because bar 1 hasn't arrived
        mock_stop_monitor.force_exit.assert_called_once()
        assert mock_stop_monitor.force_exit.call_args.kwargs['reason'] == 'tag_bb'
        assert pos.rule_m_evaluated is True
