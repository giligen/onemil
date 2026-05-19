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

        # _notify forwards to send_message; MagicMock has both attributes so
        # the route can be either. Assert that ONE of them was called with
        # a TAG_BB-shaped message.
        sent = (
            mock_notifier.send_message_async.call_args_list
            + mock_notifier.send_message.call_args_list
        )
        assert len(sent) == 1, (
            "exactly one Telegram send expected; "
            f"send_message_async={mock_notifier.send_message_async.call_args_list} "
            f"send_message={mock_notifier.send_message.call_args_list}"
        )
        msg = sent[0].args[0]
        assert 'TAG_BB EXIT' in msg
        assert pos.symbol in msg
        assert 'bb_close_pos' in msg

    def test_rule_m_telegram_works_with_real_async_notifier_api(
        self, orb_cfg, mock_alpaca, mock_db, mock_stop_monitor,
    ):
        """Regression: production TelegramNotifier.send_message is an async
        coroutine function. Calling it without await creates a never-awaited
        coroutine warning AND silently drops the alert. This was the
        2026-05-19 prod bug. Fix: route via self._notify which detects
        and runs the coroutine. This test simulates the real API shape:
        only send_message exists (no send_message_async), AND it's a
        coroutine function.
        """
        import warnings
        from unittest.mock import MagicMock, AsyncMock

        # Real notifier shape: send_message is async; no send_message_async.
        # Using AsyncMock to mimic coroutine return.
        async_notifier = MagicMock(spec=['send_message'])
        async_notifier.send_message = AsyncMock(return_value=True)

        eng = _make_engine(orb_cfg, mock_alpaca, mock_db, mock_stop_monitor,
                           notifier=async_notifier)
        fill_at = datetime(2025, 6, 2, 13, 35, 23, tzinfo=timezone.utc)
        pos = _build_position(fill_at=fill_at)
        eng.open_positions[pos.symbol] = pos
        bars = _bars_after_fill(fill_at, [(0, 9.95, 10.10, 9.60, 9.70, 5000)])

        # Capture warnings so we can fail on RuntimeWarning(coroutine never awaited)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            eng._evaluate_touchgo(pos.symbol, bars)
            # No "coroutine was never awaited" warning means the fix
            # routed the call through self._notify which awaited the coroutine.
            unawaited = [
                w for w in caught
                if issubclass(w.category, RuntimeWarning)
                and 'never awaited' in str(w.message)
            ]
            assert not unawaited, (
                f"coroutine was created but never awaited — Telegram alert "
                f"silently dropped: {[str(w.message) for w in unawaited]}"
            )

        # And the async send_message was actually invoked
        async_notifier.send_message.assert_called_once()
        msg = async_notifier.send_message.call_args.args[0]
        assert 'TAG_BB EXIT' in msg


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

        # Routed via _notify which calls notifier.send_message (or async equiv).
        sent = (
            mock_notifier.send_message_async.call_args_list
            + mock_notifier.send_message.call_args_list
        )
        assert len(sent) == 1
        msg = sent[0].args[0]
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
        # _notify reaches send_message (sync MagicMock returns a MagicMock,
        # not a coroutine, so we don't trigger the async path). Make it
        # raise.
        notifier.send_message.side_effect = RuntimeError('telegram down')
        eng = _make_engine(orb_cfg, mock_alpaca, mock_db, mock_stop_monitor,
                           notifier=notifier)
        fill_at = datetime(2025, 6, 2, 13, 35, 23, tzinfo=timezone.utc)
        pos = _build_position(fill_at=fill_at)
        eng.open_positions[pos.symbol] = pos
        bars = _bars_after_fill(fill_at, [(0, 9.95, 10.10, 9.60, 9.70, 5000)])

        # Should NOT raise; force_exit still called
        eng._evaluate_touchgo(pos.symbol, bars)

        mock_stop_monitor.force_exit.assert_called_once()
        notifier.send_message.assert_called_once()  # tried (and failed)

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
        # Bar event arrives with BOTH breakout bar AND bar 1, where each
        # would independently fire Rule M and Rule D. Rule M must win.
        bars = _bars_after_fill(fill_at, [
            (0, 9.95, 10.10, 9.60, 9.70, 5000),   # Rule M: weak breakout
            (1, 9.70, 9.75, 9.20, 9.25, 5000),    # Rule D: 0.83R revert
        ])

        eng._evaluate_touchgo(pos.symbol, bars)

        # Rule M fires; Rule D not evaluated (we returned before checking it)
        mock_stop_monitor.force_exit.assert_called_once()
        assert mock_stop_monitor.force_exit.call_args.kwargs['reason'] == 'tag_bb'
        assert pos.rule_m_evaluated is True
        assert pos.rule_d_evaluated is False  # short-circuited after M fired


class TestLateBarArrival:
    """Regression: if the engine first sees a bar event where last_ts is at
    or after b1_ts (e.g., engine restart mid-trade, WebSocket reconnect
    catching up multiple bars at once, or batched bar delivery), Rule M
    must STILL evaluate against the breakout bar present in bars_df.

    Pre-fix: outer guard `last_ts < b1_ts` skipped Rule M permanently.
    Post-fix: only gated on `last_ts >= bb_ts`; bb_row lookup finds the
    breakout bar in the rolling window regardless of last_ts.
    """

    def test_rule_m_fires_when_first_event_has_both_bars(self, orb_cfg,
                                                          mock_alpaca, mock_db,
                                                          mock_stop_monitor):
        eng = _make_engine(orb_cfg, mock_alpaca, mock_db, mock_stop_monitor)
        fill_at = datetime(2025, 6, 2, 13, 35, 23, tzinfo=timezone.utc)
        pos = _build_position(fill_at=fill_at)
        eng.open_positions[pos.symbol] = pos
        # Simulated late delivery: first event the engine sees contains BOTH
        # the breakout bar and bar 1. Breakout bar is weak; bar 1 is mild.
        # Pre-fix: Rule M would be skipped because last_ts == b1_ts is NOT
        # < b1_ts. Post-fix: Rule M evaluates first and fires.
        bars = _bars_after_fill(fill_at, [
            (0, 9.95, 10.10, 9.60, 9.70, 5000),   # weak breakout bar
            (1, 9.70, 9.85, 9.65, 9.80, 5000),    # no Rule D trigger (0.23R)
        ])

        eng._evaluate_touchgo(pos.symbol, bars)

        # Rule M MUST fire on the breakout bar even though last_ts == b1_ts
        mock_stop_monitor.force_exit.assert_called_once()
        assert mock_stop_monitor.force_exit.call_args.kwargs['reason'] == 'tag_bb'
        assert pos.rule_m_evaluated is True

    def test_rule_m_fires_after_restart_when_last_ts_past_b1(self, orb_cfg,
                                                                mock_alpaca,
                                                                mock_db,
                                                                mock_stop_monitor):
        """Restart scenario: engine wakes up minutes after fill, first bar
        event contains breakout_bar + bar1 + bar2. Strong breakout, no
        Rule D trigger — Rule M must STILL evaluate (and not fire because
        the close was strong).
        """
        eng = _make_engine(orb_cfg, mock_alpaca, mock_db, mock_stop_monitor)
        fill_at = datetime(2025, 6, 2, 13, 35, 23, tzinfo=timezone.utc)
        pos = _build_position(fill_at=fill_at)
        eng.open_positions[pos.symbol] = pos
        # last_ts will be at +2 min, well past b1_ts. Strong breakout.
        bars = _bars_after_fill(fill_at, [
            (0, 9.95, 10.10, 9.90, 10.08, 5000),   # strong breakout
            (1, 10.05, 10.15, 9.95, 10.10, 5000),  # no revert
            (2, 10.10, 10.20, 10.05, 10.15, 5000),
        ])

        eng._evaluate_touchgo(pos.symbol, bars)

        # Rule M was evaluated but did not fire (strong breakout)
        mock_stop_monitor.force_exit.assert_not_called()
        assert pos.rule_m_evaluated is True
        # Rule D evaluated too (last_ts well past b1_ts) — bar 1 had no revert
        assert pos.rule_d_evaluated is True
