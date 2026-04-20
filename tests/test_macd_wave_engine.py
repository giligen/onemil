"""
Unit tests for MACDWaveEngine helpers added in the execution-latency refactor.

Covers:
- _has_entry_capacity (hoisted capacity check)
- _has_conflicting_alpaca_orders (wash-trade pre-check)
- Bar event queue: normal put, Full handling, drain_bar_events, reset_daily
- check_entries(symbols=...) targeted path
"""

import queue as _q
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock
from types import SimpleNamespace

import pytest

from trading.macd_wave_engine import MACDWaveEngine, OpenPosition, CrossedStock


def _make_engine(**overrides):
    """Minimal engine with mocked dependencies. No StopMonitor unless provided."""
    cfg = {
        'universe': {}, 'entry': {}, 'macd': {},
        'sizing': {'position_size': 50000, 'max_concurrent': 3},
        'risk': {'daily_loss_limit': -5000},
        'slippage': {}, 'waves': {},
    }
    defaults = dict(
        alpaca_client=MagicMock(),
        db=MagicMock(),
        config=cfg,
    )
    defaults.update(overrides)
    return MACDWaveEngine(**defaults)


def _make_pos(order_id='', entry_time=None):
    return OpenPosition(
        symbol='X', entry_price=10.0, shares=1000, hard_stop=9.8,
        trade_id=1, order_id=order_id,
        entry_time=entry_time or datetime.now(timezone.utc),
    )


class TestHasEntryCapacity:
    def test_empty_has_capacity(self):
        e = _make_engine()
        assert e._has_entry_capacity() is True

    def test_at_max_concurrent_no_capacity(self):
        e = _make_engine()
        e.max_concurrent = 2
        e.open_positions = {
            'A': _make_pos(order_id=''),  # filled
            'B': _make_pos(order_id=''),  # filled
        }
        assert e._has_entry_capacity() is False

    def test_stale_pending_counted_until_gc(self):
        """After split: stale-pending stays counted until _gc_stale_pending runs."""
        e = _make_engine()
        e.max_concurrent = 1
        stale = _make_pos(
            order_id='pending-xyz',
            entry_time=datetime.now(timezone.utc) - timedelta(minutes=3),
        )
        e.open_positions = {'STALE': stale}
        # Pure predicate: stale order was >120s, so NOT counted as active.
        # With max=1 and 0 active, capacity is True.
        assert e._has_entry_capacity() is True
        # But the stale entry is still in open_positions until GC runs.
        assert 'STALE' in e.open_positions

    def test_gc_stale_pending_removes_when_alpaca_says_rejected(self):
        """Stale + Alpaca confirms rejected → purge."""
        e = _make_engine()
        e.alpaca.get_order = MagicMock(return_value={'status': 'rejected'})
        stale = _make_pos(order_id='x', entry_time=datetime.now(timezone.utc) - timedelta(minutes=3))
        e.open_positions = {'STALE': stale}
        e._gc_stale_pending()
        assert 'STALE' not in e.open_positions
        assert 'STALE' in e.invalidated

    def test_gc_stale_keeps_fresh_untouched(self):
        """<2min old: skipped entirely — no Alpaca call, no purge."""
        e = _make_engine()
        e.alpaca.get_order = MagicMock()
        fresh = _make_pos(order_id='abc', entry_time=datetime.now(timezone.utc) - timedelta(seconds=30))
        e.open_positions = {'FRESH': fresh}
        e._gc_stale_pending()
        assert 'FRESH' in e.open_positions
        e.alpaca.get_order.assert_not_called()

    def test_gc_stale_KEEPS_live_order_on_alpaca(self):
        """🔥 REGRESSION TEST for 2026-04-16 CDNA/BBGI orphan bug.

        Order is >2min old but Alpaca says it's still LIVE (pending_new,
        new, accepted, partially_filled). MUST NOT be purged — the old
        behavior orphaned legitimate slow-fill limit orders on thin stocks.
        """
        for live_status in ['pending_new', 'new', 'accepted',
                            'partially_filled', 'pending_replace']:
            e = _make_engine()
            e.alpaca.get_order = MagicMock(return_value={'status': live_status})
            stale = _make_pos(order_id='x',
                              entry_time=datetime.now(timezone.utc) - timedelta(minutes=5))
            e.open_positions = {'CDNA': stale}
            e._gc_stale_pending()
            assert 'CDNA' in e.open_positions, \
                f"order with status={live_status} was incorrectly purged — this is the bug"
            assert 'CDNA' not in e.invalidated, \
                f"status={live_status} should NOT invalidate"

    def test_gc_stale_KEEPS_filled_order_for_fill_check(self):
        """Stale + Alpaca says filled → leave in place so fill-check claims it."""
        e = _make_engine()
        e.alpaca.get_order = MagicMock(return_value={'status': 'filled'})
        stale = _make_pos(order_id='x',
                          entry_time=datetime.now(timezone.utc) - timedelta(minutes=5))
        e.open_positions = {'QBTX': stale}
        e._gc_stale_pending()
        assert 'QBTX' in e.open_positions  # kept for fill-check to claim

    def test_gc_stale_hard_cancels_very_old_live(self):
        """>30min AND still live → actively cancel + purge (don't strand capital)."""
        e = _make_engine()
        e.alpaca.get_order = MagicMock(return_value={'status': 'new'})
        e.alpaca.cancel_order = MagicMock()
        super_stale = _make_pos(order_id='old-1',
                                entry_time=datetime.now(timezone.utc) - timedelta(minutes=45))
        e.open_positions = {'OLD': super_stale}
        e._gc_stale_pending()
        assert 'OLD' not in e.open_positions
        assert 'OLD' in e.invalidated
        e.alpaca.cancel_order.assert_called_once_with('old-1')

    def test_gc_stale_keeps_tracking_on_alpaca_error(self):
        """Transient Alpaca error → keep tracking, retry next cycle.

        Never silently purge on a network blip — that was the failure mode
        the original code had.
        """
        e = _make_engine()
        e.alpaca.get_order = MagicMock(side_effect=Exception('network blip'))
        stale = _make_pos(order_id='x',
                          entry_time=datetime.now(timezone.utc) - timedelta(minutes=5))
        e.open_positions = {'NET': stale}
        e._gc_stale_pending()
        assert 'NET' in e.open_positions
        assert 'NET' not in e.invalidated

    def test_has_entry_capacity_is_pure(self):
        """_has_entry_capacity must not mutate state."""
        e = _make_engine()
        stale = _make_pos(
            order_id='pending-xyz',
            entry_time=datetime.now(timezone.utc) - timedelta(minutes=3),
        )
        e.open_positions = {'STALE': stale}
        invalidated_before = set(e.invalidated)
        positions_before = dict(e.open_positions)
        _ = e._has_entry_capacity()
        assert e.open_positions == positions_before
        assert e.invalidated == invalidated_before

    def test_fresh_pending_counts_as_active(self):
        e = _make_engine()
        e.max_concurrent = 1
        fresh = _make_pos(
            order_id='pending-abc',
            entry_time=datetime.now(timezone.utc) - timedelta(seconds=30),
        )
        e.open_positions = {'FRESH': fresh}
        assert e._has_entry_capacity() is False

    def test_daily_loss_limit_blocks(self):
        e = _make_engine()
        e.daily_pnl = -6000
        assert e._has_entry_capacity() is False


class TestConflictingOrdersCheck:
    """Covers both the fast (stream cache) and slow (REST) paths."""

    # --- Slow path: no stream or unhealthy stream → hit REST ---

    def test_no_existing_orders_rest(self):
        e = _make_engine()  # no order_stream attached
        e.alpaca.trading_client.get_orders.return_value = []
        assert e._has_conflicting_alpaca_orders('AAPL') is False

    def test_existing_order_blocks_rest(self):
        e = _make_engine()
        fake_order = SimpleNamespace(side=SimpleNamespace(value='buy'))
        e.alpaca.trading_client.get_orders.return_value = [fake_order]
        assert e._has_conflicting_alpaca_orders('AAPL') is True

    def test_fail_open_on_exception_rest(self):
        e = _make_engine()
        e.alpaca.trading_client.get_orders.side_effect = RuntimeError("api down")
        # Should NOT raise; returns False so Alpaca is the final gate
        assert e._has_conflicting_alpaca_orders('AAPL') is False

    # --- Fast path: healthy stream → no REST call ---

    def test_fast_path_conflict_via_stream(self):
        from trading.order_stream import OrderStreamWatcher
        e = _make_engine()
        stream = MagicMock(spec=OrderStreamWatcher)
        stream.is_healthy.return_value = True
        stream.get_open_order_symbols.return_value = {'AAPL'}
        e.order_stream = stream
        assert e._has_conflicting_alpaca_orders('AAPL') is True
        # REST must NOT be called when fast path is healthy
        e.alpaca.trading_client.get_orders.assert_not_called()

    def test_fast_path_no_conflict_via_stream(self):
        from trading.order_stream import OrderStreamWatcher
        e = _make_engine()
        stream = MagicMock(spec=OrderStreamWatcher)
        stream.is_healthy.return_value = True
        stream.get_open_order_symbols.return_value = {'MSFT', 'NVDA'}
        e.order_stream = stream
        assert e._has_conflicting_alpaca_orders('AAPL') is False
        e.alpaca.trading_client.get_orders.assert_not_called()

    def test_unhealthy_stream_falls_back_to_rest(self):
        from trading.order_stream import OrderStreamWatcher
        e = _make_engine()
        stream = MagicMock(spec=OrderStreamWatcher)
        stream.is_healthy.return_value = False  # unhealthy
        e.order_stream = stream
        e.alpaca.trading_client.get_orders.return_value = []
        assert e._has_conflicting_alpaca_orders('AAPL') is False
        # REST path WAS invoked
        e.alpaca.trading_client.get_orders.assert_called_once()


class TestBarEventQueue:
    def test_register_and_drain(self):
        e = _make_engine()
        from trading.stop_monitor import StopMonitor
        sm = MagicMock(spec=StopMonitor)
        e.stop_monitor = sm
        e.register_on_stop_monitor()
        # Capture the handler registered
        cb = sm.register_bar_handler.call_args[0][1]
        cb('AAPL', None)
        cb('MSFT', None)
        assert e.drain_bar_events() == {'AAPL', 'MSFT'}
        # Second drain is empty
        assert e.drain_bar_events() == set()

    def test_queue_full_logs_and_drops(self, caplog):
        e = _make_engine()
        from trading.stop_monitor import StopMonitor
        sm = MagicMock(spec=StopMonitor)
        e.stop_monitor = sm
        e.register_on_stop_monitor()
        cb = sm.register_bar_handler.call_args[0][1]
        # Fill the queue to cap
        for i in range(1000):
            cb(f'SYM{i}', None)
        # Next put should be dropped + logged
        import logging
        with caplog.at_level(logging.ERROR, logger='trading.macd_wave_engine'):
            cb('OVERFLOW', None)
        assert any('queue FULL' in r.message for r in caplog.records)
        assert e._bar_queue_full_logged is True

    def test_reset_daily_drains_queue(self):
        e = _make_engine()
        from trading.stop_monitor import StopMonitor
        sm = MagicMock(spec=StopMonitor)
        e.stop_monitor = sm
        e.register_on_stop_monitor()
        cb = sm.register_bar_handler.call_args[0][1]
        cb('STALE1', None)
        cb('STALE2', None)
        assert e._bar_event_queue.qsize() == 2
        e.reset_daily()
        assert e._bar_event_queue.qsize() == 0


# ---------------------------------------------------------------------------
# V4 Conviction Sizing — PROD wiring tests
# ---------------------------------------------------------------------------

def _make_engine_with_conviction(enabled, max_pos_usd=90_000, **overrides):
    """Engine factory that opts into/out of conviction sizing."""
    cfg = {
        'universe': {}, 'entry': {}, 'macd': {},
        'sizing': {
            'position_size': 50_000,
            'max_concurrent': 3,
            'conviction_sizing': {
                'enabled': enabled,
                'max_position_size_usd': max_pos_usd,
            },
        },
        'risk': {'daily_loss_limit': -5000},
        'slippage': {}, 'waves': {},
    }
    overrides.setdefault('alpaca_client', MagicMock())
    overrides.setdefault('db', MagicMock())
    overrides['config'] = cfg
    return MACDWaveEngine(**overrides)


class TestConvictionSizingConfig:
    """Config parsing + defaults."""

    def test_disabled_by_default_when_block_missing(self):
        """No `conviction_sizing:` block → enabled=False (safe default)."""
        e = _make_engine()
        assert e.conviction_sizing_enabled is False
        # Cap still defaults to $90K even when disabled
        assert e.max_position_size_usd == 90_000.0

    def test_enabled_true_loads(self):
        e = _make_engine_with_conviction(enabled=True)
        assert e.conviction_sizing_enabled is True
        assert e.max_position_size_usd == 90_000.0

    def test_custom_max_position_loads(self):
        e = _make_engine_with_conviction(enabled=True, max_pos_usd=75_000)
        assert e.max_position_size_usd == 75_000.0

    def test_startup_log_fires_when_enabled(self, caplog):
        import logging
        with caplog.at_level(logging.INFO, logger='trading.macd_wave_engine'):
            _make_engine_with_conviction(enabled=True)
        # The INFO log announces ENABLED so failures in yaml wiring are visible
        msgs = [r.message for r in caplog.records]
        assert any('Conviction sizing: ENABLED' in m for m in msgs), \
            f"Expected 'Conviction sizing: ENABLED' log, got: {msgs}"

    def test_warning_when_cap_is_nonpositive(self, caplog):
        """max_position_size_usd <= 0 is a footgun — warn at startup."""
        import logging
        with caplog.at_level(logging.WARNING, logger='trading.macd_wave_engine'):
            _make_engine_with_conviction(enabled=True, max_pos_usd=0)
        msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any('is <=0' in m for m in msgs), \
            f"Expected zero-cap warning, got: {msgs}"

    def test_warning_when_cap_below_baseline(self, caplog):
        """max_position_size_usd < position_size would cap baseline below flat."""
        import logging
        with caplog.at_level(logging.WARNING, logger='trading.macd_wave_engine'):
            _make_engine_with_conviction(enabled=True, max_pos_usd=30_000)  # baseline 50K
        msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any('BELOW baseline position_size' in m for m in msgs), \
            f"Expected below-baseline warning, got: {msgs}"

    def test_no_warning_when_cap_is_sensible(self, caplog):
        """Sensible cap (>= baseline) doesn't emit warnings."""
        import logging
        with caplog.at_level(logging.WARNING, logger='trading.macd_wave_engine'):
            _make_engine_with_conviction(enabled=True, max_pos_usd=90_000)
        # Filter to only the warnings from our conviction config code (others
        # may come from db/alpaca mock infrastructure — ignore those).
        cap_warnings = [
            r.message for r in caplog.records
            if r.levelno >= logging.WARNING
            and ('is <=0' in r.message or 'BELOW baseline' in r.message)
        ]
        assert cap_warnings == [], f"Unexpected cap warnings: {cap_warnings}"


class TestConvictionShareScaling:
    """The core contract — shares scale by conv_mult when enabled."""

    def _build_submit_kwargs(self, cross_time, vol_at_cross, entry_price=10.0):
        """Common args for _submit_entry test calls."""
        crossed = CrossedStock(
            symbol='TEST', open_price=entry_price * 0.9,
            cross_time_min=cross_time, vol_at_cross=vol_at_cross,
            crossed_at=datetime.now(timezone.utc),
        )
        # Pre-made smart_quote tuple matches the expected shape from
        # _get_smart_limit_price: (limit_price, quote_info_dict)
        smart_quote = (
            entry_price,
            {'bid': entry_price - 0.01, 'ask': entry_price,
             'quote_fetched_at': datetime.now(timezone.utc)},
        )
        return dict(
            symbol='TEST', price=entry_price, macd_hist_pct=1.0,
            crossed=crossed, smart_quote=smart_quote,
        )

    def _primed_engine(self, enabled, max_pos_usd=90_000):
        """Engine with all external boundaries mocked so _submit_entry runs end-to-end."""
        e = _make_engine_with_conviction(enabled=enabled, max_pos_usd=max_pos_usd)
        # No conflicting orders — test the sizing math, not the wash-trade path
        e._has_conflicting_alpaca_orders = MagicMock(return_value=False)
        e.alpaca.submit_bracket_order = MagicMock(return_value={'id': 'o1'})
        e.db.save_trade = MagicMock(return_value=1)
        e.db.update_trade = MagicMock()
        return e

    def test_sizing_disabled_uses_flat_position_size(self):
        """enabled=False → shares = int(50_000 / entry) regardless of conviction."""
        e = self._primed_engine(enabled=False)
        # Top-tier conviction setup (cross=1, vol=10K → conv=1.8), but sizing disabled
        e._submit_entry(**self._build_submit_kwargs(cross_time=1, vol_at_cross=10_000))
        submitted = e.alpaca.submit_bracket_order.call_args
        assert submitted is not None, "submit_bracket_order was not called"
        # $50K / $10 = 5000 shares flat (no scaling)
        assert submitted.kwargs['qty'] == 5000, \
            f"Expected flat 5000 shares, got {submitted.kwargs['qty']}"

    def test_sizing_enabled_scales_by_conviction(self):
        """enabled=True, conv=1.8 → shares = int(90K / entry) = 9000."""
        e = self._primed_engine(enabled=True)
        e._submit_entry(**self._build_submit_kwargs(cross_time=1, vol_at_cross=10_000))
        submitted = e.alpaca.submit_bracket_order.call_args
        # $90K / $10 = 9000 shares (1.8x baseline 5000)
        assert submitted.kwargs['qty'] == 9000

    def test_sizing_enabled_baseline_trade(self):
        """enabled=True, conv=1.0 (both rules miss) → shares unchanged from baseline."""
        e = self._primed_engine(enabled=True)
        # Both rules bottom tier: cross=10, vol=300K → conv=1.0, effective=$50K
        e._submit_entry(**self._build_submit_kwargs(cross_time=10, vol_at_cross=300_000))
        submitted = e.alpaca.submit_bracket_order.call_args
        assert submitted.kwargs['qty'] == 5000  # Flat baseline

    def test_sizing_enabled_partial_tier(self):
        """enabled=True, conv=1.4 (one top-tier, one bottom) → 1.4x baseline."""
        e = self._primed_engine(enabled=True)
        # cross=3 (+0.4) + vol=300K (+0.0) → conv=1.4, effective=$70K
        e._submit_entry(**self._build_submit_kwargs(cross_time=3, vol_at_cross=300_000))
        submitted = e.alpaca.submit_bracket_order.call_args
        # $70K / $10 = 7000 shares
        assert submitted.kwargs['qty'] == 7000

    def test_max_position_cap_enforced(self):
        """Cap at $60K clips a conv=1.8 setup even though raw would be $90K."""
        e = self._primed_engine(enabled=True, max_pos_usd=60_000)
        # Top-tier conv=1.8; would be $90K but cap is $60K
        e._submit_entry(**self._build_submit_kwargs(cross_time=1, vol_at_cross=10_000))
        submitted = e.alpaca.submit_bracket_order.call_args
        # $60K / $10 = 6000 shares (capped)
        assert submitted.kwargs['qty'] == 6000


class TestConvictionTelemetry:
    """Breakdown persisted to DB; EOD list populated."""

    def _run_entry(self, engine, cross_time, vol_at_cross, entry_price=10.0):
        crossed = CrossedStock(
            symbol='TEST', open_price=entry_price * 0.9,
            cross_time_min=cross_time, vol_at_cross=vol_at_cross,
            crossed_at=datetime.now(timezone.utc),
        )
        smart_quote = (
            entry_price,
            {'bid': entry_price - 0.01, 'ask': entry_price,
             'quote_fetched_at': datetime.now(timezone.utc)},
        )
        engine._has_conflicting_alpaca_orders = MagicMock(return_value=False)
        engine.alpaca.submit_bracket_order = MagicMock(return_value={'id': 'o1'})
        engine.db.save_trade = MagicMock(return_value=42)
        engine.db.update_trade = MagicMock()
        engine._submit_entry(
            symbol='TEST', price=entry_price, macd_hist_pct=1.0,
            crossed=crossed, smart_quote=smart_quote,
        )

    def test_pattern_data_includes_conviction_fields(self):
        """save_trade receives pattern_data JSON with conviction_* fields."""
        e = _make_engine_with_conviction(enabled=True)
        self._run_entry(e, cross_time=1, vol_at_cross=10_000)
        # Extract the pattern_data string passed to save_trade
        saved = e.db.save_trade.call_args.args[0]
        pat = saved['pattern_data']
        assert '"conviction_mult": 1.800' in pat
        assert '"conv_cross_speed": 0.4' in pat
        assert '"conv_vol_at_cross": 0.4' in pat

    def test_eod_traded_populated_on_entry(self):
        """Each accepted entry appends to self._eod_traded."""
        e = _make_engine_with_conviction(enabled=True)
        assert e._eod_traded == []
        self._run_entry(e, cross_time=1, vol_at_cross=10_000)
        assert len(e._eod_traded) == 1
        entry = e._eod_traded[0]
        assert entry['symbol'] == 'TEST'
        assert entry['conv_mult'] == pytest.approx(1.8)
        assert entry['conv_cross_speed'] == pytest.approx(0.4)
        assert entry['conv_vol_at_cross'] == pytest.approx(0.4)
        assert entry['effective_position'] == 90_000.0

    def test_eod_traded_cleared_by_reset_daily(self):
        """reset_daily() drops yesterday's telemetry."""
        e = _make_engine_with_conviction(enabled=True)
        self._run_entry(e, cross_time=1, vol_at_cross=10_000)
        assert len(e._eod_traded) == 1
        e.reset_daily()
        assert e._eod_traded == []

    def test_conviction_logged_on_entry(self, caplog):
        """CONVICTION line appears in INFO log."""
        import logging
        e = _make_engine_with_conviction(enabled=True)
        with caplog.at_level(logging.INFO, logger='trading.macd_wave_engine'):
            self._run_entry(e, cross_time=1, vol_at_cross=10_000)
        msgs = [r.message for r in caplog.records]
        assert any('CONVICTION 1.80' in m for m in msgs), f"Expected CONVICTION log, got: {msgs}"
        assert any('cross=+0.4 vol=+0.4' in m for m in msgs)

    def test_telegram_shows_effective_position_not_flat(self):
        """Telegram notifier must receive the SCALED position $, not flat.

        Otherwise users see flat $ in telegram while DB/broker have scaled.
        """
        e = _make_engine_with_conviction(enabled=True)
        e.notifier = MagicMock()
        self._run_entry(e, cross_time=1, vol_at_cross=10_000)  # conv=1.8 → $90K
        # Telegram notifier was called
        assert e.notifier.send_message_sync.called
        msg = e.notifier.send_message_sync.call_args.args[0]
        # Must show $90K (scaled), not $50K (flat)
        assert '$90,000' in msg, f"Expected $90,000 in telegram msg, got: {msg}"
        assert '$50,000' not in msg, f"Should NOT show flat $50K when sizing enabled, got: {msg}"
        # Telegram also annotates with conv when sizing is on
        assert 'conv 1.80' in msg

    def test_telegram_shows_flat_when_sizing_disabled(self):
        """With sizing disabled, telegram shows flat $50K — no conv annotation."""
        e = _make_engine_with_conviction(enabled=False)
        e.notifier = MagicMock()
        # Top-tier setup but sizing off → telegram shows flat baseline
        self._run_entry(e, cross_time=1, vol_at_cross=10_000)
        msg = e.notifier.send_message_sync.call_args.args[0]
        assert '$50,000' in msg
        assert 'conv' not in msg, "Should not annotate conv when sizing disabled"


class TestSmartLimitAskBuffer:
    """Post-2026-04-20 fix: smart-limit pricing uses ask + configurable
    buffer (default 30bps) instead of the legacy spread-midpoint logic.
    Rationale in _get_smart_limit_price docstring — USGG post-mortem."""

    def _engine_with_buffer(self, buffer_bps):
        e = _make_engine()
        e.smart_limit_ask_buffer_bps = float(buffer_bps)
        return e

    def test_limit_equals_ask_plus_30bps_by_default(self):
        e = _make_engine()
        assert e.smart_limit_ask_buffer_bps == 30.0
        e.alpaca.get_latest_quote.return_value = {
            'bid_price': 15.52, 'ask_price': 15.58,
            'bid_size': 500, 'ask_size': 400,
        }
        limit, info = e._get_smart_limit_price('USGG')
        # ask × 1.003 = 15.58 × 1.003 = 15.6267 → rounded 15.63
        assert limit == 15.63
        assert info['pricing'] == 'ask_plus_buffer'
        assert info['ask_buffer_bps'] == 30.0

    def test_buffer_configurable_via_yaml_read(self):
        """smart_limit_ask_buffer_bps from entry config block, not hardcoded."""
        cfg = {
            'universe': {}, 'entry': {'smart_limit_ask_buffer_bps': 50},
            'macd': {}, 'sizing': {'position_size': 50000, 'max_concurrent': 3},
            'risk': {'daily_loss_limit': -5000}, 'slippage': {}, 'waves': {},
        }
        e = MACDWaveEngine(alpaca_client=MagicMock(), db=MagicMock(), config=cfg)
        assert e.smart_limit_ask_buffer_bps == 50.0

    def test_zero_buffer_gives_ask_price(self):
        """Buffer=0 reproduces 'at ask' behavior (for A/B vs older logic)."""
        e = self._engine_with_buffer(0)
        e.alpaca.get_latest_quote.return_value = {
            'bid_price': 10.00, 'ask_price': 10.05,
            'bid_size': 100, 'ask_size': 200,
        }
        limit, info = e._get_smart_limit_price('X')
        assert limit == 10.05
        assert info['ask_buffer_bps'] == 0.0

    def test_crossed_market_falls_back_to_ask(self):
        """bid >= ask is stale/crossed — skip buffer math, use ask as-is."""
        e = self._engine_with_buffer(30)
        e.alpaca.get_latest_quote.return_value = {
            'bid_price': 10.10, 'ask_price': 10.00,  # crossed
            'bid_size': 100, 'ask_size': 100,
        }
        limit, info = e._get_smart_limit_price('X')
        assert limit == 10.00
        assert info['pricing'] == 'fallback_crossed'

    def test_missing_quote_falls_back(self):
        e = self._engine_with_buffer(30)
        e.alpaca.get_latest_quote.return_value = {
            'bid_price': 0, 'ask_price': 0, 'bid_size': 0, 'ask_size': 0,
        }
        limit, info = e._get_smart_limit_price('X')
        assert limit == 0
        assert info['pricing'] == 'fallback_no_quote'

    def test_usgg_postmortem_would_have_filled(self):
        """Regression: replay 2026-04-20 USGG quote snapshot. Under the new
        pricing, limit would have been $15.63 (vs old logic's $15.56). Given
        the actual post-signal price ran to $16.70, a $15.63 limit would
        have filled at submit — capturing the trade we missed."""
        e = self._engine_with_buffer(30)
        e.alpaca.get_latest_quote.return_value = {
            'bid_price': 15.52, 'ask_price': 15.58,
            'bid_size': 800, 'ask_size': 400,
        }
        limit, info = e._get_smart_limit_price('USGG')
        # The post-signal ask moved to $15.60+ within 1s; $15.63 limit survives
        # that latency. Old midpoint logic gave $15.56 — stranded.
        assert limit >= 15.58  # above current ask
        assert limit == 15.63


class TestBarStartToClose:
    """Alpaca bar timestamp = bar START. Helper adds 60s for actual close.

    Fixed 2026-04-15. Old DB rows had bar_close_at = bar_start_at, inflating
    bar_close_to_loop_ms metric by exactly 60_000.
    """

    def test_pd_timestamp_input(self):
        """pd.Timestamp input → datetime + 60s, UTC-aware."""
        import pandas as pd
        ts = pd.Timestamp('2026-04-15 14:04:00', tz='UTC')
        out = MACDWaveEngine._bar_start_to_close(ts)
        assert out == datetime(2026, 4, 15, 14, 5, 0, tzinfo=timezone.utc)
        assert out.tzinfo == timezone.utc

    def test_datetime_input(self):
        """Native datetime input."""
        ts = datetime(2026, 4, 15, 14, 4, 0, tzinfo=timezone.utc)
        out = MACDWaveEngine._bar_start_to_close(ts)
        assert out == datetime(2026, 4, 15, 14, 5, 0, tzinfo=timezone.utc)

    def test_string_input(self):
        """String input gets parsed."""
        out = MACDWaveEngine._bar_start_to_close('2026-04-15T14:04:00+00:00')
        assert out == datetime(2026, 4, 15, 14, 5, 0, tzinfo=timezone.utc)

    def test_naive_datetime_treated_as_utc(self):
        """Datetime without tzinfo is treated as UTC, +60s applied."""
        ts_naive = datetime(2026, 4, 15, 14, 4, 0)  # no tzinfo
        out = MACDWaveEngine._bar_start_to_close(ts_naive)
        assert out == datetime(2026, 4, 15, 14, 5, 0, tzinfo=timezone.utc)
        assert out.tzinfo == timezone.utc

    def test_minute_rollover_to_next_hour(self):
        """At HH:59:00 → close at (HH+1):00:00."""
        ts = datetime(2026, 4, 15, 14, 59, 0, tzinfo=timezone.utc)
        out = MACDWaveEngine._bar_start_to_close(ts)
        assert out == datetime(2026, 4, 15, 15, 0, 0, tzinfo=timezone.utc)


# =============================================================================
# Exit telemetry parity with bull_flag (added 2026-04-15 — see learnings from
# XNDU/MNTS trades where macd_flip exits had NULL exit_quote_bid_size,
# exit_trigger_price, exit_fill_latency_ms, exit_slippage)
# =============================================================================

class TestSubmitExitTelemetry:
    """_submit_exit must populate the same exit-microstructure fields as the
    bull_flag stop/TP exit so we can analyze WHY waves die."""

    def _engine_with_open_pos(self):
        e = _make_engine()
        # Stub out helpers we don't care about
        e.alpaca.get_latest_quote = MagicMock(return_value={
            'bid_price': 6.13, 'ask_price': 6.14,
            'bid_size': 500, 'ask_size': 800,
        })
        e.alpaca.close_position = MagicMock(return_value={'id': 'order-xyz'})
        e.alpaca.trading_client = MagicMock()
        e.alpaca.trading_client.get_orders = MagicMock(return_value=[])
        # Insert an open position to close
        pos = OpenPosition(
            symbol='MNTS', entry_price=6.12, shares=12254, hard_stop=6.00,
            trade_id=42, order_id='entry-abc',
            entry_time=datetime.now(timezone.utc),
        )
        e.open_positions['MNTS'] = pos
        return e

    def test_macd_flip_exit_populates_full_telemetry(self):
        e = self._engine_with_open_pos()
        ok = e._submit_exit('MNTS', 'macd_flip', trigger_price=6.10)
        assert ok is True

        # Find the FINAL update_trade call (the one with the full exit dict).
        calls = e.db.update_trade.call_args_list
        # The last call carries exit_price + exit_reason + the new telemetry.
        last_payload = calls[-1].args[1] if len(calls[-1].args) > 1 else calls[-1].kwargs
        assert last_payload['exit_price'] == 6.13
        assert last_payload['exit_reason'] == 'macd_flip'
        # NEW telemetry fields — these were NULL before today's fix.
        assert last_payload['exit_trigger_price'] == 6.10
        assert last_payload['exit_quote_bid'] == 6.13
        assert last_payload['exit_quote_ask'] == 6.14
        assert last_payload['exit_quote_bid_size'] == 500
        assert last_payload['exit_quote_ask_size'] == 800
        assert last_payload['exit_quote_spread'] == pytest.approx(0.01, abs=1e-9)
        assert last_payload['exit_limit_price'] == 6.13
        assert last_payload['exit_pricing_method'] == 'macd_flip_close'
        assert last_payload['exit_submitted_at'] is not None
        # fill latency is positive (we measure between two time.time() calls)
        assert last_payload['exit_fill_latency_ms'] >= 0
        # exit_slippage = limit (6.13) - actual fill (6.13) = 0
        assert last_payload['exit_slippage'] == pytest.approx(0.0, abs=1e-9)

    def test_force_close_exit_populates_telemetry_with_null_trigger(self):
        e = self._engine_with_open_pos()
        ok = e._submit_exit('MNTS', 'force_close')  # no trigger_price
        assert ok is True
        last_payload = e.db.update_trade.call_args_list[-1].args[1]
        # Force-close has no specific trigger price (EOD timer, not a signal).
        assert last_payload['exit_trigger_price'] is None
        # But all the rest is still captured.
        assert last_payload['exit_pricing_method'] == 'force_close_close'
        assert last_payload['exit_quote_bid_size'] == 500
        assert last_payload['exit_submitted_at'] is not None

    def test_no_quote_falls_back_safely(self):
        """If the broker returns a zero/missing quote, telemetry stores None
        rather than crashing."""
        e = self._engine_with_open_pos()
        e.alpaca.get_latest_quote = MagicMock(return_value={
            'bid_price': 0, 'ask_price': 0, 'bid_size': 0, 'ask_size': 0,
        })
        ok = e._submit_exit('MNTS', 'macd_flip', trigger_price=6.10)
        assert ok is True
        last_payload = e.db.update_trade.call_args_list[-1].args[1]
        assert last_payload['exit_quote_bid'] is None
        assert last_payload['exit_quote_ask'] is None
        assert last_payload['exit_quote_bid_size'] is None
        assert last_payload['exit_quote_spread'] is None
        assert last_payload['exit_limit_price'] is None
        assert last_payload['exit_slippage'] is None
        # Trigger price is still recorded (we know what bar fired the flip).
        assert last_payload['exit_trigger_price'] == 6.10
