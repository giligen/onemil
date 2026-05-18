"""
Integration tests: ORBEngine._submit_entry uses trading.buy_stop_guard
to dispatch on the right alpaca submission method.

These are the production-critical tests for tomorrow's 12:30 UTC cron-start
where ORB launches with the unified guard for the first time. If any of these
break, ORB stops working.

Scenarios covered:
  1. SUBMIT_AS_IS  — quote below stop → original submit_stop_bracket_order
  2. MARKETABLE_LIMIT — bid >= stop → submit_bracket_order (limit + brackets)
  3. REBUMP_STOP   — straddle within limit → submit_stop_bracket_order with
                     stop = ask+buffer
  4. SKIP          — ask + buf > limit → return None, no submit, no DB row
  5. Quote fetch failure → SUBMIT_AS_IS (defensive)
  6. Guard disabled by config → SUBMIT_AS_IS regardless of quote
  7. Submission exception → return None, error logged
  8. dry_run bypasses guard entirely
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database
from trading.orb_engine import ORBEngine
from trading.orb_planner import OrbTradePlan
from trading.stop_monitor import StopMonitor


# ---------------------------------------------------------------------------
# Fixtures (mirror the patterns in test_orb_engine.py)
# ---------------------------------------------------------------------------

@pytest.fixture
def orb_cfg():
    yaml_path = Path(__file__).parent.parent / 'orb.yaml'
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    cfg['strategy']['enabled'] = True
    return cfg


@pytest.fixture
def mock_alpaca():
    m = MagicMock(spec=AlpacaClient)
    m.get_open_positions.return_value = []
    m.get_account_info.return_value = {'buying_power': 100_000.0}
    # Sensible defaults; tests override per-scenario.
    m.get_latest_quote.return_value = {
        'bid_price': 4.00, 'ask_price': 4.01,
        'bid_size': 100, 'ask_size': 100,
    }
    m.submit_stop_bracket_order.return_value = {
        'id': 'sb-1', 'status': 'accepted',
    }
    m.submit_bracket_order.return_value = {
        'id': 'br-1', 'status': 'accepted',
    }
    return m


@pytest.fixture
def mock_db():
    db = MagicMock(spec=Database)
    db.save_trade.return_value = 999
    db.get_open_trades.return_value = []
    db.update_trade.return_value = True
    return db


@pytest.fixture
def mock_sm():
    sm = MagicMock(spec=StopMonitor)
    sm.polling_mode = False
    sm.drain_exit_events.return_value = []
    return sm


@pytest.fixture
def engine(orb_cfg, mock_alpaca, mock_db, mock_sm):
    return ORBEngine(
        alpaca_client=mock_alpaca,
        db=mock_db,
        stop_monitor=mock_sm,
        config=orb_cfg,
    )


def _plan(
    symbol='TEST',
    range_high=4.07,
    range_low=4.00,
    entry_price=4.08,        # = range_high × 1.003 (rounded to 2dp typically)
    stop_price=4.00,         # = range_low
    shares=100,
) -> OrbTradePlan:
    range_size = range_high - range_low
    return OrbTradePlan(
        symbol=symbol,
        range_high=range_high,
        range_low=range_low,
        range_size=range_size,
        entry_price=entry_price,
        stop_price=stop_price,
        shares=shares,
        position_dollars=shares * entry_price,
        lock_arm_at_r=1.5,
        lock_stop_r=1.0,
        risk_per_share=entry_price - stop_price,
        total_risk=(entry_price - stop_price) * shares,
        composite_score=0.5,
        quintile='Q4',
        adaptive_mult=1.0,
        range_open=range_low,
    )


# ---------------------------------------------------------------------------
# Scenario 1: SUBMIT_AS_IS — ask < stop (normal pre-breakout)
# ---------------------------------------------------------------------------

class TestSubmitAsIs:
    """Quote below the stop → standard buy-stop-bracket submission, no
    deviation from the legacy path."""

    def test_dispatches_to_stop_bracket_order(self, engine, mock_alpaca):
        mock_alpaca.get_latest_quote.return_value = {
            'bid_price': 3.75, 'ask_price': 3.76,
            'bid_size': 100, 'ask_size': 100,
        }
        oid = engine._submit_entry(_plan())
        assert oid == 'sb-1'
        mock_alpaca.submit_stop_bracket_order.assert_called_once()
        mock_alpaca.submit_bracket_order.assert_not_called()
        # The stop sent to Alpaca is the ORIGINAL stop_trigger (range_high
        # rounded), not the rebumped stop.
        kw = mock_alpaca.submit_stop_bracket_order.call_args.kwargs
        assert kw['stop_price'] == 4.07     # round(range_high, 2)
        assert kw['limit_price'] == 4.08    # round(entry_price, 2)
        assert kw['side'] == 'buy'


# ---------------------------------------------------------------------------
# Scenario 2: MARKETABLE_LIMIT — bid >= stop (breakout confirmed)
# ---------------------------------------------------------------------------

class TestMarketableLimit:
    """bid already above the breakout level → submit_bracket_order (limit
    entry + bracket legs), NOT submit_stop_bracket_order. This is the path
    that would have saved BTCZ/BMNZ on 2026-05-18 (rejected by Alpaca's
    'stop <= ask' rule)."""

    def test_dispatches_to_bracket_order_with_limit_entry(self, engine, mock_alpaca):
        mock_alpaca.get_latest_quote.return_value = {
            'bid_price': 4.10, 'ask_price': 4.12,
            'bid_size': 100, 'ask_size': 100,
        }
        oid = engine._submit_entry(_plan())
        assert oid == 'br-1'
        mock_alpaca.submit_bracket_order.assert_called_once()
        mock_alpaca.submit_stop_bracket_order.assert_not_called()
        kw = mock_alpaca.submit_bracket_order.call_args.kwargs
        assert kw['limit_price'] == 4.08    # round(entry_price, 2)
        assert kw['side'] == 'buy'
        # Bracket legs preserved (safety_sl = entry × 0.90, safety_tp = entry × 3)
        assert kw['sl_price'] > 0
        assert kw['tp_price'] > kw['limit_price']

    def test_btcz_replay(self, engine, mock_alpaca):
        """Today's BTCZ: stop $4.09, limit $4.10, bid $4.11, ask $4.12.
        Old code: REJECTED. New code: MARKETABLE_LIMIT → enters."""
        mock_alpaca.get_latest_quote.return_value = {
            'bid_price': 4.11, 'ask_price': 4.12,
            'bid_size': 100, 'ask_size': 100,
        }
        plan = _plan(
            symbol='BTCZ',
            range_high=4.09, range_low=3.88,
            entry_price=4.10, stop_price=3.88,
        )
        oid = engine._submit_entry(plan)
        assert oid == 'br-1'
        mock_alpaca.submit_bracket_order.assert_called_once()


# ---------------------------------------------------------------------------
# Scenario 3: REBUMP_STOP — straddle within limit
# ---------------------------------------------------------------------------

class TestRebumpStop:
    """Spread straddles stop AND ask + buffer <= limit → submit_stop_bracket
    with stop bumped to ask + buffer."""

    def test_bumped_stop_passed_to_alpaca(self, engine, mock_alpaca):
        # Plan: stop $4.40, limit $4.49 (= 4.40 × 1.02).
        plan = _plan(
            symbol='STRD',
            range_high=4.40, range_low=4.30,
            entry_price=4.49, stop_price=4.30,
        )
        mock_alpaca.get_latest_quote.return_value = {
            'bid_price': 4.35, 'ask_price': 4.45,    # straddle stop 4.40
            'bid_size': 100, 'ask_size': 100,
        }
        oid = engine._submit_entry(plan)
        assert oid == 'sb-1'
        mock_alpaca.submit_stop_bracket_order.assert_called_once()
        mock_alpaca.submit_bracket_order.assert_not_called()
        kw = mock_alpaca.submit_stop_bracket_order.call_args.kwargs
        # Rebumped stop = round(4.45 + 0.02, 2) = 4.47, limit unchanged
        assert kw['stop_price'] == 4.47
        assert kw['limit_price'] == 4.49


# ---------------------------------------------------------------------------
# Scenario 4: SKIP — breakout extended past limit
# ---------------------------------------------------------------------------

class TestSkip:
    """ask + buffer > limit → return None, NO order submitted, NO DB row."""

    def test_no_submit_no_db(self, engine, mock_alpaca, mock_db):
        plan = _plan(
            symbol='XTND',
            range_high=4.40, range_low=4.30,
            entry_price=4.49, stop_price=4.30,
        )
        mock_alpaca.get_latest_quote.return_value = {
            'bid_price': 4.35, 'ask_price': 4.48,    # ask + 0.02 = 4.50 > 4.49
            'bid_size': 100, 'ask_size': 100,
        }
        oid = engine._submit_entry(plan)
        assert oid is None
        mock_alpaca.submit_stop_bracket_order.assert_not_called()
        mock_alpaca.submit_bracket_order.assert_not_called()
        # No DB write on skip (the pending-trade save lives downstream of
        # the submit; we never reach it).
        mock_db.save_trade.assert_not_called()

    def test_yss_replay(self, engine, mock_alpaca):
        """YSS 2026-05-18: stop $26.44, limit $26.52. Bid $26.35, ask $26.55
        → ask + 0.02 = 26.57 > 26.52 → SKIP."""
        plan = _plan(
            symbol='YSS',
            range_high=26.44, range_low=25.00,
            entry_price=26.52, stop_price=25.00,
        )
        mock_alpaca.get_latest_quote.return_value = {
            'bid_price': 26.35, 'ask_price': 26.55,
            'bid_size': 100, 'ask_size': 100,
        }
        oid = engine._submit_entry(plan)
        assert oid is None
        mock_alpaca.submit_stop_bracket_order.assert_not_called()


# ---------------------------------------------------------------------------
# Defensive paths
# ---------------------------------------------------------------------------

class TestDefensive:
    """Failure modes that must NOT take down ORB."""

    def test_quote_fetch_raises_falls_through_to_submit_as_is(
        self, engine, mock_alpaca,
    ):
        """If get_latest_quote raises, _submit_entry must still attempt the
        submission (preserving today's behavior) — the only thing missing is
        a telemetry quote."""
        mock_alpaca.get_latest_quote.side_effect = RuntimeError("flaky")
        oid = engine._submit_entry(_plan())
        assert oid == 'sb-1'
        mock_alpaca.submit_stop_bracket_order.assert_called_once()

    def test_quote_returns_none_falls_through_to_submit_as_is(
        self, engine, mock_alpaca,
    ):
        mock_alpaca.get_latest_quote.return_value = None
        oid = engine._submit_entry(_plan())
        assert oid == 'sb-1'
        mock_alpaca.submit_stop_bracket_order.assert_called_once()

    def test_guard_disabled_by_config_always_submits_as_is(
        self, engine, mock_alpaca,
    ):
        """Config kill switch: enabled=False bypasses the guard even when
        the quote would have triggered marketable / rebump / skip. Preserves
        rollback path (legacy stop-limit-bracket-only behavior)."""
        engine._buy_stop_guard_cfg = {'enabled': False, 'rebump_buffer': 0.02}
        # Quote that would normally route to MARKETABLE_LIMIT
        mock_alpaca.get_latest_quote.return_value = {
            'bid_price': 4.20, 'ask_price': 4.21,
            'bid_size': 100, 'ask_size': 100,
        }
        oid = engine._submit_entry(_plan())
        assert oid == 'sb-1'
        mock_alpaca.submit_stop_bracket_order.assert_called_once()
        mock_alpaca.submit_bracket_order.assert_not_called()
        kw = mock_alpaca.submit_stop_bracket_order.call_args.kwargs
        assert kw['stop_price'] == 4.07     # ORIGINAL stop, no rebump

    def test_submit_exception_returns_none(self, engine, mock_alpaca):
        """If the Alpaca submit raises, _submit_entry returns None — the
        engine doesn't add a half-tracked candidate."""
        mock_alpaca.submit_stop_bracket_order.side_effect = RuntimeError("alpaca 500")
        oid = engine._submit_entry(_plan())
        assert oid is None

    def test_submit_returns_empty_dict_returns_none(self, engine, mock_alpaca):
        mock_alpaca.submit_stop_bracket_order.return_value = {}
        oid = engine._submit_entry(_plan())
        assert oid is None

    def test_dry_run_bypasses_guard(self, orb_cfg, mock_alpaca, mock_db, mock_sm):
        """dry_run mode: no quote fetch, no submission, no DB write."""
        eng = ORBEngine(
            alpaca_client=mock_alpaca,
            db=mock_db,
            stop_monitor=mock_sm,
            config=orb_cfg,
            dry_run=True,
        )
        oid = eng._submit_entry(_plan())
        assert oid is not None
        assert oid.startswith('dry-run-')
        mock_alpaca.submit_stop_bracket_order.assert_not_called()
        mock_alpaca.submit_bracket_order.assert_not_called()
        mock_db.save_trade.assert_not_called()


class TestConfigIngestion:
    """ORBEngine loads marketable_limit_fallback_cfg from config.py and
    stores it as self._buy_stop_guard_cfg. Both required keys must be
    present after init."""

    def test_buy_stop_guard_cfg_loaded(self, engine):
        cfg = engine._buy_stop_guard_cfg
        assert cfg is not None
        assert 'enabled' in cfg
        assert 'rebump_buffer' in cfg
        # Floor enforced at 0.02 by Config.marketable_limit_fallback_cfg
        assert cfg['rebump_buffer'] >= 0.02
