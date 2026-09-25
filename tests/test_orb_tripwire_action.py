"""Unit tests for the ORB latency tripwire action (docs/live_guardrails_spec_20260925.md
G2): `execution.tripwire_action` 'warn' (default, unchanged today's behaviour) vs
'dry' (forces the rest of the session into the existing `strategy.dry_run` /
`[ORB DRY] WOULD BUY` path once the first submit lands late).

Style/fixture reference: tests/test_orb_dry_run.py (same base cfg, mock alpaca,
selection-chain harness) — reused here rather than duplicated per CLAUDE.md.
"""
import logging
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

from data_sources.alpaca_client import AlpacaClient
from notifications.telegram_notifier import TelegramNotifier
from persistence.database import Database
from trading.orb_engine import ORBEngine, RangeData, CandidateState
from trading.stop_monitor import StopMonitor

from tests.test_orb_dry_run import _base_cfg, _mock_alpaca, _range, _disable_gates

# A fixed ET instant well after 09:35:00 + the default 10s tripwire threshold,
# so the test is deterministic regardless of wall-clock time when it runs.
LATE_ET = datetime(2026, 9, 25, 9, 37, 0, tzinfo=ZoneInfo('America/New_York'))


def _engine(tripwire_action):
    cfg = _base_cfg()
    cfg['strategy']['dry_run'] = False  # tripwire forces dry mid-session, not from boot
    cfg.setdefault('execution', {})['tripwire_action'] = tripwire_action
    db = MagicMock(spec=Database)
    db.get_open_trades.return_value = []
    db.get_trades_by_date.return_value = []
    a = _mock_alpaca()
    notifier = MagicMock(spec=TelegramNotifier)
    notifier.send_message.return_value = None
    eng = ORBEngine(alpaca_client=a, db=db, stop_monitor=MagicMock(spec=StopMonitor),
                     config=cfg, notifier=notifier)
    eng.pdr_veto_enabled = False
    eng.g1_veto_enabled = False
    eng.range_size_veto_enabled = False
    eng.catalyst_veto_enabled = False
    eng.skip_q1 = False
    eng.filter_threshold = -999.0
    return eng, a, db, notifier


def _seed(eng, symbol):
    eng.universe.add(symbol)
    eng.candidates[symbol] = CandidateState(symbol=symbol)
    eng.candidates[symbol].range_data = _range(symbol)
    eng._symbol_pool[symbol] = 'production'


def _fp(symbol):
    return {symbol: {'prev_day_bar': {}, 'daily_stats_20d': {}}}


def _run_tick(eng, symbol, tmp_path):
    _seed(eng, symbol)
    with _disable_gates(eng), \
            patch('trading.orb_engine.composite_score', return_value=1.0), \
            patch('trading.orb_engine.assign_quintile', return_value='Q5'), \
            patch('trading.orb_engine.__file__', str(tmp_path / 'trading' / 'orb_engine.py')), \
            patch.object(eng, '_et_now', return_value=LATE_ET):
        return eng.check_entries(feature_providers=_fp(symbol))


class TestTripwireActionDefaultsToWarn:
    def test_config_default_is_warn(self):
        eng, _, _, _ = _engine('warn')
        assert eng.tripwire_action == 'warn'

    def test_unknown_value_falls_back_to_warn(self, caplog):
        cfg = _base_cfg()
        cfg.setdefault('execution', {})['tripwire_action'] = 'banana'
        db = MagicMock(spec=Database)
        db.get_open_trades.return_value = []
        db.get_trades_by_date.return_value = []
        with caplog.at_level(logging.WARNING):
            eng = ORBEngine(alpaca_client=_mock_alpaca(), db=db,
                            stop_monitor=MagicMock(spec=StopMonitor), config=cfg)
        assert eng.tripwire_action == 'warn'
        assert any("not 'warn'/'dry'" in r.message for r in caplog.records)


class TestWarnActionUnchangedBehaviour:
    """warn (default) = today's behaviour: tripwire logs/telegrams only, real
    orders keep flowing before AND after it fires."""

    def test_first_late_submit_is_real_and_dry_mode_never_engages(self, caplog, tmp_path):
        eng, a, db, notifier = _engine('warn')
        with caplog.at_level(logging.WARNING):
            submitted = _run_tick(eng, 'WARN1', tmp_path)
        assert submitted == ['WARN1']
        assert eng.candidates['WARN1'].rejected_reason is None
        assert eng.strategy_dry_run is False
        assert eng._tripwire_forced_dry_today is False
        assert 'LATENCY TRIPWIRE' in caplog.text
        assert 'TRIPWIRE → DRY' not in caplog.text

    def test_second_submit_after_tripwire_still_real(self, tmp_path):
        eng, a, db, notifier = _engine('warn')
        _run_tick(eng, 'WARN1', tmp_path)
        submitted2 = _run_tick(eng, 'WARN2', tmp_path)
        assert submitted2 == ['WARN2']
        assert eng.candidates['WARN2'].rejected_reason is None
        assert a.submit_stop_bracket_order.call_count == 2

    def test_no_tripwire_telegram_text_sent(self, tmp_path):
        eng, a, db, notifier = _engine('warn')
        _run_tick(eng, 'WARN1', tmp_path)
        texts = [c.args[0] for c in notifier.send_message.call_args_list]
        assert not any('TRIPWIRE → DRY' in t for t in texts)


class TestDryActionForcesRestOfSession:
    """dry: the first late submit is still real (the tripwire is measured
    AFTER it), then the engine forces strategy_dry_run True for every
    subsequent pick this session — zero orders, `[ORB DRY] WOULD BUY`."""

    def test_first_late_submit_is_real_then_dry_mode_engages(self, caplog, tmp_path):
        eng, a, db, notifier = _engine('dry')
        with caplog.at_level(logging.ERROR):
            submitted = _run_tick(eng, 'DRY1', tmp_path)
        assert submitted == ['DRY1']
        assert eng.candidates['DRY1'].rejected_reason is None
        assert a.submit_stop_bracket_order.call_count == 1
        assert eng.strategy_dry_run is True
        assert eng._tripwire_forced_dry_today is True
        assert 'LATENCY TRIPWIRE' in caplog.text and 'tripwire_action=dry' in caplog.text

    def test_second_submit_after_tripwire_is_dry_zero_orders(self, tmp_path):
        eng, a, db, notifier = _engine('dry')
        _run_tick(eng, 'DRY1', tmp_path)
        submitted2 = _run_tick(eng, 'DRY2', tmp_path)
        assert submitted2 == []  # WOULD BUY only — nothing submitted this tick
        assert eng.candidates['DRY2'].rejected_reason == 'production_dry_run'
        # Only the FIRST (pre-tripwire) submit reached the real order call.
        assert a.submit_stop_bracket_order.call_count == 1

    def test_would_buy_line_present_after_tripwire(self, caplog, tmp_path):
        eng, a, db, notifier = _engine('dry')
        _run_tick(eng, 'DRY1', tmp_path)
        with caplog.at_level(logging.INFO):
            _run_tick(eng, 'DRY2', tmp_path)
        assert '[ORB DRY] WOULD BUY DRY2' in caplog.text

    def test_tripwire_telegram_text_sent(self, tmp_path):
        eng, a, db, notifier = _engine('dry')
        _run_tick(eng, 'DRY1', tmp_path)
        texts = [c.args[0] for c in notifier.send_message.call_args_list]
        assert any('TRIPWIRE → DRY for today' in t for t in texts)

    def test_reset_daily_restores_config_baseline(self, tmp_path):
        eng, a, db, notifier = _engine('dry')
        _run_tick(eng, 'DRY1', tmp_path)
        assert eng.strategy_dry_run is True
        eng.reset_daily()
        assert eng.strategy_dry_run is False  # config baseline was dry_run=False
        assert eng._tripwire_forced_dry_today is False
        assert eng._first_submit_latency_logged is False
