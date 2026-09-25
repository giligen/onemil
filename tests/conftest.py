"""Shared pytest fixtures (CLAUDE.md: fixtures live here, not duplicated
per test file).

Currently backs the BF wrapper-guard tests (2026-09-24 RGTZ parity defect,
docs/bf_wrapper_guard_spec_20260924.md): a real `TradingEngine` wired with
`MagicMock(spec=...)` collaborators so `on_stock_qualified` / `run_pattern_check`
exercise the actual production choke point, not a re-description of it.
"""
import sqlite3
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database
from trading.order_executor import OrderExecutor
from trading.pattern_detector import BullFlagDetector
from trading.position_manager import PositionManager
from trading.trade_planner import TradePlanner
from trading.trading_engine import TradingEngine


@pytest.fixture(autouse=True)
def _isolated_guardrail_state(tmp_path, monkeypatch):
    """No test ever reads or writes the PRODUCTION guardrail state file.

    trading/live_guardrail.py's boot-time `is_paused()` (called by every
    live engine at init) defaults to `data/guardrail_state.json` unless
    `ONEMIL_GUARDRAIL_STATE` is set. Point every test at a fresh temp file
    so engine tests are never gated by today's real pause state."""
    monkeypatch.setenv("ONEMIL_GUARDRAIL_STATE", str(tmp_path / "guardrail_state_test.json"))


@pytest.fixture
def bf_db(tmp_path):
    """Real Database on a temp file, closed after the test."""
    database = Database(db_path=str(tmp_path / "bf_conftest.db"))
    yield database
    database.close()


@pytest.fixture
def bf_mock_alpaca():
    """Mocked AlpacaClient (domain class, so MagicMock(spec=...) per CLAUDE.md)."""
    return MagicMock(spec=AlpacaClient)


@pytest.fixture
def bf_engine(bf_mock_alpaca, bf_db):
    """Real TradingEngine with mocked collaborators, for BF wrapper-guard tests."""
    eng = TradingEngine(
        alpaca_client=bf_mock_alpaca,
        db=bf_db,
        detector=MagicMock(spec=BullFlagDetector),
        planner=MagicMock(spec=TradePlanner),
        executor=MagicMock(spec=OrderExecutor),
        position_manager=MagicMock(spec=PositionManager),
        pattern_poll_interval=60,
        enabled=True,
    )
    eng.quality_filter_enabled = False
    eng.conviction_enabled = False
    eng.news_gate_enabled = False
    return eng


@pytest.fixture
def guardrail_trades_db(tmp_path):
    """A real trades.db (full schema, via persistence.database.Database — includes
    the `strategy` migration column) at a temp path, for tests/test_live_guardrail.py.
    Never touches production data/trades.db."""
    path = tmp_path / "guardrail_trades.db"
    db = Database(trades_path=str(path), cache_path=str(tmp_path / "guardrail_cache.db"))
    db.close()
    return path


@pytest.fixture
def insert_trade(guardrail_trades_db):
    """Insert one closed trade row into `guardrail_trades_db` for guardrail tests.

    Defaults give a coherent, riskless-to-reason-about fill (entry 100, stop 99,
    100 shares -> total_risk 100) that individual tests override as needed.
    """
    def _insert(strategy, trade_date, pnl, total_risk=None, entry_price=100.0,
               stop_loss_price=99.0, shares=100, symbol='TEST', exited_at=None):
        conn = sqlite3.connect(str(guardrail_trades_db))
        now = datetime.now(timezone.utc).isoformat()
        risk = total_risk if total_risk is not None else abs(entry_price - stop_loss_price) * shares
        conn.execute(
            "INSERT INTO trades (trade_date, symbol, side, entry_price, stop_loss_price, "
            "take_profit_price, shares, risk_per_share, total_risk, risk_reward_ratio, "
            "pnl, exited_at, strategy, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (trade_date, symbol, 'buy', entry_price, stop_loss_price,
             entry_price + 2 * (entry_price - stop_loss_price), shares,
             abs(entry_price - stop_loss_price), risk, 2.0, pnl,
             exited_at or f"{trade_date}T12:00:00", strategy, now, now))
        conn.commit()
        conn.close()
    return _insert
