"""Shared pytest fixtures (CLAUDE.md: fixtures live here, not duplicated
per test file).

Currently backs the BF wrapper-guard tests (2026-09-24 RGTZ parity defect,
docs/bf_wrapper_guard_spec_20260924.md): a real `TradingEngine` wired with
`MagicMock(spec=...)` collaborators so `on_stock_qualified` / `run_pattern_check`
exercise the actual production choke point, not a re-description of it.
"""
from unittest.mock import MagicMock

import pytest

from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database
from trading.order_executor import OrderExecutor
from trading.pattern_detector import BullFlagDetector
from trading.position_manager import PositionManager
from trading.trade_planner import TradePlanner
from trading.trading_engine import TradingEngine


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
