"""ORB _save_pending_trade recovery when save_trade raises but the row landed (2026-11-19)."""
import logging
import sqlite3
from unittest.mock import MagicMock

import pytest

from persistence.database import Database
from tests.test_orb_integration import (  # noqa: F401  (fixtures)
    orb_cfg, real_db, mock_alpaca, mock_stop_monitor, engine,
)

_RACE = sqlite3.OperationalError("cannot commit - no transaction is active")


@pytest.fixture
def plan_engine(engine):
    """Engine with a spec'd mock DB plus a real planner-built plan."""
    plan = engine.planner.build(
        symbol='BEX', range_high=100.0, range_low=95.0,
        composite_score=0.5, quintile='Q4', adaptive_mult=0.95,
    )
    engine.db = MagicMock(spec=Database)
    return engine, plan


def test_row_exists_after_raise_is_adopted(plan_engine, caplog):
    """save_trade raises, row exists by order_id -> its id is returned, no ERROR."""
    engine, plan = plan_engine
    engine.db.save_trade.side_effect = _RACE
    engine.db.get_trade_by_order_id.return_value = {'id': 827}
    with caplog.at_level(logging.WARNING):
        assert engine._save_pending_trade(plan, order_id='oid-1') == 827
    assert engine.db.save_trade.call_count == 1
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert any('adopting' in r.getMessage() for r in caplog.records)


def test_retry_once_when_row_absent(plan_engine, caplog):
    """Raise once, row absent, retry succeeds -> id returned with a retry WARNING."""
    engine, plan = plan_engine
    engine.db.save_trade.side_effect = [_RACE, 5]
    engine.db.get_trade_by_order_id.return_value = None
    with caplog.at_level(logging.WARNING):
        assert engine._save_pending_trade(plan, order_id='oid-2') == 5
    assert any('retry' in r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_double_failure_returns_none_with_error(plan_engine, caplog):
    """Raise twice, row absent -> None and exactly one ERROR."""
    engine, plan = plan_engine
    engine.db.save_trade.side_effect = [_RACE, _RACE]
    engine.db.get_trade_by_order_id.return_value = None
    with caplog.at_level(logging.WARNING):
        assert engine._save_pending_trade(plan, order_id='oid-3') is None
    assert len([r for r in caplog.records if r.levelno >= logging.ERROR]) == 1
