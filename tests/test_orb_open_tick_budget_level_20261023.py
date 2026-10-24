"""2026-10-23: first exhausted ORB open tick is a WARNING; second consecutive or one ending >= 09:34 ET is an ERROR."""
import logging
import time
from datetime import datetime
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

from data_sources.alpaca_client import AlpacaClient
from tests.test_orb_open_tick_stall_20261002 import _engine, _snap, _today_et

ET = ZoneInfo('America/New_York')


def _levels(eng, caplog, now_et):
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        eng._report_open_tick_budget_exhausted('ORB: open-tick budget (20.0s) exhausted', now_et)
    return [r.levelname for r in caplog.records if 'budget' in r.message]


def test_first_exhaustion_is_warning_second_consecutive_is_error(caplog):
    eng = _engine(MagicMock(spec=AlpacaClient))
    t = datetime(2026, 10, 23, 9, 31, 27, tzinfo=ET)
    assert _levels(eng, caplog, t) == ['WARNING']
    assert _levels(eng, caplog, t) == ['ERROR']


def test_first_exhaustion_at_0934_is_error(caplog):
    eng = _engine(MagicMock(spec=AlpacaClient))
    assert _levels(eng, caplog, datetime(2026, 10, 23, 9, 34, 10, tzinfo=ET)) == ['ERROR']


def test_successful_tick_resets_counter(caplog):
    a = MagicMock(spec=AlpacaClient)
    syms = [f"S{i}" for i in range(5)]
    a.get_snapshots.return_value = {s: _snap(10.0, 9.0, _today_et()) for s in syms}
    eng = _engine(a)
    t = datetime(2026, 10, 23, 9, 31, 0, tzinfo=ET)
    assert _levels(eng, caplog, t) == ['WARNING']
    eng.build_orb_universe_from_snapshots(syms, deadline=time.time() + 60)     # finishes inside the budget
    assert eng._open_tick_exhausted_streak == 0
    assert _levels(eng, caplog, t) == ['WARNING']


def test_session_reset_clears_counter():
    eng = _engine(MagicMock(spec=AlpacaClient))
    eng._open_tick_exhausted_streak = 3
    eng.reset_daily()
    assert eng._open_tick_exhausted_streak == 0
