"""ORBEngine bulk-subscribes the daily universe seed so StopMonitor records session-open bars.

docs/orb_ws_range_path_fix_spec_20261110.md section 2.
"""
import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database
from trading.orb_engine import ORBEngine
from trading.stop_monitor import StopMonitor


def _engine(sm):
    cfg = yaml.safe_load(open(Path(__file__).parent.parent / 'orb.yaml'))
    cfg['strategy']['enabled'] = True
    alpaca = MagicMock(spec=AlpacaClient)
    alpaca.get_snapshots.return_value = {}
    return ORBEngine(alpaca_client=alpaca, db=MagicMock(spec=Database), stop_monitor=sm, config=cfg)


def test_seed_bulk_subscribed_once_and_idempotent(caplog):
    sm = MagicMock(spec=StopMonitor)
    sm.subscribe_bars_many.side_effect = [3, 0]
    eng = _engine(sm)
    with caplog.at_level(logging.INFO):
        assert eng.bulk_subscribe_seed(['AAA', 'BBB', 'CCC']) == 3
        assert eng.bulk_subscribe_seed(['AAA', 'BBB', 'CCC']) == 0
    assert sm.subscribe_bars_many.call_count == 2
    msgs = [r.message for r in caplog.records if 'bulk-subscribed' in r.message]
    assert msgs == ['[ORB] bulk-subscribed 3 seed symbols for session-open bars']   # second call: 0 new, silent


def test_test_tickers_excluded_from_the_bulk_subscribe():
    sm = MagicMock(spec=StopMonitor)
    sm.subscribe_bars_many.return_value = 2
    eng = _engine(sm)
    eng.bulk_subscribe_seed(['AAA', 'ZAZZT', 'BBB', 'ZVZZT', None])
    assert sm.subscribe_bars_many.call_args[0][0] == ['AAA', 'BBB']


def test_universe_build_path_subscribes_the_raw_seed_before_the_gap_gate():
    """The seed reaches the websocket from build_orb_universe_from_snapshots (the pre-open scanner tick)."""
    sm = MagicMock(spec=StopMonitor)
    sm.subscribe_bars_many.return_value = 2
    eng = _engine(sm)
    eng.build_orb_universe_from_snapshots(['AAA', 'BBB'])
    sm.subscribe_bars_many.assert_called_once_with(['AAA', 'BBB'])


def test_subscribe_failure_is_loud_not_fatal(caplog):
    sm = MagicMock(spec=StopMonitor)
    sm.subscribe_bars_many.side_effect = RuntimeError('ws down')
    eng = _engine(sm)
    with caplog.at_level(logging.ERROR):
        assert eng.bulk_subscribe_seed(['AAA']) == 0
    assert any('bulk subscribe' in r.message and r.levelno == logging.ERROR for r in caplog.records)


def test_no_stop_monitor_or_empty_seed_is_a_noop():
    assert _engine(None).bulk_subscribe_seed(['AAA']) == 0
    sm = MagicMock(spec=StopMonitor)
    assert _engine(sm).bulk_subscribe_seed([]) == 0
    sm.subscribe_bars_many.assert_not_called()
