"""docs/hod_resting_entry_spec_20260925.md — resting buy-stop-limit entry (dry only).

Pure-function tests on `trading.hod_break.arm_state` / `resting_entry_fill`, engine tests on
`HodBreakEngine._evaluate_resting` (entry_mode='resting_stop_limit', dry_run=True — never submits an order),
an `entry_mode='next_open'` parity check (today's behaviour must be byte-identical), and a PARITY test against
research/hod_entry/causal_arming.py's `arm_state` (the same rule, walked offline for cell 1,438).
"""
import csv
import sqlite3
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database
from trading.stop_monitor import StopMonitor
from trading.hod_break import HodBreakParams, arm_state, resting_entry_fill
from trading.hod_break_engine import HodBreakEngine
from tests.test_hod_break import bars as bars_arrays
from tests.test_hod_break_engine import cfg, bars_df, admit

ADV20 = 1_000_000.0   # matches the 'ABC' fixture's avg_volume_daily in test_hod_break_engine.mock_db


def tape(n_consol=5, break_high=11.2, next_open=11.05):
    """open 10 -> drives to 11 (10% above open) -> n_consol bars holding 10.7-11.0 -> break bar -> next bar."""
    t = [(10.0, 10.3, 9.9, 10.3, 5000), (10.3, 10.7, 10.2, 10.7, 6000), (10.7, 11.0, 10.6, 10.95, 8000)]
    t += [(10.9, 10.98, 10.7, 10.85, 3000)] * n_consol
    t += [(10.9, break_high, 10.85, 11.1, 9000)]
    t += [(next_open, 11.1, 11.0, 11.05, 4000)]
    return t


class TestArmState:
    """Pure function: armed at the close of bar j for bar j+1, from bars 0..j only."""

    def test_arms_exactly_when_consolidation_completes(self):
        o, h, l, c, v, m = bars_arrays(tape())
        p = HodBreakParams()
        # window l[j-4:j+1]: at j=5 it includes the drive bar at idx1 (low 10.2, outside the 4% band) -> None;
        # at j=6 the window is l[2:7] (idx2's low 10.6 is inside the band) -> the earliest valid consolidation.
        assert arm_state(o, h, l, v, m, 5, ADV20, p) is None
        a = arm_state(o, h, l, v, m, 6, ADV20, p)
        assert a is not None and a['level'] == pytest.approx(11.0) and a['trigger'] == pytest.approx(11.01)
        assert a['limit'] == pytest.approx(11.0 * 1.0015) and a['stop'] == pytest.approx(10.6)

    def test_min_dist_open_pct_boundary(self):
        o, h, l, c, v, m = bars_arrays(tape())
        p = HodBreakParams(min_dist_open_pct=10.0)                       # level 11.0 = exactly 10% above open 10.0
        assert arm_state(o, h, l, v, m, 7, ADV20, p) is not None
        p2 = HodBreakParams(min_dist_open_pct=10.01)
        assert arm_state(o, h, l, v, m, 7, ADV20, p2) is None

    def test_rv_band_boundary(self):
        o, h, l, c, v, m = bars_arrays(tape())
        cumv = float(np.sum(v[:8]))                                      # cumv through bar j=7
        lo_adv = cumv / (1.0 * hb_fraction(m[7]))                        # rv == rv_lo exactly
        p = HodBreakParams()
        assert arm_state(o, h, l, v, m, 7, lo_adv, p) is not None
        assert arm_state(o, h, l, v, m, 7, lo_adv * 1.0001, p) is None   # rv just under rv_lo -> not armed

    def test_last_entry_minute_boundary(self):
        o, h, l, c, v, m = bars_arrays(tape())
        p = HodBreakParams(last_entry_minute=int(m[8]))
        assert arm_state(o, h, l, v, m, 7, ADV20, p) is not None
        p2 = HodBreakParams(last_entry_minute=int(m[8]) - 1)
        assert arm_state(o, h, l, v, m, 7, ADV20, p2) is None

    def test_rearms_after_a_no_cross_bar(self):
        """Bar 8 doesn't cross (high stays under trigger); bar 9 still consolidates -> re-armed for bar 10."""
        t = tape()
        t[8] = (10.9, 10.99, 10.85, 10.95, 9000)                         # no break: high 10.99 < trigger 11.01
        t.append((10.95, 11.2, 10.9, 11.1, 9000))                        # bar 9: the real break
        o, h, l, c, v, m = bars_arrays(t)
        p = HodBreakParams()
        a8 = arm_state(o, h, l, v, m, 7, ADV20, p)
        assert a8 is not None and h[8] < a8['trigger']                   # armed for bar 8, bar 8 doesn't cross
        a9 = arm_state(o, h, l, v, m, 8, ADV20, p)
        assert a9 is not None and a9['trigger'] == pytest.approx(a8['trigger'])   # re-armed for bar 9 with the same level


def hb_fraction(minute):
    from trading.hod_break import profile_fraction
    return profile_fraction(int(minute))


class TestRestingEntryFill:
    def test_fills_at_ask_at_or_under_limit(self):
        arm = dict(level=11.0, trigger=11.01, limit=11.0165, stop=10.7)
        assert resting_entry_fill(11.0165, arm) == pytest.approx(11.0165)
        assert resting_entry_fill(11.00, arm) == pytest.approx(11.00)

    def test_no_fill_above_limit(self):
        arm = dict(level=11.0, trigger=11.01, limit=11.0165, stop=10.7)
        assert resting_entry_fill(11.02, arm) is None


# --------------------------------------------------------------------------------------------- engine (dry only)
@pytest.fixture
def trades_db(tmp_path):
    p = tmp_path / 'trades.db'
    con = sqlite3.connect(p); con.execute("create table trades (id integer primary key, strategy text, trade_date text, pnl real, symbol text, order_status text)"); con.commit(); con.close()
    return p


@pytest.fixture
def mock_alpaca():
    a = MagicMock(spec=AlpacaClient)
    a.get_latest_quote.return_value = {'bid_price': 11.00, 'ask_price': 11.015, 'bid_size': 100, 'ask_size': 100}  # <= limit 11.0165
    a.get_1min_bars_multi.return_value = {}
    a.get_open_positions.return_value = []
    return a


@pytest.fixture
def mock_db(trades_db):
    d = MagicMock(spec=Database)
    d._trades_path = str(trades_db)
    d.get_active_universe.return_value = [{'symbol': 'ABC', 'avg_volume_daily': 1_000_000}]
    d.get_open_trades.return_value = []
    return d


@pytest.fixture
def mock_sm():
    s = MagicMock(spec=StopMonitor); s.polling_mode = False; return s


def resting_engine(mock_alpaca, mock_db, mock_sm, tmp_path, **over):
    c = cfg(dry_run=True, entry_mode='resting_stop_limit', **over)
    c['dry_ledger_path'] = str(tmp_path / 'hod_dry_entry_ledger.csv')
    e = HodBreakEngine(mock_alpaca, mock_db, mock_sm, cfg=c)
    e._roll_session()
    # 2026-09-25 live_since fix: production sets live_since from the FIRST real websocket bar
    # (HodBreakEngine._on_bar_close); these tests feed bars straight to _ingest_bars and never go through
    # that handler, so without this every synthetic bar would be (correctly) treated as pre-live and skipped.
    # Back-date live_since to before minute 0 of the session so the whole synthetic day counts as live.
    e.live_since = e._bar_close_et(0)
    return e


def read_ledger(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


class TestRestingStopLimitEngine:
    def test_dry_fill_writes_one_ledger_row_and_never_submits(self, mock_alpaca, mock_db, mock_sm, tmp_path):
        e = resting_engine(mock_alpaca, mock_db, mock_sm, tmp_path)
        admit(e); e._ingest_bars('ABC', bars_df(tape()))
        rows = read_ledger(e.dry_ledger_path)
        assert len(rows) == 1 and rows[0]['symbol'] == 'ABC' and rows[0]['filled'] == '1'
        assert float(rows[0]['fill_px']) == pytest.approx(11.015)         # the mocked ask
        assert e.candidates['ABC'].resting_filled is True
        assert not mock_alpaca.submit_bracket_order.called
        assert e.positions == {}

    def test_one_fill_per_symbol_day(self, mock_alpaca, mock_db, mock_sm, tmp_path):
        e = resting_engine(mock_alpaca, mock_db, mock_sm, tmp_path)
        admit(e); e._ingest_bars('ABC', bars_df(tape()))
        e._ingest_bars('ABC', bars_df(tape() + [(11.05, 11.3, 11.0, 11.2, 2000)], minute0=570))
        rows = read_ledger(e.dry_ledger_path)
        assert len(rows) == 1                                              # no second fill after the first

    def test_no_fill_above_limit_is_logged_and_re_arms(self, mock_alpaca, mock_db, mock_sm, tmp_path):
        mock_alpaca.get_latest_quote.return_value = {'bid_price': 11.20, 'ask_price': 11.30, 'bid_size': 100, 'ask_size': 100}  # ask > limit
        e = resting_engine(mock_alpaca, mock_db, mock_sm, tmp_path)
        admit(e); e._ingest_bars('ABC', bars_df(tape()))
        rows = read_ledger(e.dry_ledger_path)
        assert len(rows) == 1 and rows[0]['filled'] == '0'
        assert e.candidates['ABC'].resting_filled is False


class TestEntryModeDefaultParity:
    def test_default_entry_mode_is_next_open(self, mock_alpaca, mock_db, mock_sm):
        e = HodBreakEngine(mock_alpaca, mock_db, mock_sm, cfg=cfg())
        assert e.entry_mode == 'next_open'

    def test_next_open_path_is_untouched_by_the_resting_fields(self, mock_alpaca, mock_db, mock_sm):
        """entry_mode='next_open' (default) never touches resting_arm/resting_filled — byte-identical to
        pre-existing behaviour: the ordinary detect() -> _try_enter path runs and submits the bracket order."""
        mock_alpaca.submit_bracket_order.return_value = {'id': 'o1', 'status': 'accepted', 'legs': []}
        e = HodBreakEngine(mock_alpaca, mock_db, mock_sm, cfg=cfg(dry_run=False)); e._roll_session()
        admit(e); e._ingest_bars('ABC', bars_df(tape()[:-1]))
        assert 'ABC' in e.positions and mock_alpaca.submit_bracket_order.called
        assert e.candidates['ABC'].resting_arm is None and e.candidates['ABC'].resting_filled is False


class TestParityWithResearch:
    """trading.hod_break.arm_state must agree with research/hod_entry/causal_arming.py's arm_state (cell 1,438's
    research build) on a synthetic day. Skips if the research module can't be imported in this environment."""

    def test_agrees_with_causal_arming_on_a_synthetic_day(self):
        try:
            import os, sys
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'research', 'hod_entry'))
            import causal_arming as ca
        except Exception as e:
            pytest.skip(f"research/hod_entry/causal_arming.py not importable here: {e}")
        o, h, l, c, v, m = bars_arrays(tape())
        p = HodBreakParams()
        for j in range(2, len(o) - 1):
            mine = arm_state(o, h, l, v, m, j, ADV20, p)
            theirs = ca.arm_state(o, h, l, v, m, j, ADV20, p, floor=0.0, use_rv=True)
            assert (mine is None) == (theirs is None), f"disagree at j={j}: mine={mine} theirs={theirs}"
            if mine is not None:
                for k in ('level', 'trigger', 'limit', 'stop'):
                    assert mine[k] == pytest.approx(theirs[k]), f"{k} differs at j={j}: {mine[k]} vs {theirs[k]}"


def test_config_accessor_passes_entry_mode_through(monkeypatch, tmp_path):
    """2026-09-25 live finding: Config.hod_break_cfg whitelists keys; entry_mode must reach the engine."""
    import config as cfgmod
    c = cfgmod.Config()
    monkeypatch.setattr(c, '_get_yaml', lambda *a, **k: {'enabled': True, 'dry_run': True,
                                                          'entry_mode': 'resting_stop_limit',
                                                          'dry_ledger_path': str(tmp_path / 'l.csv')})
    out = c.hod_break_cfg
    assert out['entry_mode'] == 'resting_stop_limit'
    assert out['dry_ledger_path'].endswith('l.csv')
    monkeypatch.setattr(c, '_get_yaml', lambda *a, **k: {})
    assert c.hod_break_cfg['entry_mode'] == 'next_open'
