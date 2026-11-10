"""ORB post-open range sweep hardening + quiet selection observer (2026-11-10).

Spec: docs/orb_range_sweep_hardening_spec_20261110.md. Incident: vendor bars late
at the open, sweep applied 0/59 ranges, retried once and gave up silently.

Structure under test: the sweep is RE-ENTRANT (one paced pass per tick until
`sweep_deadline_et`), `done` is set only when everything filled or the deadline
passed; the final outcome is INFO / WARNING (3 classes) / one ERROR per day.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import yaml

from persistence.database import Database
from scripts import orb_selection_observer as obs
from scripts import report_common as rc
from data_sources.alpaca_client import AlpacaClient
from trading.orb_engine import ORBEngine
from trading.stop_monitor import StopMonitor

DAY = datetime(2026, 11, 9, tzinfo=timezone.utc)  # EST day: 09:30 ET == 14:30Z
T0935 = datetime(2026, 11, 9, 14, 35, 0, tzinfo=timezone.utc)
T0941 = datetime(2026, 11, 9, 14, 41, 0, tzinfo=timezone.utc)


class Clock:
    """Mutable frozen clock behind `trading.orb_engine.datetime.now`."""

    def __init__(self, now):
        self.now = now

    def __enter__(self):
        self._p = patch('trading.orb_engine.datetime')
        mock_dt = self._p.start()
        mock_dt.now = lambda tz=None: self.now
        mock_dt.combine = datetime.combine
        mock_dt.side_effect = lambda *a, **k: datetime(*a, **k)
        return self

    def __exit__(self, *exc):
        self._p.stop()


def make_bars(n_bars: int, base: float = 10.0) -> pd.DataFrame:
    """`n_bars` 1-min bars from 09:30 ET (14:30Z, EST); full range needs 5."""
    rows = []
    for i in range(n_bars):
        rows.append({'timestamp': datetime(2026, 11, 9, 14, 30 + i, tzinfo=timezone.utc),
                     'open': base, 'high': base + 0.1 * (i + 1), 'low': base - 0.05 * (i + 1),
                     'close': base + 0.05, 'volume': 10000})
    return pd.DataFrame(rows)


@pytest.fixture
def engine():
    cfg = yaml.safe_load(open(Path(__file__).parent.parent / 'orb.yaml'))
    cfg['strategy']['enabled'] = True
    alpaca = MagicMock(spec=AlpacaClient)
    alpaca.get_open_positions.return_value = []
    alpaca.get_account_info.return_value = {'buying_power': 100_000.0}
    db = MagicMock(spec=Database)
    db.save_trade.return_value = 1
    db.get_open_trades.return_value = []
    db.get_trades_by_date.return_value = []
    sm = MagicMock(spec=StopMonitor)
    sm.polling_mode = False
    sm.drain_exit_events.return_value = []
    eng = ORBEngine(alpaca_client=alpaca, db=db, stop_monitor=sm, config=cfg)
    eng.sweep_retry_delay_s = 0.01
    return eng


def _universe(engine, n):
    syms = [f"S{i:03d}" for i in range(n)]
    engine.build_universe(source_loader=lambda: syms)
    return syms


def _records(caplog, level):
    return [r for r in caplog.records if r.levelno == level and 'sweep' in r.getMessage()]


class TestRetryUntilFilled:
    def test_range_completes_on_third_pass(self, engine, caplog):
        caplog.set_level(logging.INFO)
        syms = _universe(engine, 1)
        calls = []

        def fake(symbols, lookback_minutes=60):
            calls.append(list(symbols))
            return {s: make_bars(3 if len(calls) < 3 else 5) for s in symbols}
        engine.alpaca.get_1min_bars_multi = fake
        with Clock(T0935) as clk:
            first = engine._ensure_ranges_post_open()   # pass 1 + sleep/retry 1 (3 bars each)
            assert first == set() and engine._post_open_range_sweep_done is False
            assert engine._sweep_retry_count_today == 1
            assert engine.candidates[syms[0]].range_data is None
            clk.now = datetime(2026, 11, 9, 14, 35, 6, tzinfo=timezone.utc)
            time.sleep(0.03)
            second = engine._ensure_ranges_post_open()  # re-entrant pass 3 (5 bars)
        assert second == {syms[0]}
        assert engine._post_open_range_sweep_done is True
        assert engine._sweep_retry_count_today == 2
        assert len(calls) == 3 and calls[2] == [syms[0]]
        msgs = [r.getMessage() for r in caplog.records]
        assert any('retry 2' in m for m in msgs)
        assert any('filled 1/1 ranges' in m for m in msgs)
        assert not _records(caplog, logging.WARNING) and not _records(caplog, logging.ERROR)

    def test_pacing_gate_skips_fetch_inside_delay(self, engine):
        _universe(engine, 2)
        engine.alpaca.get_1min_bars_multi = MagicMock(return_value={})
        engine.sweep_retry_delay_s = 0.01
        with Clock(T0935):
            engine._ensure_ranges_post_open()
            n = engine.alpaca.get_1min_bars_multi.call_count
            engine.sweep_retry_delay_s = 3600.0         # next tick lands inside the delay
            assert engine._ensure_ranges_post_open() == set()
        assert engine.alpaca.get_1min_bars_multi.call_count == n

    def test_deadline_with_leftovers_warns_three_classes_no_error(self, engine, caplog):
        caplog.set_level(logging.INFO)
        syms = _universe(engine, 4)

        def fake(symbols, lookback_minutes=60):
            out = {}
            for s in symbols:
                if s == syms[0]:
                    out[s] = make_bars(3)               # short_window
                elif s == syms[1]:
                    out[s] = make_bars(5).iloc[0:0]     # empty -> no_bars
                elif s == syms[2]:                      # bars exist, none at the 09:30 hour
                    b = make_bars(5)
                    b['timestamp'] = b['timestamp'] + pd.Timedelta(hours=1)
                    out[s] = b
                else:
                    out[s] = make_bars(5)
            return out
        engine.alpaca.get_1min_bars_multi = fake
        with Clock(T0941):
            filled = engine._ensure_ranges_post_open()
        assert filled == {syms[3]}
        assert engine._post_open_range_sweep_done is True
        warns = _records(caplog, logging.WARNING)
        gave_up = [r.getMessage() for r in warns if 'gave up' in r.getMessage()]
        assert len(gave_up) == 1
        msg = gave_up[0]
        assert 'short_window [S000]' in msg and 'no_bars [S001]' in msg and 'no_open_bar [S002]' in msg
        assert not _records(caplog, logging.ERROR)      # 1/4 filled but n < 10

    def test_zero_of_59_fires_error_once_per_day(self, engine, caplog):
        caplog.set_level(logging.INFO)
        _universe(engine, 59)
        engine.alpaca.get_1min_bars_multi = MagicMock(return_value={})
        with Clock(T0941):
            engine._ensure_ranges_post_open()
            errs = _records(caplog, logging.ERROR)
            assert len(errs) == 1
            m = errs[0].getMessage()
            assert 'vendor bars late' in m and 'only 0/59' in m and '09:41' in m
            # grace gate re-arms the sweep; the alarm must not repeat
            engine._post_open_range_sweep_done = False
            time.sleep(0.03)
            engine._ensure_ranges_post_open()
        assert len(_records(caplog, logging.ERROR)) == 1

    def test_nov9_shape_134_then_14_of_172_no_error(self, engine, caplog):
        caplog.set_level(logging.INFO)
        syms = _universe(engine, 172)
        first_ok, retry_ok = set(syms[:134]), set(syms[134:148])
        calls = []

        def fake(symbols, lookback_minutes=60):
            calls.append(len(symbols))
            ok = first_ok if len(calls) == 1 else retry_ok
            return {s: make_bars(5) for s in symbols if s in ok}
        engine.alpaca.get_1min_bars_multi = fake
        with Clock(T0935):
            filled = engine._ensure_ranges_post_open()
        assert len(filled) == 148 and calls == [172, 38]   # exactly one retry
        assert not _records(caplog, logging.WARNING) and not _records(caplog, logging.ERROR)
        # at the deadline the 24 leftovers warn, but 148/172 = 86 % never alarms
        with Clock(T0941):
            time.sleep(0.03)
            engine._ensure_ranges_post_open()
        assert engine._post_open_range_sweep_done is True
        assert len(_records(caplog, logging.WARNING)) >= 1
        assert not _records(caplog, logging.ERROR)

    def test_deadline_is_configurable(self):
        cfg = yaml.safe_load(open(Path(__file__).parent.parent / 'orb.yaml'))
        cfg['strategy']['enabled'] = True
        cfg['entry']['sweep_deadline_et'] = '09:38:30'
        eng = ORBEngine(alpaca_client=MagicMock(spec=AlpacaClient), db=MagicMock(spec=Database),
                        stop_monitor=MagicMock(spec=StopMonitor), config=cfg)
        assert (eng.sweep_deadline_et.hour, eng.sweep_deadline_et.minute,
                eng.sweep_deadline_et.second) == (9, 38, 30)


class TestSweepIntegration:
    def test_ranges_set_from_fed_bars_and_day_roll_ingests_nothing(self, engine):
        syms = _universe(engine, 2)
        # a) frozen 09:35:00 ET, full bars -> exact H/L on candidates
        # key the price ladder by the symbol's index in `syms`, not by the engine's iteration order
        engine.alpaca.get_1min_bars_multi = lambda symbols, lookback_minutes=60: {
            s: make_bars(5, base=10.0 + syms.index(s)) for s in symbols}
        with Clock(T0935):
            filled = engine._ensure_ranges_post_open()
        assert filled == set(syms) and engine._post_open_range_sweep_done is True
        for i, s in enumerate(syms):
            rd = engine.candidates[s].range_data
            assert rd is not None
            base = 10.0 + i
            assert rd.range_high == pytest.approx(base + 0.5)   # highest high = base + 0.1*5
            assert rd.range_low == pytest.approx(base - 0.25)   # lowest low = base - 0.05*5

    def test_day_rolled_during_retry_ingests_nothing(self, engine):
        syms = _universe(engine, 2)
        calls = []

        def fake(symbols, lookback_minutes=60):
            calls.append(1)
            if len(calls) == 2:
                engine.reset_daily()                    # daily reset mid-retry
                return {s: make_bars(5) for s in symbols}
            return {s: make_bars(3) for s in symbols}
        engine.alpaca.get_1min_bars_multi = fake
        with Clock(T0935):
            out = engine._ensure_ranges_post_open()
        assert out == set()
        assert all(engine.candidates.get(s) is None or engine.candidates[s].range_data is None
                   for s in syms)
        assert engine._post_open_range_sweep_done is False


class TestObserver:
    def _resp(self, n):
        r = MagicMock()
        r.json.return_value = {'bars': {'AAA': [{}] * n}}
        return r

    def test_window_excludes_0935_and_counts_out_of_five(self, monkeypatch):
        monkeypatch.setenv('ALPACA_API_KEY', 'k')
        monkeypatch.setenv('ALPACA_API_SECRET', 's')
        get = MagicMock(return_value=self._resp(5))
        now = datetime(2026, 11, 9, 15, 0, tzinfo=timezone.utc)
        out = obs.classify_drop('AAA', now=now, http_get=get)
        params = get.call_args.kwargs['params']
        assert params['start'] == '2026-11-09T14:30:00Z'
        assert params['end'] == '2026-11-09T14:34:59Z'      # exclusive of the 09:35 bar
        assert out.startswith('REAL DROP (5/5')
        get = MagicMock(return_value=self._resp(3))
        assert '3/5' in obs.classify_drop('AAA', now=now, http_get=get)

    def test_clean_verdict_sends_nothing_caught_sends(self):
        n = MagicMock()
        audit = {'date': '2026-11-09', 'live_range': ['A']}
        kept = {'A': {'gap': 5.0}}
        sent = obs.send_verdict(n, 'clean', audit, kept, ['A'], [], {}, [], True)
        assert sent is False and n.send_message_sync.call_count == 0
        sent = obs.send_verdict(n, 'caught', audit, kept, [], ['A'],
                                {'A': 'REAL DROP (5/5 range bars present — investigate)'}, ['A'], True)
        assert sent is True and n.send_message_sync.call_count == 1
        assert 'CAUGHT 1 REAL' in n.send_message_sync.call_args.args[0]
        sent = obs.send_verdict(n, 'ops_warning', audit, kept, [], [], {}, [], False)
        assert sent is True and n.send_message_sync.call_count == 2


class TestEodSelectionParityLine:
    def test_line_from_audit_json(self, tmp_path):
        (tmp_path / 'orb_selection_audit_20261110.json').write_text(json.dumps({
            'live_range': list('abcdefgh'), 'expected_candidates': {s: {} for s in 'abcdefghij'},
            'drop_classification': {'x': 'REAL DROP (5/5 range bars present — investigate)',
                                    'y': 'benign: vendor-stale phantom (zero bars in 9:30-9:35)'}}))
        line = rc.selection_parity_line('2026-11-10', logs_dir=tmp_path)
        assert line.startswith('selection parity: live ranged 8 / BT-eligible 10, REAL drops 1')

    def test_missing_and_corrupt_audit_never_raise(self, tmp_path):
        assert 'no observer audit' in rc.selection_parity_line('2026-11-10', logs_dir=tmp_path)
        (tmp_path / 'orb_selection_audit_20261110.json').write_text('{not json')
        assert 'unreadable' in rc.selection_parity_line('2026-11-10', logs_dir=tmp_path)
