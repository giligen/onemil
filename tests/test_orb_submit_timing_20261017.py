"""ORB submit-timing instrumentation (docs/orb_submit_timing_instrumentation_20261017.md). Logging only:
timing line format + ET conversion, missing-timestamp WARNING once, tripwire on the broker clock, burst serial
detection, EOD parse, and the real `_submit_entry` wiring (fake Alpaca response carrying `submitted_at`)."""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import yaml

from data_sources.alpaca_client import AlpacaClient, _broker_ts_iso
from persistence.database import Database
from trading import orb_submit_timing as ost
from trading.orb_engine import ORBEngine
from trading.stop_monitor import StopMonitor


def _engine():
    cfg = yaml.safe_load(open(Path(__file__).parent.parent / 'orb.yaml'))
    cfg['strategy']['enabled'] = True
    return ORBEngine(alpaca_client=MagicMock(spec=AlpacaClient), db=MagicMock(spec=Database),
                     stop_monitor=MagicMock(spec=StopMonitor), config=cfg)


def _plan(sym='CORD'):
    return SimpleNamespace(
        symbol=sym, shares=100, entry_price=10.03, stop_price=9.5, range_high=10.0, range_low=9.5,
        range_size=0.5, lock_arm_at_r=1.0, lock_stop_r=0.0, composite_score=1.0, quintile='Q3',
        adaptive_mult=1.0, pool_id='production', pool_exit={}, add_on_cfg={})


class TestTimingLine:
    def test_broker_minus_decision_and_et_conversion(self):
        # 2026-10-16 09:35:08.100 ET = 13:35:08.100 UTC (EDT, UTC-4)
        dec = datetime(2026, 10, 16, 13, 35, 8, 100000, tzinfo=timezone.utc).timestamp()
        broker = datetime(2026, 10, 16, 13, 35, 15, 300000, tzinfo=timezone.utc)
        line, delta = ost.build_timing_line('CORD', dec, dec + 7.0, dec + 7.25, broker, 'production',
                                            quote_ms=120.4, size_ms=None)
        assert delta == pytest.approx(7.2)
        assert ("[ORB] SUBMIT TIMING sym=CORD decision_ts=09:35:08.100 quote_ms=120 size_ms=NA http_ms=250 "
                "decision_to_http_start_s=7.00 broker_submitted_at=09:35:15.300 broker_minus_decision_s=7.20 "
                "path=production") == line

    def test_response_timestamp_parse_aware_utc_and_fallback(self):
        ts = ost.broker_ts_from_response({'submitted_at': '2026-10-16T13:35:15.300000+00:00'})
        assert ts == datetime(2026, 10, 16, 13, 35, 15, 300000, tzinfo=timezone.utc)
        ts2 = ost.broker_ts_from_response({'submitted_at': None, 'created_at': '2026-10-16T13:35:15Z'})
        assert ts2.second == 15

    def test_missing_timestamps_na_and_one_warning(self, caplog):
        ost._missing_warned_on = None
        with caplog.at_level(logging.WARNING):
            assert ost.broker_ts_from_response({'id': 'x'}) is None
            assert ost.broker_ts_from_response({}) is None
        assert sum('neither submitted_at nor created_at' in r.message for r in caplog.records) == 1
        line, delta = ost.build_timing_line('A', 100.0, 100.0, 100.1, None, 'addon')
        assert delta is None and 'broker_submitted_at=NA broker_minus_decision_s=NA path=addon' in line

    def test_wrapper_ts_iso_only_for_real_datetimes(self):
        assert _broker_ts_iso(datetime(2026, 10, 16, 13, 35, tzinfo=timezone.utc)).startswith('2026-10-16T13:35')
        assert _broker_ts_iso(MagicMock()) is None and _broker_ts_iso(None) is None


class TestBurst:
    def test_serial_vs_concurrent(self):
        assert ost.build_burst_line([(0.0, 1.0), (1.0, 2.0), (2.1, 3.0)]) == \
            "[ORB] SUBMIT BURST n=3 span_s=3.00 serial=yes"
        assert ost.build_burst_line([(0.0, 1.0), (0.1, 1.2)]).endswith("serial=no")
        assert ost.build_burst_line([(0.0, 1.0)]).endswith("n=1 span_s=1.00 serial=no")
        assert ost.build_burst_line([]) is None


class TestEodParse:
    def test_median_max_both_clocks(self):
        lines = [
            "x [ORB] SUBMIT TIMING sym=A decision_ts=09:35:01.000 quote_ms=1 size_ms=NA http_ms=5 "
            "decision_to_http_start_s=0.10 broker_submitted_at=09:35:05.000 broker_minus_decision_s=4.00 path=production",
            "x [ORB] SUBMIT TIMING sym=B decision_ts=09:35:01.000 quote_ms=1 size_ms=NA http_ms=5 "
            "decision_to_http_start_s=0.10 broker_submitted_at=09:35:11.000 broker_minus_decision_s=10.00 path=production",
            "x [ORB] SUBMIT TIMING sym=C decision_ts=09:35:01.000 quote_ms=1 size_ms=NA http_ms=5 "
            "decision_to_http_start_s=0.10 broker_submitted_at=NA broker_minus_decision_s=NA path=production",
            "x [ORB] SUBMIT LATENCY sym=A seconds=2.00", "x [ORB] SUBMIT LATENCY sym=B seconds=3.00"]
        assert ost.timing_summary_line(lines) == (
            "ORB submit latency: broker median 7.0s max 10.0s (n 2) | engine median 2.5s max 3.0s (n 2)")
        assert 'broker NO-DATA' in ost.timing_summary_line([])

    def test_eod_sections_orb_line_present(self):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
        import eod_sections as es
        assert es.timing_summary_line is ost.timing_summary_line


class TestEngineWiring:
    def test_submit_entry_emits_timing_line_with_broker_clock(self, caplog):
        eng = _engine()
        eng.dry_run = False
        eng._save_pending_trade = MagicMock(return_value=1)
        eng.alpaca.get_latest_quote.return_value = {'bid_price': 9.9, 'ask_price': 9.95, 'bid_size': 1, 'ask_size': 1}
        t0 = time.time()
        sub_at = datetime.fromtimestamp(t0 + 5.0, tz=timezone.utc).isoformat()
        eng.alpaca.submit_stop_bracket_order.return_value = {'id': 'oid1', 'legs': [], 'submitted_at': sub_at}
        ost._missing_warned_on = None
        with caplog.at_level(logging.INFO):
            oid = eng._submit_entry(_plan(), path='production', decision_ts=t0)
        assert oid == 'oid1'
        msgs = [r.message for r in caplog.records if 'SUBMIT TIMING' in r.message]
        assert len(msgs) == 1 and 'sym=CORD' in msgs[0] and 'path=production' in msgs[0]
        m = ost.TIMING_RE.search(msgs[0])
        assert float(m.group('b_minus_d')) == pytest.approx(5.0, abs=0.01)
        assert len(eng._broker_submit_ts) == 1 and len(eng._submit_spans) == 1

    def test_addon_pool_path_label(self, caplog):
        eng = _engine()
        eng.dry_run = False
        eng._save_pending_trade = MagicMock(return_value=1)
        eng.alpaca.get_latest_quote.return_value = {'bid_price': 9.9, 'ask_price': 9.95}
        eng.alpaca.submit_stop_bracket_order.return_value = {'id': 'o', 'legs': []}
        p = _plan('ZZ')
        p.pool_id = 'p30'
        with caplog.at_level(logging.INFO):
            eng._submit_entry(p)
        assert any('path=addon' in r.message and 'broker_submitted_at=NA' in r.message for r in caplog.records)

    def _tripwire_engine(self, engine_delay, broker_delay, paper=False):
        eng = _engine()
        eng.alpaca.is_paper = paper
        eng.tripwire_action = 'warn'
        eng.notifier = None
        eng._notify = MagicMock()
        target = datetime(2026, 10, 16, 9, 35, tzinfo=__import__('zoneinfo').ZoneInfo('America/New_York'))
        eng._et_now = lambda: target + __import__('datetime').timedelta(seconds=engine_delay)
        eng._first_submit_latency_logged = False
        if broker_delay is not None:
            eng._broker_submit_ts = [(target + __import__('datetime').timedelta(seconds=broker_delay))
                                     .astimezone(timezone.utc)]
        return eng

    def test_tripwire_uses_broker_clock_when_present(self, caplog):
        eng = self._tripwire_engine(engine_delay=7.0, broker_delay=12.0)
        with caplog.at_level(logging.INFO):
            eng._check_first_submit_latency()
        trip = [r for r in caplog.records if 'LATENCY TRIPWIRE: first order submit' in r.message]
        assert len(trip) == 1 and 'clock=broker' in trip[0].message and '12.0s after' in trip[0].message
        assert 'clock=broker' in eng._notify.call_args[0][0]

    def test_tripwire_engine_clock_fallback_labelled(self, caplog):
        eng = self._tripwire_engine(engine_delay=7.0, broker_delay=None)
        with caplog.at_level(logging.INFO):
            eng._check_first_submit_latency()
        ok = [r for r in caplog.records if 'first order submit 7.0s after 09:35 ET' in r.message]
        assert len(ok) == 1 and 'clock=engine' in ok[0].message
        assert not any('TRIPWIRE' in r.message for r in caplog.records)

    def test_tripwire_broker_clock_within_threshold_does_not_trip(self, caplog):
        eng = self._tripwire_engine(engine_delay=30.0, broker_delay=6.0)
        with caplog.at_level(logging.INFO):
            eng._check_first_submit_latency()
        assert not any('TRIPWIRE' in r.message for r in caplog.records)
        assert any('clock=broker' in r.message and 'first order submit 6.0s' in r.message for r in caplog.records)

    def test_paper_endpoint_decides_on_engine_clock_and_logs_both(self, caplog):
        eng = self._tripwire_engine(engine_delay=7.0, broker_delay=15.0, paper=True)
        with caplog.at_level(logging.INFO):
            eng._check_first_submit_latency()
        assert not any('TRIPWIRE' in r.message for r in caplog.records)
        eng._notify.assert_not_called()
        ok = [r.message for r in caplog.records if 'first order submit 7.0s after 09:35 ET' in r.message]
        assert len(ok) == 1 and 'clock=engine' in ok[0]
        assert 'engine=7.0s broker=15.0s (decides on engine: paper endpoint queues orders ~7 s)' in ok[0]

    def test_live_endpoint_trips_on_broker_clock_and_logs_both(self, caplog):
        eng = self._tripwire_engine(engine_delay=7.0, broker_delay=12.0, paper=False)
        with caplog.at_level(logging.INFO):
            eng._check_first_submit_latency()
        trip = [r.message for r in caplog.records if 'LATENCY TRIPWIRE: first order submit' in r.message]
        assert len(trip) == 1 and 'clock=broker' in trip[0]
        assert 'engine=7.0s broker=12.0s (decides on broker' in trip[0]
        assert '12s' in eng._notify.call_args[0][0]

    def test_live_without_submitted_at_uses_engine_clock(self, caplog):
        eng = self._tripwire_engine(engine_delay=7.0, broker_delay=None, paper=False)
        with caplog.at_level(logging.INFO):
            eng._check_first_submit_latency()
        ok = [r.message for r in caplog.records if 'first order submit 7.0s after 09:35 ET' in r.message]
        assert len(ok) == 1 and 'clock=engine' in ok[0] and 'broker=NA' in ok[0]

    def test_flush_burst_logs_and_detects_serial(self, caplog):
        eng = _engine()
        eng._submit_spans = [(0.0, 1.0), (1.0, 2.0)]
        with caplog.at_level(logging.INFO):
            eng._flush_submit_burst(0)
        assert any(r.message == "[ORB] SUBMIT BURST n=2 span_s=2.00 serial=yes" for r in caplog.records)


def test_parse_orb_log_collects_timing_lines():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    import eod_sections as es
    text = ("2026-10-16 13:35:09 | INFO | [ORB] SUBMIT TIMING sym=A decision_ts=09:35:01.000 quote_ms=1 size_ms=NA "
            "http_ms=5 decision_to_http_start_s=0.10 broker_submitted_at=09:35:05.000 broker_minus_decision_s=4.00 "
            "path=production\nunrelated line\n")
    parsed = es.parse_orb_log(text)
    assert len(parsed["timing_lines"]) == 1
    assert "broker median 4.0s" in es.timing_summary_line(parsed["timing_lines"])
