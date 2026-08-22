"""
Tests for data_sources/news_stream.py — Alpaca real-time news listener.

All tests mock the websocket layer (fake connect factories / raw frames);
no live connection is made except the optional NEWS_STREAM_LIVE_TEST-gated
integration test at the bottom (never runs in the default suite).
"""

import asyncio
import json
import logging
import os
import threading
import time
from datetime import date, datetime, timedelta, timezone

import pytest

from data_sources import news_stream
from data_sources.news_stream import (
    LatencyStats,
    NewsEvent,
    NewsStreamListener,
    TodayNewsIndex,
    nearest_rank_percentile,
    parse_iso8601,
    parse_news_message,
    run_shadow,
)


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
def make_listener(**overrides) -> NewsStreamListener:
    """Listener with test credentials and any overrides."""
    kwargs = dict(api_key='test-key', api_secret='test-secret')
    kwargs.update(overrides)
    return NewsStreamListener(**kwargs)


def news_msg(article_id=1, symbols=('AAPL',), headline='HL',
             created_at='2026-08-21T13:30:00Z',
             updated_at='2026-08-21T13:30:00Z', source='benzinga') -> dict:
    return {'T': 'n', 'id': article_id, 'symbols': list(symbols),
            'headline': headline, 'created_at': created_at,
            'updated_at': updated_at, 'source': source}


RECV_AT = datetime(2026, 8, 21, 13, 30, 2, tzinfo=timezone.utc)


class FakeClock:
    """Deterministic monotonic clock for staleness/gap tests."""

    def __init__(self, start=1000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, s: float) -> None:
        self.now += s


class FakeWebSocket:
    """Async-context-manager fake ws: yields queued frames then raises."""

    def __init__(self, frames, hang_when_empty=False):
        self.frames = list(frames)
        self.sent = []
        self.closed = False
        self.hang_when_empty = hang_when_empty

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self.closed = True
        return False

    async def send(self, msg):
        self.sent.append(json.loads(msg))

    async def recv(self):
        if self.frames:
            return self.frames.pop(0)
        if self.hang_when_empty:
            await asyncio.Event().wait()  # cancelled by wait_for timeout
        raise ConnectionError('fake socket closed')


# ---------------------------------------------------------------------- #
# parse_iso8601
# ---------------------------------------------------------------------- #
class TestParseIso8601:
    def test_z_suffix_parses_to_utc(self):
        dt = parse_iso8601('2026-08-21T13:30:00Z')
        assert dt == datetime(2026, 8, 21, 13, 30, tzinfo=timezone.utc)

    def test_offset_form_normalized_to_utc(self):
        dt = parse_iso8601('2026-08-21T09:30:00-04:00')
        assert dt == datetime(2026, 8, 21, 13, 30, tzinfo=timezone.utc)

    def test_empty_returns_none(self):
        assert parse_iso8601('') is None

    def test_garbage_returns_none_and_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            assert parse_iso8601('not-a-timestamp') is None
        assert 'unparseable timestamp' in caplog.text

    def test_naive_assumed_utc_with_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            dt = parse_iso8601('2026-08-21T13:30:00')
        assert dt == datetime(2026, 8, 21, 13, 30, tzinfo=timezone.utc)
        assert 'assuming UTC' in caplog.text


# ---------------------------------------------------------------------- #
# parse_news_message / NewsEvent
# ---------------------------------------------------------------------- #
class TestParseNewsMessage:
    def test_full_message(self):
        ev = parse_news_message(news_msg(), received_at=RECV_AT)
        assert ev.article_id == '1'
        assert ev.headline == 'HL'
        assert ev.symbols == ('AAPL',)
        assert ev.source == 'benzinga'
        assert ev.created_at == datetime(2026, 8, 21, 13, 30,
                                         tzinfo=timezone.utc)
        assert ev.received_at == RECV_AT
        assert ev.is_update is False

    def test_multi_symbol_one_event_full_set(self):
        """One article -> ONE event with ALL symbols (complex-group solver)."""
        ev = parse_news_message(
            news_msg(symbols=['tsla', 'AAPL', 'aapl ', 'MSTR']),
            received_at=RECV_AT)
        assert ev.symbols == ('AAPL', 'MSTR', 'TSLA')  # deduped, upper, sorted

    def test_empty_symbols_kept_as_empty_tuple(self):
        ev = parse_news_message(news_msg(symbols=[]), received_at=RECV_AT)
        assert ev.symbols == ()

    def test_non_news_type_returns_none(self):
        assert parse_news_message({'T': 'success', 'msg': 'connected'}) is None

    def test_missing_id_dropped_with_warning(self, caplog):
        msg = news_msg()
        del msg['id']
        with caplog.at_level(logging.WARNING):
            assert parse_news_message(msg, received_at=RECV_AT) is None
        assert 'without id' in caplog.text

    def test_recv_latency_math(self):
        ev = parse_news_message(news_msg(created_at='2026-08-21T13:30:00Z'),
                                received_at=RECV_AT)
        assert ev.recv_latency_s == pytest.approx(2.0)

    def test_recv_latency_none_when_created_at_unparseable(self):
        ev = parse_news_message(news_msg(created_at='bogus'),
                                received_at=RECV_AT)
        assert ev.recv_latency_s is None


# ---------------------------------------------------------------------- #
# nearest_rank_percentile / LatencyStats
# ---------------------------------------------------------------------- #
class TestLatencyStats:
    def test_nearest_rank_percentile(self):
        vals = [float(i) for i in range(1, 11)]  # 1..10
        assert nearest_rank_percentile(vals, 90.0) == 9.0
        assert nearest_rank_percentile(vals, 50.0) == 5.0
        assert nearest_rank_percentile([3.5], 90.0) == 3.5

    def test_nearest_rank_percentile_empty_raises(self):
        with pytest.raises(ValueError):
            nearest_rank_percentile([], 90.0)

    def test_median_p90_min_max(self):
        clock = FakeClock()
        stats = LatencyStats(time_fn=clock)
        for v in range(1, 11):  # latencies 1..10s
            stats.record_event(float(v))
            clock.advance(1.0)
        snap = stats.snapshot()
        assert snap['events_total'] == 10
        assert snap['latency_count'] == 10
        assert snap['latency_median_s'] == pytest.approx(5.5)
        assert snap['latency_p90_s'] == pytest.approx(9.0)
        assert snap['latency_min_s'] == pytest.approx(1.0)
        assert snap['latency_max_s'] == pytest.approx(10.0)

    def test_staleness_gaps(self):
        clock = FakeClock()
        stats = LatencyStats(gap_threshold_s=60.0, time_fn=clock)
        stats.record_event(1.0)          # t=1000
        clock.advance(10.0)
        stats.record_event(1.0)          # gap 10
        clock.advance(90.0)
        stats.record_event(1.0)          # gap 90 -> over threshold
        snap = stats.snapshot()
        assert snap['max_gap_s'] == pytest.approx(90.0)
        assert snap['gaps_over_60s'] == 1

    def test_none_latency_counts_event_not_latency(self):
        stats = LatencyStats(time_fn=FakeClock())
        stats.record_event(None)
        snap = stats.snapshot()
        assert snap['events_total'] == 1
        assert snap['latency_count'] == 0
        assert snap['latency_median_s'] is None

    def test_empty_snapshot(self):
        snap = LatencyStats(time_fn=FakeClock()).snapshot()
        assert snap['events_total'] == 0
        assert snap['latency_p90_s'] is None

    def test_rolling_window_caps_memory(self):
        stats = LatencyStats(window=5, time_fn=FakeClock())
        for v in range(100):
            stats.record_event(float(v))
        snap = stats.snapshot()
        assert snap['latency_count'] == 5
        assert snap['latency_min_s'] == 95.0  # only last 5 retained
        assert snap['events_total'] == 100


# ---------------------------------------------------------------------- #
# TodayNewsIndex
# ---------------------------------------------------------------------- #
class TestTodayNewsIndex:
    def _event(self, symbols=('AAPL',), article_id='1',
               created_at=None, received_at=RECV_AT) -> NewsEvent:
        return NewsEvent(article_id=article_id, headline='HL',
                         symbols=tuple(symbols), source='benzinga',
                         created_at=created_at, updated_at=None,
                         received_at=received_at)

    def test_add_and_query(self):
        idx = TodayNewsIndex(date_provider=lambda: date(2026, 8, 21))
        created = datetime(2026, 8, 21, 13, 29, tzinfo=timezone.utc)
        idx.add(self._event(symbols=('AAPL', 'TSLA'), created_at=created))
        idx.add(self._event(symbols=('AAPL',), article_id='2'))
        assert idx.symbols_with_news_today() == {'AAPL', 'TSLA'}
        assert len(idx.articles_for('AAPL')) == 2
        assert len(idx.articles_for('TSLA')) == 1
        assert idx.first_article_ts('AAPL') == created

    def test_lookup_case_insensitive(self):
        idx = TodayNewsIndex(date_provider=lambda: date(2026, 8, 21))
        idx.add(self._event())
        assert len(idx.articles_for('aapl')) == 1

    def test_first_article_ts_falls_back_to_received_at(self):
        idx = TodayNewsIndex(date_provider=lambda: date(2026, 8, 21))
        idx.add(self._event(created_at=None))
        assert idx.first_article_ts('AAPL') == RECV_AT

    def test_unknown_symbol(self):
        idx = TodayNewsIndex(date_provider=lambda: date(2026, 8, 21))
        assert idx.articles_for('ZZZZ') == []
        assert idx.first_article_ts('ZZZZ') is None

    def test_et_day_roll_clears_index(self):
        day = {'d': date(2026, 8, 21)}
        idx = TodayNewsIndex(date_provider=lambda: day['d'])
        idx.add(self._event())
        assert idx.symbols_with_news_today() == {'AAPL'}
        day['d'] = date(2026, 8, 22)  # ET midnight passes
        assert idx.symbols_with_news_today() == set()
        assert idx.articles_for('AAPL') == []
        idx.add(self._event(article_id='9'))
        assert idx.symbols_with_news_today() == {'AAPL'}


# ---------------------------------------------------------------------- #
# Listener construction
# ---------------------------------------------------------------------- #
class TestConstruction:
    def test_requires_credentials(self, monkeypatch):
        monkeypatch.delenv('ALPACA_API_KEY', raising=False)
        monkeypatch.delenv('ALPACA_API_SECRET', raising=False)
        with pytest.raises(ValueError, match='ALPACA_API_KEY'):
            NewsStreamListener()

    def test_env_credentials_accepted(self, monkeypatch):
        monkeypatch.setenv('ALPACA_API_KEY', 'k')
        monkeypatch.setenv('ALPACA_API_SECRET', 's')
        listener = NewsStreamListener()
        assert listener._api_key == 'k'


# ---------------------------------------------------------------------- #
# Frame handling: parse, dedup, updates, index, telemetry, log line
# ---------------------------------------------------------------------- #
class TestHandleRaw:
    def test_news_frame_delivers_event_and_indexes(self):
        listener = make_listener()
        events = listener._handle_raw(
            json.dumps([news_msg(symbols=['AAPL', 'TSLA'])]),
            received_at=RECV_AT)
        assert len(events) == 1
        assert events[0].symbols == ('AAPL', 'TSLA')
        assert listener.symbols_with_news_today() == {'AAPL', 'TSLA'}
        assert listener.last_event_ts == RECV_AT

    def test_duplicate_id_same_update_dropped(self):
        listener = make_listener()
        raw = json.dumps([news_msg()])
        assert len(listener._handle_raw(raw, received_at=RECV_AT)) == 1
        assert listener._handle_raw(raw, received_at=RECV_AT) == []
        stats = listener.latency_stats()
        assert stats['duplicates_dropped'] == 1
        assert stats['unique_articles'] == 1
        assert stats['events_total'] == 1

    def test_update_tracked_separately_and_flagged(self):
        listener = make_listener()
        listener._handle_raw(json.dumps([news_msg()]), received_at=RECV_AT)
        updated = news_msg(headline='HL v2',
                           updated_at='2026-08-21T13:35:00Z')
        events = listener._handle_raw(json.dumps([updated]),
                                      received_at=RECV_AT)
        assert len(events) == 1
        assert events[0].is_update is True
        stats = listener.latency_stats()
        assert stats['updates_seen'] == 1
        assert stats['duplicates_dropped'] == 0
        assert stats['unique_articles'] == 1

    def test_bad_json_dropped_with_warning(self, caplog):
        listener = make_listener()
        with caplog.at_level(logging.WARNING):
            assert listener._handle_raw('{not json') == []
        assert 'undecodable frame' in caplog.text

    def test_single_dict_frame_accepted(self):
        listener = make_listener()
        events = listener._handle_raw(json.dumps(news_msg()),
                                      received_at=RECV_AT)
        assert len(events) == 1

    def test_non_list_frame_dropped_with_warning(self, caplog):
        listener = make_listener()
        with caplog.at_level(logging.WARNING):
            assert listener._handle_raw('"just a string"') == []
        assert 'unexpected frame shape' in caplog.text

    def test_error_frame_logged_error(self, caplog):
        listener = make_listener()
        with caplog.at_level(logging.ERROR):
            listener._handle_raw(
                json.dumps([{'T': 'error', 'code': 401, 'msg': 'auth'}]))
        assert 'server error frame' in caplog.text

    def test_unknown_type_warned(self, caplog):
        listener = make_listener()
        with caplog.at_level(logging.WARNING):
            listener._handle_raw(json.dumps([{'T': 'weird'}]))
        assert 'unknown message type' in caplog.text

    def test_control_frames_ignored_quietly(self):
        listener = make_listener()
        raw = json.dumps([{'T': 'success', 'msg': 'authenticated'},
                          {'T': 'subscription', 'news': ['*']}])
        assert listener._handle_raw(raw) == []

    def test_latency_log_line_format(self, caplog):
        """The telemetry line the EOD dive greps for."""
        listener = make_listener()
        with caplog.at_level(logging.INFO, logger='data_sources.news_stream'):
            listener._handle_raw(
                json.dumps([news_msg(symbols=['AAPL', 'TSLA'])]),
                received_at=RECV_AT)
        line = next(r.getMessage() for r in caplog.records
                    if 'recv_latency=' in r.getMessage())
        assert '[NEWS-STREAM] AAPL,TSLA recv_latency=2.00s' in line

    def test_latency_stats_from_stream(self):
        listener = make_listener()
        for i, lat in enumerate([1, 2, 3]):
            msg = news_msg(article_id=100 + i,
                           created_at='2026-08-21T13:30:00Z')
            listener._handle_raw(
                json.dumps([msg]),
                received_at=RECV_AT.replace(second=lat) - timedelta(seconds=2))
        snap = listener.latency_stats()
        assert snap['latency_count'] == 3
        assert snap['latency_median_s'] == pytest.approx(0.0)


# ---------------------------------------------------------------------- #
# Consumer bus fault isolation
# ---------------------------------------------------------------------- #
class TestConsumerBus:
    def test_raising_consumer_does_not_stop_others(self, caplog):
        listener = make_listener()
        seen = []

        def bad_consumer(event):
            raise RuntimeError('boom')

        def good_consumer(event):
            seen.append(event.article_id)

        listener.register_callback(bad_consumer)
        listener.register_callback(good_consumer)
        with caplog.at_level(logging.ERROR):
            events = listener._handle_raw(json.dumps([news_msg()]),
                                          received_at=RECV_AT)
        assert len(events) == 1          # stream loop survived
        assert seen == ['1']             # later consumer still ran
        assert 'consumer bad_consumer raised' in caplog.text

    def test_all_consumers_receive_full_symbol_set(self):
        listener = make_listener()
        received = []
        listener.register_callback(lambda ev: received.append(ev.symbols))
        listener._handle_raw(
            json.dumps([news_msg(symbols=['GME', 'AMC', 'KOSS'])]),
            received_at=RECV_AT)
        assert received == [('AMC', 'GME', 'KOSS')]


# ---------------------------------------------------------------------- #
# Staleness
# ---------------------------------------------------------------------- #
class TestStaleness:
    def test_stale_before_start(self):
        assert make_listener().is_stale() is True

    def test_fresh_after_frame_then_stale(self):
        clock = FakeClock()
        listener = make_listener(time_fn=clock, stale_after_s=60.0)
        listener._started_mono = clock()
        listener._last_frame_mono = clock()
        assert listener.is_stale() is False
        clock.advance(59.0)
        assert listener.is_stale() is False
        clock.advance(2.0)
        assert listener.is_stale() is True
        assert listener.is_stale(threshold_s=120.0) is False

    def test_started_but_silent_goes_stale(self):
        clock = FakeClock()
        listener = make_listener(time_fn=clock, stale_after_s=60.0)
        listener._started_mono = clock()   # started, no frame ever
        clock.advance(61.0)
        assert listener.is_stale() is True


# ---------------------------------------------------------------------- #
# Backoff / reconnect
# ---------------------------------------------------------------------- #
class TestBackoff:
    def test_capped_exponential_ladder(self):
        listener = make_listener(backoff_base_s=1.0, backoff_max_s=60.0)
        delays = [listener._next_backoff_s(a) for a in range(1, 9)]
        assert delays == [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 60.0, 60.0]

    def test_run_forever_reconnects_after_failures(self, caplog):
        """Two failed connects -> WARNINGs with growing delay, then a good
        connection delivers a news frame; consumer stops the loop."""
        listener = make_listener(backoff_base_s=0.01, backoff_max_s=0.05)
        calls = {'n': 0}
        good_ws = FakeWebSocket([json.dumps([news_msg()])])

        def connect(url):
            calls['n'] += 1
            if calls['n'] <= 2:
                raise ConnectionError('refused %d' % calls['n'])
            return good_ws

        listener._connect_factory = connect
        listener.register_callback(
            lambda ev: listener._stop_requested.set())
        with caplog.at_level(logging.WARNING):
            asyncio.run(asyncio.wait_for(listener._run_forever(), timeout=10))
        warnings = [r.getMessage() for r in caplog.records
                    if 'reconnect #' in r.getMessage()]
        assert len(warnings) == 2
        assert 'reconnect #1 in 0.0s' in warnings[0]
        assert calls['n'] == 3
        assert listener.latency_stats()['reconnects'] == 2
        # auth + subscribe were sent on the live connection
        assert good_ws.sent[0]['action'] == 'auth'
        assert good_ws.sent[0]['key'] == 'test-key'
        assert good_ws.sent[1] == {'action': 'subscribe', 'news': ['*']}

    def test_attempt_resets_after_successful_connection(self):
        """A good connection resets the ladder: next failure backs off at
        base delay again, not the escalated one."""
        listener = make_listener(backoff_base_s=0.01, backoff_max_s=10.0)
        delays = []
        orig_sleep = listener._interruptible_sleep

        async def spy_sleep(delay):
            delays.append(delay)
            await orig_sleep(0.0)

        listener._interruptible_sleep = spy_sleep
        calls = {'n': 0}

        def connect(url):
            calls['n'] += 1
            if calls['n'] in (1, 2):        # two failures: 0.01, 0.02
                raise ConnectionError('down')
            if calls['n'] == 3:             # success (empty ws -> drops)
                return FakeWebSocket([])
            if calls['n'] == 4:             # failure AFTER success
                raise ConnectionError('down again')
            listener._stop_requested.set()
            raise ConnectionError('end of test')

        listener._connect_factory = connect
        asyncio.run(asyncio.wait_for(listener._run_forever(), timeout=10))
        # ladder: 0.01, 0.02, then reset -> disconnect after success is
        # attempt 1 again (0.01) ... escalating from there
        assert delays[0] == pytest.approx(0.01)
        assert delays[1] == pytest.approx(0.02)
        assert delays[2] == pytest.approx(0.01)  # reset proven

    def test_recv_timeout_polls_and_honours_stop(self, monkeypatch):
        """A silent socket doesn't wedge the loop: recv poll times out,
        stop flag is noticed, shutdown is clean."""
        monkeypatch.setattr(news_stream, '_RECV_POLL_TIMEOUT_S', 0.02)
        listener = make_listener()
        ws = FakeWebSocket([], hang_when_empty=True)
        listener._connect_factory = lambda url: ws

        async def run():
            task = asyncio.ensure_future(listener._run_forever())
            await asyncio.sleep(0.06)
            listener._stop_requested.set()
            await asyncio.wait_for(task, timeout=5)

        asyncio.run(run())
        assert ws.closed is True


# ---------------------------------------------------------------------- #
# Thread lifecycle
# ---------------------------------------------------------------------- #
class TestThreadLifecycle:
    def test_start_stream_stop(self, monkeypatch):
        monkeypatch.setattr(news_stream, '_RECV_POLL_TIMEOUT_S', 0.02)
        got_event = threading.Event()
        ws = FakeWebSocket([json.dumps([news_msg(symbols=['NVDA'])])],
                           hang_when_empty=True)
        listener = make_listener(connect_factory=lambda url: ws)
        listener.register_callback(lambda ev: got_event.set())
        listener.start()
        try:
            assert got_event.wait(timeout=5.0), 'event not delivered'
            assert listener._thread.daemon is True
            assert listener.symbols_with_news_today() == {'NVDA'}
            assert listener.is_stale() is False
        finally:
            listener.stop(timeout_s=5.0)
        assert listener._thread is None
        assert ws.closed is True

    def test_double_start_raises(self, monkeypatch):
        monkeypatch.setattr(news_stream, '_RECV_POLL_TIMEOUT_S', 0.02)
        ws = FakeWebSocket([], hang_when_empty=True)
        listener = make_listener(connect_factory=lambda url: ws)
        listener.start()
        try:
            with pytest.raises(RuntimeError, match='already started'):
                listener.start()
        finally:
            listener.stop(timeout_s=5.0)

    def test_stop_without_start_is_safe(self):
        make_listener().stop()


# ---------------------------------------------------------------------- #
# Shadow runner
# ---------------------------------------------------------------------- #
class TestShadowRunner:
    def test_shadow_print_event(self, capsys):
        ev = parse_news_message(news_msg(symbols=['AAPL', 'TSLA']),
                                received_at=RECV_AT)
        news_stream._shadow_print_event(ev)
        out = capsys.readouterr().out
        assert 'AAPL,TSLA' in out
        assert 'lat=2.00s' in out
        assert 'HL' in out

    def test_run_shadow_prints_stats_and_exits(self, capsys, monkeypatch):
        monkeypatch.setattr(news_stream, '_RECV_POLL_TIMEOUT_S', 0.02)
        ws = FakeWebSocket([json.dumps([news_msg()])], hang_when_empty=True)
        listener = make_listener(connect_factory=lambda url: ws)
        run_shadow(stats_interval_s=0.05, listener=listener,
                   max_seconds=0.15)
        listener.stop(timeout_s=5.0)
        out = capsys.readouterr().out
        assert 'shadow mode' in out
        assert '[NEWS-STREAM] stats' in out
        assert 'final stats' in out

    def test_main_requires_shadow_flag(self, capsys):
        with pytest.raises(SystemExit):
            news_stream.main([])

    def test_main_shadow_invokes_runner(self, monkeypatch):
        called = {}
        monkeypatch.setattr(
            news_stream, 'run_shadow',
            lambda stats_interval_s, max_seconds: called.update(
                interval=stats_interval_s, max_s=max_seconds))
        rc = news_stream.main(['--shadow', '--stats-interval', '30',
                               '--max-seconds', '5'])
        assert rc == 0
        assert called == {'interval': 30.0, 'max_s': 5.0}


# ---------------------------------------------------------------------- #
# Config template
# ---------------------------------------------------------------------- #
class TestConfigTemplate:
    def test_template_has_news_stream_disabled(self):
        import yaml
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, 'config.yaml.template')) as fh:
            cfg = yaml.safe_load(fh)
        assert cfg['news_stream'] == {'enabled': False}


# ---------------------------------------------------------------------- #
# OPTIONAL live integration test — never runs in the default suite.
# Enable manually:  NEWS_STREAM_LIVE_TEST=1 pytest tests/test_news_stream.py -k live
# ---------------------------------------------------------------------- #
@pytest.mark.integration
@pytest.mark.skipif(os.environ.get('NEWS_STREAM_LIVE_TEST') != '1',
                    reason='set NEWS_STREAM_LIVE_TEST=1 for a real 30s '
                           'connection test')
def test_live_30s_connection():
    """Real 30s connection to the Alpaca news stream (credentials from env)."""
    from dotenv import load_dotenv
    load_dotenv()
    listener = NewsStreamListener()
    listener.start()
    try:
        time.sleep(30)
        assert listener.is_stale(threshold_s=30) is False, \
            'no frames within 30s of connecting'
        print('live stats:', listener.latency_stats())
    finally:
        listener.stop()
