"""
Tests for scanner/realtime_scanner.py - RealtimeScanner.

Covers:
- _load_universe
- _run_premarket_cycle: detects gap-ups, skips non-gappers
- _run_intraday_cycle: qualifies stocks, detects close calls
- _print_intraday_output (verbose and non-verbose)
"""

import pytest
from unittest.mock import MagicMock, patch, call
from io import StringIO

from data_sources.alpaca_client import AlpacaClient
from data_sources.news_provider import NewsProvider
from persistence.database import Database
from scanner.criteria import ScannerCriteria, ScanCandidate
from scanner.realtime_scanner import RealtimeScanner


@pytest.fixture
def mock_alpaca():
    """Create a mock AlpacaClient with spec."""
    return MagicMock(spec=AlpacaClient)


@pytest.fixture
def mock_news():
    """Create a mock NewsProvider with spec."""
    m = MagicMock(spec=NewsProvider)
    m.classify_news.return_value = {'has_news': False, 'catalyst': None, 'headline': '', 'reason': ''}
    return m


@pytest.fixture
def mock_db():
    """Create a mock Database with spec."""
    return MagicMock(spec=Database)


@pytest.fixture
def criteria():
    """Create a real ScannerCriteria with default thresholds."""
    return ScannerCriteria()


@pytest.fixture
def scanner(mock_alpaca, mock_news, mock_db, criteria):
    """Create a RealtimeScanner with mocked dependencies."""
    return RealtimeScanner(
        alpaca_client=mock_alpaca,
        news_provider=mock_news,
        db=mock_db,
        criteria=criteria,
        verbose=False,
    )


@pytest.fixture
def verbose_scanner(mock_alpaca, mock_news, mock_db, criteria):
    """Create a RealtimeScanner with verbose=True."""
    return RealtimeScanner(
        alpaca_client=mock_alpaca,
        news_provider=mock_news,
        db=mock_db,
        criteria=criteria,
        verbose=True,
    )


# =============================================================================
# _load_universe
# =============================================================================

class TestLoadUniverse:
    """Tests for RealtimeScanner._load_universe."""

    def test_loads_universe_and_profiles(self, scanner, mock_db):
        """_load_universe populates _universe and _volume_profiles from DB."""
        mock_db.get_active_universe.return_value = [
            {'symbol': 'AAA', 'price_close': 5.0, 'float_shares': 1_000_000},
            {'symbol': 'BBB', 'price_close': 8.0, 'float_shares': 2_000_000},
        ]
        mock_db.get_all_volume_profiles.return_value = {
            'AAA': {'09:30': 50000, '09:45': 60000},
        }

        scanner._load_universe()

        assert len(scanner._universe) == 2
        assert scanner._universe[0]['symbol'] == 'AAA'
        assert 'AAA' in scanner._volume_profiles
        assert scanner._volume_profiles['AAA']['09:30'] == 50000

    def test_empty_universe(self, scanner, mock_db):
        """_load_universe handles empty universe gracefully."""
        mock_db.get_active_universe.return_value = []
        mock_db.get_all_volume_profiles.return_value = {}

        scanner._load_universe()

        assert scanner._universe == []
        assert scanner._volume_profiles == {}


# =============================================================================
# _run_premarket_cycle
# =============================================================================

class TestRunPremarketCycle:
    """Tests for RealtimeScanner._run_premarket_cycle.

    Premarket is pure gap detection — no news/LLM calls.
    """

    def test_detects_gap_ups(self, scanner, mock_alpaca, mock_news, mock_db):
        """Premarket cycle detects stocks gapping up (no news check)."""
        scanner._universe = [
            {'symbol': 'GAP', 'price_close': 5.0, 'company_name': 'Gap Co', 'float_shares': 1_000_000},
        ]
        mock_alpaca.get_latest_trades.return_value = {
            'GAP': {'price': 5.50},  # 10% gap
        }

        scanner._run_premarket_cycle()

        assert 'GAP' in scanner._premarket_gap_symbols
        mock_news.has_interesting_news.assert_not_called()
        mock_db.save_scan_result.assert_called_once()
        saved = mock_db.save_scan_result.call_args[0][0]
        assert saved['symbol'] == 'GAP'
        assert saved['phase'] == 'premarket'
        assert saved['has_news'] == 0
        assert saved['qualified'] == 1

    def test_skips_non_gappers(self, scanner, mock_alpaca, mock_news, mock_db):
        """Premarket cycle skips stocks that don't gap enough."""
        scanner._universe = [
            {'symbol': 'FLAT', 'price_close': 10.0, 'company_name': 'Flat Co', 'float_shares': 1_000_000},
        ]
        mock_alpaca.get_latest_trades.return_value = {
            'FLAT': {'price': 10.10},  # 1% gap, below 2% threshold
        }

        scanner._run_premarket_cycle()

        assert 'FLAT' not in scanner._premarket_gap_symbols
        mock_news.has_interesting_news.assert_not_called()
        mock_db.save_scan_result.assert_not_called()

    def test_skips_zero_price_trade(self, scanner, mock_alpaca, mock_db):
        """Premarket cycle skips stocks with zero or missing trade price."""
        scanner._universe = [
            {'symbol': 'ZERO', 'price_close': 5.0, 'company_name': '', 'float_shares': 0},
        ]
        mock_alpaca.get_latest_trades.return_value = {
            'ZERO': {'price': 0},
        }

        scanner._run_premarket_cycle()

        mock_db.save_scan_result.assert_not_called()

    def test_skips_missing_trade(self, scanner, mock_alpaca, mock_db):
        """Premarket cycle skips stocks with no trade data."""
        scanner._universe = [
            {'symbol': 'MISS', 'price_close': 5.0, 'company_name': '', 'float_shares': 0},
        ]
        mock_alpaca.get_latest_trades.return_value = {}

        scanner._run_premarket_cycle()

        mock_db.save_scan_result.assert_not_called()

    def test_skips_zero_prev_close(self, scanner, mock_alpaca, mock_db):
        """Premarket cycle skips stocks with zero prev_close to avoid division by zero."""
        scanner._universe = [
            {'symbol': 'NOCLOSE', 'price_close': 0, 'company_name': '', 'float_shares': 0},
        ]
        mock_alpaca.get_latest_trades.return_value = {
            'NOCLOSE': {'price': 5.0},
        }

        scanner._run_premarket_cycle()

        mock_db.save_scan_result.assert_not_called()


# =============================================================================
# _run_intraday_cycle
# =============================================================================

class TestRunIntradayCycle:
    """Tests for RealtimeScanner._run_intraday_cycle."""

    def _setup_universe(self, scanner, mock_alpaca, mock_news, symbol='MOMO',
                        price_close=4.0, trade_price=5.0, bar_volume=100_000,
                        avg_volume=10_000, has_news=True, headline="Catalyst"):
        """Set up universe, trades, bars, volume profiles, and news for intraday test."""
        from datetime import datetime
        import pytz

        scanner._universe = [
            {'symbol': symbol, 'price_close': price_close,
             'company_name': 'Momo Co', 'float_shares': 2_000_000},
        ]
        mock_alpaca.get_latest_trades.return_value = {
            symbol: {'price': trade_price},
        }
        # Compute a bar timestamp in the current 15-min bucket (ET)
        now_et = datetime.now(pytz.timezone('US/Eastern'))
        bucket = f"{now_et.hour:02d}:{(now_et.minute // 15) * 15:02d}"
        bar_ts = now_et.replace(minute=(now_et.minute // 15) * 15, second=0, microsecond=0)

        mock_alpaca.get_current_bars.return_value = {
            symbol: {'volume': bar_volume, 'timestamp': bar_ts},
        }
        # Volume profile keyed by bucket derived from bar timestamp
        scanner._volume_profiles = {symbol: {}}
        scanner._volume_profiles[symbol][bucket] = avg_volume

        mock_news.has_interesting_news.return_value = (has_news, headline)
        mock_news.classify_news.return_value = {
            'has_news': has_news, 'catalyst': has_news if has_news else None,
            'headline': headline or '', 'reason': 'test',
        }

    @patch('scanner.realtime_scanner.datetime')
    def test_qualifies_stock(self, mock_dt, scanner, mock_alpaca, mock_news, mock_db):
        """Intraday cycle qualifies a stock meeting all criteria."""
        import pytz
        from datetime import datetime as real_datetime

        # Fix the ET time to 10:00 for predictable bucket
        fake_now = real_datetime(2026, 3, 13, 10, 0, 0,
                                 tzinfo=pytz.timezone('US/Eastern'))
        mock_dt.now.return_value = fake_now
        mock_dt.side_effect = lambda *a, **kw: real_datetime(*a, **kw)

        scanner._universe = [
            {'symbol': 'MOMO', 'price_close': 4.0,
             'company_name': 'Momo Co', 'float_shares': 2_000_000},
        ]
        mock_alpaca.get_latest_trades.return_value = {
            'MOMO': {'price': 5.0},  # 25% change
        }
        fake_bar_ts = real_datetime(2026, 3, 13, 10, 0, 0,
                                    tzinfo=pytz.timezone('US/Eastern'))
        mock_alpaca.get_current_bars.return_value = {
            'MOMO': {'volume': 100_000, 'timestamp': fake_bar_ts},
        }
        scanner._volume_profiles = {'MOMO': {'10:00': 10_000}}
        mock_news.has_interesting_news.return_value = (True, "Big news")
        mock_news.classify_news.return_value = {'has_news': True, 'catalyst': True, 'headline': 'Big news', 'reason': 'test'}

        scanner._run_intraday_cycle()

        mock_db.save_scan_result.assert_called_once()
        saved = mock_db.save_scan_result.call_args[0][0]
        assert saved['symbol'] == 'MOMO'
        assert saved['phase'] == 'intraday'
        assert saved['qualified'] == 1

    @patch('scanner.realtime_scanner.datetime')
    def test_detects_close_calls(self, mock_dt, verbose_scanner, mock_alpaca, mock_news, mock_db, capsys):
        """Intraday cycle identifies close calls (5 of 6 criteria met)."""
        import pytz
        from datetime import datetime as real_datetime

        fake_now = real_datetime(2026, 3, 13, 10, 0, 0,
                                 tzinfo=pytz.timezone('US/Eastern'))
        mock_dt.now.return_value = fake_now
        mock_dt.side_effect = lambda *a, **kw: real_datetime(*a, **kw)

        # Stock meets all criteria EXCEPT news (volume + price trigger news check,
        # but news returns False)
        verbose_scanner._universe = [
            {'symbol': 'NEAR', 'price_close': 4.0,
             'company_name': 'Near Co', 'float_shares': 2_000_000},
        ]
        mock_alpaca.get_latest_trades.return_value = {
            'NEAR': {'price': 5.0},  # 25% change
        }
        fake_bar_ts = real_datetime(2026, 3, 13, 10, 0, 0,
                                    tzinfo=pytz.timezone('US/Eastern'))
        mock_alpaca.get_current_bars.return_value = {
            'NEAR': {'volume': 100_000, 'timestamp': fake_bar_ts},
        }
        verbose_scanner._volume_profiles = {'NEAR': {'10:00': 10_000}}
        mock_news.has_interesting_news.return_value = (False, None)

        verbose_scanner._run_intraday_cycle()

        # Not qualified -> not saved
        mock_db.save_scan_result.assert_not_called()

        # Verbose output should show close call
        captured = capsys.readouterr()
        assert "Close calls" in captured.out
        assert "NEAR" in captured.out

    # =========================================================================
    # Parallelism: bars + trades fetch run concurrently
    # =========================================================================
    # Intraday cycle does TWO independent broad-universe API calls.
    # 2026-04-27 incident showed sequential calls compound during 9:30-10:00 ET
    # Alpaca congestion (60s + 60s = 120s vs 60s cycle budget). Now parallelized
    # via ThreadPoolExecutor(max_workers=2). These tests guard against accidental
    # regression to sequential.

    @patch('scanner.realtime_scanner.datetime')
    def test_intraday_cycle_runs_both_api_calls(
        self, mock_dt, scanner, mock_alpaca, mock_news, mock_db
    ):
        """Happy path: both bars and trades fetched, called once each with
        the same universe symbol list."""
        import pytz
        from datetime import datetime as real_datetime

        fake_now = real_datetime(2026, 3, 13, 10, 0, 0,
                                 tzinfo=pytz.timezone('US/Eastern'))
        mock_dt.now.return_value = fake_now
        mock_dt.side_effect = lambda *a, **kw: real_datetime(*a, **kw)

        scanner._universe = [
            {'symbol': 'MOMO', 'price_close': 4.0,
             'company_name': 'Momo Co', 'float_shares': 2_000_000},
        ]
        bar_ts = real_datetime(2026, 3, 13, 10, 0, 0,
                               tzinfo=pytz.timezone('US/Eastern'))
        mock_alpaca.get_current_bars.return_value = {
            'MOMO': {'volume': 100_000, 'timestamp': bar_ts},
        }
        mock_alpaca.get_latest_trades.return_value = {
            'MOMO': {'price': 5.0},
        }
        scanner._volume_profiles = {'MOMO': {'10:00': 10_000}}
        mock_news.has_interesting_news.return_value = (True, "n")
        mock_news.classify_news.return_value = {
            'has_news': True, 'catalyst': True,
            'headline': 'n', 'reason': 'test',
        }

        # Should not raise
        scanner._run_intraday_cycle()

        # Both calls were made exactly once with the same universe symbols
        mock_alpaca.get_current_bars.assert_called_once_with(['MOMO'])
        mock_alpaca.get_latest_trades.assert_called_once_with(['MOMO'])

    @patch('scanner.realtime_scanner.datetime')
    def test_intraday_cycle_runs_calls_in_parallel(
        self, mock_dt, scanner, mock_alpaca, mock_news
    ):
        """The proof: bars + trades execute concurrently, not sequentially.

        Each mocked call sleeps 0.5s. If sequential, total >= 1.0s.
        If parallel (correct), total ~0.5s + small thread overhead.
        Generous bound (0.9s) avoids flakiness on slow CI but still
        catches any regression to sequential execution.
        """
        import pytz
        from datetime import datetime as real_datetime
        import time as _time

        fake_now = real_datetime(2026, 3, 13, 10, 0, 0,
                                 tzinfo=pytz.timezone('US/Eastern'))
        mock_dt.now.return_value = fake_now
        mock_dt.side_effect = lambda *a, **kw: real_datetime(*a, **kw)

        scanner._universe = [
            {'symbol': 'PARA', 'price_close': 4.0,
             'company_name': 'Para Co', 'float_shares': 2_000_000},
        ]
        bar_ts = real_datetime(2026, 3, 13, 10, 0, 0,
                               tzinfo=pytz.timezone('US/Eastern'))

        def slow_bars(*_args, **_kw):
            _time.sleep(0.5)
            return {'PARA': {'volume': 100_000, 'timestamp': bar_ts}}

        def slow_trades(*_args, **_kw):
            _time.sleep(0.5)
            return {'PARA': {'price': 5.0}}

        mock_alpaca.get_current_bars.side_effect = slow_bars
        mock_alpaca.get_latest_trades.side_effect = slow_trades
        scanner._volume_profiles = {'PARA': {'10:00': 10_000}}
        mock_news.has_interesting_news.return_value = (False, None)

        t0 = _time.perf_counter()
        scanner._run_intraday_cycle()
        elapsed = _time.perf_counter() - t0

        # Sequential would be ≥1.0s. Parallel is ~0.5s + thread overhead.
        # 0.9s is the discrimination boundary. Sequential regression hits >1.0s.
        assert elapsed < 0.9, (
            f"Intraday cycle took {elapsed:.2f}s with two 0.5s API calls — "
            f"this means bars + trades ran sequentially, not in parallel. "
            f"Parallel execution should complete in ~0.5s + thread overhead."
        )

    @patch('scanner.realtime_scanner.datetime')
    def test_intraday_cycle_propagates_bars_error(
        self, mock_dt, scanner, mock_alpaca, mock_news
    ):
        """Error in get_current_bars propagates up. Caller's wrapper
        (cbe780a) catches at cycle boundary and skips the minute."""
        import pytz
        from datetime import datetime as real_datetime
        import pytest as _pt
        from data_sources.alpaca_client import AlpacaAPIError

        fake_now = real_datetime(2026, 3, 13, 10, 0, 0,
                                 tzinfo=pytz.timezone('US/Eastern'))
        mock_dt.now.return_value = fake_now
        mock_dt.side_effect = lambda *a, **kw: real_datetime(*a, **kw)

        scanner._universe = [
            {'symbol': 'BAR', 'price_close': 4.0,
             'company_name': 'Bar Co', 'float_shares': 2_000_000},
        ]
        mock_alpaca.get_current_bars.side_effect = AlpacaAPIError(
            "simulated bars failure")
        mock_alpaca.get_latest_trades.return_value = {
            'BAR': {'price': 5.0},
        }

        with _pt.raises(AlpacaAPIError, match="simulated bars failure"):
            scanner._run_intraday_cycle()

    @patch('scanner.realtime_scanner.datetime')
    def test_intraday_cycle_propagates_trades_error(
        self, mock_dt, scanner, mock_alpaca, mock_news
    ):
        """Error in get_latest_trades propagates up. Symmetric to bars
        error path — confirms either side can fail without losing the
        exception in the parallel pool."""
        import pytz
        from datetime import datetime as real_datetime
        import pytest as _pt
        from data_sources.alpaca_client import AlpacaAPIError

        fake_now = real_datetime(2026, 3, 13, 10, 0, 0,
                                 tzinfo=pytz.timezone('US/Eastern'))
        mock_dt.now.return_value = fake_now
        mock_dt.side_effect = lambda *a, **kw: real_datetime(*a, **kw)

        scanner._universe = [
            {'symbol': 'TRD', 'price_close': 4.0,
             'company_name': 'Trd Co', 'float_shares': 2_000_000},
        ]
        bar_ts = real_datetime(2026, 3, 13, 10, 0, 0,
                               tzinfo=pytz.timezone('US/Eastern'))
        mock_alpaca.get_current_bars.return_value = {
            'TRD': {'volume': 100_000, 'timestamp': bar_ts},
        }
        mock_alpaca.get_latest_trades.side_effect = AlpacaAPIError(
            "simulated trades failure")

        with _pt.raises(AlpacaAPIError, match="simulated trades failure"):
            scanner._run_intraday_cycle()

    @patch('scanner.realtime_scanner.datetime')
    def test_intraday_cycle_both_fail_propagates_one_exception(
        self, mock_dt, scanner, mock_alpaca, mock_news
    ):
        """When BOTH bars and trades raise, exactly one exception
        propagates (the bars one — it is .result()'d first). The trades
        exception is consumed by the executor's shutdown. The cycle
        wrapper at the caller (cbe780a) catches whatever surfaces and
        skips the minute; we only need to guarantee that an exception
        does propagate (not silently swallowed)."""
        import pytz
        from datetime import datetime as real_datetime
        import pytest as _pt
        from data_sources.alpaca_client import AlpacaAPIError

        fake_now = real_datetime(2026, 3, 13, 10, 0, 0,
                                 tzinfo=pytz.timezone('US/Eastern'))
        mock_dt.now.return_value = fake_now
        mock_dt.side_effect = lambda *a, **kw: real_datetime(*a, **kw)

        scanner._universe = [
            {'symbol': 'BOTH', 'price_close': 4.0,
             'company_name': 'Both Co', 'float_shares': 2_000_000},
        ]
        mock_alpaca.get_current_bars.side_effect = AlpacaAPIError(
            "bars fail")
        mock_alpaca.get_latest_trades.side_effect = AlpacaAPIError(
            "trades fail")

        with _pt.raises(AlpacaAPIError, match="bars fail"):
            scanner._run_intraday_cycle()

    @patch('scanner.realtime_scanner.datetime')
    def test_intraday_cycle_drains_bar_events_during_wait(
        self, mock_dt, scanner, mock_alpaca, mock_news
    ):
        """TMCR-class fix (2026-05-01): while bars+trades broad-universe
        calls are in-flight (30-90s in prod), incoming WS bar events for
        already-qualified crossed_stocks must be drained on the main
        thread — otherwise the breakout bar queues for tens of seconds
        and the buy-stop is placed too late (TMCR 4/9 cost +$24K
        unrealized winner).

        This test slows bars/trades by 0.4s and asserts the engine drain
        runs at least once during that window."""
        import pytz
        from datetime import datetime as real_datetime
        import time as _time
        from unittest.mock import MagicMock

        fake_now = real_datetime(2026, 3, 13, 10, 0, 0,
                                 tzinfo=pytz.timezone('US/Eastern'))
        mock_dt.now.return_value = fake_now
        mock_dt.side_effect = lambda *a, **kw: real_datetime(*a, **kw)

        # Slow API so the wait loop has time to drain
        bar_ts = real_datetime(2026, 3, 13, 10, 0, 0,
                               tzinfo=pytz.timezone('US/Eastern'))

        def slow_bars(*_a, **_kw):
            _time.sleep(0.4)
            return {'TMCR': {'volume': 100_000, 'timestamp': bar_ts}}

        def slow_trades(*_a, **_kw):
            _time.sleep(0.4)
            return {'TMCR': {'price': 5.0}}

        mock_alpaca.get_current_bars.side_effect = slow_bars
        mock_alpaca.get_latest_trades.side_effect = slow_trades

        scanner._universe = [
            {'symbol': 'TMCR', 'price_close': 4.0,
             'company_name': 'TMCR Co', 'float_shares': 2_000_000},
        ]
        scanner._volume_profiles = {'TMCR': {'10:00': 10_000}}
        mock_news.has_interesting_news.return_value = (False, None)

        # Inject a trading_engine whose _drain_bar_events we can count
        fake_engine = MagicMock()
        fake_engine.enabled = True
        fake_engine._drain_bar_events.return_value = None
        scanner.trading_engine = fake_engine

        scanner._run_intraday_cycle()

        # Drain must have fired at least twice during the ~0.4s wait
        # (poll is 0.5s, but futures aren't both done immediately, so
        # at least one drain happens before the wait completes).
        assert fake_engine._drain_bar_events.call_count >= 1, (
            f"Expected drain to run during bars+trades wait, but "
            f"call_count was {fake_engine._drain_bar_events.call_count}. "
            f"TMCR-class latency fix regressed."
        )


# =============================================================================
# _print_intraday_output
# =============================================================================

class TestPrintIntradayOutput:
    """Tests for RealtimeScanner._print_intraday_output."""

    def test_prints_qualified(self, scanner, capsys):
        """Prints qualified stocks with header when results exist."""
        qualified = [
            ScanCandidate(
                symbol="WIN",
                prev_close=4.0,
                current_price=5.5,
                intraday_change_pct=37.5,
                relative_volume=10.0,
                float_shares=1_000_000,
                news_headline="Great news",
            ),
        ]
        scanner._print_intraday_output(
            bucket="10:00", symbols=["WIN", "LOSE"],
            vol_5x=1, move_10pct=1, news=1,
            qualified=qualified, close_calls=[],
        )
        captured = capsys.readouterr()
        assert "QUALIFIED: 1" in captured.out
        assert "WIN" in captured.out

    def test_verbose_no_qualified(self, verbose_scanner, capsys):
        """Verbose mode prints summary even when no stocks qualify."""
        verbose_scanner._print_intraday_output(
            bucket="10:00", symbols=["A", "B", "C"],
            vol_5x=0, move_10pct=0, news=0,
            qualified=[], close_calls=[],
        )
        captured = capsys.readouterr()
        assert "QUALIFIED: 0" in captured.out
        assert "Universe: 3" in captured.out

    def test_non_verbose_no_output_when_empty(self, scanner, capsys):
        """Non-verbose mode prints nothing when no stocks qualify."""
        scanner._print_intraday_output(
            bucket="10:00", symbols=["A", "B"],
            vol_5x=0, move_10pct=0, news=0,
            qualified=[], close_calls=[],
        )
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_verbose_close_calls(self, verbose_scanner, capsys):
        """Verbose mode prints close call details."""
        close_call = ScanCandidate(
            symbol="ALMOST",
            current_price=5.0,
            intraday_change_pct=15.0,
            relative_volume=6.0,
            float_shares=2_000_000,
        )
        close_call.criteria_met = {
            'price_range': True,
            'float': True,
            'gap': True,
            'relative_volume': True,
            'intraday_change': True,
            'has_news': False,
        }

        verbose_scanner._print_intraday_output(
            bucket="10:30", symbols=["ALMOST"],
            vol_5x=1, move_10pct=1, news=0,
            qualified=[], close_calls=[close_call],
        )
        captured = capsys.readouterr()
        assert "Close calls" in captured.out
        assert "ALMOST" in captured.out
        assert "has_news" in captured.out


# =============================================================================
# Trading Engine Integration
# =============================================================================

class TestTradingEngineHandoff:
    """Tests for scanner → trading engine handoff."""

    def test_scanner_accepts_trading_engine(self, mock_alpaca, mock_news, mock_db, criteria):
        """Scanner can be created with trading_engine parameter."""
        mock_engine = MagicMock()
        scanner = RealtimeScanner(
            alpaca_client=mock_alpaca,
            news_provider=mock_news,
            db=mock_db,
            criteria=criteria,
            trading_engine=mock_engine,
        )
        assert scanner.trading_engine is mock_engine

    def test_scanner_works_without_trading_engine(self, scanner):
        """Scanner works normally without trading_engine."""
        assert scanner.trading_engine is None

    @patch('scanner.realtime_scanner.datetime')
    def test_qualified_stock_handed_to_trading_engine(
        self, mock_dt, mock_alpaca, mock_news, mock_db, criteria
    ):
        """When a stock qualifies, on_stock_qualified is called on trading engine."""
        import pytz
        from datetime import datetime as real_datetime

        # Mock datetime.now(ET) to return 10:00 ET
        ET = pytz.timezone('US/Eastern')
        fake_now = real_datetime(2026, 3, 13, 10, 0, 0, tzinfo=ET)
        mock_dt.now.return_value = fake_now
        mock_dt.side_effect = lambda *a, **kw: real_datetime(*a, **kw)

        mock_engine = MagicMock()
        scanner = RealtimeScanner(
            alpaca_client=mock_alpaca,
            news_provider=mock_news,
            db=mock_db,
            criteria=criteria,
            trading_engine=mock_engine,
            verbose=False,
        )

        # Setup universe with a stock that will qualify
        scanner._universe = [{
            'symbol': 'HOT',
            'price_close': 5.0,
            'company_name': 'Hot Inc.',
            'float_shares': 2_000_000,
        }]
        # Use bucket matching bar timestamp: 10:00 ET
        scanner._volume_profiles = {'HOT': {'10:00': 10000}}

        # Mock API responses so stock qualifies
        # Bar timestamp = 10:00 ET (matching volume profile bucket)
        bar_ts = real_datetime(2026, 3, 13, 10, 0, 0, tzinfo=ET)
        mock_alpaca.get_current_bars.return_value = {
            'HOT': {'open': 5.0, 'high': 6.5, 'low': 5.0, 'close': 6.0, 'volume': 100000,
                     'timestamp': bar_ts},
        }
        mock_alpaca.get_latest_trades.return_value = {
            'HOT': {'price': 6.0, 'size': 100, 'timestamp': '2026-03-13T14:30:00Z'},
        }
        # News check returns True (qualifies)
        mock_news.has_interesting_news.return_value = (True, "Big catalyst news")
        mock_news.classify_news.return_value = {'has_news': True, 'catalyst': True, 'headline': 'Big catalyst news', 'reason': 'test'}
        mock_db.save_scan_result.return_value = 1

        scanner._run_intraday_cycle()

        # Verify trading engine was notified
        mock_engine.on_stock_qualified.assert_called_once()
        call_args = mock_engine.on_stock_qualified.call_args
        assert call_args[0][0] == 'HOT'  # first positional arg is symbol


# =============================================================================
# reset_daily wiring
# =============================================================================

class TestResetDailyWiring:
    """Tests that scanner calls trading_engine.reset_daily() at startup."""

    def test_run_calls_reset_daily_on_trading_engine(self, mock_alpaca, mock_news, mock_db, criteria):
        """run() calls reset_daily() on trading engine after loading universe."""
        from trading.trading_engine import TradingEngine

        mock_engine = MagicMock(spec=TradingEngine)
        scanner = RealtimeScanner(
            alpaca_client=mock_alpaca,
            news_provider=mock_news,
            db=mock_db,
            criteria=criteria,
            trading_engine=mock_engine,
        )
        mock_db.get_active_universe.return_value = [
            {'symbol': 'AAA', 'price_close': 5.0, 'float_shares': 1_000_000},
        ]
        mock_db.get_all_volume_profiles.return_value = {}
        mock_alpaca.is_trading_day.return_value = True
        mock_alpaca.is_short_trading_day.return_value = False

        # Patch datetime so we get "after market close" and run() exits early
        with patch('scanner.realtime_scanner.datetime') as mock_dt:
            import pytz
            from datetime import datetime as real_datetime
            ET = pytz.timezone('US/Eastern')
            fake_now = real_datetime(2026, 3, 13, 16, 30, 0, tzinfo=ET)
            mock_dt.now.return_value = fake_now
            mock_dt.side_effect = lambda *a, **kw: real_datetime(*a, **kw)

            scanner.run()

        mock_engine.reset_daily.assert_called_once()

    def test_run_no_error_without_trading_engine(self, scanner, mock_alpaca, mock_db):
        """run() works fine when trading_engine is None (no crash)."""
        mock_db.get_active_universe.return_value = [
            {'symbol': 'AAA', 'price_close': 5.0, 'float_shares': 1_000_000},
        ]
        mock_db.get_all_volume_profiles.return_value = {}
        mock_alpaca.is_trading_day.return_value = True
        mock_alpaca.is_short_trading_day.return_value = False

        with patch('scanner.realtime_scanner.datetime') as mock_dt:
            import pytz
            from datetime import datetime as real_datetime
            ET = pytz.timezone('US/Eastern')
            fake_now = real_datetime(2026, 3, 13, 16, 30, 0, tzinfo=ET)
            mock_dt.now.return_value = fake_now
            mock_dt.side_effect = lambda *a, **kw: real_datetime(*a, **kw)

            scanner.run()  # Should not raise

    def test_run_test_cycle_calls_reset_daily(self, mock_alpaca, mock_news, mock_db, criteria):
        """run_test_cycle() calls reset_daily() on trading engine."""
        from trading.trading_engine import TradingEngine

        mock_engine = MagicMock(spec=TradingEngine)
        scanner = RealtimeScanner(
            alpaca_client=mock_alpaca,
            news_provider=mock_news,
            db=mock_db,
            criteria=criteria,
            trading_engine=mock_engine,
        )
        mock_db.get_active_universe.return_value = [
            {'symbol': 'AAA', 'price_close': 5.0, 'float_shares': 1_000_000},
        ]
        mock_db.get_all_volume_profiles.return_value = {}

        # Mock API calls used by run_test_cycle
        mock_alpaca.get_latest_trades.return_value = {}
        mock_alpaca.get_current_bars.return_value = {}
        mock_db.get_scan_results.return_value = []

        scanner.run_test_cycle()

        mock_engine.reset_daily.assert_called_once()


class TestIsTradingDay:
    """Tests for _is_trading_day — calendar-based guard."""

    def test_skips_non_trading_day(self, scanner, mock_alpaca):
        """Returns False on weekends/holidays."""
        mock_alpaca.is_trading_day.return_value = False
        assert scanner._is_trading_day() is False

    def test_skips_short_trading_day(self, scanner, mock_alpaca):
        """Returns False on short days (e.g., Black Friday)."""
        mock_alpaca.is_trading_day.return_value = True
        mock_alpaca.is_short_trading_day.return_value = True
        assert scanner._is_trading_day() is False

    def test_allows_normal_trading_day(self, scanner, mock_alpaca):
        """Returns True on a normal full trading day."""
        mock_alpaca.is_trading_day.return_value = True
        mock_alpaca.is_short_trading_day.return_value = False
        assert scanner._is_trading_day() is True

    def test_allows_on_calendar_api_error(self, scanner, mock_alpaca):
        """Returns True (proceed cautiously) if calendar API fails."""
        mock_alpaca.is_trading_day.side_effect = Exception("API down")
        assert scanner._is_trading_day() is True

    def test_run_exits_early_on_non_trading_day(self, scanner, mock_alpaca):
        """Scanner.run() returns immediately on non-trading day."""
        mock_alpaca.is_trading_day.return_value = False
        # _load_universe should NOT be called
        scanner.run()
        mock_alpaca.is_trading_day.assert_called_once()
        # Universe should be empty (never loaded)
        assert scanner._universe == []


# =============================================================================
# Main Loop Behavior (Bug #1, #2, #3 fixes)
# =============================================================================

class TestMainLoopBehavior:
    """Tests for the rewritten run() loop — force close, 60s ticks, shutdown."""

    def _make_scanner_with_engine(self, mock_alpaca, mock_news, mock_db, criteria,
                                   shutdown_event=None):
        """Create scanner with a mock trading engine and shutdown event."""
        from trading.trading_engine import TradingEngine
        mock_engine = MagicMock(spec=TradingEngine)
        mock_engine.enabled = True
        scanner = RealtimeScanner(
            alpaca_client=mock_alpaca,
            news_provider=mock_news,
            db=mock_db,
            criteria=criteria,
            trading_engine=mock_engine,
            shutdown_event=shutdown_event,
        )
        return scanner, mock_engine

    @patch('scanner.realtime_scanner.datetime')
    def test_force_close_called_at_force_close_time(
        self, mock_dt, mock_alpaca, mock_news, mock_db, criteria
    ):
        """Engine _force_close_all() called when past 15:45 ET."""
        import pytz
        from datetime import datetime as real_datetime
        import threading

        shutdown = threading.Event()
        scanner, mock_engine = self._make_scanner_with_engine(
            mock_alpaca, mock_news, mock_db, criteria, shutdown_event=shutdown)

        ET = pytz.timezone('US/Eastern')

        # Setup: universe loaded, trading day
        mock_db.get_active_universe.return_value = [
            {'symbol': 'AAA', 'price_close': 5.0, 'float_shares': 1_000_000},
        ]
        mock_db.get_all_volume_profiles.return_value = {}
        mock_alpaca.is_trading_day.return_value = True
        mock_alpaca.is_short_trading_day.return_value = False
        mock_alpaca.get_latest_trades.return_value = {}
        mock_alpaca.get_current_bars.return_value = {}

        # Time sequence: 15:46 (past force close) → 16:00 (market close)
        call_count = [0]
        def fake_now(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 4:
                return real_datetime(2026, 3, 16, 15, 46, 0, tzinfo=ET)
            return real_datetime(2026, 3, 16, 16, 0, 0, tzinfo=ET)
        mock_dt.now.side_effect = fake_now
        mock_dt.side_effect = lambda *a, **kw: real_datetime(*a, **kw)

        mock_engine._is_past_force_close_time.return_value = True

        # Trigger shutdown on sleep to avoid infinite loop
        def trigger_close(timeout):
            shutdown.set()
            return True
        scanner._interruptible_sleep = MagicMock(side_effect=trigger_close)

        scanner.run()

        mock_engine._force_close_all.assert_called()

    @patch('scanner.realtime_scanner.datetime')
    def test_pattern_check_called_every_tick(
        self, mock_dt, mock_alpaca, mock_news, mock_db, criteria
    ):
        """run_pattern_check() called multiple times within one 15-min bucket."""
        import pytz
        from datetime import datetime as real_datetime
        import threading

        shutdown = threading.Event()
        scanner, mock_engine = self._make_scanner_with_engine(
            mock_alpaca, mock_news, mock_db, criteria, shutdown_event=shutdown)

        ET = pytz.timezone('US/Eastern')

        mock_db.get_active_universe.return_value = [
            {'symbol': 'AAA', 'price_close': 5.0, 'float_shares': 1_000_000},
        ]
        mock_db.get_all_volume_profiles.return_value = {}
        mock_alpaca.is_trading_day.return_value = True
        mock_alpaca.is_short_trading_day.return_value = False
        mock_alpaca.get_latest_trades.return_value = {}
        mock_alpaca.get_current_bars.return_value = {}

        mock_engine._is_past_force_close_time.return_value = False

        # Stay in same bucket (10:00-10:14) for 3 ticks, then close market
        tick = [0]
        def fake_now(*args, **kwargs):
            tick[0] += 1
            if tick[0] <= 6:
                # Vary minutes within the same 15-min bucket
                minute = min(tick[0], 14)
                return real_datetime(2026, 3, 16, 10, minute, 0, tzinfo=ET)
            return real_datetime(2026, 3, 16, 16, 0, 0, tzinfo=ET)
        mock_dt.now.side_effect = fake_now
        mock_dt.side_effect = lambda *a, **kw: real_datetime(*a, **kw)

        # Let sleep pass without blocking
        scanner._interruptible_sleep = MagicMock(return_value=False)

        scanner.run()

        # Pattern check should be called on every tick (not just bucket changes)
        assert mock_engine.run_pattern_check.call_count >= 2

    def test_shutdown_event_breaks_loop(self, mock_alpaca, mock_news, mock_db, criteria):
        """Setting shutdown event causes run() to exit + force-close."""
        import threading

        shutdown = threading.Event()
        scanner, mock_engine = self._make_scanner_with_engine(
            mock_alpaca, mock_news, mock_db, criteria, shutdown_event=shutdown)

        mock_db.get_active_universe.return_value = [
            {'symbol': 'AAA', 'price_close': 5.0, 'float_shares': 1_000_000},
        ]
        mock_db.get_all_volume_profiles.return_value = {}
        mock_alpaca.is_trading_day.return_value = True
        mock_alpaca.is_short_trading_day.return_value = False
        mock_alpaca.get_latest_trades.return_value = {}
        mock_alpaca.get_current_bars.return_value = {}

        # Set shutdown before first iteration
        shutdown.set()

        with patch('scanner.realtime_scanner.datetime') as mock_dt:
            import pytz
            from datetime import datetime as real_datetime
            ET = pytz.timezone('US/Eastern')
            fake_now = real_datetime(2026, 3, 16, 10, 0, 0, tzinfo=ET)
            mock_dt.now.return_value = fake_now
            mock_dt.side_effect = lambda *a, **kw: real_datetime(*a, **kw)

            scanner.run()

        # Post-loop safety net should force close
        mock_engine._force_close_all.assert_called()

    def test_interruptible_sleep_returns_true_on_shutdown(
        self, mock_alpaca, mock_news, mock_db, criteria
    ):
        """_interruptible_sleep returns True when event is set."""
        import threading

        shutdown = threading.Event()
        scanner = RealtimeScanner(
            alpaca_client=mock_alpaca,
            news_provider=mock_news,
            db=mock_db,
            criteria=criteria,
            shutdown_event=shutdown,
        )

        shutdown.set()
        assert scanner._interruptible_sleep(60) is True

    @patch('scanner.realtime_scanner.time_mod')
    def test_interruptible_sleep_without_event_uses_time_sleep(
        self, mock_time, mock_alpaca, mock_news, mock_db, criteria
    ):
        """Falls back to time_mod.sleep when no shutdown event."""
        scanner = RealtimeScanner(
            alpaca_client=mock_alpaca,
            news_provider=mock_news,
            db=mock_db,
            criteria=criteria,
            shutdown_event=None,
        )

        result = scanner._interruptible_sleep(30)
        assert result is False
        mock_time.sleep.assert_called_once_with(30)

    def test_premarket_wait_interruptible(self, mock_alpaca, mock_news, mock_db, criteria):
        """Shutdown during pre-market _sleep_until causes early return."""
        import threading

        shutdown = threading.Event()
        scanner, mock_engine = self._make_scanner_with_engine(
            mock_alpaca, mock_news, mock_db, criteria, shutdown_event=shutdown)

        mock_db.get_active_universe.return_value = [
            {'symbol': 'AAA', 'price_close': 5.0, 'float_shares': 1_000_000},
        ]
        mock_db.get_all_volume_profiles.return_value = {}
        mock_alpaca.is_trading_day.return_value = True
        mock_alpaca.is_short_trading_day.return_value = False

        # Time is 08:00 — needs to wait for 09:30
        with patch('scanner.realtime_scanner.datetime') as mock_dt:
            import pytz
            from datetime import datetime as real_datetime
            ET = pytz.timezone('US/Eastern')
            fake_now = real_datetime(2026, 3, 16, 8, 0, 0, tzinfo=ET)
            mock_dt.now.return_value = fake_now
            mock_dt.side_effect = lambda *a, **kw: real_datetime(*a, **kw)

            # Set shutdown so _interruptible_sleep returns True immediately
            shutdown.set()

            scanner.run()

        # Should have returned early — no gap scan or intraday cycle
        mock_alpaca.get_latest_trades.assert_not_called()
        # No force close since we returned before the loop
        mock_engine._force_close_all.assert_not_called()


# ===========================================================================
# Bug B: Volume bucket computed from bar timestamp, not wall clock
# ===========================================================================

class TestVolumeBucketFromBarTimestamp:
    """Verify _run_intraday_cycle computes the volume bucket from the bar's
    timestamp rather than datetime.now(), so the bucket key matches the
    completed bar returned by get_current_bars()."""

    @patch('scanner.realtime_scanner.datetime')
    def test_bucket_from_bar_timestamp_not_now(
        self, mock_dt, scanner, mock_alpaca, mock_news, mock_db
    ):
        """When wall clock is 10:16 but bar timestamp is 10:00, bucket = 10:00."""
        import pytz
        from datetime import datetime as real_datetime

        # Wall clock: 10:16 ET (just past the bucket boundary)
        fake_now = real_datetime(2026, 3, 13, 10, 16, 0,
                                 tzinfo=pytz.timezone('US/Eastern'))
        mock_dt.now.return_value = fake_now
        mock_dt.side_effect = lambda *a, **kw: real_datetime(*a, **kw)

        # Bar timestamp: 10:00 ET (the completed bar)
        bar_ts = real_datetime(2026, 3, 13, 10, 0, 0,
                               tzinfo=pytz.timezone('US/Eastern'))

        scanner._universe = [
            {'symbol': 'MOMO', 'price_close': 4.0,
             'company_name': 'Momo Co', 'float_shares': 2_000_000},
        ]
        mock_alpaca.get_latest_trades.return_value = {
            'MOMO': {'price': 5.0},
        }
        mock_alpaca.get_current_bars.return_value = {
            'MOMO': {'volume': 100_000, 'timestamp': bar_ts},
        }
        # Profile has 10:00 bucket (matching bar) but NOT 10:15 (wall clock bucket)
        scanner._volume_profiles = {'MOMO': {'10:00': 10_000}}
        mock_news.has_interesting_news.return_value = (True, "Big news")
        mock_news.classify_news.return_value = {'has_news': True, 'catalyst': True, 'headline': 'Big news', 'reason': 'test'}

        scanner._run_intraday_cycle()

        # Should qualify (rvol = 100k/10k = 10x) using 10:00 bucket
        mock_db.save_scan_result.assert_called_once()
        saved = mock_db.save_scan_result.call_args[0][0]
        assert saved['symbol'] == 'MOMO'
        assert saved['qualified'] == 1

    @patch('scanner.realtime_scanner.datetime')
    def test_wrong_bucket_misses_qualification(
        self, mock_dt, scanner, mock_alpaca, mock_news, mock_db
    ):
        """If bucket were computed from wall clock, volume profile lookup would
        miss and relative_volume would be 0 — stock would NOT qualify."""
        import pytz
        from datetime import datetime as real_datetime

        fake_now = real_datetime(2026, 3, 13, 10, 16, 0,
                                 tzinfo=pytz.timezone('US/Eastern'))
        mock_dt.now.return_value = fake_now
        mock_dt.side_effect = lambda *a, **kw: real_datetime(*a, **kw)

        bar_ts = real_datetime(2026, 3, 13, 10, 0, 0,
                               tzinfo=pytz.timezone('US/Eastern'))

        scanner._universe = [
            {'symbol': 'MOMO', 'price_close': 4.0,
             'company_name': 'Momo Co', 'float_shares': 2_000_000},
        ]
        mock_alpaca.get_latest_trades.return_value = {
            'MOMO': {'price': 5.0},
        }
        mock_alpaca.get_current_bars.return_value = {
            'MOMO': {'volume': 100_000, 'timestamp': bar_ts},
        }
        # Only 10:15 bucket exists — bar is from 10:00, so it should NOT match
        # This test proves that if we used wall clock (10:15), it would find
        # the profile and qualify, but with bar timestamp (10:00) it misses.
        scanner._volume_profiles = {'MOMO': {'10:15': 10_000}}
        mock_news.has_interesting_news.return_value = (True, "Big news")
        mock_news.classify_news.return_value = {'has_news': True, 'catalyst': True, 'headline': 'Big news', 'reason': 'test'}

        scanner._run_intraday_cycle()

        # avg_vol = 0 for bucket 10:00, so relative_volume = 0 → not qualified
        mock_db.save_scan_result.assert_not_called()
        mock_news.classify_news.return_value = {'has_news': True, 'catalyst': True, 'headline': 'Big news', 'reason': 'test'}


class TestNextAlignedWakeup:
    """Static helper that picks the next ":offset_secs past minute" wakeup target.

    Used by the main loop to align cycle work with bar boundaries (cycle fires
    at HH:MM:01 instead of drifting 0-60s).
    """

    def test_returns_next_minute_plus_offset_when_mid_minute(self, monkeypatch):
        """At HH:MM:30, next aligned wakeup is HH:(MM+1):01."""
        import time as time_mod
        # Pin time to 1234567890.0 → second 30 of some minute
        # 1234567890 % 60 = 30 → minute_start = 1234567860, +1 = 1234567861 (in past)
        # → target = 1234567861 + 60 = 1234567921
        monkeypatch.setattr(time_mod, 'time', lambda: 1234567890.0)
        target = RealtimeScanner._next_aligned_wakeup(offset_secs=1.0)
        assert target == 1234567921.0
        # 31 seconds in the future
        assert target - 1234567890.0 == 31.0

    def test_targets_current_minute_offset_when_before_offset(self, monkeypatch):
        """At HH:MM:00.5 (before :01), target is HH:MM:01.0 — same minute, 0.5s away."""
        import time as time_mod
        monkeypatch.setattr(time_mod, 'time', lambda: 1234567860.5)
        target = RealtimeScanner._next_aligned_wakeup(offset_secs=1.0)
        assert target == 1234567861.0
        assert target - 1234567860.5 == 0.5

    def test_skips_to_next_minute_when_at_offset_boundary(self, monkeypatch):
        """At HH:MM:01.0 exactly, target jumps to HH:(MM+1):01.0 (target<=now → +60s)."""
        import time as time_mod
        monkeypatch.setattr(time_mod, 'time', lambda: 1234567861.0)
        target = RealtimeScanner._next_aligned_wakeup(offset_secs=1.0)
        assert target == 1234567921.0  # +60s
        assert target - 1234567861.0 == 60.0

    def test_custom_offset_works(self, monkeypatch):
        """offset_secs=2.0 means align to HH:MM:02."""
        import time as time_mod
        monkeypatch.setattr(time_mod, 'time', lambda: 1234567860.0)  # at minute boundary
        target = RealtimeScanner._next_aligned_wakeup(offset_secs=2.0)
        assert target == 1234567862.0
        assert target - 1234567860.0 == 2.0

    def test_target_always_in_future(self, monkeypatch):
        """For ANY input second, the returned target is strictly greater than now."""
        import time as time_mod
        # Sample many sub-second offsets to flush rounding edge cases
        for sub in (0.0, 0.001, 0.5, 0.999, 1.0, 1.001, 30.0, 59.999):
            now = 1234567860.0 + sub
            monkeypatch.setattr(time_mod, 'time', lambda n=now: n)
            target = RealtimeScanner._next_aligned_wakeup(offset_secs=1.0)
            assert target > now, f"target {target} not > now {now} (sub={sub})"
            # And target is within (0, 60] seconds away
            assert 0 < target - now <= 60.0, f"unexpected delta {target - now} (sub={sub})"
