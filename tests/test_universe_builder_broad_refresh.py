"""Tests for UniverseBuilder._refresh_broad_daily_bars (ORB seed pool).

This step was added 2026-05-18 as part of a post-mortem after prod's first
LIVE ORB day. Prod's `daily_bars` table had stagnated to ~4.9K symbols (vs
dev's ~12.5K) because the broad daily-bars refresh was a SIDE EFFECT of
the dev-only onemil-orb-backtest.timer. Making it a first-class step of
the universe-rebuild cron gives both nodes broad fresh rows on every
weekday morning — independent of any BT cron.

Tests pin:
  1. Pre-filter cutoffs match ORB live's seed query (close $1-50, vol≥500K).
  2. Caller-level chunking persists partial progress on chunk failure.
  3. Failure modes log ERROR but do NOT abort the universe build.
"""
from __future__ import annotations

import logging
from datetime import date
from unittest.mock import MagicMock

import pytest

from batch.universe_builder import (
    BROAD_BARS_CHUNK_SIZE,
    BROAD_BARS_LOOKBACK_DAYS,
    BROAD_BARS_MAX_CLOSE,
    BROAD_BARS_MIN_CLOSE,
    UniverseBuilder,
)
from data_sources.alpaca_client import AlpacaClient
from data_sources.float_provider import FloatProvider
from persistence.database import Database


@pytest.fixture
def builder():
    """Builder with all I/O dependencies mocked via spec= for safety."""
    return UniverseBuilder(
        alpaca_client=MagicMock(spec=AlpacaClient),
        float_provider=MagicMock(spec=FloatProvider),
        db=MagicMock(spec=Database),
    )


def _bar(d=date(2026, 5, 15), o=10.0, h=10.5, l=9.5, c=10.2, v=750_000):
    """Build a daily-bar dict in the shape get_daily_bars_range returns."""
    return {'date': d, 'open': o, 'high': h, 'low': l, 'close': c, 'volume': v}


# ---------------------------------------------------------------------------
# Module-level constants — pinned API ORB live depends on
# ---------------------------------------------------------------------------


class TestModuleConstants:
    def test_min_close_matches_orb_seed_query(self):
        # scanner.realtime_scanner._orb_universe_source uses
        # `close BETWEEN 1.0 AND 50.0` — DO NOT desync from this.
        assert BROAD_BARS_MIN_CLOSE == 1.0

    def test_max_close_matches_orb_seed_query(self):
        assert BROAD_BARS_MAX_CLOSE == 50.0

    def test_lookback_covers_long_weekend(self):
        # 3-day weekend (Fri close → Mon morning cron) + 1 holiday = 4 days.
        # 7d gives a buffer for cron outages.
        assert BROAD_BARS_LOOKBACK_DAYS >= 5

    def test_chunk_size_matches_alpaca_internal(self):
        # AlpacaClient.get_daily_bars_range chunks at 200 internally; our
        # caller-level chunks should match so each batch is a single API call.
        assert BROAD_BARS_CHUNK_SIZE == 200


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_persists_bars_with_correct_window(self, builder):
        builder.alpaca.get_daily_bars_range.return_value = {
            'AAA': [_bar(d=date(2026, 5, 15))],
            'BBB': [_bar(d=date(2026, 5, 15))],
        }
        builder._refresh_broad_daily_bars(['AAA', 'BBB'], lookback_days=7)

        builder.alpaca.get_daily_bars_range.assert_called_once()
        args, _ = builder.alpaca.get_daily_bars_range.call_args
        symbols, start, end = args
        assert symbols == ['AAA', 'BBB']
        assert (end - start).days == 7

    def test_persists_bars_to_db_with_correct_schema(self, builder):
        builder.alpaca.get_daily_bars_range.return_value = {
            'AAA': [_bar()],
        }
        rows_written = builder._refresh_broad_daily_bars(['AAA'], lookback_days=7)

        assert rows_written == 1
        builder.db.save_daily_bars.assert_called_once()
        rows = builder.db.save_daily_bars.call_args[0][0]
        assert len(rows) == 1
        r = rows[0]
        assert r['symbol'] == 'AAA'
        assert isinstance(r['date'], str)  # ISO format for save_daily_bars
        for k in ('open', 'high', 'low', 'close', 'volume'):
            assert k in r

    def test_iso_date_string_passthrough(self, builder):
        """If alpaca returns date as ISO string already, do not mangle it."""
        builder.alpaca.get_daily_bars_range.return_value = {
            'AAA': [{'date': '2026-05-15', 'open': 1, 'high': 2, 'low': 1, 'close': 1.5, 'volume': 1}],
        }
        rows = builder._refresh_broad_daily_bars(['AAA'], lookback_days=7)
        assert rows == 1
        assert builder.db.save_daily_bars.call_args[0][0][0]['date'] == '2026-05-15'

    def test_multiple_bars_per_symbol_all_persisted(self, builder):
        """7-day lookback returns multiple bars per symbol — all persisted."""
        builder.alpaca.get_daily_bars_range.return_value = {
            'AAA': [
                _bar(d=date(2026, 5, 11)),
                _bar(d=date(2026, 5, 12)),
                _bar(d=date(2026, 5, 13)),
                _bar(d=date(2026, 5, 14)),
                _bar(d=date(2026, 5, 15)),
            ],
        }
        rows = builder._refresh_broad_daily_bars(['AAA'], lookback_days=7)
        assert rows == 5


# ---------------------------------------------------------------------------
# Caller-level chunking — partial-progress on failure
# ---------------------------------------------------------------------------


class TestChunking:
    def test_chunks_at_caller_level(self, builder):
        """501 symbols → 3 chunks at 200 each, 3 API calls."""
        # All chunks return one bar each
        builder.alpaca.get_daily_bars_range.side_effect = [
            {f'S{i:03d}': [_bar()] for i in range(0, 200)},
            {f'S{i:03d}': [_bar()] for i in range(200, 400)},
            {f'S{i:03d}': [_bar()] for i in range(400, 501)},
        ]
        symbols = [f'S{i:03d}' for i in range(501)]
        rows = builder._refresh_broad_daily_bars(symbols, lookback_days=7)
        assert rows == 501
        assert builder.alpaca.get_daily_bars_range.call_count == 3

    def test_failed_chunk_does_not_block_others(self, builder, caplog):
        """If chunk 2/3 raises, chunks 1 and 3 still persist."""
        builder.alpaca.get_daily_bars_range.side_effect = [
            {'AAA': [_bar()]},                 # chunk 1 OK
            RuntimeError("transient API error"),  # chunk 2 fails
            {'CCC': [_bar()]},                 # chunk 3 OK
        ]
        symbols = [f'S{i:03d}' for i in range(401)]  # 3 chunks
        with caplog.at_level(logging.ERROR):
            rows = builder._refresh_broad_daily_bars(symbols, lookback_days=7)

        # 2 chunks succeeded, 1 row each = 2 rows persisted
        assert rows == 2
        assert builder.db.save_daily_bars.call_count == 2
        # Error logged for chunk 2
        assert any(
            "Step 9" in r.message and "FAILED" in r.message
            for r in caplog.records
        )


# ---------------------------------------------------------------------------
# Failure modes — must NOT abort the build, MUST log loudly
# ---------------------------------------------------------------------------


class TestFailureModes:
    def test_empty_symbols_warns_returns_zero(self, builder, caplog):
        with caplog.at_level(logging.WARNING):
            rows = builder._refresh_broad_daily_bars([], lookback_days=7)
        assert rows == 0
        builder.alpaca.get_daily_bars_range.assert_not_called()
        builder.db.save_daily_bars.assert_not_called()
        assert any(
            "Step 9" in r.message and "No symbols" in r.message
            for r in caplog.records
        )

    def test_all_chunks_fail_returns_zero(self, builder, caplog):
        builder.alpaca.get_daily_bars_range.side_effect = RuntimeError("alpaca down")
        with caplog.at_level(logging.ERROR):
            rows = builder._refresh_broad_daily_bars(['AAA'], lookback_days=7)
        assert rows == 0
        builder.db.save_daily_bars.assert_not_called()
        assert any("FAILED" in r.message for r in caplog.records)

    def test_save_failure_does_not_block_remaining_chunks(self, builder, caplog):
        """DB save failure on chunk 1 should not prevent chunk 2 from trying."""
        builder.alpaca.get_daily_bars_range.side_effect = [
            {'AAA': [_bar()]},
            {'CCC': [_bar()]},
        ]
        # First save raises, second succeeds
        builder.db.save_daily_bars.side_effect = [
            RuntimeError("disk lock"),
            None,
        ]
        symbols = [f'S{i:03d}' for i in range(201)]  # 2 chunks
        with caplog.at_level(logging.ERROR):
            rows = builder._refresh_broad_daily_bars(symbols, lookback_days=7)
        assert builder.db.save_daily_bars.call_count == 2
        # Only chunk 2's row counted
        assert rows == 1
        assert any("save_daily_bars" in r.message and "FAILED" in r.message
                   for r in caplog.records)

    def test_empty_api_response_no_persist_no_crash(self, builder):
        """Alpaca returns {} for a chunk — skip, do not crash."""
        builder.alpaca.get_daily_bars_range.return_value = {}
        rows = builder._refresh_broad_daily_bars(['AAA'], lookback_days=7)
        assert rows == 0
        builder.db.save_daily_bars.assert_not_called()


# ---------------------------------------------------------------------------
# Pre-filter integration with build() — verify ORB-eligible symbols are picked
# ---------------------------------------------------------------------------


class TestBuildPipelinePrefilter:
    """The build() pipeline pre-filters Step 2's lossy daily_bars dict before
    calling _refresh_broad_daily_bars. Pin that the cutoffs match the ORB
    seed query — anything outside must be excluded.
    """

    def _stub_build_dependencies(self, builder, daily_bars_dict):
        """Set up the minimum mocks so build() reaches Step 9.

        _cache_volume_profiles uses signal.SIGALRM which is Unix-only; stub
        it out so this test runs on Windows dev machines too. The behavior
        under test is the pre-filter logic between Step 2 and Step 9, not
        the volume-profile step.
        """
        builder.alpaca.get_all_tradeable_assets.return_value = [
            {'symbol': s, 'company_name': '', 'exchange': '', 'asset_class': 'us_equity'}
            for s in daily_bars_dict.keys()
        ]
        builder.alpaca.get_daily_bars.return_value = daily_bars_dict
        builder.alpaca.get_daily_bars_range.return_value = {}  # SPY + broad both
        builder.db.get_active_universe.return_value = []
        builder.db.get_symbols_needing_float_update.return_value = []
        builder.db.get_volume_profile_count.return_value = 0
        # No-op the Unix-only volume profile step (orthogonal to Step 9 logic)
        builder._cache_volume_profiles = MagicMock(return_value=None)

    def test_symbol_at_lower_close_bound_included(self, builder):
        self._stub_build_dependencies(builder, {
            'EDGE_LO': {'close': BROAD_BARS_MIN_CLOSE, 'volume': 750_000},
        })
        builder.build()
        # Find the broad-refresh call (Step 8's SPY call comes first)
        broad_calls = [
            c for c in builder.alpaca.get_daily_bars_range.call_args_list
            if c.args and isinstance(c.args[0], list) and 'SPY' not in c.args[0]
        ]
        assert broad_calls, "Step 9 never called get_daily_bars_range"
        assert 'EDGE_LO' in broad_calls[0].args[0]

    def test_symbol_at_upper_close_bound_included(self, builder):
        self._stub_build_dependencies(builder, {
            'EDGE_HI': {'close': BROAD_BARS_MAX_CLOSE, 'volume': 750_000},
        })
        builder.build()
        broad_calls = [
            c for c in builder.alpaca.get_daily_bars_range.call_args_list
            if c.args and isinstance(c.args[0], list) and 'SPY' not in c.args[0]
        ]
        assert broad_calls
        assert 'EDGE_HI' in broad_calls[0].args[0]

    def test_symbol_below_min_close_excluded(self, builder):
        self._stub_build_dependencies(builder, {
            'TOO_LOW': {'close': 0.50, 'volume': 750_000},
            'KEEP':    {'close': 10.0, 'volume': 750_000},
        })
        builder.build()
        broad_calls = [
            c for c in builder.alpaca.get_daily_bars_range.call_args_list
            if c.args and isinstance(c.args[0], list) and 'SPY' not in c.args[0]
        ]
        assert broad_calls
        seed = broad_calls[0].args[0]
        assert 'TOO_LOW' not in seed
        assert 'KEEP' in seed

    def test_symbol_above_max_close_excluded(self, builder):
        self._stub_build_dependencies(builder, {
            'TOO_HIGH': {'close': 75.0, 'volume': 750_000},
            'KEEP':     {'close': 10.0, 'volume': 750_000},
        })
        builder.build()
        broad_calls = [
            c for c in builder.alpaca.get_daily_bars_range.call_args_list
            if c.args and isinstance(c.args[0], list) and 'SPY' not in c.args[0]
        ]
        assert broad_calls
        seed = broad_calls[0].args[0]
        assert 'TOO_HIGH' not in seed
        assert 'KEEP' in seed

    def test_thin_volume_symbol_still_included(self, builder):
        """Step 2's volume is 20d AVG; a thin-avg stock may spike yesterday.
        We persist regardless and let ORB's live query enforce per-day vol."""
        self._stub_build_dependencies(builder, {
            'THIN_AVG': {'close': 10.0, 'volume': 100_000},
            'KEEP':     {'close': 10.0, 'volume': 750_000},
        })
        builder.build()
        broad_calls = [
            c for c in builder.alpaca.get_daily_bars_range.call_args_list
            if c.args and isinstance(c.args[0], list) and 'SPY' not in c.args[0]
        ]
        assert broad_calls
        seed = broad_calls[0].args[0]
        # Both included — only close band gates Step 9; vol gated downstream.
        assert 'THIN_AVG' in seed
        assert 'KEEP' in seed

    def test_missing_volume_key_still_included(self, builder):
        """Bar without 'volume' key — should still be included because Step 9
        no longer filters on volume (price-band only). Defensive: don't crash."""
        self._stub_build_dependencies(builder, {
            'NO_VOL': {'close': 10.0},  # missing 'volume'
            'KEEP':   {'close': 10.0, 'volume': 750_000},
        })
        builder.build()
        broad_calls = [
            c for c in builder.alpaca.get_daily_bars_range.call_args_list
            if c.args and isinstance(c.args[0], list) and 'SPY' not in c.args[0]
        ]
        assert broad_calls
        seed = broad_calls[0].args[0]
        assert 'NO_VOL' in seed
        assert 'KEEP' in seed

    # Note: missing 'close' key in Step 2's daily_bars would crash earlier
    # in _filter_by_price (Step 3) before Step 9 runs — a pre-existing
    # fragility, not part of the broad-refresh contract. Our prefilter
    # uses bar.get('close', 0) defensively but Step 3 is the actual gate.
