"""docs/hod_entry_quote_capture_fix_20261113.md — HOD entry-quote SIZES and L2 depth at the fill.

Root cause (2026-11-13): HOD symbols are PRINT-watched (`StopMonitor.subscribe_trades_quotes` -> `_print_quotes`
(bid, ask, ts)), never QuoteWatch-ed; `_snapshot_cross_quote` read the sizes from `get_quote_watch_snapshot`, which is
None for them in production (the 10/31 unit test mocked it) -> entry_quote_bid_size/ask_size NULL on every HOD row
while bid/ask were captured. Fix: the print-watch cache keeps the sizes of the same quote (`get_print_quote_sizes`)
and the cross snapshot falls back to it. L2 depth (`entry_l2_depth`, a PAID Databento snapshot) is launched once per
order behind `l2_at_fill` (default OFF).
"""
import logging
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tests.test_hod_exec_instruments_20261031 import _arm_and_cross, _fill, _update_calls
from tests.test_hod_live_resting import live_engine
from trading.stop_monitor import StopMonitor


# ---------------------------------------------------------------- StopMonitor: the print-quote cache keeps sizes
class TestPrintQuoteSizes:
    def _sm(self):
        sm = StopMonitor.__new__(StopMonitor)
        import threading
        sm._print_watch_lock = threading.Lock()
        sm._print_watch_symbols = {'ABC'}
        sm._print_quotes = {}
        sm._print_quote_sizes = {}
        return sm

    def test_sizes_are_cached_beside_the_quote_and_dropped_with_it(self):
        sm = self._sm()
        with sm._print_watch_lock:
            sm._print_quotes['ABC'] = (10.98, 11.02, 1.0)
            sm._print_quote_sizes['ABC'] = (300, 200)
        assert sm.get_print_quote('ABC') == (10.98, 11.02, 1.0)        # 3-tuple shape unchanged for its readers
        assert sm.get_print_quote_sizes('ABC') == (300, 200)
        assert sm.get_print_quote_sizes('ZZZ') is None

    def test_quote_handler_writes_both_caches(self):
        """The real quote handler path: a print-watched symbol's quote lands in BOTH caches (bid/ask/ts and sizes)."""
        sm = self._sm()
        import asyncio
        sm._watches = {}; sm._quote_watches = {}; sm._watch_lock = __import__('threading').Lock()
        sm._last_data_ts = 0.0
        quote = SimpleNamespace(symbol='ABC', bid_price=10.98, ask_price=11.02, bid_size=3, ask_size=2)
        # _on_quote is the live handler; it reads watches under _watch_lock and the print set unlocked.
        asyncio.run(sm._on_quote(quote))
        bid, ask, ts = sm.get_print_quote('ABC')
        assert (bid, ask) == (10.98, 11.02) and ts > 0
        assert sm.get_print_quote_sizes('ABC') == (3, 2)

    def test_unsubscribe_drops_the_sizes_too(self):
        sm = self._sm()
        sm._print_quotes['ABC'] = (1.0, 1.1, 1.0); sm._print_quote_sizes['ABC'] = (5, 6)
        sm._print_watch_symbols = {'ABC'}
        # mirror the unsubscribe body: both caches are popped under the lock
        with sm._print_watch_lock:
            sm._print_watch_symbols.discard('ABC'); sm._print_quotes.pop('ABC', None); sm._print_quote_sizes.pop('ABC', None)
        assert sm.get_print_quote('ABC') is None and sm.get_print_quote_sizes('ABC') is None


# ---------------------------------------------------------------- engine: sizes at the cross from the print cache
def _no_quote_watch(sm, bs=300, asz=200):
    """Production shape for a HOD symbol: NO QuoteWatch (snapshot None) but a print-watch quote WITH sizes."""
    sm.get_quote_watch_snapshot.return_value = None
    sm.get_print_quote_sizes.return_value = (bs, asz)


class TestCrossSizesFallback:
    def test_sizes_come_from_the_print_cache_when_there_is_no_quote_watch(
            self, hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path):
        e = live_engine(hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path)
        _no_quote_watch(hod_live_sm, bs=300, asz=200)
        cand = _arm_and_cross(e, hod_live_sm, time.time())
        lo = cand.live_order
        assert (lo['cross_bid_size'], lo['cross_ask_size']) == (300, 200)
        _fill(e, hod_live_stream, cand)
        eq = [u for u in _update_calls(hod_live_db, 7) if 'entry_quote_bid' in u]
        assert len(eq) == 1 and eq[0]['entry_quote_bid_size'] == 300 and eq[0]['entry_quote_ask_size'] == 200

    def test_zero_sizes_stay_null_and_the_quote_is_still_written(
            self, hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path):
        e = live_engine(hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path)
        _no_quote_watch(hod_live_sm, bs=0, asz=0)
        cand = _arm_and_cross(e, hod_live_sm, time.time())
        _fill(e, hod_live_stream, cand)
        eq = [u for u in _update_calls(hod_live_db, 7) if 'entry_quote_bid' in u]
        assert len(eq) == 1 and eq[0]['entry_quote_bid'] > 0
        assert eq[0]['entry_quote_bid_size'] is None and eq[0]['entry_quote_ask_size'] is None

    def test_no_print_sizes_accessor_is_tolerated(self, hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path):
        """An older StopMonitor without get_print_quote_sizes: bid/ask still land, sizes NULL, no exception."""
        e = live_engine(hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path)
        hod_live_sm.get_quote_watch_snapshot.return_value = None
        del hod_live_sm.get_print_quote_sizes
        cand = _arm_and_cross(e, hod_live_sm, time.time())
        _fill(e, hod_live_stream, cand)
        eq = [u for u in _update_calls(hod_live_db, 7) if 'entry_quote_bid' in u]
        assert len(eq) == 1 and eq[0]['entry_quote_bid_size'] is None


# ---------------------------------------------------------------- engine: L2 depth behind l2_at_fill
class TestL2AtFill:
    def test_off_by_default_launches_nothing(self, hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path):
        e = live_engine(hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path)
        assert e.l2_at_fill is False
        _no_quote_watch(hod_live_sm)
        cand = _arm_and_cross(e, hod_live_sm, time.time())
        with patch('data_sources.l2_depth.log_l2_async') as l2:
            _fill(e, hod_live_stream, cand)
        l2.assert_not_called()

    def test_on_launches_once_per_order_with_the_trade_id(self, hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path):
        e = live_engine(hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path, l2_at_fill=True)
        _no_quote_watch(hod_live_sm)
        cand = _arm_and_cross(e, hod_live_sm, time.time())
        with patch('data_sources.l2_depth.log_l2_async') as l2:
            _fill(e, hod_live_stream, cand)
            lo = cand.live_order or {}
            # a second booking pass on the same order (partial top-up) must not relaunch
            e._launch_l2_at_fill('ABC', 7, lo if lo else {'l2_launched': True})
        assert l2.call_count == 1
        sym, when, tid, fn = l2.call_args.args
        assert sym == 'ABC' and tid == 7 and fn is hod_live_db.update_trade and l2.call_args.kwargs == {'column': 'entry_l2_depth'}

    def test_launch_failure_is_one_warning_and_the_fill_is_still_booked(
            self, hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path, caplog):
        e = live_engine(hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path, l2_at_fill=True)
        _no_quote_watch(hod_live_sm)
        cand = _arm_and_cross(e, hod_live_sm, time.time())
        with patch('data_sources.l2_depth.log_l2_async', side_effect=RuntimeError('no databento')), caplog.at_level(logging.WARNING):
            _fill(e, hod_live_stream, cand)
        warns = [r for r in caplog.records if 'L2 depth snapshot launch failed' in r.getMessage()]
        assert len(warns) == 1 and warns[0].levelno == logging.WARNING
        eq = [u for u in _update_calls(hod_live_db, 7) if 'entry_quote_bid' in u]
        assert len(eq) == 1                                              # the NBBO write still happened

    def test_missing_cross_snapshot_does_not_suppress_the_depth_snapshot(
            self, hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path):
        e = live_engine(hod_live_alpaca, hod_live_db, hod_live_sm, hod_live_stream, tmp_path, l2_at_fill=True)
        lo = {}                                      # one order, no cross snapshot taken
        with patch('data_sources.l2_depth.log_l2_async') as l2:
            e._write_entry_quote('ABC', 7, lo)       # NULL quote columns + one WARNING, but the depth snapshot launches
            e._write_entry_quote('ABC', 7, lo)       # second booking pass on the same order: no relaunch
        assert l2.call_count == 1 and lo['l2_launched'] is True
