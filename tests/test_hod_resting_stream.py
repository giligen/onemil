"""docs/hod_resting_entry_spec_20260925.md — tape-accurate dry fills for the resting stop-limit entry.

Extends tests/test_hod_resting_entry.py (which covers the bar-level fallback) with the live trade-print
path: StopMonitor.subscribe_trades_quotes()/.unsubscribe() (a capped pool separate from _watches/
_quote_watches) and HodBreakEngine._on_trade_print (the same fill rule, trading.hod_break.resting_entry_fill,
fed a real print + the quote cached at that instant instead of a bar's high). Still dry only: no order is
ever submitted from this path.
"""
import asyncio
import logging
import os
import time
import types
from datetime import timedelta
from unittest.mock import MagicMock

import pytest

from trading.stop_monitor import StopMonitor
from trading.hod_break_engine import Candidate
from tests.test_hod_resting_entry import (
    tape, ADV20, mock_alpaca, mock_db, mock_sm, trades_db, resting_engine, read_ledger,
)
from tests.test_hod_break import bars as bars_arrays
from tests.test_hod_break_engine import admit, bars_df


# --------------------------------------------------------------------------------------------- StopMonitor pool
@pytest.fixture
def raw_monitor(mock_alpaca):
    """A real, unstarted StopMonitor (no WebSocket thread) — exercises the print-watch bookkeeping alone."""
    return StopMonitor(api_key='k', api_secret='s', alpaca_client=mock_alpaca)


class TestSubscribeTradesQuotesPool:
    def test_subscribe_adds_new_symbols_and_returns_count(self, raw_monitor):
        n = raw_monitor.subscribe_trades_quotes(['ABC', 'DEF'])
        assert n == 2 and raw_monitor._print_watch_symbols == {'ABC', 'DEF'}

    def test_subscribe_skips_already_subscribed(self, raw_monitor):
        raw_monitor.subscribe_trades_quotes(['ABC'])
        assert raw_monitor.subscribe_trades_quotes(['ABC', 'DEF']) == 1

    def test_cap_warns_and_skips_beyond_200(self, raw_monitor, caplog):
        symbols = [f'S{i}' for i in range(201)]
        with caplog.at_level(logging.WARNING):
            n = raw_monitor.subscribe_trades_quotes(symbols)
        assert n == 200 and len(raw_monitor._print_watch_symbols) == 200
        assert any('cap' in r.message.lower() for r in caplog.records)

    def test_unsubscribe_removes_from_pool_and_clears_quote_cache(self, raw_monitor):
        raw_monitor.subscribe_trades_quotes(['ABC'])
        raw_monitor._print_quotes['ABC'] = (10.0, 10.05, time.time())
        raw_monitor.unsubscribe(['ABC'])
        assert 'ABC' not in raw_monitor._print_watch_symbols
        assert raw_monitor.get_print_quote('ABC') is None


class TestOnQuoteAndOnTradeRouting:
    def test_on_quote_caches_only_for_print_watched_symbols(self, raw_monitor):
        raw_monitor.subscribe_trades_quotes(['ABC'])
        q_watched = types.SimpleNamespace(symbol='ABC', bid_price=10.0, ask_price=10.05, bid_size=1, ask_size=1)
        q_other = types.SimpleNamespace(symbol='XYZ', bid_price=5.0, ask_price=5.05, bid_size=1, ask_size=1)
        asyncio.run(raw_monitor._on_quote(q_watched))
        asyncio.run(raw_monitor._on_quote(q_other))
        assert raw_monitor.get_print_quote('ABC') == (10.0, 10.05, pytest.approx(time.time(), abs=2))
        assert raw_monitor.get_print_quote('XYZ') is None

    def test_on_trade_dispatches_only_to_subscribed_symbols(self, raw_monitor):
        raw_monitor.subscribe_trades_quotes(['ABC'])
        seen = []
        raw_monitor.register_trade_print_handler('h1', lambda sym, px, sz, ts: seen.append((sym, px, sz)))
        asyncio.run(raw_monitor._on_trade(types.SimpleNamespace(symbol='ABC', price=11.02, size=250)))
        asyncio.run(raw_monitor._on_trade(types.SimpleNamespace(symbol='ZZZ', price=1.0, size=1)))
        assert seen == [('ABC', 11.02, 250)]


# --------------------------------------------------------------------------------------------- engine wiring
class TestEngineSubscribesAndUnsubscribes:
    def test_registers_trade_print_handler(self, mock_alpaca, mock_db, mock_sm, tmp_path):
        e = resting_engine(mock_alpaca, mock_db, mock_sm, tmp_path)
        e.register_on_stop_monitor()
        mock_sm.register_trade_print_handler.assert_called_once_with(e.STRATEGY_NAME, e._on_trade_print)

    def test_subscribes_when_armed_and_not_yet_crossed(self, mock_alpaca, mock_db, mock_sm, tmp_path):
        """Real reproduction of the live path (2026-09-25 fix: arm_state no longer needs bar j+1 to exist —
        the next minute is derived as m[j] + 1). Bars arrive ONE AT A TIME, exactly like the websocket feeding
        _on_bar_close -> _ingest_bars per bar. The consolidation completes at bar 6 (test_arms_exactly_when_
        consolidation_completes): after bar 6 closes — with bar 7 not yet fed/existing — the order must already
        be armed for bar 7, and the print-watch subscribe must fire at that same moment."""
        from tests.test_hod_break_engine import bars_df
        e = resting_engine(mock_alpaca, mock_db, mock_sm, tmp_path)
        admit(e)
        df = bars_df(tape())
        for i in range(7):                                              # feed bars 0..6 one at a time; bar 7+ do NOT exist yet
            e._ingest_bars('ABC', df.iloc[i:i + 1])
        cand = e.candidates['ABC']
        assert cand.resting_arm is not None and cand.resting_arm['idx'] == 6      # armed at the close of bar 6, from bars 0..6 only
        assert cand.resting_arm['trigger'] == pytest.approx(11.01)
        # bars 0..5 end each call unarmed (idempotent unsubscribe, harmless — StopMonitor no-ops on a symbol
        # it isn't tracking); subscribe fires exactly once, on the 7th call, the moment bar 6 arms it.
        mock_sm.subscribe_trades_quotes.assert_called_once_with(['ABC'])
        assert mock_sm.unsubscribe.call_args.args == (['ABC'],)   # last pre-arm call, not a fill/loss-of-arm unsubscribe

    def test_arms_one_at_a_time_then_fills_on_the_next_minutes_print(self, mock_alpaca, mock_db, mock_sm, tmp_path):
        """Continues the bar-by-bar feed past arming: the first qualifying trade print in the following minute
        (the tape-accurate _on_trade_print path) resolves the resting order at the ask."""
        from tests.test_hod_break_engine import bars_df
        e = resting_engine(mock_alpaca, mock_db, mock_sm, tmp_path)
        admit(e)
        df = bars_df(tape())
        for i in range(7):
            e._ingest_bars('ABC', df.iloc[i:i + 1])
        cand = e.candidates['ABC']
        assert cand.resting_arm is not None and cand.resting_arm['idx'] == 6
        unsub_calls_before = mock_sm.unsubscribe.call_count
        mock_sm.get_print_quote.return_value = (11.00, 11.015, time.time())       # ask <= limit 11.0165
        e._on_trade_print('ABC', 11.02, 250, time.time())                         # first print >= trigger, in the following minute
        assert cand.resting_filled is True and cand.resting_arm is None
        rows = read_ledger(e.dry_ledger_path)
        assert len(rows) == 1 and rows[0]['filled'] == '1' and rows[0]['tape_accurate'] == '1'
        assert mock_sm.unsubscribe.call_count == unsub_calls_before + 1           # the fill unsubscribes the print watch
        assert mock_sm.unsubscribe.call_args.args == (['ABC'],)

    def test_unsubscribes_when_the_arm_is_lost_at_the_next_bar_close(self, mock_alpaca, mock_db, mock_sm, tmp_path):
        from tests.test_hod_break import bars as bars_arrays
        e = resting_engine(mock_alpaca, mock_db, mock_sm, tmp_path)
        t = tape()[:7] + [(10.9, 10.95, 8.0, 10.9, 9000)]   # bar 7: high < trigger (no cross), low crashes -> fails to re-arm
        o, h, l, c, v, m = bars_arrays(t)
        cand = Candidate(symbol='ABC', day_open=10.0, adv20=ADV20, subscribed=True, backfill_ok=True)
        e._evaluate_resting(cand, o, h, l, v, m, e.params)
        assert cand.resting_arm is None and cand.resting_filled is False
        mock_sm.unsubscribe.assert_called_once_with(['ABC'])


# --------------------------------------------------------------------------------------------- tape resolution
ARM = dict(level=11.0, trigger=11.01, limit=11.0165, stop=10.7, idx=6, arm_ts='t')


def armed_engine(mock_alpaca, mock_db, mock_sm, tmp_path):
    e = resting_engine(mock_alpaca, mock_db, mock_sm, tmp_path)
    cand = Candidate(symbol='ABC', day_open=10.0, adv20=ADV20, subscribed=True, backfill_ok=True)
    cand.resting_arm = dict(ARM)
    e.candidates['ABC'] = cand
    return e, cand


class TestTradePrintResolution:
    def test_print_below_trigger_does_nothing(self, mock_alpaca, mock_db, mock_sm, tmp_path):
        e, cand = armed_engine(mock_alpaca, mock_db, mock_sm, tmp_path)
        e._on_trade_print('ABC', 11.00, 100, time.time())
        assert cand.resting_arm == ARM and cand.resting_filled is False
        mock_sm.unsubscribe.assert_not_called()

    def test_first_print_at_or_above_trigger_fills_when_ask_le_limit(self, mock_alpaca, mock_db, mock_sm, tmp_path):
        e, cand = armed_engine(mock_alpaca, mock_db, mock_sm, tmp_path)
        mock_sm.get_print_quote.return_value = (11.00, 11.015, time.time())   # ask <= limit 11.0165
        e._on_trade_print('ABC', 11.02, 250, time.time())
        assert cand.resting_filled is True and cand.resting_arm is None
        rows = read_ledger(e.dry_ledger_path)
        assert len(rows) == 1 and rows[0]['filled'] == '1' and rows[0]['tape_accurate'] == '1'
        assert float(rows[0]['fill_px']) == pytest.approx(11.015)
        mock_sm.unsubscribe.assert_called_once_with(['ABC'])

    def test_ask_above_limit_no_fills_and_stays_armed(self, mock_alpaca, mock_db, mock_sm, tmp_path):
        e, cand = armed_engine(mock_alpaca, mock_db, mock_sm, tmp_path)
        mock_sm.get_print_quote.return_value = (11.20, 11.30, time.time())   # ask > limit
        e._on_trade_print('ABC', 11.02, 250, time.time())
        assert cand.resting_filled is False
        assert cand.resting_arm is not None and cand.resting_arm['trigger'] == pytest.approx(11.01)
        assert cand.resting_tape_cross_idx == ARM['idx']
        rows = read_ledger(e.dry_ledger_path)
        assert len(rows) == 1 and rows[0]['filled'] == '0' and rows[0]['tape_accurate'] == '1'
        mock_sm.unsubscribe.assert_not_called()

    def test_second_print_after_no_fill_is_a_no_op(self, mock_alpaca, mock_db, mock_sm, tmp_path):
        e, cand = armed_engine(mock_alpaca, mock_db, mock_sm, tmp_path)
        mock_sm.get_print_quote.return_value = (11.20, 11.30, time.time())
        e._on_trade_print('ABC', 11.02, 250, time.time())
        e._on_trade_print('ABC', 11.05, 300, time.time())
        assert len(read_ledger(e.dry_ledger_path)) == 1

    def test_bar_close_skips_the_bar_level_fallback_after_a_tape_no_fill(self, mock_alpaca, mock_db, mock_sm, tmp_path):
        from tests.test_hod_break import bars as bars_arrays
        e, cand = armed_engine(mock_alpaca, mock_db, mock_sm, tmp_path)
        mock_sm.get_print_quote.return_value = (11.20, 11.30, time.time())
        e._on_trade_print('ABC', 11.02, 250, time.time())            # tape NO FILL, arm stays, cross_idx marked
        assert len(read_ledger(e.dry_ledger_path)) == 1
        cand.resting_scanned_idx = 6                                  # bars 0..6 already scanned (matches the manual arm at idx=6)
        o, h, l, c, v, m = bars_arrays(tape()[:8])                    # bar 7 = the official break bar (h >= trigger)
        e._evaluate_resting(cand, o, h, l, v, m, e.params)
        mock_alpaca.get_latest_quote.assert_not_called()               # the bar-level fallback (_quote -> get_latest_quote) never ran
        assert len(read_ledger(e.dry_ledger_path)) == 1                # no duplicate/fallback ledger row
        assert cand.resting_tape_cross_idx is None                     # consumed


class TestEntryModeNextOpenUnchanged:
    def test_on_trade_print_is_a_no_op_outside_resting_stop_limit(self, mock_alpaca, mock_db, mock_sm):
        from trading.hod_break_engine import HodBreakEngine
        from tests.test_hod_break_engine import cfg
        e = HodBreakEngine(mock_alpaca, mock_db, mock_sm, cfg=cfg())   # default entry_mode='next_open'
        e._on_trade_print('ABC', 999.0, 100, time.time())              # must not raise; no candidates dict needed
        mock_sm.get_print_quote.assert_not_called()


# --------------------------------------------------------------------------------------------- live_since gating
# 2026-09-25 live-dry defects (17:15-17:16 UTC): a day backfill/catch-up walked historical closed bars through
# _evaluate_resting and 'filled' them against the CURRENT quote (AAON: fill 88.39 below trigger 89.47). Fix:
# a bar may only arm if it closed after HodBreakEngine.live_since (set once, from the first real websocket bar).
class TestLiveSinceGating:
    def test_backfill_bars_never_arm_and_write_no_ledger_rows(self, mock_alpaca, mock_db, mock_sm, tmp_path, caplog):
        """Startup catch-up: live_since is still None (no websocket bar has arrived yet) when a full historical
        day is handed to _evaluate_resting — none of it may arm or touch the ledger."""
        e = resting_engine(mock_alpaca, mock_db, mock_sm, tmp_path)
        e.live_since = None                                            # undo the fixture's back-date: simulate pre-live startup
        admit(e)
        with caplog.at_level(logging.INFO):
            e._ingest_bars('ABC', bars_df(tape()))
        cand = e.candidates['ABC']
        assert cand.resting_arm is None and cand.resting_filled is False
        assert not os.path.exists(e.dry_ledger_path)
        assert any('skipped' in r.message and 'backfill' in r.message for r in caplog.records)

    def test_bar_closing_at_or_after_live_since_arms(self, mock_alpaca, mock_db, mock_sm, tmp_path):
        e = resting_engine(mock_alpaca, mock_db, mock_sm, tmp_path)
        e.live_since = e._bar_close_et(570 + 6)                        # == bar 6's own close time (boundary: not skipped)
        admit(e)
        df = bars_df(tape())
        for i in range(7):
            e._ingest_bars('ABC', df.iloc[i:i + 1])
        cand = e.candidates['ABC']
        assert cand.resting_arm is not None and cand.resting_arm['idx'] == 6

    def test_bar_closing_before_live_since_is_skipped_not_armed(self, mock_alpaca, mock_db, mock_sm, tmp_path):
        e = resting_engine(mock_alpaca, mock_db, mock_sm, tmp_path)
        e.live_since = e._bar_close_et(570 + 6) + timedelta(minutes=1)  # one minute AFTER bar 6 closes
        admit(e)
        df = bars_df(tape())
        for i in range(7):
            e._ingest_bars('ABC', df.iloc[i:i + 1])
        cand = e.candidates['ABC']
        assert cand.resting_arm is None                                 # bar 6 would otherwise have armed it


class TestFallbackFillFixes:
    """Bug 2: the bar-level fallback (no print stream data) resolved a cross against 'the current quote'
    unconditionally — if that ask sat below the trigger (stale/backfill quote) the dry ledger recorded a fill
    BELOW the order's own trigger, which a real buy-stop-limit can never do. Fix: fill at max(ask, trigger),
    and only when the quote is fresh (StopMonitor._quote already refuses one older than max_quote_age_s=5s)."""

    def test_fallback_ask_below_trigger_fills_at_the_trigger(self, mock_alpaca, mock_db, mock_sm, tmp_path):
        e, cand = armed_engine(mock_alpaca, mock_db, mock_sm, tmp_path)   # ARM: trigger 11.01, limit 11.0165
        cand.resting_scanned_idx = 6
        mock_alpaca.get_latest_quote.return_value = {'bid_price': 10.85, 'ask_price': 10.90, 'bid_size': 1, 'ask_size': 1}  # ask < trigger
        o, h, l, c, v, m = bars_arrays(tape(n_consol=4))                  # 4 consol bars -> the break bar (h >= trigger) lands at index 7
        e._evaluate_resting(cand, o, h, l, v, m, e.params)
        assert cand.resting_filled is True
        rows = read_ledger(e.dry_ledger_path)
        assert len(rows) == 1 and rows[0]['filled'] == '1' and rows[0]['tape_accurate'] == '0'
        assert float(rows[0]['fill_px']) == pytest.approx(ARM['trigger'])   # filled at the trigger, never below it
        assert float(rows[0]['ask']) == pytest.approx(10.90)                # the raw ask is still logged, unmodified

    def test_fallback_stale_quote_is_no_fill_fail_closed(self, mock_alpaca, mock_db, mock_sm, tmp_path):
        e, cand = armed_engine(mock_alpaca, mock_db, mock_sm, tmp_path)
        cand.resting_scanned_idx = 6
        stale_ts = (e._et_now() - timedelta(seconds=30)).isoformat()
        mock_alpaca.get_latest_quote.return_value = {'bid_price': 11.00, 'ask_price': 11.015, 'bid_size': 1, 'ask_size': 1, 'timestamp': stale_ts}
        o, h, l, c, v, m = bars_arrays(tape(n_consol=4))
        e._evaluate_resting(cand, o, h, l, v, m, e.params)
        assert cand.resting_filled is False                                # no fill (fail closed) — a re-arm attempt for the next bar is separate behaviour
        rows = read_ledger(e.dry_ledger_path)
        assert len(rows) == 1 and rows[0]['filled'] == '0' and rows[0]['fill_px'] == ''
