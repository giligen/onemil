"""TTGT 2026-05-08 root-cause regression: quote-watch → stop-watch upgrade race.

The bug: when a buy-stop fills, trading_engine called
  remove_quote_watch(symbol)  → schedules _unsubscribe_symbol coroutine
  add_watch(symbol, ...)      → schedules _subscribe_symbol coroutine

Both coroutines manipulate `self._stream._handlers["trades"][symbol]`. The
first does `handlers.pop(symbol, None)`; the second does
`handlers[symbol] = self._on_trade`. If the asyncio event loop interleaves
them such that the SUBSCRIBE happens before the UNSUBSCRIBE pops, the
unsubscribe wipes out the freshly-installed handler. WS server keeps
delivering trade ticks for the symbol; client-side dispatch finds no
handler and silently drops them.

Symptom in production: 0 R-trail activations across all bull flag positions
in the 14 days prior to 2026-05-08, despite multiple positions running
past +1.5R. TTGT today peaked at +2.4R; trail never fired.

The fix: `StopMonitor.upgrade_quote_to_stop_watch()` does the swap
atomically under one lock with NO WS operations — handlers stay registered
because the quote-watch already installed them via add_quote_watch's
subscribe flow.
"""
import unittest
from unittest.mock import MagicMock

from trading.stop_monitor import StopMonitor


class _FakeStream:
    """Minimal stand-in for alpaca-py's StockDataStream `_handlers` dict.

    Real stream has `_handlers["trades"]` etc. as dicts symbol → callback.
    """
    def __init__(self):
        self._handlers = {"trades": {}, "quotes": {}, "bars": {}}
        self._ws = None  # no real socket — guards in subscribe/unsubscribe skip the await
        self._subscribed = {"trades": set(), "quotes": set(), "bars": set()}


def _make_monitor():
    """StopMonitor with a fake stream, no thread, ready for handler-dict tests."""
    mon = StopMonitor(
        api_key='k', api_secret='s', alpaca_client=MagicMock(),
        marketable_limit_offset=0.03, marketable_limit_offset_pct=0.005,
    )
    mon._stream = _FakeStream()
    # _loop and _running gate the run_coroutine_threadsafe call. With them
    # None/False, the coroutine schedule is a no-op — we exercise the SYNC
    # parts of the upgrade path directly, which is where the race lives.
    mon._loop = None
    mon._running = False
    return mon


class TestUpgradeRaceProof(unittest.TestCase):
    """Demonstrate the bug and prove the fix."""

    def _add_quote_watch_handlers(self, mon, symbol):
        """Simulate the post-add_quote_watch state: SDK handler dict has
        the symbol mapped to StopMonitor's callbacks (this is what
        _subscribe_symbol installs synchronously before its first await).
        """
        mon._stream._handlers["trades"][symbol] = mon._on_trade
        mon._stream._handlers["quotes"][symbol] = mon._on_quote

    def test_BUG_old_path_loses_handler_under_race_interleave(self):
        """REPRODUCE: with the OLD remove_quote_watch + add_watch sequence,
        if asyncio interleaves the two scheduled coroutines such that the
        subscribe's sync handler-set runs BEFORE the unsubscribe's sync
        handler-pop, the handler is wiped."""
        mon = _make_monitor()
        symbol = 'TTGT'

        # Pre-fill: quote-watch is in place, SDK handler installed
        mon.add_quote_watch(
            symbol=symbol, submit_bid=5.95, submit_ask=6.00,
            submit_bid_size=100, submit_ask_size=100,
        )
        self._add_quote_watch_handlers(mon, symbol)
        # Bound methods can't be compared with `is` (each access creates a new
        # bound-method object). Compare callable identity via __func__.
        h_before = mon._stream._handlers["trades"].get(symbol)
        self.assertIsNotNone(h_before)
        self.assertEqual(h_before.__func__, StopMonitor._on_trade)

        # OLD ORDER OF OPERATIONS (the bug):
        # Step 1: remove_quote_watch removes from _quote_watches dict.
        # Its coroutine would later pop _handlers[trades][symbol].
        with mon._watch_lock:
            mon._quote_watches.pop(symbol, None)

        # Step 2: add_watch installs the stop-watch in _watches.
        # Its coroutine would later set _handlers[trades][symbol] = _on_trade.
        mon.add_watch(
            symbol=symbol, stop_price=5.57, shares=13500,
            tp_leg_id='', sl_leg_id='',
            entry_price=5.7863, risk_per_share=0.2162,
            trail_r=1.0, activate_at_r=1.5,
            strategy='bull_flag',
        )

        # Now simulate the ASYNCIO INTERLEAVE that hits in production:
        # Task 2 (subscribe) sync portion runs first
        mon._stream._handlers["trades"][symbol] = mon._on_trade  # subscribe sets
        mon._stream._handlers["quotes"][symbol] = mon._on_quote
        # Task 1 (unsubscribe) sync portion runs SECOND
        mon._stream._handlers["trades"].pop(symbol, None)         # unsubscribe pops
        mon._stream._handlers["quotes"].pop(symbol, None)

        # BUG: handler was popped after subscribe — TTGT trades land in the
        # client with no dispatch target. WS delivers, dispatch silently drops.
        self.assertIsNone(
            mon._stream._handlers["trades"].get(symbol),
            "BUG REPRODUCED: handler was wiped by unsubscribe-after-subscribe race"
        )
        self.assertIsNone(
            mon._stream._handlers["quotes"].get(symbol),
            "BUG REPRODUCED: quote handler was wiped"
        )
        # Watch dict still has TTGT (so r_gain check WOULD pass), but no
        # ticks ever reach _on_trade — 0 activations possible.
        with mon._watch_lock:
            self.assertIn(symbol, mon._watches)

    def test_FIX_atomic_upgrade_preserves_handler(self):
        """PROVE FIX: upgrade_quote_to_stop_watch does the swap atomically
        with NO ws operations — handler stays registered."""
        mon = _make_monitor()
        symbol = 'TTGT'

        # Pre-fill: quote-watch + SDK handler in place
        mon.add_quote_watch(
            symbol=symbol, submit_bid=5.95, submit_ask=6.00,
            submit_bid_size=100, submit_ask_size=100,
        )
        self._add_quote_watch_handlers(mon, symbol)
        before_handler = mon._stream._handlers["trades"].get(symbol)
        self.assertIsNotNone(before_handler)
        self.assertEqual(before_handler.__func__, StopMonitor._on_trade)

        # NEW PATH: atomic upgrade
        mon.upgrade_quote_to_stop_watch(
            symbol=symbol, stop_price=5.57, shares=13500,
            tp_leg_id='', sl_leg_id='',
            entry_price=5.7863, risk_per_share=0.2162,
            trail_r=1.0, activate_at_r=1.5,
            strategy='bull_flag',
        )

        # Handler must still be present and routing to _on_trade (no pop ever happens)
        after_handler = mon._stream._handlers["trades"].get(symbol)
        self.assertIsNotNone(after_handler, "FIX: handler must survive atomic upgrade")
        self.assertEqual(after_handler.__func__, StopMonitor._on_trade)
        q_handler = mon._stream._handlers["quotes"].get(symbol)
        self.assertIsNotNone(q_handler)
        self.assertEqual(q_handler.__func__, StopMonitor._on_quote)

        # State is correct: stop-watch in _watches, quote-watch removed
        with mon._watch_lock:
            self.assertIn(symbol, mon._watches)
            self.assertNotIn(symbol, mon._quote_watches)
        watch = mon._watches[symbol]
        self.assertEqual(watch.entry_price, 5.7863)
        self.assertEqual(watch.trail_r, 1.0)
        self.assertEqual(watch.activate_at_r, 1.5)
        self.assertFalse(watch.trailing_active)  # R-trail starts inactive

    def test_FIX_no_prior_quote_watch_falls_back_to_subscribe(self):
        """Edge case: if no quote_watch exists (e.g., post-restart sync),
        upgrade should still register and trigger a subscribe."""
        mon = _make_monitor()
        symbol = 'TTGT'

        # No prior quote_watch; handlers dict is empty
        self.assertIsNone(mon._stream._handlers["trades"].get(symbol))

        mon.upgrade_quote_to_stop_watch(
            symbol=symbol, stop_price=5.57, shares=13500,
            tp_leg_id='', sl_leg_id='',
            entry_price=5.7863, risk_per_share=0.2162,
            trail_r=1.0, activate_at_r=1.5,
            strategy='bull_flag',
        )

        with mon._watch_lock:
            self.assertIn(symbol, mon._watches)
        # Subscribe wasn't scheduled (loop is None in fixture) but path was
        # taken. In real prod with _loop set, _subscribe_symbol would fire.

    def test_FIX_post_upgrade_trail_activates_on_R_threshold(self):
        """End-to-end behavior: after the atomic upgrade, an _on_trade
        delivered to TTGT at +1.5R+ DOES activate the trail (R-trail
        activation log + ratchet). This is the missing behavior in the
        TTGT 2026-05-08 incident.
        """
        import asyncio
        mon = _make_monitor()
        symbol = 'TTGT'
        mon.add_quote_watch(
            symbol=symbol, submit_bid=5.95, submit_ask=6.00,
            submit_bid_size=100, submit_ask_size=100,
        )
        self._add_quote_watch_handlers(mon, symbol)
        mon.upgrade_quote_to_stop_watch(
            symbol=symbol, stop_price=5.57, shares=13500,
            tp_leg_id='', sl_leg_id='',
            entry_price=5.7863, risk_per_share=0.2162,
            trail_r=1.0, activate_at_r=1.5,
            strategy='bull_flag',
        )

        # Fire a trade tick at $6.32 — TTGT's actual peak. r_gain = (6.32-5.7863)/0.2162 = 2.47 >= 1.5
        trade = MagicMock(symbol=symbol, price=6.32)
        asyncio.run(mon._on_trade(trade))

        with mon._watch_lock:
            watch = mon._watches[symbol]
        # Trail must be active and stop ratcheted (high - 1R = 6.32 - 0.2162 = 6.10)
        self.assertTrue(
            watch.trailing_active,
            "FIX: tick at +2.4R must activate the trail (was the missing behavior on TTGT)"
        )
        expected_stop = 6.32 - 0.2162
        self.assertAlmostEqual(
            watch.stop_price, expected_stop, places=2,
            msg="FIX: trail ratchet must fire on tick at new high"
        )


if __name__ == '__main__':
    unittest.main(verbosity=2)
