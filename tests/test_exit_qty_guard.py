"""trading/exit_qty_guard.py — the broker-truth sell-qty guard shared by every live exit path.

9/25 CDNA incident: HOD's registry still showed 57 sh open after StopMonitor's own stop-loss had
already flattened the broker position; force_close_all sold the registry's 57 sh again with no
broker check, taking the shared live account short. These are the guard's unit-level contract
tests; tests/test_hod_break_engine.py exercises it wired into force_close_all end to end.
"""
from unittest.mock import MagicMock

from trading.exit_qty_guard import get_signed_broker_qty, resolve_broker_capped_sell_qty


class TestResolveBrokerCappedSellQty:
    def test_registry_and_broker_agree_sells_registry_qty(self):
        assert resolve_broker_capped_sell_qty('CDNA', registry_qty=57, broker_qty_signed=57, tag='[HOD]') == 57

    def test_broker_flat_skips_entirely(self):
        """registry 57 / broker 0 -> None (skip, never sell into flat)."""
        assert resolve_broker_capped_sell_qty('CDNA', registry_qty=57, broker_qty_signed=0, tag='[HOD]') is None

    def test_broker_holds_fewer_shares_clamps_down(self):
        """registry 57 / broker 30 -> sell 30, never the stale registry qty."""
        assert resolve_broker_capped_sell_qty('CDNA', registry_qty=57, broker_qty_signed=30, tag='[HOD]') == 30

    def test_broker_already_short_skips_entirely(self):
        """A prior over-exit already shorted the account -- must never sell MORE into it."""
        assert resolve_broker_capped_sell_qty('CDNA', registry_qty=57, broker_qty_signed=-57, tag='[HOD]') is None

    def test_broker_holds_more_shares_sells_the_brokers_qty_not_the_stale_registry(self):
        # APT/MLTX 2026-05-11 class: a partial-fill race the registry missed. Selling the broker's
        # full qty still only reaches flat (never short) and avoids stranding an orphan residual.
        assert resolve_broker_capped_sell_qty('CDNA', registry_qty=30, broker_qty_signed=57, tag='[HOD]') == 57

    def test_skip_and_clamp_both_notify(self):
        calls = []
        resolve_broker_capped_sell_qty('CDNA', 57, 0, '[HOD]', notify_fn=calls.append)
        assert calls and 'CDNA' in calls[0] and 'no position' in calls[0]
        calls.clear()
        resolve_broker_capped_sell_qty('CDNA', 57, 30, '[HOD]', notify_fn=calls.append)
        assert calls and 'CDNA' in calls[0]

    def test_silent_common_case_does_not_notify(self):
        calls = []
        resolve_broker_capped_sell_qty('CDNA', 57, 57, '[HOD]', notify_fn=calls.append)
        assert calls == []

    def test_notify_failure_is_logged_not_raised(self):
        def boom(_msg): raise RuntimeError('telegram down')
        # Must not raise even though notify_fn blew up -- the guard's own decision still holds.
        assert resolve_broker_capped_sell_qty('CDNA', 57, 0, '[HOD]', notify_fn=boom) is None


class TestGetSignedBrokerQty:
    def test_returns_signed_qty_for_symbol(self):
        alpaca = MagicMock()
        alpaca.get_open_positions.return_value = [{'symbol': 'CDNA', 'qty': -57}, {'symbol': 'VECO', 'qty': 11}]
        assert get_signed_broker_qty(alpaca, 'CDNA') == -57
        assert get_signed_broker_qty(alpaca, 'VECO') == 11

    def test_symbol_not_found_is_flat(self):
        alpaca = MagicMock(); alpaca.get_open_positions.return_value = [{'symbol': 'VECO', 'qty': 11}]
        assert get_signed_broker_qty(alpaca, 'CDNA') == 0

    def test_broker_error_is_unknown_none_not_flat(self):
        """Review B1 (2026-10-03): an API error is UNKNOWN (None), never 0 -- 0 is reserved for 'genuinely absent'."""
        alpaca = MagicMock(); alpaca.get_open_positions.side_effect = RuntimeError('rate limited')
        assert get_signed_broker_qty(alpaca, 'CDNA') is None
