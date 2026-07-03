"""Source-code regression guard: there is NO periodic intraday orphan
reconciler in ORB or bull flag.

History (2026-06-05 → 2026-06-06):
  L7 added _maybe_reconcile_orphans hooks to ORB's check_exits and
  bull flag's run_pattern_check, throttled to 60s. Code review found:
    - ORB hook called sync_positions(), which race-A re-rehydrated
      the exit_pending_verification row the engine had just released.
      Reconciler was then bypassed because the symbol was back in
      open_positions.
    - Bull flag hook passed tracked_symbols=_traded_symbols. Bull
      flag adds symbols to _traded_symbols at entry-fill time
      (trading_engine.py:1552), so by the time _maybe_reconcile_orphans
      runs, the orphan symbol is already in tracked → reconciler
      skips. The hook was a no-op for bull flag's own orphans.

This file pins, via source inspection, that those hooks do not regrow.
If a future change re-introduces periodic intraday reconciliation
without addressing race A and bull flag's tracked-symbols semantics,
this test calls it out.

The MACD wave reconciler call (in sync_positions at the end) is left
in place because MACD's tracked_symbols=open_positions semantics +
race-A fix in the recovery loop together make it correct.
"""
from pathlib import Path

ROOT = Path(__file__).parent.parent


def _read(rel: str) -> str:
    return (ROOT / rel).read_text()


class TestNoPeriodicHooksRegrew:

    def test_orb_no_maybe_reconcile_orphans_method(self):
        src = _read("trading/orb_engine.py")
        assert "_maybe_reconcile_orphans" not in src, (
            "ORB's _maybe_reconcile_orphans regrew. Race A: the method "
            "called sync_positions() which re-rehydrated "
            "exit_pending_verification rows and bypassed the reconciler. "
            "Don't re-add without first fixing the racy interaction with "
            "sync_positions's recovery loop."
        )

    def test_bull_flag_no_maybe_reconcile_orphans_method(self):
        src = _read("trading/trading_engine.py")
        assert "_maybe_reconcile_orphans" not in src, (
            "Bull flag's _maybe_reconcile_orphans regrew. It passed "
            "tracked_symbols=_traded_symbols, which already includes the "
            "orphan symbol (added at entry-fill) — making the reconciler "
            "skip the orphan. Don't re-add without first redefining what "
            "'actively-managed' means for bull flag."
        )


class TestRaceAGuardsPresent:
    """The race-A fix lives in sync_positions recovery loops as an
    explicit skip of exit_pending_verification rows. If someone removes
    the guard while the active-statuses + reconciler ownership model
    stays the same, the race re-emerges. Pin the guard via source
    inspection."""

    def test_macd_wave_skips_exit_pending_verification_in_recovery(self):
        src = _read("trading/macd_wave_engine.py")
        assert "exit_pending_verification" in src, (
            "MACD wave's sync_positions recovery loop must skip "
            "exit_pending_verification rows — they're owned by the "
            "orphan_reconciler, not by the engine. See plan "
            "mellow-sniffing-abelson + L1/L2."
        )
        # Pin the specific guard pattern is present
        assert (
            "trade.get('order_status') == 'exit_pending_verification'"
            in src
        ), (
            "The Race-A guard in MACD wave's recovery loop was removed "
            "or rewritten. Re-adding sync-time rehydrate for these rows "
            "creates a stale-stop StopMonitor watch and bypasses the "
            "reconciler."
        )

    def test_orb_skips_exit_pending_verification_in_recovery(self):
        src = _read("trading/orb_engine.py")
        assert "exit_pending_verification" in src
        # ORB stores the DB status in db_status before branching
        assert "db_status == 'exit_pending_verification'" in src, (
            "The Race-A guard in ORB sync_positions's State A rehydrate "
            "was removed or rewritten. See plan mellow-sniffing-abelson."
        )
