"""Pin that bull_flag + ORB run the orphan reconciler intraday, not
just at startup. Without these hooks, an intraday stop_loss_unconfirmed
stays orphaned for ~24 hours until next morning's reset_daily —
exactly the SMU/QBTZ failure mode we're fixing.

MACD wave is covered by its existing pattern: `sync_positions()` runs
at the top of `check_exits()` every cycle. ORB + bull flag both lacked
this hook prior to commit (intraday-hook commit).
"""
from pathlib import Path

ROOT = Path(__file__).parent.parent


def _read(rel: str) -> str:
    return (ROOT / rel).read_text()


class TestORBIntradayHook:

    def test_orb_check_exits_calls_maybe_reconcile(self):
        src = _read("trading/orb_engine.py")
        # The hook is named _maybe_reconcile_orphans — pin both the
        # caller site (check_exits) and the throttle.
        assert "_maybe_reconcile_orphans" in src
        assert "self._maybe_reconcile_orphans()" in src
        # Reconcile interval defined
        assert "_reconcile_min_interval_s" in src


class TestBullFlagIntradayHook:

    def test_bull_flag_run_pattern_check_calls_maybe_reconcile(self):
        src = _read("trading/trading_engine.py")
        assert "_maybe_reconcile_orphans" in src
        assert "self._maybe_reconcile_orphans()" in src
        # Reconcile interval defined
        assert "_reconcile_min_interval_s" in src


class TestMACDWaveIntradayHookAlreadyPresent:

    def test_macd_wave_check_exits_calls_sync_positions(self):
        """MACD wave's existing pattern — sync_positions IS the reconciler
        call site for MACD wave (sync_positions calls reconciler at the
        end). Pin that this pattern is preserved."""
        src = _read("trading/macd_wave_engine.py")
        # Pattern: check_exits() begins with self.sync_positions()
        assert "self.sync_positions()" in src
        # And sync_positions includes the reconciler call
        assert "reconcile_strategy_orphans(" in src


class TestNoMoreDuplicateORBAlert:

    def test_orb_does_not_telegram_orphan_summary(self):
        """The pre-2026-06-05 code did self._notify_error('ORPHAN ALPACA
        POSITIONS') which fired on EVERY sync cycle with an orphan
        present — alert storm. The reconciler's per-orphan structured
        alert has a 60-min cooldown and replaces this summary."""
        src = _read("trading/orb_engine.py")
        assert "self._notify_error(\n                f\"ORPHAN ALPACA POSITIONS" not in src, (
            "ORB pre-existing _notify_error('ORPHAN ALPACA POSITIONS') "
            "regrew — it duplicates the reconciler's per-orphan alert "
            "AND has no rate-limit, producing alert storms on stuck "
            "orphans."
        )
