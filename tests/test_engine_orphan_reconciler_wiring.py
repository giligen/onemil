"""Source-code inspection — every engine's sync path must delegate to
the shared orphan_reconciler.

Mirrors the buy_stop_guard / two_tier_filter / orb_touchgo_filter parity
pattern. Catches the regression where someone deletes the reconciler
call and reverts to a per-engine orphan loop with its own blind spot.
"""
from pathlib import Path

ROOT = Path(__file__).parent.parent


def _read(rel: str) -> str:
    return (ROOT / rel).read_text()


class TestReconcilerImported:

    def test_macd_wave_imports_reconciler(self):
        src = _read("trading/macd_wave_engine.py")
        assert "from trading.orphan_reconciler import" in src
        assert "reconcile_strategy_orphans" in src

    def test_orb_imports_reconciler(self):
        src = _read("trading/orb_engine.py")
        assert "from trading.orphan_reconciler import" in src
        assert "reconcile_strategy_orphans" in src

    def test_bull_flag_imports_reconciler(self):
        src = _read("trading/trading_engine.py")
        assert "from trading.orphan_reconciler import" in src
        assert "reconcile_strategy_orphans" in src


class TestOldOrphanLogicRemoved:
    """The pre-fix ORB sync called close_position directly in the in-engine
    auto-close block; once we've delegated to the reconciler, the in-engine
    close call must NOT come back."""

    def test_orb_no_inline_orphan_close_call(self):
        src = _read("trading/orb_engine.py")
        # The pre-fix block used the literal "alpaca.close_position(sym)"
        # inside the off-hours auto-close branch. If this regrows, the
        # reconciler is being bypassed (potentially racing with it).
        assert "ORB sync auto-close: orphan" not in src, (
            "The in-engine ORB auto-close branch was reintroduced — "
            "delegate to trading.orphan_reconciler instead."
        )

    def test_orb_no_inline_owned_check_loop(self):
        src = _read("trading/orb_engine.py")
        # The old loop iterated `for sym in orphans: if sym not in orb_owned`
        # using _orb_owned_symbols(lookback_days=4). If that pattern regrows,
        # the deadlock we just fixed is back.
        assert "_orb_owned_symbols(lookback_days=4)" not in src, (
            "The broken ownership check (lookback against open_trades only) "
            "must not be re-used for orphan recovery."
        )
