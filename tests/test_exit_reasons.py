"""
Unit tests for trading.exit_reasons.

Two contracts to defend:

  1. **Backward compat**: every string value that has EVER been written to
     trades.exit_reason in production must remain a defined ExitReason
     member. Renaming a member without a backfill would break analytics
     SQL (`GROUP BY exit_reason`) and silently mis-attribute trades.

  2. **No drift**: every inline `exit_reason='...'` (or `'exit_reason':
     '...'`) literal across the BF / ORB / MACD / BT / stop-monitor code
     paths must resolve to an ExitReason member string. Otherwise a typo
     in a new branch (`stop-loss` instead of `stop_loss`) silently writes
     a novel value that's invisible to the daily summary.

The drift test parses each .py file with ast and walks the tree —
catches both kwargs (`exit_reason='x'`) and dict literals
(`{'exit_reason': 'x'}`) without false positives on dunders or comments.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from trading.exit_reasons import (
    ExitReason,
    is_attributed,
    is_historical_only,
    is_known,
    needs_reconcile,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Enum membership + helpers
# ---------------------------------------------------------------------------


class TestEnumMembership:
    """Every observed-in-prod exit_reason must be a defined member."""

    # Snapshot of `SELECT exit_reason, COUNT(*) FROM trades` on
    # data/trades.db as of 2026-06-12. Update when introducing a new
    # value (and only after adding it to the enum).
    OBSERVED_IN_PROD = (
        'stop_loss', 'tag_bb', 'force_close', 'trail_stop',
        'sync_reconcile', 'lock_stop', 'unknown_exit',
        'macd_flip', 'thin_liquidity_reject', 'take_profit',
        'stop_loss_timeout', 'stop_loss_bracket_sl_race',
    )

    @pytest.mark.parametrize('value', OBSERVED_IN_PROD)
    def test_prod_value_is_defined(self, value):
        assert is_known(value), (
            f"{value!r} exists in production DB but is NOT in ExitReason. "
            f"Renaming would break analytics; add a member if needed."
        )

    def test_no_member_string_collision(self):
        """Two members must not share the same .value (would break
        analytics aggregation)."""
        values = [m.value for m in ExitReason]
        assert len(values) == len(set(values)), (
            f"Duplicate member values in ExitReason: {values}"
        )

    def test_all_members_are_lowercase_snake(self):
        """All values are lowercase snake_case (matches existing DB rows
        and the rest of the codebase)."""
        for m in ExitReason:
            assert m.value == m.value.lower(), m.value
            assert ' ' not in m.value, m.value


class TestCategorizationHelpers:
    def test_is_known_returns_false_for_none_and_empty(self):
        assert is_known(None) is False
        assert is_known('') is False
        assert is_known('made_up_reason_2027') is False

    def test_is_attributed_excludes_leak_paths(self):
        assert is_attributed(ExitReason.STOP_LOSS.value) is True
        assert is_attributed(ExitReason.TRAIL_STOP.value) is True
        assert is_attributed(ExitReason.FORCE_CLOSE.value) is True
        # Leak paths — analytics must NOT count these as clean attributions
        assert is_attributed(ExitReason.UNKNOWN_EXIT.value) is False
        assert is_attributed(ExitReason.STOP_LOSS_UNCONFIRMED.value) is False
        assert is_attributed(ExitReason.SYNC_RECONCILE.value) is False
        assert is_attributed(ExitReason.STOP_LOSS_TIMEOUT.value) is False

    def test_needs_reconcile_flags_leaks_and_history(self):
        assert needs_reconcile(ExitReason.UNKNOWN_EXIT.value) is True
        assert needs_reconcile(ExitReason.STOP_LOSS_UNCONFIRMED.value) is True
        assert needs_reconcile(ExitReason.SYNC_RECONCILE.value) is True
        assert needs_reconcile(ExitReason.STOP_LOSS_TIMEOUT.value) is True
        assert needs_reconcile(ExitReason.STOP_LOSS.value) is False
        assert needs_reconcile(ExitReason.FORCE_CLOSE.value) is False
        assert needs_reconcile(None) is False
        assert needs_reconcile('bogus') is False

    def test_historical_only_is_a_subset_of_needs_reconcile(self):
        for m in ExitReason:
            if is_historical_only(m.value):
                assert needs_reconcile(m.value), (
                    f"{m.value}: historical implies needs_reconcile"
                )

    def test_str_enum_subclass_str_compat(self):
        """A member should be usable wherever a str is — important for
        legacy assertions like `exit_reason == 'stop_loss'` to keep
        passing after the refactor."""
        assert ExitReason.STOP_LOSS == 'stop_loss'
        assert {'exit_reason': ExitReason.STOP_LOSS.value} == {
            'exit_reason': 'stop_loss'
        }


# ---------------------------------------------------------------------------
# AST drift prevention
# ---------------------------------------------------------------------------


# Files that emit exit_reason at runtime. Adding a new emitter file
# without adding it here is a deliberate choice — but the new file
# should still go through ExitReason; this list keeps the scan finite.
EXIT_REASON_EMITTER_FILES = (
    'trading/orb_engine.py',
    'trading/trading_engine.py',
    'trading/stop_monitor.py',
    'trading/macd_wave_engine.py',
    'backtest.py',
)


def _find_exit_reason_string_literals(path: Path) -> list[tuple[int, str]]:
    """Walk the AST of `path` and return [(lineno, value), ...] for
    every place where an inline string literal is assigned to an
    `exit_reason` kwarg or dict key. Ignores:
      * `partial_exit_reason` (separate column, different taxonomy)
      * docstrings + comments (AST doesn't surface them at these nodes)
      * variable assignment like `exit_reason = ExitReason.X.value`
        (those are ast.Attribute, not ast.Constant)
    """
    tree = ast.parse(path.read_text())
    findings: list[tuple[int, str]] = []

    for node in ast.walk(tree):
        # kwarg form: foo(exit_reason='stop_loss')
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if (kw.arg == 'exit_reason'
                        and isinstance(kw.value, ast.Constant)
                        and isinstance(kw.value.value, str)):
                    findings.append((kw.value.lineno, kw.value.value))

        # dict literal form: {'exit_reason': 'stop_loss'}
        if isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if (isinstance(k, ast.Constant)
                        and k.value == 'exit_reason'
                        and isinstance(v, ast.Constant)
                        and isinstance(v.value, str)):
                    findings.append((v.lineno, v.value))

    return findings


class TestNoStringLiteralDrift:
    """Every inline exit_reason='...' or 'exit_reason': '...' literal in
    emitter files must match an ExitReason member. If a new contributor
    adds a typo'd literal (`'stop-loss'` or `'stoploss'`), this test
    catches it before it lands in production DB."""

    @pytest.mark.parametrize('rel', EXIT_REASON_EMITTER_FILES)
    def test_emitter_file_has_no_unknown_literal(self, rel):
        path = REPO_ROOT / rel
        assert path.exists(), f"missing emitter file: {rel}"
        findings = _find_exit_reason_string_literals(path)
        unknown = [(ln, v) for ln, v in findings if not is_known(v)]
        assert not unknown, (
            f"{rel} contains inline exit_reason literal(s) not defined in "
            f"trading/exit_reasons.py::ExitReason. Add the member there "
            f"first (or use ExitReason.X.value at the call site).\n"
            f"  Offending lines: {unknown}"
        )
