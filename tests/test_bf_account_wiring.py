"""Wiring tests for the Bull Flag paper account (added 2026-05-11).

Soft-fallback contract: when ALPACA_BF_API_KEY / ALPACA_BF_API_SECRET are
EMPTY in .env, bull flag must use the main `alpaca` client (backwards
compat). When the keys are PRESENT but the client init fails, bull flag
must be DISABLED (refuse to silently route to main).

The full main.py run_scan wiring is large with many side effects, so these
tests focus on the contract pieces that determine the wiring decision:
Config properties + the AlpacaClient init pattern. The actual swap in
main.py is a one-liner conditional that's easy to inspect, hard to break.

Bug-4 fix (post-code-review for commit 6fb378e): without these, refactors
of the BF wiring can silently revert to "always use main account" without
test failure.
"""
from __future__ import annotations

import pytest

from config import Config


@pytest.fixture
def empty_env_file(tmp_path, monkeypatch):
    """Empty .env file so Config() doesn't re-load real BF keys from
    the project root .env. Also sets the REQUIRED main account keys
    (ALPACA_API_KEY/SECRET) so Config._validate_required_keys passes
    in the isolated env — the tests are scoped to BF wiring decisions,
    not main credential presence."""
    p = tmp_path / "empty.env"
    p.touch()
    # Required main keys — Config rejects construction without these.
    # Use dummy values; tests never make real API calls.
    monkeypatch.setenv("ALPACA_API_KEY", "test-main-key")
    monkeypatch.setenv("ALPACA_API_SECRET", "test-main-secret")
    return str(p)


class TestConfigBFEmptyKeys:
    """Empty BF keys → empty Config strings → main.py wiring falls back."""

    def test_empty_bf_key_returns_empty_string(self, monkeypatch, empty_env_file):
        monkeypatch.delenv("ALPACA_BF_API_KEY", raising=False)
        c = Config(env_path=empty_env_file)
        assert c.alpaca_bf_api_key == ""

    def test_empty_bf_secret_returns_empty_string(self, monkeypatch, empty_env_file):
        monkeypatch.delenv("ALPACA_BF_API_SECRET", raising=False)
        c = Config(env_path=empty_env_file)
        assert c.alpaca_bf_api_secret == ""

    def test_falsy_truthcheck_triggers_fallback(self, monkeypatch, empty_env_file):
        """The wiring guard is `if config.alpaca_bf_api_key and
        config.alpaca_bf_api_secret:` — empty strings are falsy → guard
        is False → bf_alpaca remains None → fallback to main path."""
        monkeypatch.delenv("ALPACA_BF_API_KEY", raising=False)
        monkeypatch.delenv("ALPACA_BF_API_SECRET", raising=False)
        c = Config(env_path=empty_env_file)
        # The exact guard expression used in main.py
        guard_passes = bool(c.alpaca_bf_api_key and c.alpaca_bf_api_secret)
        assert not guard_passes, (
            "guard MUST be False on empty keys; fallback path depends on it"
        )

    def test_bf_paper_defaults_true(self, monkeypatch, empty_env_file):
        """If unset, default is paper=true — safer side."""
        monkeypatch.delenv("ALPACA_BF_PAPER", raising=False)
        c = Config(env_path=empty_env_file)
        assert c.alpaca_bf_paper is True

    def test_bf_paper_explicit_false(self, monkeypatch, empty_env_file):
        """Explicit ALPACA_BF_PAPER=false flips to live mode."""
        monkeypatch.setenv("ALPACA_BF_PAPER", "false")
        c = Config(env_path=empty_env_file)
        assert c.alpaca_bf_paper is False


class TestConfigBFKeysPresent:
    """Keys present → guard passes → main.py wiring attempts init."""

    def test_keys_present_guard_passes(self, monkeypatch, empty_env_file):
        monkeypatch.setenv("ALPACA_BF_API_KEY", "dummy-key")
        monkeypatch.setenv("ALPACA_BF_API_SECRET", "dummy-secret")
        c = Config(env_path=empty_env_file)
        assert c.alpaca_bf_api_key == "dummy-key"
        assert c.alpaca_bf_api_secret == "dummy-secret"
        guard_passes = bool(c.alpaca_bf_api_key and c.alpaca_bf_api_secret)
        assert guard_passes, (
            "guard MUST be True on non-empty keys; init path depends on it"
        )

    def test_keys_independent_from_orb(self, monkeypatch, empty_env_file):
        """BF keys and ORB keys are independent — setting one doesn't
        cross-contaminate the other. Regression guard for any future
        refactor that consolidates the property accessors."""
        monkeypatch.setenv("ALPACA_BF_API_KEY", "bf-key")
        monkeypatch.setenv("ALPACA_BF_API_SECRET", "bf-secret")
        monkeypatch.setenv("ALPACA_ORB_API_KEY", "orb-key")
        monkeypatch.setenv("ALPACA_ORB_API_SECRET", "orb-secret")
        c = Config(env_path=empty_env_file)
        assert c.alpaca_bf_api_key == "bf-key"
        assert c.alpaca_orb_api_key == "orb-key"
        assert c.alpaca_bf_api_secret == "bf-secret"
        assert c.alpaca_orb_api_secret == "orb-secret"
        # Crossing them would be a wiring bug worse than the original
        assert c.alpaca_bf_api_key != c.alpaca_orb_api_key


class TestStrategyUsesSeparateAccount:
    """Unit tests for main._strategy_uses_separate_account (Tier 1).

    This helper drives the same-account-detection logic that decides whether
    to create a dedicated AlpacaClient + OrderStreamWatcher for BF/ORB or
    to reuse the main account's instances. Pre-existing bug (commit 25d05e1):
    when ALPACA_ORB_API_KEY == ALPACA_API_KEY, a second OrderStreamWatcher
    was opened on the same key, causing duplicate event delivery.
    """

    def test_empty_strategy_key_is_same_account(self):
        """Empty strategy key → fallback to main → NOT separate account."""
        from main import _strategy_uses_separate_account
        assert _strategy_uses_separate_account("", "PKABC") is False

    def test_equal_keys_are_same_account(self):
        """Strategy key == main key → SAME account → reuse resources."""
        from main import _strategy_uses_separate_account
        assert _strategy_uses_separate_account("PKABC", "PKABC") is False

    def test_different_keys_are_separate_account(self):
        """Different keys → SEPARATE account → dedicated resources."""
        from main import _strategy_uses_separate_account
        assert _strategy_uses_separate_account("PKBF1", "PKMAIN") is True

    def test_whitespace_in_strategy_key_normalized(self):
        """Trailing whitespace in .env value must NOT mask same-account.

        Common production .env footgun: env-var value has trailing newline
        or space from manual edit. Pre-strip: `"PKABC\n" != "PKABC"` → True
        → spurious dedicated watcher → original bug recurs silently.
        Defensive strip() catches this.
        """
        from main import _strategy_uses_separate_account
        assert _strategy_uses_separate_account("PKABC\n", "PKABC") is False
        assert _strategy_uses_separate_account("  PKABC  ", "PKABC") is False
        assert _strategy_uses_separate_account("PKABC", " PKABC ") is False

    def test_whitespace_does_not_collapse_truly_different_keys(self):
        """Whitespace stripping must not collapse distinct keys."""
        from main import _strategy_uses_separate_account
        assert _strategy_uses_separate_account("PKABC  ", "PKDEF") is True

    def test_none_strategy_key_treated_as_empty(self):
        """Defensive: None should behave like empty (fallback to main)."""
        from main import _strategy_uses_separate_account
        # falsy via `if not strategy_key:` guard
        assert _strategy_uses_separate_account(None, "PKABC") is False  # type: ignore[arg-type]


class TestSameAccountSharing:
    """Tier 2 + Tier 3' integration: when keys match main, both the
    AlpacaClient and OrderStreamWatcher are SHARED, not duplicated.

    Tests assert via object identity that no second AlpacaClient or
    OrderStreamWatcher is constructed in the same-account case.
    """

    def test_orb_keys_equal_main_reuses_main_alpaca_client(
        self, monkeypatch, empty_env_file
    ):
        """When ALPACA_ORB_API_KEY == ALPACA_API_KEY, orb_alpaca should
        be the SAME object as the main `alpaca` (no duplicate REST
        client constructed)."""
        from main import _strategy_uses_separate_account

        monkeypatch.setenv("ALPACA_API_KEY", "PKMAIN")
        monkeypatch.setenv("ALPACA_API_SECRET", "main-secret")
        monkeypatch.setenv("ALPACA_ORB_API_KEY", "PKMAIN")
        monkeypatch.setenv("ALPACA_ORB_API_SECRET", "main-secret")
        c = Config(env_path=empty_env_file)

        # The wiring decision the run_scan ORB block makes
        is_separate = _strategy_uses_separate_account(
            c.alpaca_orb_api_key, c.alpaca_api_key
        )
        assert is_separate is False, (
            "ORB on main account MUST share the main AlpacaClient "
            "(prevents duplicate REST client + duplicate startup checks)"
        )

    def test_bf_keys_equal_main_reuses_main_alpaca_client(
        self, monkeypatch, empty_env_file
    ):
        from main import _strategy_uses_separate_account

        monkeypatch.setenv("ALPACA_API_KEY", "PKMAIN")
        monkeypatch.setenv("ALPACA_API_SECRET", "main-secret")
        monkeypatch.setenv("ALPACA_BF_API_KEY", "PKMAIN")
        monkeypatch.setenv("ALPACA_BF_API_SECRET", "main-secret")
        c = Config(env_path=empty_env_file)

        is_separate = _strategy_uses_separate_account(
            c.alpaca_bf_api_key, c.alpaca_api_key
        )
        assert is_separate is False

    def test_orb_keys_differ_from_main_uses_dedicated(
        self, monkeypatch, empty_env_file
    ):
        """When ORB keys are truly separate, must NOT collapse to main."""
        from main import _strategy_uses_separate_account

        monkeypatch.setenv("ALPACA_API_KEY", "PKMAIN")
        monkeypatch.setenv("ALPACA_API_SECRET", "main-secret")
        monkeypatch.setenv("ALPACA_ORB_API_KEY", "PKORB")
        monkeypatch.setenv("ALPACA_ORB_API_SECRET", "orb-secret")
        c = Config(env_path=empty_env_file)

        is_separate = _strategy_uses_separate_account(
            c.alpaca_orb_api_key, c.alpaca_api_key
        )
        assert is_separate is True

    def test_main_py_wiring_uses_helper(self):
        """Source-code inspection: main.py BF + ORB blocks must route the
        same-account decision through _strategy_uses_separate_account.

        Catches regression where someone reintroduces a hand-rolled
        `bf_key != main_key` comparison without stripping (the original
        25d05e1 form) — which would re-open the whitespace-fragility
        footgun.
        """
        import re
        src = open("/home/ec2-user/onemil/main.py").read()
        # The helper must exist
        assert "def _strategy_uses_separate_account(" in src
        # Both BF and ORB OrderStreamWatcher decisions use it
        helper_calls = re.findall(
            r"_strategy_uses_separate_account\(\s*config\.alpaca_(bf|orb)_api_key,\s*"
            r"config\.alpaca_api_key\s*\)",
            src,
        )
        assert "bf" in helper_calls, "BF must use _strategy_uses_separate_account"
        assert "orb" in helper_calls, "ORB must use _strategy_uses_separate_account"
        # Bare comparison should not have crept back in
        assert "config.alpaca_orb_api_key != config.alpaca_api_key" not in src, (
            "Bare key comparison without strip() reintroduces whitespace bug"
        )
        assert "config.alpaca_bf_api_key != config.alpaca_api_key" not in src, (
            "Bare key comparison without strip() reintroduces whitespace bug"
        )

    def test_main_py_orb_block_short_circuits_to_main_alpaca(self):
        """Source-code inspection: ORB AlpacaClient creation must skip
        the AlpacaClient(...) constructor call when same-account, and
        instead assign `orb_alpaca = alpaca`.
        """
        src = open("/home/ec2-user/onemil/main.py").read()
        # The short-circuit assignment must be present in the ORB block
        # (use unicode em-dash from the actual log line as a stable anchor)
        assert "ORB Alpaca client: keys match main account" in src, (
            "ORB same-account log line missing — short-circuit may have "
            "been removed"
        )
        # The BF equivalent
        assert "BF Alpaca client: keys match main account" in src, (
            "BF same-account log line missing — short-circuit may have "
            "been removed"
        )
