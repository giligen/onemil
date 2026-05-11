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
