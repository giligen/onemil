"""Unit tests for scripts/check_paper_accounts.py — the read-only probe that must catch a
LIVE-account key pasted into ALPACA_HOD_API_KEY / ALPACA_ORB_API_KEY while *_PAPER=true.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import check_paper_accounts as cpa  # noqa: E402

from data_sources.alpaca_client import AlpacaClient


def _mock_client(is_paper=True, account_number="PA123", equity=100000.0, buying_power=200000.0,
                  positions=None, connects=True):
    c = MagicMock(spec=AlpacaClient)
    c.is_paper = is_paper
    c.test_connection.return_value = connects
    c.get_account_info.return_value = {
        "equity": equity, "buying_power": buying_power, "account_number": account_number,
    }
    c.get_open_positions.return_value = positions or []
    return c


def test_skips_when_neither_key_nor_secret_set(caplog):
    with caplog.at_level("INFO"):
        n = cpa.probe_strategy("hod", "", "", True, "PALIVE0")
    assert n == 0
    assert "skipping" in caplog.text


def test_fails_on_partial_credentials():
    assert cpa.probe_strategy("hod", "key-only", "", True, "PALIVE0") == 1
    assert cpa.probe_strategy("hod", "", "secret-only", True, "PALIVE0") == 1


def test_fails_when_connection_test_fails():
    with patch.object(cpa, "AlpacaClient", return_value=_mock_client(connects=False)):
        assert cpa.probe_strategy("hod", "k", "s", True, "PALIVE0") == 1


def test_clean_distinct_paper_account_is_zero_problems():
    with patch.object(cpa, "AlpacaClient", return_value=_mock_client(
            is_paper=True, account_number="PAHOD1")):
        assert cpa.probe_strategy("hod", "k", "s", True, "PALIVE0") == 0


def test_account_matching_main_live_account_while_paper_true_is_a_problem(caplog):
    """The exact danger this script exists for: the 'paper' key is really the main LIVE
    account's key, so its account_number matches the main account's."""
    with caplog.at_level("ERROR"):
        with patch.object(cpa, "AlpacaClient", return_value=_mock_client(
                is_paper=True, account_number="PALIVE0")):
            n = cpa.probe_strategy("hod", "k", "s", True, "PALIVE0")
    assert n >= 1
    assert "DO NOT enable this book" in caplog.text


def test_is_paper_mismatch_is_flagged():
    """Defensive check: client.is_paper disagreeing with the requested flag (a future wiring
    bug, e.g. hardcoding paper=True) must be caught even without the main-account collision."""
    with patch.object(cpa, "AlpacaClient", return_value=_mock_client(
            is_paper=False, account_number="PAHOD1")):
        assert cpa.probe_strategy("hod", "k", "s", True, "PALIVE0") >= 1


def test_explicit_live_configuration_warns_but_is_not_itself_a_failure(caplog):
    """*_PAPER=false is a deliberate live configuration, not the bug this script hunts for —
    it must not silently pass either, so it gets a WARNING, not a returned problem count."""
    with caplog.at_level("WARNING"):
        with patch.object(cpa, "AlpacaClient", return_value=_mock_client(
                is_paper=False, account_number="PAHOD1")):
            n = cpa.probe_strategy("hod", "k", "s", False, "PALIVE0")
    assert n == 0
    assert "configured to trade its account LIVE" in caplog.text
