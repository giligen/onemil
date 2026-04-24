"""Regression for S1 / S2 — orb_backtest.py mid-day CLI validation.

S1: `--include-today-provisional` only makes sense when `--end` is ET-today.
    Previously the flag fetched bars for `date.today()` regardless of --end.
S2: server runs UTC; `date.today()` between 00:00-04:00 UTC returns UTC's
    next calendar day, not ET's. Use `today_et()` throughout.

These tests don't spin up the full BT pipeline — they exercise the CLI
validation surface directly by monkeypatching `today_et` and parsing
argv. Full end-to-end is exercised by the live post-close timer.
"""
from __future__ import annotations

import sys
from datetime import date
from unittest.mock import patch, MagicMock

import pytest


def _fake_today_et(today_iso):
    """Monkeypatch target — pretend today_et() returns a specific date."""
    return lambda now_et=None: date.fromisoformat(today_iso)


class TestIncludeTodayProvisionalValidation:
    """S1: --include-today-provisional requires --end to match today_et()."""

    def test_flag_without_end_defaults_to_today_et_and_passes(self, monkeypatch):
        """Default `--end` = today in ET; the flag combo is valid and doesn't
        raise. We stop execution at a mock to avoid hitting Alpaca."""
        monkeypatch.setattr('orb_backtest.today_et', _fake_today_et('2026-04-24'))
        # Short-circuit everything after validation — we only care that
        # parsing + validation succeeds.
        monkeypatch.setattr('orb_backtest.Config', MagicMock())
        monkeypatch.setattr('orb_backtest.AlpacaClient', MagicMock())
        monkeypatch.setattr('orb_backtest._latest_features_csv', lambda: None)
        monkeypatch.setattr('orb_backtest._trading_days_between',
                            lambda *a, **kw: [])
        monkeypatch.setattr('orb_backtest._symbols_in_daily_cache',
                            lambda: [])
        monkeypatch.setattr('orb_backtest.Database', MagicMock())
        monkeypatch.setattr('orb_backtest.regen_features', lambda **kw: None)
        monkeypatch.setattr('orb_backtest.run_pipeline_bt', lambda s: None)
        with patch.object(sys, 'argv', [
            'orb_backtest.py', '--include-today-provisional',
        ]):
            from orb_backtest import main
            main()  # should not raise

    def test_flag_with_explicit_today_end_passes(self, monkeypatch):
        monkeypatch.setattr('orb_backtest.today_et', _fake_today_et('2026-04-24'))
        monkeypatch.setattr('orb_backtest.Config', MagicMock())
        monkeypatch.setattr('orb_backtest.AlpacaClient', MagicMock())
        monkeypatch.setattr('orb_backtest._latest_features_csv', lambda: None)
        monkeypatch.setattr('orb_backtest._trading_days_between',
                            lambda *a, **kw: [])
        monkeypatch.setattr('orb_backtest._symbols_in_daily_cache',
                            lambda: [])
        monkeypatch.setattr('orb_backtest.Database', MagicMock())
        monkeypatch.setattr('orb_backtest.regen_features', lambda **kw: None)
        monkeypatch.setattr('orb_backtest.run_pipeline_bt', lambda s: None)
        with patch.object(sys, 'argv', [
            'orb_backtest.py', '--end', '2026-04-24',
            '--include-today-provisional',
        ]):
            from orb_backtest import main
            main()  # should not raise

    def test_flag_with_past_end_raises(self, monkeypatch):
        """S1 bug: previously the flag fetched `date.today()` regardless of
        --end. Now we fail loudly instead of doing the wrong thing."""
        monkeypatch.setattr('orb_backtest.today_et', _fake_today_et('2026-04-24'))
        with patch.object(sys, 'argv', [
            'orb_backtest.py', '--end', '2026-04-20',
            '--include-today-provisional',
        ]):
            from orb_backtest import main
            with pytest.raises(SystemExit) as excinfo:
                main()
            assert '--include-today-provisional' in str(excinfo.value)
            assert '2026-04-20' in str(excinfo.value)

    def test_flag_with_future_end_raises(self, monkeypatch):
        """Symmetric: future --end is also rejected."""
        monkeypatch.setattr('orb_backtest.today_et', _fake_today_et('2026-04-24'))
        with patch.object(sys, 'argv', [
            'orb_backtest.py', '--end', '2026-04-30',
            '--include-today-provisional',
        ]):
            from orb_backtest import main
            with pytest.raises(SystemExit) as excinfo:
                main()
            assert '2026-04-30' in str(excinfo.value)

    def test_flag_omitted_with_past_end_is_fine(self, monkeypatch):
        """Historical BT without the flag is totally normal."""
        monkeypatch.setattr('orb_backtest.today_et', _fake_today_et('2026-04-24'))
        monkeypatch.setattr('orb_backtest.Config', MagicMock())
        monkeypatch.setattr('orb_backtest.AlpacaClient', MagicMock())
        monkeypatch.setattr('orb_backtest._latest_features_csv', lambda: None)
        monkeypatch.setattr('orb_backtest._trading_days_between',
                            lambda *a, **kw: [])
        monkeypatch.setattr('orb_backtest.Database', MagicMock())
        monkeypatch.setattr('orb_backtest.regen_features', lambda **kw: None)
        monkeypatch.setattr('orb_backtest.run_pipeline_bt', lambda s: None)
        with patch.object(sys, 'argv', [
            'orb_backtest.py', '--end', '2026-04-20',
        ]):
            from orb_backtest import main
            main()  # should not raise
