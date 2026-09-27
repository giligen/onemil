"""Unit tests for main._apply_bf_enable_gate.

Regression test for the 2026-09-27 boot-rehearsal defect: main.py used to
force `trading_engine.enabled = True` unconditionally right after creating
the bull-flag TradingEngine, silently overriding a `trading.enabled: false`
config pause (e.g. the 9/25 BF pause) on every boot that passes --flag.
"""
import logging
from unittest.mock import MagicMock

from main import _apply_bf_enable_gate
from trading.trading_engine import TradingEngine


def _stub_config(trading_enabled: bool):
    config = MagicMock()
    config.trading_enabled = trading_enabled
    return config


def test_apply_bf_enable_gate_false_disables_and_warns(caplog):
    """config.trading_enabled=False must disable the engine and WARN once."""
    trading_engine = MagicMock(spec=TradingEngine)
    config = _stub_config(False)

    with caplog.at_level(logging.WARNING, logger="main"):
        result = _apply_bf_enable_gate(trading_engine, config, "main account")

    assert result is False
    assert trading_engine.enabled is False
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "DISABLED" in warnings[0].message
    assert "trading.enabled=false" in warnings[0].message


def test_apply_bf_enable_gate_true_enables_and_infos(caplog):
    """config.trading_enabled=True must enable the engine and log INFO."""
    trading_engine = MagicMock(spec=TradingEngine)
    config = _stub_config(True)

    with caplog.at_level(logging.INFO, logger="main"):
        result = _apply_bf_enable_gate(trading_engine, config, "BF paper account")

    assert result is True
    assert trading_engine.enabled is True
    infos = [r for r in caplog.records if r.levelno == logging.INFO]
    assert any("ENABLED" in r.message and "BF paper account" in r.message for r in infos)
