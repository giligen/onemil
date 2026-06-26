"""Tests for the touchgo audit log (trading/touchgo_audit.py) + its wiring
into ORBEngine._evaluate_touchgo.

The audit exists to diagnose the EIDO/OSCR/TSDD false-positive class: live
firing Rule M on a bar whose consolidated bb_close_pos was >= 0.5. The
integration test replays exactly that shape — a breakout bar that closes in
the TOP half (should NOT fire) — and asserts the record captures the real
bb_close_pos so the comparison script can catch a future divergence.
"""
from __future__ import annotations

import json
import importlib
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pandas as pd
import pytest

import trading.touchgo_audit as ta


@pytest.fixture
def audit_path(tmp_path, monkeypatch):
    p = tmp_path / "sub" / "touchgo_audit.jsonl"   # sub/ tests dir creation
    monkeypatch.setenv("ORB_TOUCHGO_AUDIT_PATH", str(p))
    monkeypatch.setenv("ORB_TOUCHGO_AUDIT", "1")
    # reset one-shot warn flag
    ta._warned = False
    return p


class TestRecord:
    def test_writes_valid_json_line(self, audit_path):
        ta.record({"symbol": "TSDD", "rule": "M", "fired": False, "bb_close_pos": 0.583})
        lines = audit_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        obj = json.loads(lines[0])
        assert obj["symbol"] == "TSDD"
        assert obj["bb_close_pos"] == 0.583
        assert obj["schema_version"] == ta.SCHEMA_VERSION

    def test_appends_multiple(self, audit_path):
        ta.record({"symbol": "A", "rule": "M"})
        ta.record({"symbol": "B", "rule": "D"})
        lines = audit_path.read_text().strip().splitlines()
        assert [json.loads(x)["symbol"] for x in lines] == ["A", "B"]

    def test_creates_parent_dir(self, audit_path):
        assert not audit_path.parent.exists()
        ta.record({"symbol": "X"})
        assert audit_path.exists()

    def test_disabled_writes_nothing(self, audit_path, monkeypatch):
        monkeypatch.setenv("ORB_TOUCHGO_AUDIT", "0")
        ta.record({"symbol": "X"})
        assert not audit_path.exists()

    def test_serialises_datetimes_via_default_str(self, audit_path):
        ta.record({"symbol": "X", "ts": datetime(2026, 6, 23, 13, 35, tzinfo=timezone.utc)})
        obj = json.loads(audit_path.read_text().strip())
        assert "2026-06-23" in obj["ts"]

    def test_never_raises_on_unwritable_path(self, monkeypatch):
        # Point at a path whose parent cannot be created (a file as a dir)
        monkeypatch.setenv("ORB_TOUCHGO_AUDIT", "1")
        monkeypatch.setenv("ORB_TOUCHGO_AUDIT_PATH", "/dev/null/cannot/exist.jsonl")
        ta._warned = False
        # Must not raise
        ta.record({"symbol": "X"})


class TestEngineWiring:
    """Drive _evaluate_touchgo with a TSDD-shaped bar and assert the audit
    record carries the real bb_close_pos."""

    def _engine(self):
        from trading.orb_engine import ORBEngine
        from trading.orb_touchgo_filter import load_touchgo_config
        eng = ORBEngine.__new__(ORBEngine)
        eng.open_positions = {}
        eng.touchgo_cfg = load_touchgo_config({})   # defaults: thr 0.5, market, age 15
        eng.stop_monitor = None
        eng.notifier = None
        return eng

    def _pos(self, bb_ts):
        from trading.orb_engine import OpenPosition
        return OpenPosition(
            symbol="TSDD", entry_price=8.48, stop_price=8.31, shares=1000,
            trade_id=999, order_id="",   # '' == filled
            entry_time=bb_ts,            # filled at the breakout bar (age 0)
            range_high=8.45, range_low=8.31,
            lock_arm_at_r=1.75, lock_stop_r=0.5,
            composite_score=0.4, quintile="Q4",
            breakout_bar_ts=bb_ts,
        )

    def test_rule_m_no_fire_top_half_close_audited(self, audit_path):
        eng = self._engine()
        bb_ts = pd.Timestamp("2026-06-23 13:35:00", tz="UTC")
        pos = self._pos(bb_ts)
        eng.open_positions["TSDD"] = pos
        # Breakout bar closes in the TOP half → bb_close_pos 0.583 → Rule M must NOT fire
        bars = pd.DataFrame([{
            "timestamp": bb_ts, "open": 8.40, "high": 8.455,
            "low": 8.395, "close": 8.430, "volume": 50000,
        }])
        eng._evaluate_touchgo("TSDD", bars)

        # Position not exited (no fire); audit record written
        assert pos.rule_m_evaluated is True
        rec = json.loads(audit_path.read_text().strip().splitlines()[0])
        assert rec["rule"] == "M"
        assert rec["fired"] is False
        # (8.430-8.395)/(8.455-8.395) = 0.583
        assert rec["bb_close_pos"] == pytest.approx(0.583, abs=0.005)
        assert rec["symbol"] == "TSDD"
        assert rec["breakout_bar_ts"].startswith("2026-06-23T13:35")
        assert rec["rule_m_threshold"] == 0.5

    def test_rule_m_fire_bottom_half_close_audited(self, audit_path):
        eng = self._engine()
        bb_ts = pd.Timestamp("2026-06-23 13:35:00", tz="UTC")
        pos = self._pos(bb_ts)
        eng.stop_monitor = MagicMock()   # capture the force_exit
        eng.open_positions["TSDD"] = pos
        # Breakout bar closes in the BOTTOM half → bb_close_pos ~0.17 → Rule M fires
        bars = pd.DataFrame([{
            "timestamp": bb_ts, "open": 8.45, "high": 8.46,
            "low": 8.40, "close": 8.41, "volume": 50000,
        }])
        eng._evaluate_touchgo("TSDD", bars)

        rec = json.loads(audit_path.read_text().strip().splitlines()[0])
        assert rec["rule"] == "M"
        assert rec["fired"] is True
        assert rec["bb_close_pos"] == pytest.approx(0.1667, abs=0.005)
        eng.stop_monitor.force_exit.assert_called_once()
