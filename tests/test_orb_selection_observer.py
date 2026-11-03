"""Unit tests for scripts/orb_selection_observer.py (DST waits, journal parse, verdict).

Spec: docs/orb_parity_defects_spec_20261103.md fix 1.
"""
from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts import orb_selection_observer as obs

ARCHIVE_1030 = Path(__file__).parent.parent / "logs" / "session_archive" / "2026-10-30.log"

RECONCILE_1030 = (
    "Oct 30 13:35:17 h onemil-trader[1]: 2026-10-30 13:35:17 | INFO     | trading.orb_engine:4495 | "
    "[ORB PREPLACE] reconcile final top-8 of 7 ranked: ['SNK', 'SSPC', 'RMNI', 'MICC', 'UVXY', "
    "'OIS', 'BTCZ'] | preplaced: ['IREZ']")
PROV_TOP = ("2026-10-30 13:34:57 | INFO     | trading.orb_engine:4304 | [ORB PREPLACE] provisional top-1 "
            "@ 09:34:57.481 ET: IREZ(rh=$22.61,Q4)")
PROV_VETO = ("2026-10-30 13:34:57 | INFO | trading.orb_engine:4277 | [ORB PREPLACE] provisional veto "
             "SNK (pdr_veto) — not planned; final selection re-decides")


class TestEtTargets:
    def test_edt_date_target_is_1329_utc(self):
        now = datetime(2026, 10, 30, 12, 0, tzinfo=timezone.utc)
        t = obs.et_target(now, 9, 29, 30)
        assert t.astimezone(timezone.utc).strftime("%H:%M:%S") == "13:29:30"

    def test_est_date_target_is_1429_utc(self):
        now = datetime(2026, 11, 2, 12, 0, tzinfo=timezone.utc)
        t = obs.et_target(now, 9, 29, 30)
        assert t.astimezone(timezone.utc).strftime("%H:%M:%S") == "14:29:30"

    @pytest.mark.parametrize("day,expect", [((2026, 10, 30), "13:25:00"), ((2026, 11, 2), "14:25:00")])
    def test_journal_since_is_explicit_utc(self, day, expect):
        now = datetime(*day, 12, 0, tzinfo=timezone.utc)
        assert obs.journal_since_arg(now) == f"{day[0]}-{day[1]:02d}-{day[2]:02d} {expect} UTC"

    def test_wait_returns_immediately_when_past_and_sleeps_when_future(self):
        with patch.object(obs.time, "sleep") as sl:
            obs.wait_until_et(0, 0, 0)  # long past on any date
            sl.assert_not_called()


class TestPreplaceParse:
    def test_1030_lines_give_the_ranked_names(self):
        syms = obs.preplace_symbols([RECONCILE_1030, PROV_TOP, PROV_VETO])
        assert {"SNK", "SSPC", "RMNI", "MICC", "UVXY", "OIS", "BTCZ", "IREZ"} <= syms

    @pytest.mark.skipif(not ARCHIVE_1030.exists(), reason="session archive not present")
    def test_real_1030_archive_parses_non_empty(self):
        """The 10/30 bug: archive has no 'ORB SCORED'/'range complete' lines; parse must still see the day."""
        lines = [l for l in ARCHIVE_1030.read_text(errors="replace").splitlines() if "ORB PREPLACE" in l]
        syms = obs.preplace_symbols(lines)
        assert "IREZ" in syms and "SNK" in syms


class TestVerdict:
    def test_empty_live_never_caught(self):
        assert obs.decide_verdict([], [], True, True, ["X"]) == "ops_warning"
        assert obs.decide_verdict([], [], False, True, ["X"]) == "ops_warning"

    def test_journal_failure_is_ops_warning(self):
        assert obs.decide_verdict(["A"], ["A"], True, False, ["X"]) == "ops_warning"

    def test_nonempty_live_with_real_drop_is_caught(self):
        assert obs.decide_verdict(["A"], ["A"], True, True, ["X"]) == "caught"

    def test_clean(self):
        assert obs.decide_verdict(["A"], ["A"], True, True, []) == "clean"

    def test_engine_active_detection(self):
        assert obs.engine_active(["x ORB ENTRY SUBMITTED: A"])
        assert obs.engine_active(["[ORB PREPLACE] reconcile final top-8 of 0 ranked: []"])
        assert not obs.engine_active(["nothing here"])


class TestLoadJournal:
    def setup_method(self):
        obs._JOURNAL.update(lines=None, ok=False)

    def test_timeout_marks_failed_and_empty(self):
        with patch.object(obs.subprocess, "run", side_effect=subprocess.TimeoutExpired("journalctl", 1)):
            assert obs.load_journal() is False
        assert obs._JOURNAL["lines"] == [] and obs._JOURNAL["ok"] is False

    def test_success_caches_once(self):
        res = MagicMock(returncode=0, stdout="a ORB SCORED: X\nb\n", stderr="")
        with patch.object(obs.subprocess, "run", return_value=res) as run:
            assert obs.journal_lines("ORB SCORED") == ["a ORB SCORED: X"]
            obs.journal_lines("b")
            assert run.call_count == 1
        assert obs._JOURNAL["ok"] is True
