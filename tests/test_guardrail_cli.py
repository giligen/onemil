"""tests/test_guardrail_cli.py — scripts/guardrail.py (docs/live_guardrails_spec_20260925.md).

--check on a temp trades.db with a losing book writes the pause state and
calls the (mocked) Telegram helper; --clear stores the reason. Never touches
production data/trades.db or data/guardrail_state.json.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import guardrail as cli  # noqa: E402

from trading import live_guardrail as gr  # noqa: E402


@pytest.fixture(autouse=True)
def _no_config_reads(monkeypatch):
    """Every CLI test drives risk/band through fixed values, never real
    orb.yaml/config.yaml — the CLI's own config wiring is not what these
    tests are checking."""
    monkeypatch.setattr(cli, "stage_risk_usd", lambda book: 100.0)
    monkeypatch.setattr(cli, "band_p5", lambda book, n: None)


def test_check_pauses_losing_book_and_notifies(guardrail_trades_db, insert_trade, tmp_path, monkeypatch):
    # -6 x stage risk $100 = -$600 single-session threshold (rule 3).
    insert_trade("orb", "2026-06-01", pnl=-700.0)

    sent = MagicMock(spec=lambda text, script=gr.TELEGRAM_SCRIPT: None)
    monkeypatch.setattr(gr, "send_guardrail_telegram", sent)

    state_path = tmp_path / "guardrail_state.json"
    rc = cli.run_check(db_path=guardrail_trades_db, state_path=state_path, notify=True)

    assert rc == 1  # a new pause happened
    assert gr.is_paused("orb", path=state_path) is True
    state = gr.load_state(state_path)
    assert state["orb"]["rule"] == gr.RULE_SINGLE_SESSION
    sent.assert_called_once()
    assert "[GUARDRAIL] orb PAUSED" in sent.call_args[0][0]


def test_check_leaves_healthy_book_unpaused(guardrail_trades_db, insert_trade, tmp_path, monkeypatch):
    insert_trade("bull_flag", "2026-06-01", pnl=50.0)
    sent = MagicMock(spec=lambda text, script=gr.TELEGRAM_SCRIPT: None)
    monkeypatch.setattr(gr, "send_guardrail_telegram", sent)

    state_path = tmp_path / "guardrail_state.json"
    rc = cli.run_check(db_path=guardrail_trades_db, state_path=state_path, notify=True)

    assert rc == 0
    assert gr.is_paused("bull_flag", path=state_path) is False
    sent.assert_not_called()


def test_check_pauses_hod_break_scaled_to_its_own_risk_usd(guardrail_trades_db, insert_trade, tmp_path, capsys):
    """2026-09-25: hod_break is PAUSABLE now that it places real resting orders — thresholds scale to
    stage_risk_usd (hod_break.risk_usd, not trading.risk_per_trade), same -3x/x4 trailing-20-session rule
    as bull_flag. -5,000 on one fill <= -3*100*4 = -1,200 trips it."""
    insert_trade("hod_break", "2026-06-01", pnl=-5000.0)
    state_path = tmp_path / "guardrail_state.json"
    cli.run_check(db_path=guardrail_trades_db, state_path=state_path, notify=False)
    assert gr.is_paused("hod_break", path=state_path) is True
    out = capsys.readouterr().out
    assert "hod_break: n=1" in out and "PAUSED" in out


def test_clear_stores_reason(tmp_path, monkeypatch, capsys):
    state_path = tmp_path / "guardrail_state.json"
    stats = gr.LedgerStats(book="orb", n_fills=1, total_usd=-700.0, mean_r=-7.0,
                           trailing_40_mean_r=-7.0, trailing_40_n=1,
                           trailing_20_session_usd=-700.0, worst_month="2026-06",
                           worst_month_usd=-700.0, worst_session_date="2026-06-01",
                           worst_session_usd=-700.0, first_fill_date="2026-06-01")
    check = gr.evaluate_pause(stats, stage_risk_usd=100.0, band_p5=None)
    gr.pause_book(check, stage_risk_usd=100.0, path=state_path, notify=False)
    assert gr.is_paused("orb", path=state_path) is True

    monkeypatch.setattr(gr, "STATE_PATH", state_path)
    monkeypatch.setattr(sys, "argv", ["guardrail.py", "--clear", "orb", "latency fix rehearsed"])
    rc = cli.main()
    assert rc == 0
    assert gr.is_paused("orb", path=state_path) is False
    out = capsys.readouterr().out
    assert "cleared by" in out
    assert gr.load_state(state_path)["orb"]["cleared_reason"] == "latency fix rehearsed"


def test_resolve_state_path_uses_production_default_when_env_unset(monkeypatch):
    """gr.resolve_state_path(None): with ONEMIL_GUARDRAIL_STATE unset and
    STATE_PATH untouched, scripts/guardrail.py --check resolves to the real
    production file — the env override never silently changes prod behavior."""
    monkeypatch.delenv("ONEMIL_GUARDRAIL_STATE", raising=False)
    assert gr.resolve_state_path(None) == gr.STATE_PATH
    assert gr.resolve_state_path(None) == gr.ROOT / "data" / "guardrail_state.json"


def test_eod_guardrail_section_renders(guardrail_trades_db, insert_trade, tmp_path, monkeypatch):
    """scripts/eod_report.py's guardrail_section() reuses trading/live_guardrail.py
    (via scripts/guardrail.py's own stage_risk_usd/band_p5) — no duplicated logic.
    Runs entirely against a temp DB, never data/trades.db."""
    insert_trade("orb", "2026-06-01", pnl=-50.0)
    insert_trade("bull_flag", "2026-06-01", pnl=200.0)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import eod_report as er  # noqa: E402  (local import: needs the sys.path insert above)

    monkeypatch.setattr(gr, "TRADES_DB", guardrail_trades_db)
    monkeypatch.setattr(gr, "STATE_PATH", tmp_path / "state.json")

    out = er.guardrail_section()
    assert out.startswith("GUARDRAIL:")
    assert "orb: n=1" in out
    assert "bull_flag: n=1" in out
    assert "hod_break: no live fills yet" in out
    assert "band p5" in out


def test_check_cli_smoke(guardrail_trades_db, insert_trade, tmp_path, monkeypatch):
    """--check via main() end to end (state path patched to tmp, no real Telegram)."""
    insert_trade("orb", "2026-06-01", pnl=10.0)
    monkeypatch.setattr(gr, "TRADES_DB", guardrail_trades_db)
    monkeypatch.setattr(gr, "STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(gr, "send_guardrail_telegram", MagicMock())
    monkeypatch.setattr(sys, "argv", ["guardrail.py", "--check"])
    rc = cli.main()
    assert rc == 0
