"""Unit + integration tests for scripts/eod_report.py (the deterministic daily report)."""
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import eod_report as er  # noqa: E402


def _trade(symbol, strategy, pnl, status="closed", exit_price=1.0, reason="stop"):
    return {"symbol": symbol, "strategy": strategy, "pnl": pnl, "order_status": status,
            "exit_price": exit_price, "exit_reason": reason}


def test_books_section_groups_by_strategy_and_sums_closed_only():
    rows = [_trade("AAA", "orb", -100), _trade("BBB", "orb", 250, reason="lock"),
            _trade("CCC", "orb", None, status="filled", exit_price=None),
            _trade("DDD", "bull_flag", 40), _trade("EEE", "bull_flag", 0, status="cancelled")]
    txt = er.books_section(rows, "2026-09-21")
    assert "orb: 3 fills, 2 closed, $+150, 1 still open" in txt
    assert "bull_flag: 1 fills, 1 closed, $+40" in txt
    assert "AAA -100 stop" in txt and "BBB +250 lock" in txt


def test_books_section_empty():
    assert er.books_section([], "2026-09-21") == "BOOKS 2026-09-21: no fills."


def test_last_matching_line_missing_log_warns(tmp_path, caplog):
    out = er.last_matching_line(tmp_path / "nope.log", ("GREEN",))
    assert "missing" in out and "log missing" in caplog.text


def test_last_matching_line_picks_last_verdict(tmp_path):
    p = tmp_path / "g.log"
    p.write_text("noise\n[GREEN 1/10] a\nmore\n[RED DAY] b\n")
    assert er.last_matching_line(p, ("GREEN", "RED")) == "[RED DAY] b"


def test_boot_section_skips_weekend():
    with patch.object(er, "yaml_flag", return_value="true"):
        assert "NEXT BOOT 2026-09-21" in er.boot_section("2026-09-18")  # Friday -> Monday


def test_phrase_with_llm_rejects_bad_prefix(caplog):
    fake = subprocess.CompletedProcess(args=[], returncode=0, stdout="hello", stderr="")
    with patch.object(er.subprocess, "run", return_value=fake):
        assert er.phrase_with_llm("x") == ""
    assert "rejected" in caplog.text


def test_phrase_with_llm_accepts_and_truncates():
    fake = subprocess.CompletedProcess(args=[], returncode=0, stdout="[EOD] " + "y" * 5000, stderr="")
    with patch.object(er.subprocess, "run", return_value=fake):
        out = er.phrase_with_llm("x")
    assert out.startswith("[EOD]") and len(out) == er.TELEGRAM_MAX


def test_main_falls_back_to_raw_summary_and_writes_file(tmp_path, monkeypatch, capsys, caplog):
    """Integration: DB -> assemble -> LLM failure -> raw summary printed, file written, no send."""
    monkeypatch.setattr(er, "ROOT", tmp_path)
    monkeypatch.setattr(er.Database, "get_trades_by_date",
                        lambda self, d: [_trade("ZZZ", "orb", 12.0)])
    monkeypatch.setattr(er.Database, "__init__", lambda self: None)
    monkeypatch.setattr(er, "assemble", lambda day, trades: f"SUMMARY {day} {len(trades)} trades")
    monkeypatch.setattr(er, "phrase_with_llm", lambda s: "")
    sent = []
    monkeypatch.setattr(er, "send_telegram", lambda t: sent.append(t))
    monkeypatch.setattr(sys, "argv", ["eod_report.py", "--date", "2026-09-21", "--no-send"])
    assert er.main() == 0
    out = capsys.readouterr().out
    assert out.startswith("[EOD] SUMMARY 2026-09-21 1 trades")
    assert (tmp_path / "logs/eod/2026-09-21.md").read_text().startswith("SUMMARY")
    assert "falling back" in caplog.text and sent == []


def test_hygiene_section_survives_journal_timeout(monkeypatch, caplog):
    """journalctl timing out on a huge journal must not take the report down (2026-09-25 incident)."""
    def fake_run(cmd, **kwargs):
        if cmd and cmd[0] == "journalctl":
            raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 20))
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="active", stderr="")
    monkeypatch.setattr(er.subprocess, "run", fake_run)
    txt = er.hygiene_section()
    assert "(journal check timed out)" in txt
    assert "timed out" in caplog.text


def test_assemble_renders_ramp_and_boot_despite_all_subprocess_timeouts(tmp_path, monkeypatch):
    """Every subprocess (journalctl, systemctl, ramp scripts) timing out still yields RAMP: and NEXT BOOT."""
    monkeypatch.setattr(er, "ROOT", tmp_path)
    monkeypatch.setattr(er, "parity_section", lambda day: "PARITY: ok")
    monkeypatch.setattr(er, "guardrail_section", lambda: "GUARDRAIL: ok")
    monkeypatch.setattr(er, "research_section", lambda: "RESEARCH: ok")

    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 20))
    monkeypatch.setattr(er.subprocess, "run", fake_run)

    out = er.assemble("2026-09-18", [])
    assert "RAMP:" in out
    assert "NEXT BOOT" in out


def test_parity_section_reads_bf_json(tmp_path, monkeypatch):
    monkeypatch.setattr(er, "ROOT", tmp_path)
    (tmp_path / "logs/bf_parity").mkdir(parents=True)
    (tmp_path / "logs/bf_parity/bf_parity_2026-09-21.json").write_text(
        json.dumps({"status": "OK", "n_bt_trades": 2, "n_live_rows": 2, "bt_stale": False}))
    txt = er.parity_section("2026-09-21")
    assert "BF parity OK (bt 2, live 2, stale=False)" in txt and "freeze: none" in txt


@pytest.mark.integration
def test_script_runs_end_to_end_no_send_no_llm():
    """System-ish: the real script against the real DB and checkers, nothing sent."""
    r = subprocess.run([sys.executable, "scripts/eod_report.py", "--no-send", "--no-llm",
                        "--date", "2026-09-18"], capture_output=True, text=True, timeout=400,
                       cwd=Path(__file__).resolve().parents[1])
    assert r.returncode == 0, r.stderr[-500:]
    assert r.stdout.startswith("[EOD]") and "RAMP:" in r.stdout and "NEXT BOOT" in r.stdout
