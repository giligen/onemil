"""tests/test_live_guardrail.py — docs/live_guardrails_spec_20260925.md G1.

Covers: ledger from a temp trades.db across three books, each pause rule
firing exactly at its frozen threshold (and not just past it), that a
simulated config/stage change cannot reset the ledger start, state-file
round-trip, and a logged clear-with-reason. Never touches production
data/trades.db or data/guardrail_state.json (every path is tmp_path).
"""
from __future__ import annotations

import json

import pytest

from trading import live_guardrail as gr


# --------------------------------------------------------------- live_record

def test_live_record_three_books(guardrail_trades_db, insert_trade):
    """orb, bull_flag, hod_break each ledger independently from the same DB."""
    insert_trade('orb', '2026-06-01', pnl=-50.0)
    insert_trade('orb', '2026-06-02', pnl=125.0)
    insert_trade('bull_flag', '2026-06-01', pnl=200.0)
    insert_trade('hod_break', '2026-06-01', pnl=0.0)  # ledgered like any other book; pnl 0.0 trips no threshold

    orb = gr.live_record('orb', stage_risk_usd=100.0, db_path=guardrail_trades_db)
    bf = gr.live_record('bull_flag', stage_risk_usd=100.0, db_path=guardrail_trades_db)
    hod = gr.live_record('hod_break', stage_risk_usd=100.0, db_path=guardrail_trades_db)

    assert orb.n_fills == 2 and orb.total_usd == pytest.approx(75.0)
    assert orb.first_fill_date == '2026-06-01'
    assert bf.n_fills == 1 and bf.total_usd == pytest.approx(200.0)
    assert hod.n_fills == 1 and hod.total_usd == pytest.approx(0.0)


def test_live_record_empty_book_is_honest_zero(guardrail_trades_db):
    """A book with zero fills in the DB reports n_fills=0, not an error."""
    stats = gr.live_record('bull_flag', stage_risk_usd=100.0, db_path=guardrail_trades_db)
    assert stats.n_fills == 0
    assert stats.first_fill_date is None
    assert stats.trailing_40_mean_r is None


def test_config_change_does_not_reset_ledger(guardrail_trades_db, insert_trade):
    """Ledger start is the book's first-ever fill — a later 'stage start' or
    config-change date must never move it (spec: 'never reset by stage or
    config'). live_record() takes no such argument at all, so a caller
    cannot accidentally reset it; this asserts the origin point directly."""
    insert_trade('orb', '2026-01-05', pnl=-30.0)
    insert_trade('orb', '2026-05-20', pnl=10.0)  # e.g. a config change landed here
    insert_trade('orb', '2026-06-01', pnl=40.0)

    assert gr.first_fill_date('orb', db_path=guardrail_trades_db) == '2026-01-05'
    stats = gr.live_record('orb', stage_risk_usd=100.0, db_path=guardrail_trades_db)
    assert stats.n_fills == 3
    assert stats.first_fill_date == '2026-01-05'
    assert stats.total_usd == pytest.approx(20.0)


def test_trade_risk_usd_prefers_total_risk_then_reconstructs_then_stage(caplog):
    row_with_total = {'pnl': 50.0, 'total_risk': 200.0}
    assert gr.trade_risk_usd(row_with_total, stage_risk_usd=375.0) == 200.0

    row_reconstruct = {'entry_price': 10.0, 'stop_loss_price': 9.0, 'shares': 100}
    assert gr.trade_risk_usd(row_reconstruct, stage_risk_usd=375.0) == pytest.approx(100.0)

    with caplog.at_level('WARNING'):
        row_fallback = {}
        assert gr.trade_risk_usd(row_fallback, stage_risk_usd=375.0) == 375.0
    assert any('stage risk' in r.message for r in caplog.records)


# --------------------------------------------------------- pause rule (G1)

def _stats(book='orb', n_fills=40, trailing_40_mean_r=0.0, trailing_40_n=20,
          trailing_20_session_usd=0.0, worst_session_usd=0.0,
          worst_session_date='2026-06-01') -> gr.LedgerStats:
    return gr.LedgerStats(
        book=book, n_fills=n_fills, total_usd=0.0, mean_r=trailing_40_mean_r,
        trailing_40_mean_r=trailing_40_mean_r, trailing_40_n=trailing_40_n,
        trailing_20_session_usd=trailing_20_session_usd, worst_month=None,
        worst_month_usd=None, worst_session_date=worst_session_date,
        worst_session_usd=worst_session_usd, first_fill_date='2026-01-01', sessions=[])


def test_pause_rule_band_p5_fires_exactly_at_threshold():
    at = _stats(trailing_40_mean_r=-0.50, trailing_40_n=20)
    check = gr.evaluate_pause(at, stage_risk_usd=100.0, band_p5=-0.50)
    assert check.should_pause and check.rule == gr.RULE_BAND_P5

    above = _stats(trailing_40_mean_r=-0.49, trailing_40_n=20)
    check2 = gr.evaluate_pause(above, stage_risk_usd=100.0, band_p5=-0.50)
    assert not check2.should_pause


def test_pause_rule_band_p5_requires_min_fills():
    """Below p5 but only 19 fills feeding the mean -> rule 1 does not fire."""
    stats = _stats(trailing_40_mean_r=-0.90, trailing_40_n=19)
    check = gr.evaluate_pause(stats, stage_risk_usd=100.0, band_p5=-0.50)
    assert not check.should_pause


def test_pause_rule_trailing_20_session_fires_exactly_at_threshold():
    # -3 * 100 * 8 (orb mult) = -2400
    at = _stats(trailing_20_session_usd=-2400.0)
    check = gr.evaluate_pause(at, stage_risk_usd=100.0, band_p5=None)
    assert check.should_pause and check.rule == gr.RULE_TRAILING_20_SESSION

    above = _stats(trailing_20_session_usd=-2399.0)
    check2 = gr.evaluate_pause(above, stage_risk_usd=100.0, band_p5=None)
    assert not check2.should_pause


def test_pause_rule_single_session_fires_exactly_at_threshold():
    # -6 * 100 = -600
    at = _stats(worst_session_usd=-600.0)
    check = gr.evaluate_pause(at, stage_risk_usd=100.0, band_p5=None)
    assert check.should_pause and check.rule == gr.RULE_SINGLE_SESSION

    above = _stats(worst_session_usd=-599.0)
    check2 = gr.evaluate_pause(above, stage_risk_usd=100.0, band_p5=None)
    assert not check2.should_pause


def test_pause_rule_bull_flag_uses_x4_session_mult():
    # -3 * 100 * 4 (bf mult) = -1200
    at = _stats(book='bull_flag', trailing_20_session_usd=-1200.0)
    check = gr.evaluate_pause(at, stage_risk_usd=100.0, band_p5=None)
    assert check.should_pause and check.rule == gr.RULE_TRAILING_20_SESSION


def test_hod_break_pauses_now_that_it_places_real_orders():
    """2026-09-25: hod_break moved from reported-only to PAUSABLE_BOOKS (real resting orders,
    docs/hod_live_resting_orders_spec_20260925.md) — deep negative trips the band-p5 rule same as any
    other pausable book, scaled to hod_break's own risk_usd via SESSION_MULT['hod_break']."""
    stats = _stats(book='hod_break', trailing_40_mean_r=-5.0, trailing_40_n=100,
                   trailing_20_session_usd=-1_000_000.0, worst_session_usd=-1_000_000.0)
    check = gr.evaluate_pause(stats, stage_risk_usd=100.0, band_p5=-0.1)
    assert check.should_pause and check.rule == gr.RULE_BAND_P5


def test_hod_break_with_zero_live_fills_is_reported_but_not_paused(guardrail_trades_db):
    """Before hod_break's first live fill: live_record still reports it like any other book (n_fills=0, not an
    error), and at its real Monday config (risk_usd=50, docs/hod_live_resting_orders_spec_20260925.md item 4)
    an empty trailing window trips nothing — "pause-capable from its first fill" without a separate empty-book
    gate. (stage_risk_usd itself collapsing to $0 on a config-read failure is a separate, deliberately
    maximally-conservative fail-safe — see scripts/guardrail.py stage_risk_usd's own docstring — not this case.)"""
    stats = gr.live_record('hod_break', stage_risk_usd=50.0, db_path=guardrail_trades_db)
    assert stats.n_fills == 0 and stats.book == 'hod_break'
    check = gr.evaluate_pause(stats, stage_risk_usd=50.0, band_p5=None)
    assert not check.should_pause


# ------------------------------------------------------------- state file

def test_state_file_roundtrip(tmp_path):
    path = tmp_path / 'guardrail_state.json'
    stats = _stats(trailing_20_session_usd=-9999.0)
    check = gr.evaluate_pause(stats, stage_risk_usd=100.0, band_p5=None)
    assert check.should_pause

    gr.pause_book(check, stage_risk_usd=100.0, path=path, notify=False)
    assert path.exists()
    reloaded = gr.load_state(path)
    assert reloaded['orb']['paused_by_guardrail'] is True
    assert reloaded['orb']['rule'] == gr.RULE_TRAILING_20_SESSION
    assert 'at_utc' in reloaded['orb']
    assert gr.is_paused('orb', path=path) is True
    assert gr.is_paused('bull_flag', path=path) is False  # untouched book


def test_pause_book_is_idempotent_on_at_utc(tmp_path):
    path = tmp_path / 'guardrail_state.json'
    stats = _stats(trailing_20_session_usd=-9999.0)
    check = gr.evaluate_pause(stats, stage_risk_usd=100.0, band_p5=None)
    gr.pause_book(check, stage_risk_usd=100.0, path=path, notify=False)
    first_at = gr.load_state(path)['orb']['at_utc']
    gr.pause_book(check, stage_risk_usd=100.0, path=path, notify=False)
    assert gr.load_state(path)['orb']['at_utc'] == first_at
    assert len(gr.load_state(path)['orb']['history']) == 2


def test_clear_pause_is_logged_and_requires_reason(tmp_path):
    path = tmp_path / 'guardrail_state.json'
    stats = _stats(worst_session_usd=-600.0)
    check = gr.evaluate_pause(stats, stage_risk_usd=100.0, band_p5=None)
    gr.pause_book(check, stage_risk_usd=100.0, path=path, notify=False)

    with pytest.raises(ValueError):
        gr.clear_pause('orb', '', path=path)

    entry = gr.clear_pause('orb', 'latency defect fixed and rehearsed', by='owner', path=path)
    assert entry['paused_by_guardrail'] is False
    assert entry['cleared_by'] == 'owner'
    assert entry['cleared_reason'] == 'latency defect fixed and rehearsed'
    history = entry['history']
    assert [h['action'] for h in history] == ['pause', 'clear']
    assert history[-1]['reason'] == 'latency defect fixed and rehearsed'
    assert gr.is_paused('orb', path=path) is False


def test_pause_book_sends_telegram_once(tmp_path, monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        class R:
            returncode = 0
            stdout = ''
            stderr = ''
        return R()

    monkeypatch.setattr(gr.subprocess, 'run', fake_run)
    path = tmp_path / 'guardrail_state.json'
    stats = _stats(worst_session_usd=-600.0)
    check = gr.evaluate_pause(stats, stage_risk_usd=100.0, band_p5=None)

    gr.pause_book(check, stage_risk_usd=100.0, path=path, notify=True)
    assert len(calls) == 1
    assert '[GUARDRAIL] orb PAUSED' in calls[0][-1]

    # Re-pausing an already-paused book must not re-notify.
    gr.pause_book(check, stage_risk_usd=100.0, path=path, notify=True)
    assert len(calls) == 1


def test_load_state_missing_file_is_all_unpaused(tmp_path):
    path = tmp_path / 'does_not_exist.json'
    assert gr.load_state(path) == {}
    assert gr.is_paused('orb', path=path) is False


def test_load_state_corrupt_file_logs_error_and_fails_open(tmp_path, caplog):
    path = tmp_path / 'corrupt.json'
    path.write_text('{not json')
    with caplog.at_level('ERROR'):
        state = gr.load_state(path)
    assert state == {}
    assert any('unreadable' in r.message for r in caplog.records)
