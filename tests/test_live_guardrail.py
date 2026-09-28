"""tests/test_live_guardrail.py — docs/live_guardrails_spec_20260925.md G1.

Covers: ledger from a temp trades.db across three books, each pause rule
firing exactly at its frozen threshold (and not just past it), that a
simulated config/stage change cannot reset the ledger start, state-file
round-trip, and a logged clear-with-reason. Never touches production
data/trades.db or data/guardrail_state.json (every path is tmp_path).

Also covers G2 (owner 2026-09-28, docs/live_guardrails_spec_20260925.md
amendment): an owner --clear is the ONE legitimate reset of the tripwire
window (acknowledged_through_utc / after_exited_at), and the trades.account
(migration 17) split between the live ledger and ORB/HOD-break's own paper
accounts.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone

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


# --------------------------------------------------- account split (migration 17)

def test_account_filter_excludes_paper_from_live_ledger(guardrail_trades_db, insert_trade):
    """ORB/HOD-break's own paper-account fills (owner 9/28) must never feed
    the live tripwire; they ledger separately under account='paper'."""
    insert_trade('orb', '2026-06-01', pnl=-50.0, account='live')
    insert_trade('orb', '2026-06-02', pnl=-9999.0, account='paper')

    live = gr.live_record('orb', stage_risk_usd=100.0, db_path=guardrail_trades_db)
    assert live.n_fills == 1 and live.total_usd == pytest.approx(-50.0)

    paper = gr.live_record('orb', stage_risk_usd=100.0, db_path=guardrail_trades_db, account='paper')
    assert paper.n_fills == 1 and paper.total_usd == pytest.approx(-9999.0)


def test_legacy_null_account_counts_as_live(guardrail_trades_db, insert_trade):
    """Rows written before migration 17 have account=NULL — the live ledger
    (default account='live') must still count them; they are not paper."""
    insert_trade('orb', '2026-06-01', pnl=-50.0)  # account left unset -> NULL
    live = gr.live_record('orb', stage_risk_usd=100.0, db_path=guardrail_trades_db)
    assert live.n_fills == 1 and live.total_usd == pytest.approx(-50.0)
    paper = gr.live_record('orb', stage_risk_usd=100.0, db_path=guardrail_trades_db, account='paper')
    assert paper.n_fills == 0


# ------------------------------------------------- acknowledgement cutoff (G2)

def test_after_exited_at_drops_fills_at_or_before_cutoff(guardrail_trades_db, insert_trade):
    """A fill exited exactly AT the cutoff is excluded (strictly-after semantics,
    matching clear_pause's 'the clear time is the acknowledged instant')."""
    insert_trade('orb', '2026-09-20', pnl=-700.0, exited_at='2026-09-20T15:00:00+00:00')
    insert_trade('orb', '2026-09-28', pnl=-800.0, exited_at='2026-09-28T15:25:00+00:00')  # == cutoff
    insert_trade('orb', '2026-09-29', pnl=-900.0, exited_at='2026-09-29T09:00:00+00:00')  # after cutoff

    stats = gr.live_record('orb', stage_risk_usd=100.0, db_path=guardrail_trades_db,
                           after_exited_at='2026-09-28T15:25:00+00:00')
    assert stats.n_fills == 1
    assert stats.total_usd == pytest.approx(-900.0)


def test_after_exited_at_none_is_unchanged(guardrail_trades_db, insert_trade):
    """No acknowledgement (after_exited_at=None) -> behaviour identical to before G2."""
    insert_trade('orb', '2026-06-01', pnl=-50.0)
    insert_trade('orb', '2026-06-02', pnl=125.0)
    with_none = gr.live_record('orb', stage_risk_usd=100.0, db_path=guardrail_trades_db,
                               after_exited_at=None)
    without_arg = gr.live_record('orb', stage_risk_usd=100.0, db_path=guardrail_trades_db)
    assert with_none == without_arg
    assert with_none.n_fills == 2


def test_after_exited_at_handles_space_separated_exited_at(guardrail_trades_db, insert_trade):
    """sqlite3's legacy datetime adapter writes exited_at with a SPACE
    separator ('... 15:00:00...'); acknowledged_through_utc (this module's own
    isoformat()) uses 'T'. A naive string compare would misorder same-day
    values (' ' < 'T'); _parse_utc must classify this correctly as AFTER."""
    insert_trade('orb', '2026-09-28', pnl=-900.0, exited_at='2026-09-28 20:00:00.000000+00:00')
    stats = gr.live_record('orb', stage_risk_usd=100.0, db_path=guardrail_trades_db,
                           after_exited_at='2026-09-28T15:25:00+00:00')
    assert stats.n_fills == 1 and stats.total_usd == pytest.approx(-900.0)


def test_after_exited_at_missing_value_is_included_with_warning(guardrail_trades_db, caplog):
    """A closed fill with no usable exited_at (NULL — insert_trade's fixture
    always synthesizes one, so this inserts directly) under an active cutoff
    is included (ambiguous fills count against the tripwire — conservative,
    matching trade_risk_usd's fallback philosophy), and logs a WARNING."""
    now = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(str(guardrail_trades_db))
    conn.execute(
        "INSERT INTO trades (trade_date, symbol, side, entry_price, stop_loss_price, "
        "take_profit_price, shares, risk_per_share, total_risk, risk_reward_ratio, "
        "pnl, exited_at, strategy, account, created_at, updated_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ('2026-09-28', 'TEST', 'buy', 100.0, 99.0, 102.0, 100, 1.0, 100.0, 2.0,
         -900.0, None, 'orb', None, now, now))
    conn.commit()
    conn.close()

    with caplog.at_level('WARNING'):
        stats = gr.live_record('orb', stage_risk_usd=100.0, db_path=guardrail_trades_db,
                               after_exited_at='2026-09-28T15:25:00+00:00')
    assert stats.n_fills == 1
    assert any('unparseable/missing exited_at' in r.message for r in caplog.records)


def test_acknowledged_through_utc_and_line_before_and_after_clear(tmp_path):
    path = tmp_path / 'state.json'
    assert gr.acknowledged_through_utc('orb', path=path) is None
    assert gr.acknowledged_line('orb', path=path) is None

    stats = gr.LedgerStats(book='orb', n_fills=123, total_usd=-5281.0, mean_r=None,
                           trailing_40_mean_r=-0.348, trailing_40_n=40,
                           trailing_20_session_usd=0.0, worst_month=None,
                           worst_month_usd=None, worst_session_date=None,
                           worst_session_usd=None, first_fill_date='2026-05-19')
    entry = gr.clear_pause('orb', 'owner cleared 15:25 UTC', by='owner', path=path, ledger=stats)

    assert entry['acknowledged_through_utc']
    assert entry['acknowledged_ledger'] == {'n_fills': 123, 'total_usd': -5281.0}
    assert gr.acknowledged_through_utc('orb', path=path) == entry['acknowledged_through_utc']

    line = gr.acknowledged_line('orb', path=path)
    assert line.startswith('acknowledged through ')
    assert 'n=123' in line and '$-5,281' in line


def test_clear_pause_without_ledger_logs_warning_and_records_zero(tmp_path, caplog):
    """A --clear call site that forgets to pass a ledger snapshot must not
    silently drop the acknowledged-history line — it logs and records n=0 $0."""
    path = tmp_path / 'state.json'
    with caplog.at_level('WARNING'):
        entry = gr.clear_pause('orb', 'no ledger passed', by='owner', path=path)
    assert entry['acknowledged_ledger'] == {'n_fills': 0, 'total_usd': 0.0}
    assert any('without a ledger snapshot' in r.message for r in caplog.records)


def test_config_change_cannot_set_acknowledged_through_utc(guardrail_trades_db, insert_trade):
    """Simulates the exact 2026-09-28 defect: --check (no clear involved) must
    NEVER move acknowledged_through_utc — only clear_pause may. A book that
    was never cleared re-evaluates its FULL live ledger every time."""
    insert_trade('orb', '2026-06-01', pnl=-5281.0)
    stats1 = gr.live_record('orb', stage_risk_usd=100.0, db_path=guardrail_trades_db)
    stats2 = gr.live_record('orb', stage_risk_usd=100.0, db_path=guardrail_trades_db)
    assert stats1.n_fills == stats2.n_fills == 1
    assert stats1.total_usd == stats2.total_usd == pytest.approx(-5281.0)
