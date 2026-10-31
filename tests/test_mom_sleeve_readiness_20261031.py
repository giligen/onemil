"""MOM sleeve readiness for Monday 2026-11-02 (docs/mom_sleeve_monday_readiness_review_20261031.md).

Unit: the catch-up rotation rule (`rotation_due`). Integration (scripts/momentum_sleeve.run on real tmp state files):
Tuesday catch-up after a crashed Monday, Thursday no-op, Monday 11/2 unchanged, the EST pre-open cron line is a clean
no-op (no 'crashed' ERROR), LIVE same-day re-run after a crash neither double-books a fill nor refuses its own
already-placed buys on buying power, the LIVE rotation marker is persisted before the ledger code, and a LIVE
completeness refusal is labelled LIVE.
"""
import json
import os
import sys
from datetime import date, datetime
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'tests'))

from notifications.telegram_notifier import TelegramNotifier  # noqa: E402
from trading import momentum_live as ml  # noqa: E402
import test_momentum_sleeve as tps  # noqa: E402
import test_momentum_sleeve_live as tls  # noqa: E402
from test_momentum_sleeve_live import world  # noqa: E402,F401  (pytest fixture)

UTC = ZoneInfo('UTC')


def week(*days):
    """Sessions of one week as dates in 2026 (month, day) pairs."""
    return [date(2026, m, d) for m, d in days]


WK44 = week((10, 26), (10, 27), (10, 28), (10, 29), (10, 30))
WK45 = week((11, 2), (11, 3), (11, 4), (11, 5), (11, 6))


# ------------------------------------------------------------------ unit: rotation_due
def test_monday_nov2_is_due_as_before():
    due, reason = tps.runner.rotation_due(date(2026, 11, 2), WK45, '2026-10-19')
    assert due and reason == 'first session of week 45'
    assert tps.runner.rotation_due(date(2026, 11, 2), WK45, '2026-10-26')[0]
    assert tps.runner.rotation_due(date(2026, 11, 2), WK45, None)[0]


def test_tuesday_after_monday_crash_is_a_catch_up():
    due, reason = tps.runner.rotation_due(date(2026, 10, 27), WK44, '2026-10-19')
    assert due and reason == 'catch-up (last 2026-10-19, week 44)'
    assert tps.runner.rotation_due(date(2026, 10, 28), WK44, '2026-10-19')[0]          # Wednesday too


def test_thursday_and_friday_after_crash_wait_for_monday():
    for d in (29, 30):
        due, reason = tps.runner.rotation_due(date(2026, 10, d), WK44, '2026-10-19')
        assert not due and 'waits for Monday' in reason


def test_rotation_done_this_week_is_not_due():
    for d in (26, 27, 28):
        due, reason = tps.runner.rotation_due(date(2026, 10, d), WK44, '2026-10-26')
        assert not due and 'already done this week' in reason


def test_monday_holiday_tuesday_first_session_and_non_session():
    tue_week = week((10, 13), (10, 14), (10, 15), (10, 16))
    assert tps.runner.rotation_due(date(2026, 10, 13), tue_week, '2026-10-05') == (True, 'first session of week 42')
    assert tps.runner.rotation_due(date(2026, 10, 12), tue_week, '2026-10-05')[0] is False


def test_unreadable_last_rebalance_warns_and_counts_as_none(caplog):
    with caplog.at_level('WARNING'):
        assert tps.runner.rotation_due(date(2026, 10, 27), WK44, 'garbage')[0]
    assert 'unreadable' in caplog.text


def test_new_rule_equals_old_rule_on_every_first_session_of_2026():
    """Parity: with last week's rotation done, the new rule fires exactly where the old one did."""
    days = [d.date() for d in pd.bdate_range('2026-01-05', '2026-12-31')]
    by_week = {}
    for d in days:
        by_week.setdefault(d.isocalendar()[:2], []).append(d)
    last = None
    for sessions in by_week.values():
        for d in sessions:
            old = tps.runner.is_rebalance_session(d, sessions)
            new = tps.runner.rotation_due(d, sessions, str(last) if last else None)[0]
            assert new == old, d
            if new:
                last = d                                 # the rotation completed that session


# ------------------------------------------------------------------ integration: paper run() calendar paths
def _paper(monkeypatch, tmp_path, last):
    c, args, panel, dates, _ = tps._run_world(monkeypatch, tmp_path)
    spy, _ = tps.make_panel({'SPY': (400, 0.0005, 1e8, 0.005)})          # the weekly row needs an SPY close
    pd.concat([panel, spy], ignore_index=True).to_parquet(tps.runner.cache_file(dates[-1].date()), index=False)
    (tmp_path / 'state.json').write_text(json.dumps({'cash': 20000.0, 'positions': {}, 'peak_equity': 20000.0,
                                                     'last_rebalance': last}))
    args.force, args.submit = False, False
    return c, args


def test_paper_tuesday_catch_up_plans_and_logs(monkeypatch, tmp_path, caplog, capsys):
    c, args = _paper(monkeypatch, tmp_path, '2026-10-19')
    with caplog.at_level('INFO'):
        assert tps.runner.run(args, c, None, datetime(2026, 10, 27, 15, 0, tzinfo=UTC)) == 0
    assert 'rotation due: catch-up (last 2026-10-19, week 44)' in caplog.text
    assert 'ORDERS (20)' in capsys.readouterr().out


def test_paper_thursday_after_crash_is_noop(monkeypatch, tmp_path, capsys):
    c, args = _paper(monkeypatch, tmp_path, '2026-10-19')
    monkeypatch.setattr(tps.runner, 'fetch_assets', lambda client: pytest.fail('no work on a Thursday'))
    assert tps.runner.run(args, c, None, datetime(2026, 10, 29, 15, 0, tzinfo=UTC)) == 0
    assert 'waits for Monday' in capsys.readouterr().out


def test_paper_monday_nov2_est_lines(monkeypatch, tmp_path, caplog, capsys):
    """11/2 is EST: the 13:45 UTC line (08:45 ET) is a clean INFO no-op, the 14:45 UTC line (09:45 ET) trades."""
    c, args = _paper(monkeypatch, tmp_path, '2026-10-19')
    args.submit = True
    monkeypatch.setattr(tps.runner, 'execute', lambda *a, **k: [])
    monkeypatch.setattr(tps.runner, 'sync_state_from_broker', lambda client, state, eq: None)
    monkeypatch.setattr(tps.runner, 'official_opens', lambda *a, **k: {})
    with caplog.at_level('INFO'):
        assert tps.runner.run(args, c, None, datetime(2026, 11, 2, 13, 45, tzinfo=UTC)) == 0
    assert 'submit window opens 09:31 ET (now 08:45 ET)' in capsys.readouterr().out
    assert not [r for r in caplog.records if r.levelname == 'ERROR']
    c.trading_client.submit_order.assert_not_called()
    assert tps.runner.run(args, c, None, datetime(2026, 11, 2, 14, 45, tzinfo=UTC)) == 0
    assert json.loads((tmp_path / 'state.json').read_text())['last_rebalance'] == '2026-11-02'
    assert tps.runner.run(args, c, None, datetime(2026, 11, 2, 14, 50, tzinfo=UTC)) == 0       # retry = no-op
    assert 'already done this week' in capsys.readouterr().out


def test_paper_after_close_submit_still_refused(monkeypatch, tmp_path):
    c, args = _paper(monkeypatch, tmp_path, '2026-10-19')
    args.submit = True
    with pytest.raises(RuntimeError, match='outside 09:31-15:30'):
        tps.runner.run(args, c, None, datetime(2026, 11, 2, 21, 0, tzinfo=UTC))   # 16:00 ET


# ------------------------------------------------------------------ integration: LIVE same-day re-run safety
def test_live_rerun_after_crash_in_poll_books_existing_buys_without_bp_refusal(world, monkeypatch):
    args, asof, tmp = world
    c, b = tls.live_client()
    real_poll = tls.runner.poll_fills
    calls = {'n': 0}

    def poll_crash_on_buys(client, coids, **k):
        calls['n'] += 1
        if any('-b' in x for x in coids) and calls['n'] == 1:
            raise RuntimeError('network died while polling the buys')
        return real_poll(client, coids, **k)
    monkeypatch.setattr(tls.runner, 'poll_fills', poll_crash_on_buys)
    with pytest.raises(RuntimeError):
        tls.runner.run(args(submit=True, force=False), c, None, tls.NOW)
    assert len(b.orders) == 20 and json.load(open(ml.LIVE_STATE_PATH))['positions'] == {}
    b.buying_power = 0.0                                 # the 20 buys consumed it
    assert tls.runner.run(args(submit=True, force=False), c, None, tls.NOW) == 0
    st = json.load(open(ml.LIVE_STATE_PATH))
    assert len(b.orders) == 20                           # nothing re-submitted
    assert len(st['positions']) == 20 and st['cash'] == pytest.approx(0.0, abs=1.0)
    assert st['last_rebalance'] == '2026-10-05' and len(st['booked_coids']) == 20


def test_live_rerun_never_books_the_same_fill_twice(world, caplog):
    args, asof, tmp = world
    c, b = tls.live_client()
    assert tls.runner.run(args(submit=True, force=False), c, None, tls.NOW) == 0
    st = json.load(open(ml.LIVE_STATE_PATH))
    before = dict(st['positions'])
    st.pop('last_rebalance')                             # a crash between the booking and the marker
    json.dump(st, open(ml.LIVE_STATE_PATH, 'w'))
    low = sorted(before)[:10]                            # these marks fell -> the re-plan wants small top-up buys
    b.positions = [tls.pos(s, q, px=45.0 if s in low else 50.0) for s, q in before.items()]
    with caplog.at_level('WARNING'):
        assert tls.runner.run(args(submit=True, force=False), c, None, tls.NOW) == 0
    after = json.load(open(ml.LIVE_STATE_PATH))['positions']
    for s in low:
        assert after[s] == pytest.approx(before[s]), s   # the existing buy was NOT booked a second time
    assert 'already booked in the sleeve state' in caplog.text


def test_live_marker_persisted_before_ledger_code(world, monkeypatch, capsys):
    args, asof, tmp = world
    c, b = tls.live_client()
    monkeypatch.setattr(tls.runner, 'official_opens', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('boom')))
    with pytest.raises(RuntimeError):
        tls.runner.run(args(submit=True, force=False), c, None, tls.NOW)
    assert json.load(open(ml.LIVE_STATE_PATH))['last_rebalance'] == '2026-10-05'
    n = len(b.orders)
    assert tls.runner.run(args(submit=True, force=False), c, None, tls.NOW) == 0
    assert len(b.orders) == n and 'already done this week' in capsys.readouterr().out


def test_live_completeness_refusal_is_labelled_live(world, caplog):
    args, asof, tmp = world
    os.remove(tls.runner.completeness_path(asof))
    c, b = tls.live_client()
    notifier = MagicMock(spec=TelegramNotifier)
    with caplog.at_level('ERROR'):
        assert tls.runner.run(args(submit=True), c, notifier, tls.NOW) == 2
    assert 'MOM LIVE REFUSED: completeness' in caplog.text and not b.orders
    assert notifier.send_message_sync.call_args[0][0].startswith('[MOM LIVE]')
    assert not os.path.exists(ml.LIVE_STATE_PATH)        # state untouched
