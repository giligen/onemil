"""Ramp FREEZE state (trading/ramp_freeze.py) — Gate-1 of docs/scaling_plan_2026.md.

A freeze is "we do not know what we are running": the stage clock STOPS, no
ADVANCE, size unchanged, and clearing is manual and logged.
"""
from __future__ import annotations

import argparse
import json

import pytest

from trading import ramp_freeze as rf


@pytest.fixture
def path(tmp_path):
    return tmp_path / 'ramp_freeze.json'


class TestLoadSave:
    def test_missing_file_is_unfrozen(self, path):
        st = rf.load_state(path)
        assert set(st) == set(rf.BOOKS)
        assert not st['bf'].frozen and not st['orb'].frozen
        assert st['bf'].frozen_dates == []

    def test_corrupt_file_reads_unfrozen_and_logs(self, path, caplog):
        path.write_text('{not json')
        with caplog.at_level('ERROR'):
            st = rf.load_state(path)
        assert not st['orb'].frozen
        assert 'unreadable' in caplog.text

    def test_round_trip(self, path):
        rf.set_freeze('orb', 'mult drift', day='2026-09-18', by='tester',
                      path=path, notify=False)
        raw = json.loads(path.read_text())
        assert raw['orb']['frozen'] is True
        assert raw['orb']['since'] == '2026-09-18'
        assert raw['orb']['frozen_dates'] == ['2026-09-18']
        assert rf.is_frozen('orb', path) and not rf.is_frozen('bf', path)

    def test_frozen_line_is_the_checker_output(self, path):
        rf.set_freeze('orb', 'pm_mult drift', day='2026-09-18', by='cron',
                      path=path, notify=False)
        line = rf.get('orb', path).line()
        assert line.startswith('FROZEN since 2026-09-18: pm_mult drift')
        assert '--clear-freeze orb' in line and 'set by cron' in line

    def test_unknown_book_rejected(self, path):
        with pytest.raises(ValueError):
            rf.get('ignition', path)


class TestSetFreeze:
    def test_since_is_sticky_reason_updates(self, path):
        rf.set_freeze('bf', 'first breach', day='2026-09-14', path=path,
                      notify=False)
        s = rf.set_freeze('bf', 'second breach', day='2026-09-16', path=path,
                          notify=False)
        assert s.since == '2026-09-14'          # the clock stopped THEN
        assert s.reason == 'second breach'
        assert s.frozen_dates == ['2026-09-14', '2026-09-16']
        assert [h['action'] for h in s.history] == ['freeze', 'freeze']

    def test_telegram_sent_once_on_first_freeze(self, path, monkeypatch):
        sent = []
        monkeypatch.setattr(rf, 'send_freeze_telegram',
                            lambda *a, **k: sent.append(a) or True)
        rf.set_freeze('orb', 'a', day='2026-09-14', path=path, notify=True)
        rf.set_freeze('orb', 'b', day='2026-09-15', path=path, notify=True)
        assert len(sent) == 1 and sent[0][0] == 'orb'

    def test_telegram_failure_never_breaks_the_freeze(self, path, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError('no network')
        monkeypatch.setattr(rf.subprocess, 'run', boom)
        s = rf.set_freeze('orb', 'x', day='2026-09-14', path=path, notify=True)
        assert s.frozen and rf.is_frozen('orb', path)

    def test_telegram_message_carries_the_prefix(self, path, monkeypatch):
        seen = {}

        class R:
            returncode = 0
            stdout = stderr = ''

        def fake_run(cmd, **kw):
            seen['cmd'] = cmd
            return R()
        monkeypatch.setattr(rf.subprocess, 'run', fake_run)
        assert rf.send_freeze_telegram('bf', 'exit_reason mismatch', '2026-09-18')
        assert rf.TELEGRAM_PREFIX in seen['cmd'][-1]
        assert 'BF FROZEN' in seen['cmd'][-1]
        assert seen['cmd'][1].endswith('send_telegram_alert.py')

    def test_telegram_nonzero_rc_is_reported_not_raised(self, path, monkeypatch):
        class R:
            returncode = 1
            stdout = ''
            stderr = 'not configured'
        monkeypatch.setattr(rf.subprocess, 'run', lambda *a, **k: R())
        assert rf.send_freeze_telegram('bf', 'r', '2026-09-18') is False


class TestClearFreeze:
    def test_clear_is_manual_and_logged(self, path):
        rf.set_freeze('bf', 'breach', day='2026-09-14', path=path, notify=False)
        s = rf.clear_freeze('bf', 'cache rebuilt, re-verified', by='owner',
                            path=path, day='2026-09-19')
        assert not s.frozen and s.since is None
        last = s.history[-1]
        assert last == {**last, 'action': 'clear', 'by': 'owner',
                        'reason': 'cache rebuilt, re-verified',
                        'was_frozen_since': '2026-09-14'}
        assert rf.get('bf', path).frozen_dates == ['2026-09-14']  # still excluded

    def test_clear_requires_a_reason(self, path):
        rf.set_freeze('bf', 'breach', day='2026-09-14', path=path, notify=False)
        with pytest.raises(ValueError):
            rf.clear_freeze('bf', '   ', path=path)
        assert rf.is_frozen('bf', path)

    def test_clear_when_not_frozen_warns(self, path, caplog):
        with caplog.at_level('WARNING'):
            rf.clear_freeze('orb', 'nothing to do', path=path)
        assert 'not frozen' in caplog.text

    def test_cli_helper_round_trip(self, path, monkeypatch):
        monkeypatch.setattr(rf, 'FREEZE_PATH', path)
        rf.set_freeze('orb', 'breach', day='2026-09-14', path=path, notify=False)
        ap = argparse.ArgumentParser()
        rf.add_clear_freeze_arg(ap)
        args = ap.parse_args(['--clear-freeze', 'orb', 'explained + fixed'])
        line = rf.handle_clear_freeze(args.clear_freeze, path=path)
        assert 'FREEZE CLEARED on ORB' in line and 'explained + fixed' in line
        assert not rf.is_frozen('orb', path)

    def test_no_clear_freeze_flag_defaults_to_none(self):
        ap = argparse.ArgumentParser()
        rf.add_clear_freeze_arg(ap)
        assert ap.parse_args([]).clear_freeze is None


class TestSessionClock:
    def test_frozen_dates_are_excluded(self, path):
        rf.set_freeze('orb', 'b', day='2026-09-16', path=path, notify=False)
        rf.clear_freeze('orb', 'fixed', path=path, day='2026-09-17')
        sessions = ['2026-09-14', '2026-09-15', '2026-09-16', '2026-09-17']
        assert rf.frozen_sessions('orb', sessions[0], sessions[-1], path) == \
            ['2026-09-16']
        assert rf.unfrozen_sessions('orb', sessions, path) == \
            ['2026-09-14', '2026-09-15', '2026-09-17']

    def test_still_frozen_freezes_every_later_weekday(self, path):
        rf.set_freeze('bf', 'b', day='2026-09-16', path=path, notify=False)
        sessions = ['2026-09-14', '2026-09-15', '2026-09-16', '2026-09-17',
                    '2026-09-18']
        # weekend days are not sessions and never appear
        assert rf.frozen_sessions('bf', '2026-09-14', '2026-09-21', path) == \
            ['2026-09-16', '2026-09-17', '2026-09-18', '2026-09-21']
        assert rf.unfrozen_sessions('bf', sessions, path) == \
            ['2026-09-14', '2026-09-15']

    def test_freeze_before_the_stage_start_does_not_leak_in(self, path):
        rf.set_freeze('bf', 'b', day='2026-08-03', path=path, notify=False)
        rf.clear_freeze('bf', 'fixed', path=path, day='2026-08-04')
        assert rf.frozen_sessions('bf', '2026-09-07', '2026-09-11', path) == []

    def test_empty_session_list(self, path):
        assert rf.unfrozen_sessions('bf', [], path) == []


class TestIdentity:
    def test_current_user_prefers_env(self, monkeypatch):
        monkeypatch.setenv('SUDO_USER', 'owner')
        assert rf.current_user() == 'owner'

    def test_current_user_never_raises(self, monkeypatch):
        for k in ('SUDO_USER', 'USER', 'LOGNAME'):
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setattr(rf.getpass, 'getuser',
                            lambda: (_ for _ in ()).throw(OSError('nope')))
        assert rf.current_user() == 'unknown'
