"""Daily green-check + streak logic (2026-07-05 reporting ship).

The streak in logs/green_streak.json IS the ramp advancement gate
(2026-07-06 policy), so its semantics are pinned here: greens increment,
a red resets, weekend gaps don't break it, same-day reruns are idempotent.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
import report_common as rc
from daily_green_check import build_message


class TestStreakUpdate:
    def test_greens_increment(self, tmp_path):
        p = tmp_path / 's.json'
        assert rc.streak_update('2026-07-06', True, [], path=p) == 1
        assert rc.streak_update('2026-07-07', True, [], path=p) == 2
        assert rc.streak_update('2026-07-08', True, [], path=p) == 3

    def test_red_resets(self, tmp_path):
        p = tmp_path / 's.json'
        rc.streak_update('2026-07-06', True, [], path=p)
        rc.streak_update('2026-07-07', True, [], path=p)
        assert rc.streak_update('2026-07-08', False, ['boom'], path=p) == 0
        assert rc.streak_update('2026-07-09', True, [], path=p) == 1

    def test_weekend_gap_does_not_break(self, tmp_path):
        p = tmp_path / 's.json'
        rc.streak_update('2026-07-10', True, [], path=p)   # Friday
        assert rc.streak_update('2026-07-13', True, [], path=p) == 2  # Monday

    def test_same_day_rerun_idempotent(self, tmp_path):
        p = tmp_path / 's.json'
        rc.streak_update('2026-07-06', False, ['x'], path=p)
        # rerun after fixing — overwrites the day, doesn't double-count
        assert rc.streak_update('2026-07-06', True, [], path=p) == 1
        days = json.loads(p.read_text())['days']
        assert len(days) == 1

    def test_reasons_persisted_for_weekly_report(self, tmp_path):
        p = tmp_path / 's.json'
        rc.streak_update('2026-07-06', False, ['unattributed exits: [X]'], path=p)
        st = rc.read_streak(path=p)
        assert st['days'][0]['reasons'] == ['unattributed exits: [X]']


class TestGreenVerdict:
    def _patch(self, monkeypatch, rows=(), bt=(), journal=(), bt_max='2099-01-01'):
        monkeypatch.setattr(rc, 'load_live_rows',
                            lambda day, strategy=None:
                            [r for r in rows
                             if strategy is None or r.get('strategy') == strategy])
        monkeypatch.setattr(rc, 'load_bt_selected', lambda day: list(bt))
        # hermetic: never let the real CSV's staleness leak into verdicts
        monkeypatch.setattr(rc, 'bt_data_max_date', lambda: bt_max)
        monkeypatch.setattr(rc, 'journal_grep', lambda pat, day:
                            [ln for ln in journal if pat in ln])
        monkeypatch.setattr(rc, 'read_selection_audit', lambda day: [])

    def test_clean_day_is_green(self, monkeypatch):
        rows = [{'symbol': 'AAA', 'strategy': 'orb', 'entry_price': 10.0,
                 'exit_price': 10.5, 'shares': 100, 'exit_reason': 'lock_stop',
                 'order_status': 'filled'}]
        self._patch(monkeypatch, rows=rows, bt=[{'symbol': 'AAA'}])
        v = rc.green_verdict('2026-07-06')
        assert v['green'] is True

    def test_unknown_exit_is_red(self, monkeypatch):
        rows = [{'symbol': 'AAA', 'strategy': 'orb', 'entry_price': 10.0,
                 'exit_price': 9.0, 'shares': 100, 'exit_reason': 'unknown_exit',
                 'order_status': 'filled'}]
        self._patch(monkeypatch, rows=rows)
        v = rc.green_verdict('2026-07-06')
        assert v['green'] is False
        assert any('unattributed' in r for r in v['reasons'])

    def test_bt_pick_never_ordered_is_red(self, monkeypatch):
        """The CRCD class — BT selected it, live has no row at all."""
        self._patch(monkeypatch, rows=[], bt=[{'symbol': 'CRCD'}])
        v = rc.green_verdict('2026-06-30')
        assert v['green'] is False
        assert any('CRCD' in r for r in v['reasons'])

    def test_bt_pick_missing_but_spread_explained_is_green(self, monkeypatch):
        self._patch(monkeypatch, rows=[], bt=[{'symbol': 'WIDE'}],
                    journal=['... WIDE skipped — spread 210bps > 150bps ...'])
        v = rc.green_verdict('2026-07-06')
        assert v['green'] is True

    def test_pending_verification_is_red(self, monkeypatch):
        rows = [{'symbol': 'BBB', 'strategy': 'orb', 'entry_price': 5.0,
                 'exit_price': None, 'shares': 100, 'exit_reason': None,
                 'order_status': 'exit_pending_verification'}]
        self._patch(monkeypatch, rows=rows)
        v = rc.green_verdict('2026-07-06')
        assert v['green'] is False

    def test_exhaust_composite_reason_is_green(self, monkeypatch):
        """exhaust+trail_stop must classify as attributed (taxonomy fix)."""
        rows = [{'symbol': 'CCC', 'strategy': 'bull_flag', 'entry_price': 8.0,
                 'exit_price': 9.0, 'shares': 50,
                 'exit_reason': 'exhaust+trail_stop', 'order_status': 'filled'}]
        self._patch(monkeypatch, rows=rows)
        assert rc.green_verdict('2026-07-06')['green'] is True

    def test_stale_bt_data_skips_parity_not_greenwashes(self, monkeypatch):
        """BT CSV behind the checked day -> parity SKIPPED and visible,
        day still evaluable on the other gates."""
        rows = [{'symbol': 'AAA', 'strategy': 'orb', 'entry_price': 10.0,
                 'exit_price': 10.5, 'shares': 100, 'exit_reason': 'lock_stop',
                 'order_status': 'filled'}]
        self._patch(monkeypatch, rows=rows, bt=[{'symbol': 'ZZZ'}],
                    bt_max='2026-07-01')
        v = rc.green_verdict('2026-07-02')
        assert v['green'] is True
        assert v['bt_stale'] is True
        assert 'SKIPPED' in v['checks']['bt_parity']


class TestMessage:
    def test_green_message_is_one_line(self):
        v = {'day': '2026-07-06', 'green': True, 'reasons': [], 'checks': {},
             'n_live_rows': 4, 'n_bt_selected': 3}
        msg = build_message(v, streak=4, pnl={'orb': 123.0})
        assert msg.count('\n') == 0
        assert 'GREEN 4/10' in msg

    def test_red_message_carries_reasons(self):
        v = {'day': '2026-07-06', 'green': False,
             'reasons': ['BT picks never ordered live: [CRCD]'],
             'checks': {'bt_parity': 'x'}, 'n_live_rows': 2, 'n_bt_selected': 3}
        msg = build_message(v, streak=0, pnl={})
        assert 'RED DAY' in msg and 'CRCD' in msg and 'reset' in msg
