"""Daily green-check + streak logic (2026-07-05 reporting ship).

The streak in logs/green_streak.json IS the ramp advancement gate
(2026-07-06 policy), so its semantics are pinned here: greens increment,
a red resets, weekend gaps don't break it, same-day reruns are idempotent.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
import report_common as rc
from daily_green_check import build_message

CLEAN_PARITY = {'date': 'x', 'bt_available': True, 'live_final': ['AAA'], 'bt_top8': ['AAA'], 'differences': [],
                'n_unexplained': 0, 'fills_outside_final': [], 'summary': 'ORB pick parity x: matched 1/8 | unexplained 0'}


class TestStreakUpdate:
    @pytest.fixture(autouse=True)
    def _day_one_before_fixtures(self, monkeypatch):
        """These fixtures use July dates: lower the day-one floor (the day-one rule has its own class)."""
        import datetime as dt
        monkeypatch.setattr(rc, 'ORB_DAY_ONE', dt.date(2026, 1, 1))

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

    def test_green_to_red_downgrade_blocked(self, tmp_path):
        """Journal evidence decays — a re-run must not false-red a day
        already adjudicated green (2026-07-10 smoke-test incident)."""
        p = tmp_path / 's.json'
        assert rc.streak_update('2026-07-09', True, [], path=p) == 1
        # later re-run: journald rotated the explaining line away
        s = rc.streak_update('2026-07-09', False,
                             ["BT picks never ordered live: ['IQMX']"], path=p)
        assert s == 1                       # streak preserved
        rec = rc.read_streak(p)
        assert rec['days'][-1]['green'] is True   # record untouched

    def test_green_to_red_downgrade_forced(self, tmp_path):
        p = tmp_path / 's.json'
        rc.streak_update('2026-07-09', True, [], path=p)
        s = rc.streak_update('2026-07-09', False, ['real bug found'],
                             path=p, allow_downgrade=True)
        assert s == 0

    def test_red_to_green_readjudication_allowed(self, tmp_path):
        """The 7/9 false-alarm direction: finding evidence flips red→green
        without force."""
        p = tmp_path / 's.json'
        rc.streak_update('2026-07-09', False, ['x'], path=p)
        assert rc.streak_update('2026-07-09', True, [], path=p) == 1

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

    def test_missing_bt_is_red_with_the_reason_never_skipped(self, monkeypatch):
        """BT behind the checked day -> RED naming the build timer; no bt_stale branch, no SKIPPED string."""
        rows = [{'symbol': 'AAA', 'strategy': 'orb', 'entry_price': 10.0,
                 'exit_price': 10.5, 'shares': 100, 'exit_reason': 'lock_stop',
                 'order_status': 'filled'}]
        self._patch(monkeypatch, rows=rows, bt=[{'symbol': 'ZZZ'}],
                    bt_max='2026-07-01')
        v = rc.green_verdict('2026-07-02')
        assert v['green'] is False and v['bt_missing'] is True and 'bt_stale' not in v
        assert any(r.startswith('nightly BT missing for 2026-07-02') and 'onemil-orb-backtest.timer' in r
                   for r in v['reasons'])
        assert 'SKIPPED' not in str(v)


class TestSizingAttribution:
    """EoD validation of the 2026-07-13 mult ships (quintile correction +
    news-gated PM mult): recorded pm_mult must recompute from recorded
    inputs via the SHARED helper; news drift is soft-flagged."""

    def _row(self, sym, pm_mult, has_news, pm_dv, pnl=100.0, quintile='Q2'):
        import json
        return {'symbol': sym, 'strategy': 'orb', 'pnl': pnl,
                'fill_price': 10.0,
                'pattern_data': json.dumps({
                    'quintile': quintile, 'adaptive_mult': 1.5,
                    'pm_mult': pm_mult, 'has_news': has_news,
                    'n_articles': 2 if has_news else 0,
                    'pm_dollar_vol': pm_dv})}

    def _patch(self, monkeypatch, rows, eod_newsy=None):
        monkeypatch.setattr(rc, 'load_live_rows',
                            lambda day, strategy=None: list(rows))
        monkeypatch.setattr(rc, 'eod_news_recheck',
                            lambda syms, day: eod_newsy or {})
        # hermetic cum query: empty DB slice
        import sqlite3
        class _FakeConn:
            row_factory = None
            def execute(self, *a): return []
            def close(self): pass
        monkeypatch.setattr(sqlite3, 'connect', lambda *a, **k: _FakeConn())

    def test_correct_mults_no_mismatch(self, monkeypatch):
        rows = [self._row('AAA', 2.0, True, 9e6),     # news+pm -> 2.0 ✓
                self._row('BBB', 1.0, False, 9e6),    # pm only -> 1.0 ✓
                self._row('CCC', 1.0, True, 1e6)]     # news only -> 1.0 ✓
        self._patch(monkeypatch, rows)
        attr = rc.sizing_attribution('2026-07-13')
        assert attr['mult_mismatches'] == []

    def test_wrong_mult_flagged(self, monkeypatch):
        rows = [self._row('BAD', 1.5, False, 9e6)]    # should be 1.0
        self._patch(monkeypatch, rows)
        attr = rc.sizing_attribution('2026-07-13')
        assert len(attr['mult_mismatches']) == 1
        assert 'BAD' in attr['mult_mismatches'][0]

    def test_fail_open_none_news_is_valid(self, monkeypatch):
        """has_news=None + pm_mult=1.0 is CORRECT fail-open, not a bug."""
        rows = [self._row('FOP', 1.0, None, 9e6)]
        self._patch(monkeypatch, rows)
        attr = rc.sizing_attribution('2026-07-13')
        assert attr['mult_mismatches'] == []

    def test_news_drift_soft_flagged(self, monkeypatch):
        """Live said no news, EoD re-query finds pre-9:31 articles."""
        rows = [self._row('DRF', 1.0, False, 9e6)]
        self._patch(monkeypatch, rows, eod_newsy={'DRF': True})
        attr = rc.sizing_attribution('2026-07-13')
        assert len(attr['news_drift']) == 1
        assert 'DRF' in attr['news_drift'][0]
        assert attr['mult_mismatches'] == []   # drift is soft, not hard

    def test_pre_ship_rows_skipped(self, monkeypatch):
        """Rows without pm_mult in pattern_data (pre-ship) don't false-flag."""
        import json
        rows = [{'symbol': 'OLD', 'strategy': 'orb', 'pnl': 5.0,
                 'fill_price': 9.0,
                 'pattern_data': json.dumps({'quintile': 'Q4'})}]
        self._patch(monkeypatch, rows)
        attr = rc.sizing_attribution('2026-07-01')
        assert attr['mult_mismatches'] == []

    def test_lag_audit_clean_and_flagged(self, monkeypatch, tmp_path):
        """Pool-level Benzinga indexing-lag audit: snapshot no-news symbols
        that the fully-indexed EoD re-query finds newsy = lag events;
        sizing-material when above the PM$ cut."""
        import json
        snap = {'day': '2026-07-13', 'flags': {'AAA': 0, 'BBB': 2, 'CCC': None},
                'pm_dollar_vols': {'AAA': 9e6, 'BBB': 9e6, 'CCC': 1e6}}
        logs = tmp_path / 'logs'
        logs.mkdir()
        (logs / 'orb_news_flags_2026-07-13.json').write_text(json.dumps(snap))
        monkeypatch.setattr(rc, 'ROOT', tmp_path)
        # fully-indexed EoD view: AAA's premarket article is now visible
        monkeypatch.setattr(rc, 'eod_news_recheck',
                            lambda syms, day: {s: s == 'AAA' for s in syms})
        audit = rc.news_lag_audit('2026-07-13')
        assert audit['available'] and audit['n_checked'] == 3
        assert audit['lag_symbols'] == ['AAA']
        assert audit['material'] == ['AAA']     # $9M > cut -> mis-sized
        line = rc.news_lag_line(audit)
        assert 'NEWS LAG' in line and 'AAA' in line and 'MATERIAL' in line
        # clean day
        monkeypatch.setattr(rc, 'eod_news_recheck',
                            lambda syms, day: {s: False for s in syms})
        clean = rc.news_lag_audit('2026-07-13')
        assert clean['lag_symbols'] == []
        assert 'clean' in rc.news_lag_line(clean)

    def test_lag_audit_no_snapshot_silent(self, monkeypatch, tmp_path):
        monkeypatch.setattr(rc, 'ROOT', tmp_path)
        audit = rc.news_lag_audit('2026-07-13')
        assert audit['available'] is False
        assert rc.news_lag_line(audit) == ''

    def test_mismatch_turns_day_red_in_main_wiring(self):
        """Source-level pin: daily_green_check main() appends mult
        mismatches to reasons and flips green."""
        import inspect
        import daily_green_check as dgc
        src = inspect.getsource(dgc.main)
        assert 'mult_mismatches' in src
        assert "v['green'] = False" in src


class TestDecisionParity:
    """Field-level BT↔live composite comparison (2026-07-17 z-param class)."""

    def _setup(self, monkeypatch, tmp_path, live_comp, csv_gap):
        import yaml
        # real yaml params, fake features CSV with one candidate
        (tmp_path / 'analysis_results').mkdir()
        real = yaml.safe_load(open('orb.yaml'))
        feats = real['filter']['features']
        row = {k: feats[k]['mean'] for k in feats}   # all-mean -> comp 0
        row['gap_pct'] = csv_gap
        import pandas as pd
        pd.DataFrame([{**row, 'symbol': 'TST', 'date': '2026-07-16'}]).to_csv(
            tmp_path / 'analysis_results' / 'orb_features_20260716.csv',
            index=False)
        (tmp_path / 'orb.yaml').write_text(open('orb.yaml').read())
        monkeypatch.setattr(rc, 'ROOT', tmp_path)
        monkeypatch.setattr(rc, 'journal_grep', lambda pat, day:
            [f'x ORB SCORED: TST comp={live_comp:.4f} Q4 | rest']
            if pat == 'ORB SCORED' else [])

    def test_matching_composites_clean(self, monkeypatch, tmp_path):
        import yaml
        feats = yaml.safe_load(open('orb.yaml'))['filter']['features']
        self._setup(monkeypatch, tmp_path, live_comp=0.0,
                    csv_gap=feats['gap_pct']['mean'])
        dp = rc.decision_parity('2026-07-16')
        assert dp['available'] and dp['n_compared'] == 1
        assert dp['mismatches'] == []

    def test_param_drift_flagged(self, monkeypatch, tmp_path):
        """Live logging a different composite than the shared code path
        computes = the ASPI 7/14 bug — must be caught."""
        import yaml
        feats = yaml.safe_load(open('orb.yaml'))['filter']['features']
        self._setup(monkeypatch, tmp_path, live_comp=0.0931,   # drifted
                    csv_gap=feats['gap_pct']['mean'])
        dp = rc.decision_parity('2026-07-16')
        assert len(dp['mismatches']) == 1 and 'TST' in dp['mismatches'][0]

    def test_composite_drift_is_carried_by_pick_parity_not_the_green_check(self):
        import inspect
        import daily_green_check as dgc
        src = inspect.getsource(dgc.main)
        assert 'pick_parity_verdict' in src and 'decision_parity' not in src


class TestCompositeDriftClassification:
    """Decision-relevance refinement (2026-07-23, owner-approved after
    the 7/21 VIVK streak-reset: vendor bar revision on a 174% gapper,
    3.5σ deep reject both sides, zero decisions changed, streak burned
    at 10/10)."""

    def test_within_tolerance_ignored(self):
        assert rc.classify_composite_drift(-3.5203, -3.5180) is None
        assert rc.classify_composite_drift(0.30, 0.301) is None

    def test_vivk_class_is_soft(self):
        # deep reject both sides, small drift -> warn, never red
        assert rc.classify_composite_drift(-3.5203, -3.5016) == 'soft'

    def test_aspi_class_hard_only_when_decision_changes(self):
        # 0.410 vs 0.317 near the gate: soft without a decision flip...
        assert rc.classify_composite_drift(0.410, 0.317) == 'soft'
        # ...hard once the threshold sits between the two (pass/fail differs)
        assert rc.classify_composite_drift(0.410, 0.317, threshold=0.35) == 'hard'

    def test_selection_flip_always_hard(self):
        assert rc.classify_composite_drift(0.010, -0.010) == 'hard'
        # even a deep drift that crosses zero
        assert rc.classify_composite_drift(0.6, -0.6) == 'hard'

    def test_near_zero_same_side_is_soft(self):
        """10/16 RGC: live -0.4672 vs BT -0.4513, both reject, no decision changed."""
        assert rc.classify_composite_drift(-0.4672, -0.4513) == 'soft'
        sev, why = rc.composite_drift_decision(-0.4672, -0.4513)
        assert sev == 'soft' and 'decision_changed' not in why
        assert rc.classify_composite_drift(0.45, 0.48) == 'soft'

    def test_top_n_membership_difference_is_hard(self):
        sev, why = rc.composite_drift_decision(
            0.9, 0.8, live_in_top=True, bt_in_top=False)
        assert sev == 'hard' and 'decision_changed' in why and 'top-N' in why
        assert rc.classify_composite_drift(
            0.9, 0.8, live_in_top=True, bt_in_top=True) == 'soft'

    def test_pass_fail_flip_reason_text(self):
        sev, why = rc.composite_drift_decision(0.5, 0.3, threshold=0.4)
        assert sev == 'hard' and 'decision_changed' in why

    def test_big_delta_is_hard(self):
        sev, why = rc.composite_drift_decision(-3.5, -3.3)
        assert sev == 'hard' and 'decision_changed' in why

    def test_huge_drift_on_deep_reject_still_hard(self):
        # pipeline-broken magnitude: not explainable by data revision
        assert rc.classify_composite_drift(-3.5, -3.2) == 'hard'

    def test_band_boundary(self):
        # same side, no decision change -> soft
        assert rc.classify_composite_drift(0.52, 0.55) == 'soft'


class TestBtStalenessOnFeatures:
    def test_features_date_wins_over_trades(self, tmp_path, monkeypatch):
        """Zero-selection day (7/22 incident): features CSV covers the
        day, trades CSV doesn't -> NOT stale."""
        ar = tmp_path / 'analysis_results'
        ar.mkdir()
        (ar / 'orb_features_x.csv').write_text(
            'date,symbol\n2026-07-22,AAA\n')
        (ar / 'orb_features_corrmatrix_x.csv').write_text('date\njunk\n')
        tr = tmp_path / 'orb_static_lock_trades.csv'
        tr.write_text('date,symbol\n2026-07-21,BBB\n')
        monkeypatch.setattr(rc, 'ROOT', tmp_path)
        monkeypatch.setattr(rc, 'latest_bt_trades_csv', lambda: str(tr))
        assert rc.bt_data_max_date() == '2026-07-22'

    def test_falls_back_to_trades_csv(self, tmp_path, monkeypatch):
        tr = tmp_path / 'orb_static_lock_trades.csv'
        tr.write_text('date,symbol\n2026-07-21,BBB\n')
        monkeypatch.setattr(rc, 'ROOT', tmp_path)   # no analysis_results
        monkeypatch.setattr(rc, 'latest_bt_trades_csv', lambda: str(tr))
        assert rc.bt_data_max_date() == '2026-07-21'


class TestFillParity:
    """8/14 IREX lesson: BT-FILLED pick that live ORDERED but never
    FILLED must be a red flag (the old gate only required ORDERED —
    the 7/30 monster passed silently as time_stop_canceled)."""

    def _patch(self, monkeypatch, rows=(), bt=(), bt_max='2099-01-01'):
        monkeypatch.setattr(rc, 'load_live_rows',
                            lambda day, strategy=None: list(rows))
        monkeypatch.setattr(rc, 'load_bt_selected', lambda day: list(bt))
        monkeypatch.setattr(rc, 'bt_data_max_date', lambda: bt_max)
        monkeypatch.setattr(rc, 'journal_grep', lambda pat, day: [])
        monkeypatch.setattr(rc, 'read_selection_audit', lambda day: [])

    def test_bt_filled_live_unfilled_is_red(self, monkeypatch):
        rows = [{'symbol': 'IREX', 'strategy': 'orb', 'entry_price': 10.97,
                 'fill_price': None, 'exit_price': None, 'shares': 683,
                 'exit_reason': None, 'order_status': 'time_stop_canceled'}]
        self._patch(monkeypatch, rows=rows, bt=[{'symbol': 'IREX'}])
        v = rc.green_verdict('2026-07-30')
        assert v['green'] is False
        assert any('fill-parity' in r for r in v['reasons'])

    def test_filled_row_passes(self, monkeypatch):
        rows = [{'symbol': 'AAA', 'strategy': 'orb', 'entry_price': 10.0,
                 'fill_price': 10.01, 'exit_price': 10.5, 'shares': 100,
                 'exit_reason': 'lock_stop', 'order_status': 'closed',
                 'pnl': 49.0}]
        self._patch(monkeypatch, rows=rows, bt=[{'symbol': 'AAA'}])
        assert rc.green_verdict('2026-07-30')['green'] is True

    def test_live_only_cancel_not_flagged(self, monkeypatch):
        # live ordered something BT never filled — legitimate no-trigger
        rows = [{'symbol': 'NOPE', 'strategy': 'orb', 'entry_price': 5.0,
                 'fill_price': None, 'exit_price': None, 'shares': 100,
                 'exit_reason': None, 'order_status': 'time_stop_canceled'}]
        self._patch(monkeypatch, rows=rows, bt=[])
        assert rc.green_verdict('2026-07-30')['green'] is True

    def test_missing_bt_has_no_fill_parity_and_no_skipped(self, monkeypatch):
        self._patch(monkeypatch, rows=[], bt=[], bt_max='2026-07-01')
        v = rc.green_verdict('2026-07-30')
        assert 'SKIPPED' not in str(v['checks']) and v['green'] is False


class TestParityFreeze:
    """Gate-1 of docs/scaling_plan_2026.md: a HARD parity breach FREEZES the
    ORB ramp (size unchanged, entries continue, stage clock stops)."""

    def test_hard_markers_are_classified(self):
        from daily_green_check import parity_breaches
        hard = [
            "pm_mult drift vs recompute: ['ABC 2.0 vs 1.0']",
            "pick parity: 1 unexplained ['DEF'], live fills outside the final set []",
            "fill-parity: BT FILLED but live order never filled: ['GHI']",
            "unattributed exits: ['JKL']",
            "exit_pending_verification: ['MNO']",
            "floored-stop drift: ['STU']",
        ]
        assert parity_breaches(hard) == hard

    def test_non_parity_reds_do_not_freeze(self):
        from daily_green_check import parity_breaches
        assert parity_breaches([
            "CONTEXT: trader restarted 3x today — missed picks likely due to "
            "service outage, not selection",
            "news drift vs EoD recheck: ['XYZ']",
        ]) == []
        assert parity_breaches([]) == []

    def _patch_main(self, monkeypatch, tmp_path, reasons):
        import daily_green_check as dgc
        from trading import ramp_freeze as rf
        monkeypatch.setattr(rf, 'FREEZE_PATH', tmp_path / 'freeze.json')
        monkeypatch.setattr(rc, 'green_verdict', lambda d: {
            'day': d, 'green': not reasons, 'reasons': list(reasons),
            'checks': {}, 'n_live_rows': 1, 'n_bt_selected': 1,
            'bt_missing': False})
        monkeypatch.setattr(rc, 'sizing_attribution',
                            lambda d: {'mult_mismatches': [], 'rows': []})
        monkeypatch.setattr(rc, 'decision_parity',
                            lambda d: {'mismatches': [], 'warnings': [],
                                       'n_compared': 0})
        monkeypatch.setattr(dgc, 'wait_for_bt', lambda *a, **k: True)
        monkeypatch.setattr(dgc, 'pick_parity_verdict', lambda d: dict(CLEAN_PARITY, date=d))
        monkeypatch.setattr(rc, 'streak_update', lambda *a, **k: 0)
        monkeypatch.setattr(rc, 'read_streak', lambda *a, **k: {'streak': 0})
        monkeypatch.setattr(rc, 'realized_pnl', lambda d: {})
        monkeypatch.setattr(rc, 'news_lag_audit', lambda d: {})
        monkeypatch.setattr(rc, 'news_lag_line', lambda a: '')
        monkeypatch.setattr(rc, 'bf_rails_status', lambda d: {})
        monkeypatch.setattr(rc, 'bf_rails_line', lambda a: '')
        monkeypatch.setattr(rc, 'sizing_block', lambda a: '')
        monkeypatch.setattr(rc, 'send_telegram', lambda m: True)
        monkeypatch.setattr(rf, 'send_freeze_telegram', lambda *a, **k: True)
        return dgc, rf

    def test_main_freezes_orb_on_a_hard_breach(self, monkeypatch, tmp_path):
        dgc, rf = self._patch_main(
            monkeypatch, tmp_path, ["pick parity: 1 unexplained ['DEF'], live fills outside the final set []"])
        monkeypatch.setattr(sys, 'argv',
                            ['daily_green_check.py', '--date', '2026-09-18',
                             '--no-telegram'])
        assert dgc.main() == 1
        st = rf.get('orb')
        assert st.frozen and st.since == '2026-09-18'
        assert 'pick parity' in st.reason

    def test_main_does_not_freeze_a_green_day(self, monkeypatch, tmp_path):
        dgc, rf = self._patch_main(monkeypatch, tmp_path, [])
        monkeypatch.setattr(sys, 'argv',
                            ['daily_green_check.py', '--date', '2026-09-18',
                             '--no-telegram'])
        assert dgc.main() == 0
        assert not rf.is_frozen('orb')

    def test_dry_run_never_writes_freeze_state(self, monkeypatch, tmp_path):
        dgc, rf = self._patch_main(
            monkeypatch, tmp_path, ["unattributed exits: ['JKL']"])
        monkeypatch.setattr(sys, 'argv',
                            ['daily_green_check.py', '--date', '2026-09-18',
                             '--no-telegram', '--dry-run'])
        assert dgc.main() == 1
        assert not rf.is_frozen('orb')


class TestOrbOnlyGreenCheckAndFreezeReconcile:
    """2026-10-06: the ORB green check judged HOD's paper exits by ORB's exit-reason table (10/5: 'unattributed
    exits' for ten HOD symbols, ORB had 0 fills) and froze the ramp. It loads strategy='orb' rows only, and a
    re-run of the same day clears the freeze it raised when no hard breach is left."""

    HOD = {'symbol': 'SPAL', 'strategy': 'hod_break', 'entry_price': 10.0, 'exit_price': 9.0, 'shares': 10,
           'exit_reason': 'hod_paper_stop_xyz', 'order_status': 'exit_pending_verification'}

    def _patch(self, monkeypatch, rows, bt=()):
        monkeypatch.setattr(rc, 'load_live_rows', lambda day, strategy=None:
                            [r for r in rows if strategy is None or r.get('strategy') == strategy])
        monkeypatch.setattr(rc, 'load_bt_selected', lambda day: list(bt))
        monkeypatch.setattr(rc, 'bt_data_max_date', lambda: '2099-01-01')
        monkeypatch.setattr(rc, 'journal_grep', lambda pat, day: [])
        monkeypatch.setattr(rc, 'read_selection_audit', lambda day: [])

    def test_green_check_asks_for_orb_rows_only(self, monkeypatch):
        asked = []
        monkeypatch.setattr(rc, 'load_live_rows', lambda day, strategy=None: asked.append(strategy) or [])
        monkeypatch.setattr(rc, 'load_bt_selected', lambda day: [])
        monkeypatch.setattr(rc, 'bt_data_max_date', lambda: '2099-01-01')
        monkeypatch.setattr(rc, 'journal_grep', lambda pat, day: [])
        monkeypatch.setattr(rc, 'read_selection_audit', lambda day: [])
        rc.green_verdict('2026-10-05')
        assert asked[0] == 'orb'

    def test_hod_rows_are_judged_nowhere_here(self, monkeypatch):
        self._patch(monkeypatch, rows=[self.HOD])
        v = rc.green_verdict('2026-10-05')
        assert v['green'] is True and v['reasons'] == [] and v['n_live_rows'] == 0

    def test_orb_unknown_exit_is_still_red_next_to_hod_rows(self, monkeypatch):
        orb = {'symbol': 'AAA', 'strategy': 'orb', 'entry_price': 10.0, 'exit_price': 9.0, 'shares': 100,
               'exit_reason': 'unknown_exit', 'order_status': 'filled'}
        self._patch(monkeypatch, rows=[self.HOD, orb])
        v = rc.green_verdict('2026-10-05')
        assert any("unattributed exits: ['AAA']" in r for r in v['reasons']) and 'SPAL' not in str(v['reasons'])

    def _freeze_orb(self, tmp_path, monkeypatch, day, reason):
        from trading import ramp_freeze as rf
        monkeypatch.setattr(rf, 'FREEZE_PATH', tmp_path / 'freeze.json')
        monkeypatch.setattr(rf, 'send_freeze_telegram', lambda *a, **k: True)
        rf.set_freeze('orb', reason, day=day, notify=False)
        return rf

    def test_rerun_clears_a_freeze_whose_reasons_are_gone(self, monkeypatch, tmp_path):
        import daily_green_check as dgc
        rf = self._freeze_orb(tmp_path, monkeypatch, '2026-10-05',
                              "unattributed exits: ['SPAL']; BT picks never ordered live: ['DFDV']")
        line = dgc.reconcile_freeze('2026-10-05', [], notify=False)
        assert not rf.is_frozen('orb') and 'cleared by code' in line and 'frozen since 2026-10-05' in line
        assert '2026-10-05' in rf.get('orb').frozen_dates   # the session still does not count toward the stage

    def test_surviving_breach_keeps_the_freeze_and_the_reason_says_so(self, monkeypatch, tmp_path):
        import daily_green_check as dgc
        rf = self._freeze_orb(tmp_path, monkeypatch, '2026-10-05',
                              "unattributed exits: ['SPAL']; BT picks never ordered live: ['DFDV']")
        line = dgc.reconcile_freeze('2026-10-05', ["BT picks never ordered live: ['DFDV']"], notify=False)
        st = rf.get('orb')
        assert st.frozen and st.reason == "BT picks never ordered live: ['DFDV']" and st.since == '2026-10-05'
        assert 'unattributed' not in st.reason and "-> frozen since 2026-10-05" in line

    def test_a_freeze_raised_by_another_day_is_never_cleared(self, monkeypatch, tmp_path):
        import daily_green_check as dgc
        rf = self._freeze_orb(tmp_path, monkeypatch, '2026-10-02', "fill-parity: ['IREX']")
        line = dgc.reconcile_freeze('2026-10-05', [], notify=False)
        assert rf.is_frozen('orb') and 'unchanged' in line

    def test_dry_run_reports_without_writing(self, monkeypatch, tmp_path):
        import daily_green_check as dgc
        rf = self._freeze_orb(tmp_path, monkeypatch, '2026-10-05', "unattributed exits: ['SPAL']")
        assert 'would CLEAR' in dgc.reconcile_freeze('2026-10-05', [], dry_run=True)
        assert rf.is_frozen('orb')


# ---------------------------------------------------------------------------
# ONE counter (2026-11-12): pick parity, BT wait, day-one streak, never "skipped"
# ---------------------------------------------------------------------------
import datetime as _dt
import logging

import daily_green_check as dgc
from scripts import orb_pick_parity as opp

DAY_ONE = _dt.date(2026, 11, 12)


def _res(diffs=(), final=('AAA', 'BBB'), bt_available=True, leaks=()):
    """A pick-parity result dict (shape of orb_pick_parity.run_parity)."""
    diffs = [dict(symbol=s, side='bt_only', reason=r, detail='') for s, r in diffs]
    return {'date': '2026-11-12', 'bt_available': bt_available, 'live_final': list(final), 'bt_top8': list(final),
            'differences': diffs, 'n_unexplained': sum(d['reason'] == 'unexplained' for d in diffs),
            'fills_outside_final': list(leaks), 'summary': 'ORB pick parity 2026-11-12: summary'}


def _verdict():
    return {'day': '2026-11-12', 'green': True, 'reasons': [], 'checks': {}, 'n_live_rows': 2, 'n_bt_selected': 2}


class TestPickParityVerdict:
    def test_clean_is_green(self):
        v = _verdict()
        assert dgc.apply_pick_parity(v, _res()) is False and v['green'] is True

    def test_one_unexplained_is_red(self):
        v = _verdict()
        assert dgc.apply_pick_parity(v, _res(diffs=[('AAA', 'unexplained')])) is False
        assert v['green'] is False and 'pick parity: 1 unexplained' in v['reasons'][0]
        assert dgc.parity_breaches(v['reasons'])   # a hard breach: freezes the ramp

    def test_fill_outside_final_is_red(self):
        v = _verdict()
        dgc.apply_pick_parity(v, _res(leaks=['XMAX']))
        assert v['green'] is False and 'XMAX' in v['reasons'][0]

    def test_explained_differences_stay_green(self):
        v = _verdict()
        dgc.apply_pick_parity(v, _res(diffs=[('AAA', 'range_missing')], final=('AAA', 'BBB')))
        assert v['green'] is True

    def test_vendor_no_range_day_is_not_counted(self):
        v = _verdict()
        res = _res(diffs=[('AAA', 'range_missing'), ('BBB', 'range_missing')], final=())
        res['bt_top8'] = ['AAA', 'BBB']
        assert dgc.is_no_range_day(res) and dgc.apply_pick_parity(v, res) is True

    def test_bt_side_unavailable_is_red_with_one_error(self, caplog):
        """The nightly BT was waited for, so a parity run that still cannot read its BT side is OUR defect: RED, never
        'not counted' (one ERROR names it)."""
        v = _verdict()
        res = _res(bt_available=False)
        res['bt_why'] = 'bt_ranked empty'
        with caplog.at_level(logging.ERROR, logger='daily_green_check'):
            assert dgc.apply_pick_parity(v, res) is False
        assert v['green'] is False
        assert any('BT side unavailable' in r and 'bt_ranked empty' in r for r in v['reasons'])
        assert sum(1 for r in caplog.records if r.levelno == logging.ERROR) == 1


class TestWaitForBt:
    def test_returns_when_bt_appears(self):
        seq = iter([None, '2026-11-11', '2026-11-12'])
        slept = []
        assert dgc.wait_for_bt('2026-11-12', 45, sleep=slept.append, bt_max_fn=lambda: next(seq)) is True
        assert slept == [60, 60]

    def test_gives_up_after_the_window(self):
        slept = []
        assert dgc.wait_for_bt('2026-11-12', 3, sleep=slept.append, bt_max_fn=lambda: '2026-11-11') is False
        assert len(slept) == 3


class TestDayOneStreak:
    def test_only_days_from_day_one_count(self, tmp_path):
        p = tmp_path / 's.json'
        recs = [{'day': f'2026-08-{d:02d}', 'green': True, 'reasons': []} for d in range(3, 8)]
        recs += [{'day': f'2026-11-{d:02d}', 'green': True, 'reasons': []} for d in (2, 3, 4, 5, 6, 9, 10, 11)]
        p.write_text(json.dumps({'streak': 13, 'needed': 10, 'days': recs}))
        assert rc.read_streak(p)['streak'] == 0          # stored 13 is never trusted
        assert rc.streak_update('2026-11-12', True, [], path=p) == 1
        assert len(rc.read_streak(p)['days']) == 14       # earlier records untouched
        assert rc.streak_update('2026-11-13', True, [], path=p) == 2

    def test_red_resets_and_not_counted_is_neutral(self, tmp_path):
        p = tmp_path / 's.json'
        rc.streak_update('2026-11-12', True, [], path=p)
        rc.streak_update('2026-11-13', True, [], path=p)
        assert rc.streak_update('2026-11-16', False, [], path=p, counted=False) == 2
        assert rc.read_streak(p)['days'][-1]['counted'] is False
        assert rc.streak_update('2026-11-17', True, [], path=p) == 3
        assert rc.streak_update('2026-11-18', False, ['x'], path=p) == 0

    def test_label(self):
        assert rc.green_label(3) == '[GREEN 3/10 since 11/12, A2]'
        assert '[GREEN 1/10 since 11/12, A2]' in build_message(_verdict(), 1, {})

    def test_ramp_check_reads_nothing_before_day_one(self, tmp_path, monkeypatch):
        import orb_ramp_check as orc
        g = tmp_path / 'g.json'
        g.write_text(json.dumps({'days': [{'day': '2026-11-11', 'green': False, 'reasons': ['fill-parity: x']},
                                          {'day': '2026-11-12', 'green': False, 'reasons': ['pick parity: y']}]}))
        monkeypatch.setattr(orc, 'GREEN', g)
        assert list(orc.load_parity('2026-09-01')) == ['2026-11-12']


class TestMainEndToEnd:
    """Whole `main(['--dry-run', ...])` path on a fixture day: tmp trades.db + tmp BT book + real run_parity compare."""

    DAY = '2026-11-12'

    def _wire(self, monkeypatch, tmp_path, bt_max, live_final=('AAA',), bt_ranked=('AAA',), fills=()):
        db = tmp_path / 't.db'
        con = sqlite3.connect(db)
        con.execute('create table trades (symbol, strategy, trade_date, fill_price, filled_at, order_status, pattern_data)')
        for s in fills:
            con.execute("insert into trades values (?,?,?,?,?,?,?)",
                        (s, 'orb', self.DAY, 10.0, f'{self.DAY}T14:36:00+00:00', 'filled', '{"pool_id": "production"}'))
        con.commit()
        con.close()
        book = tmp_path / 'book.csv'
        book.write_text('symbol,date,entered,params_refit_ts\n' + ''.join(f'{s},{self.DAY},1,20261101_200000\n' for s in bt_ranked))
        bt = {'ranked': list(bt_ranked), 'picks': list(bt_ranked), 'n': 8, 'comp': {s: (0.5, 'Q5') for s in bt_ranked},
              'vetoes': {}, 'day_symbols': sorted(set(bt_ranked) | set(live_final)), 'range_size': {s: 3.0 for s in bt_ranked},
              'params_refit_ts': '20261101_200000', 'params_source': 'pit'}
        live = {'final': sorted(live_final), 'provisional': [], 'submitted': sorted(live_final),
                'scored': {x: (0.5, 'Q5') for x in live_final}, 'range_done': sorted(live_final), 'final_ranges': {},
                'vetoes': {}, 'notes': {}, 'seen': ' '.join(live_final), 'has_reconcile': True}
        monkeypatch.setattr(opp, 'parse_live', lambda text: live)
        log_text = ''
        monkeypatch.setattr(dgc, 'pick_parity_verdict', lambda d: opp.run_parity(
            d, bt_fn=lambda x: (bt, ''), log_fn=lambda x: log_text, fills_fn=lambda x: opp.live_fills(x, db),
            book_path=book, emit=False))
        monkeypatch.setattr(rc, 'bt_data_max_date', lambda: bt_max)
        monkeypatch.setattr(rc, 'load_live_rows', lambda day, strategy=None: [])
        monkeypatch.setattr(rc, 'load_bt_selected', lambda day: [{'symbol': s} for s in bt_ranked])
        monkeypatch.setattr(rc, 'journal_grep', lambda pat, day: [])
        monkeypatch.setattr(rc, 'read_selection_audit', lambda day: [])
        monkeypatch.setattr(rc, 'sizing_attribution', lambda d: {'mult_mismatches': [], 'rows': []})
        monkeypatch.setattr(rc, 'realized_pnl', lambda d: {'orb': 12.0})
        monkeypatch.setattr(rc, 'news_lag_audit', lambda d: {})
        monkeypatch.setattr(rc, 'news_lag_line', lambda a: '')
        monkeypatch.setattr(rc, 'bf_rails_status', lambda d: {})
        monkeypatch.setattr(rc, 'bf_rails_line', lambda a: '')
        monkeypatch.setattr(rc, 'sizing_block', lambda a: '')
        monkeypatch.setattr(rc, 'STREAK_PATH', tmp_path / 'streak.json')
        monkeypatch.setattr(dgc.time, 'sleep', lambda s: None)
        from trading import ramp_freeze as rf
        monkeypatch.setattr(rf, 'FREEZE_PATH', tmp_path / 'freeze.json')

    def test_clean_day_message_end_to_end(self, monkeypatch, tmp_path, capsys, caplog):
        self._wire(monkeypatch, tmp_path, bt_max=self.DAY, fills=('AAA',))
        with caplog.at_level(logging.INFO):
            assert dgc.main(['--dry-run', '--date', self.DAY, '--wait-bt-min', '0']) == 0
        out = capsys.readouterr().out
        assert '[GREEN 0/10 since 11/12, A2]' in out
        assert 'pick parity clean' in out and 'SKIPPED' not in out
        assert not (tmp_path / 'streak.json').exists()      # dry run writes nothing
        assert not [r for r in caplog.records if r.levelno >= logging.ERROR]

    def test_fill_outside_final_is_red_end_to_end(self, monkeypatch, tmp_path, capsys):
        self._wire(monkeypatch, tmp_path, bt_max=self.DAY, fills=('ZZZ',))
        assert dgc.main(['--dry-run', '--date', self.DAY, '--wait-bt-min', '0']) == 1
        out = capsys.readouterr().out
        assert 'RED DAY' in out and 'ZZZ' in out and 'SKIPPED' not in out

    def test_missing_bt_is_red_with_exactly_one_error(self, monkeypatch, tmp_path, capsys, caplog):
        self._wire(monkeypatch, tmp_path, bt_max='2026-11-11')
        with caplog.at_level(logging.INFO):
            assert dgc.main(['--dry-run', '--date', self.DAY, '--wait-bt-min', '1']) == 1
        out = capsys.readouterr().out
        errs = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert len(errs) == 1 and 'onemil-orb-backtest.timer' in errs[0].getMessage()
        assert 'RED DAY' in out and f'nightly BT missing for {self.DAY}' in out and 'SKIPPED' not in out
