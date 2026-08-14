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

    def test_sizing_block_appended_to_green(self):
        v = {'day': '2026-07-13', 'green': True, 'reasons': [], 'checks': {},
             'n_live_rows': 1, 'n_bt_selected': 1}
        msg = build_message(v, streak=1, pnl={'orb': 50.0},
                            sizing_txt='sizing: AAA Q2 q1.5×pm2.0 (news✓ $9.0M)')
        assert 'sizing: AAA' in msg and 'GREEN' in msg


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

    def test_wired_as_hard_gate(self):
        import inspect
        import daily_green_check as dgc
        src = inspect.getsource(dgc.main)
        assert 'decision_parity' in src and 'composite drift' in src


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

    def test_aspi_class_stays_hard(self):
        # the P0 the gate was built for: near-threshold drift
        assert rc.classify_composite_drift(0.410, 0.317) == 'hard'

    def test_selection_flip_always_hard(self):
        assert rc.classify_composite_drift(0.010, -0.010) == 'hard'
        # even a deep drift that crosses zero
        assert rc.classify_composite_drift(0.6, -0.6) == 'hard'

    def test_near_threshold_band_hard_both_signs(self):
        assert rc.classify_composite_drift(-0.30, -0.28) == 'hard'
        assert rc.classify_composite_drift(0.45, 0.48) == 'hard'

    def test_huge_drift_on_deep_reject_still_hard(self):
        # pipeline-broken magnitude: not explainable by data revision
        assert rc.classify_composite_drift(-3.5, -3.2) == 'hard'

    def test_band_boundary(self):
        # exactly at band edge (0.5): outside band -> soft
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

    def test_stale_bt_skips_fill_parity(self, monkeypatch):
        self._patch(monkeypatch, rows=[], bt=[], bt_max='2026-07-01')
        v = rc.green_verdict('2026-07-30')
        assert v['checks']['fill_parity'].startswith('SKIPPED')
