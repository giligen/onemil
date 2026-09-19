"""Ramp stage-start table (trading/ramp_stage.py).

A ramp checker measures stage P&L / sessions / parity defects from the stage
start. Both checkers used to hardcode their book's LAUNCH date, so the
2026-09-21 boot — a NEW stage for both books by docs/scaling_plan_2026.md
("BF has ZERO live trades under the config that boots Monday — its stage clock
starts Monday"; ORB lands 8 slots + catalyst-off + the latency fix + 50 bps at
once) — would have kept counting the previous stage's trades and reds.
"""
import importlib.util
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from trading import ramp_stage as rs   # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(ROOT, 'scripts', f'{name}.py'))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class TestTable:
    @pytest.mark.parametrize('book', ['bf', 'orb'])
    def test_current_stage_starts_2026_09_21(self, book):
        """Monday's boot is a new stage for BOTH books."""
        start, reason = rs.current(book)
        assert start == '2026-09-21'
        assert reason.strip()

    def test_orb_reason_names_the_change(self):
        _, reason = rs.current('orb')
        assert '8-slot' in reason and 'catalyst-off' in reason

    def test_bf_reason_names_the_change(self):
        _, reason = rs.current('bf')
        assert 'ADV' in reason or 'min_daily_volume' in reason

    @pytest.mark.parametrize('book,launch', [('bf', '2026-09-07'),
                                             ('orb', '2026-08-17')])
    def test_history_keeps_the_launch_date(self, book, launch):
        """History is never rewritten — the old stage stays documented."""
        starts = [r['start'] for r in rs.history(book)]
        assert launch in starts
        assert starts == sorted(starts)          # chronological
        assert rs.previous(book)[0] == launch

    def test_rows_are_iso_dates_with_reasons(self):
        from datetime import date
        for book in rs.BOOKS:
            for row in rs.history(book):
                date.fromisoformat(row['start'])
                assert row['reason'].strip()

    def test_unknown_book_raises(self):
        with pytest.raises(ValueError):
            rs.current('macd_wave')


class TestResolve:
    def test_default_is_the_table(self):
        assert rs.resolve('orb') == rs.current('orb')
        assert rs.resolve('bf', None) == rs.current('bf')

    def test_override_wins_and_is_labelled(self):
        start, reason = rs.resolve('orb', '2026-08-17')
        assert start == '2026-08-17'
        assert 'override' in reason.lower()

    def test_malformed_override_raises(self):
        """A silently-wrong window would silently mis-measure every gate."""
        with pytest.raises(ValueError):
            rs.resolve('bf', 'monday')

    def test_line_flags_the_excluded_previous_stage(self):
        line = rs.line('orb', *rs.current('orb'))
        assert '2026-09-21' in line and '2026-08-17' in line
        assert 'OUT of this window' in line


class TestCheckersConsumeIt:
    """Both checkers read the SAME table — no second hardcoded launch date."""

    @pytest.mark.parametrize('name,book', [('bf_ramp_check', 'bf'),
                                           ('orb_ramp_check', 'orb')])
    def test_checker_default_is_the_table(self, name, book):
        mod = _load(name)
        assert mod.ramp_stage is rs
        src = open(os.path.join(ROOT, 'scripts', f'{name}.py')).read()
        # the old module-level launch constants are gone
        assert 'LAUNCH =' not in src and 'B_PLUS_LIVE =' not in src
        assert "ramp_stage.add_stage_start_arg(ap, '%s')" % book in src

    @pytest.mark.parametrize('name,book', [('bf_ramp_check', 'bf'),
                                           ('orb_ramp_check', 'orb')])
    def test_arg_default_none_so_resolve_decides(self, name, book):
        import argparse
        ap = argparse.ArgumentParser()
        rs.add_stage_start_arg(ap, book)
        a = ap.parse_args([])
        assert a.stage_start is None
        assert rs.resolve(book, a.stage_start)[0] == '2026-09-21'
        assert rs.resolve(book, '2026-01-02')[0] == '2026-01-02'
