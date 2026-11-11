"""ORB day one = 2026-11-12 (docs/orb_day_one_20261112_spec_20261111.md, PREREG_1709 Amendment 2).

Unit: stage table, exclusion csv, the one ORB_DAY_ONE constant in the report helpers.
Integration: a real temp trades.db through orb_ramp_check's loader + exclusion split + stats.
"""
import importlib.util
import os
import sqlite3
import sys
from datetime import date

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from trading import ramp_stage as rs  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, 'scripts', name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


orc = _load('orb_ramp_check')
bs = _load('book_status')
hs = _load('hourly_summary')
import report_common as rc  # noqa: E402


def _make_db(path, rows):
    con = sqlite3.connect(path)
    con.execute("create table trades (id integer primary key, trade_date text, symbol text, strategy text, "
                "pnl real, entry_price real, fill_price real, total_risk real, exited_at text, exit_reason text, "
                "order_status text)")
    for i, (d, sym, pnl) in enumerate(rows):
        con.execute("insert into trades (trade_date, symbol, strategy, pnl, entry_price, fill_price, total_risk, "
                    "exited_at, exit_reason, order_status) values (?,?,?,?,?,?,?,?,?,?)",
                    (d, sym, 'orb', pnl, 10.0, 10.0, 100.0, d + 'T20:00:00', 'time_stop', 'filled'))
    con.commit()
    con.close()


class TestStageTable:
    def test_resolve_orb_is_day_one_with_a2_reason(self):
        start, reason = rs.resolve('orb', None)
        assert start == '2026-11-12' == rs.ORB_DAY_ONE.isoformat()
        assert 'amend2_parity' in reason and 'PREREG_1709 Amendment 2' in reason
        assert 'eafb557, 2a55c82, ff2a6fa' in reason and '2026-09-29..11-10' in reason

    def test_old_stages_kept_and_line_prints_it(self):
        assert [r['start'] for r in rs.history('orb')][-3:] == ['2026-08-17', '2026-09-21', '2026-11-12']
        assert 'amend2_parity' in rs.line('orb', *rs.current('orb'))

    def test_suffix_helper(self):
        assert rs.orb_label('ORB $+5') == 'ORB $+5 (since 11/12, A2)'


class TestExclusions:
    def test_csv_parsed_with_crdu_and_bex(self):
        ex = orc.load_exclusions()
        assert ('2026-10-27', 'CRDU') in ex and ('2026-11-09', 'BEX') in ex
        assert all(r.strip() for r in ex.values())

    def test_excluded_removed_and_listed(self):
        fills = [{'trade_date': '2026-11-09', 'symbol': 'BEX', 'pnl': 900.0},
                 {'trade_date': '2026-11-13', 'symbol': 'ABC', 'pnl': 50.0}]
        kept, dropped = orc.split_excluded(fills, orc.load_exclusions())
        assert [f['symbol'] for f in kept] == ['ABC']
        assert [(f['symbol'], bool(r)) for f, r in dropped] == [('BEX', True)]

    def test_monster_computed_after_exclusion(self):
        fills = [{'trade_date': '2026-10-27', 'symbol': 'CRDU', 'pnl': 2000.0, 'entry_price': 10, 'fill_price': 10,
                  'total_risk': 100},
                 {'trade_date': '2026-11-13', 'symbol': 'ABC', 'pnl': 50.0, 'entry_price': 10, 'fill_price': 10,
                  'total_risk': 100}]
        kept, _ = orc.split_excluded(fills, orc.load_exclusions())
        s = orc.compute_stats(kept, 10000, None, 1, ['2026-11-13'], {})
        assert s.best_fill_pnl == 50.0 and s.pnl == 50.0

    def test_missing_file_warns_and_excludes_nothing(self, tmp_path, capsys):
        assert orc.load_exclusions(tmp_path / 'nope.csv') == {}
        assert 'WARNING' in capsys.readouterr().out


class TestReportsCountFromDayOne:
    def test_book_status_orb_section_counts_only_post_day_one(self, tmp_path):
        db = tmp_path / 'trades.db'
        _make_db(db, [('2026-11-10', 'OLD', -500.0), ('2026-11-12', 'NEW', 30.0)])
        sec = bs.section_orb(db, '2026-11-12', [], tmp_path / 'none.json')
        assert sec['today'] == {'fills': 1, 'pnl': 30.0}
        assert sec['wtd'] == {'fills': 1, 'pnl': 30.0}          # 11/10 is in the same week but pre-parity
        assert bs.ORB_DAY_ONE is rs.ORB_DAY_ONE and bs.ORB_DAY_ONE_SUFFIX == '(since 11/12, A2)'

    def test_book_status_daily_floor(self, tmp_path):
        db = tmp_path / 'trades.db'
        _make_db(db, [('2026-11-10', 'OLD', -500.0), ('2026-11-12', 'NEW', 30.0)])
        out = bs.book_daily(db, '2026-11-12', 'orb', None, rs.ORB_DAY_ONE.isoformat())
        assert [d['date'] for d in out['daily']] == ['2026-11-12']

    def test_hourly_summary_drops_pre_day_one_orb_fills(self):
        fills = [{'symbol': 'A', 'filled_date': '2026-11-11'}, {'symbol': 'B', 'filled_date': '2026-11-12'},
                 {'symbol': 'C', 'filled_date': ''}]
        assert [f['symbol'] for f in hs.since_orb_day_one(fills)] == ['B', 'C']

    def test_report_common_realized_and_pre_parity_line(self, monkeypatch):
        rows = {'2026-11-10': [{'strategy': 'orb', 'pnl': -100.0}],
                '2026-11-12': [{'strategy': 'orb', 'pnl': 40.0}, {'strategy': 'hod_break', 'pnl': 5.0}]}
        monkeypatch.setattr(rc, 'load_live_rows', lambda d, strategy=None:
                            [r for r in rows.get(d, []) if strategy in (None, r['strategy'])])
        assert 'orb' not in rc.realized_pnl('2026-11-10')
        assert rc.realized_pnl('2026-11-12') == {'orb': 40.0, 'hod_break': 5.0}
        assert rc.orb_pre_parity_window_in(['2026-11-10', '2026-11-12'])
        assert not rc.orb_pre_parity_window_in(['2026-11-12', '2026-11-13'])
        assert rc.orb_pre_parity_pnl(['2026-11-10', '2026-11-12']) == (-100.0, 1)


class TestRampCheckSummaryIntegration:
    def test_stage_fills_exclude_pre_day_one_and_list_exclusions(self, tmp_path, monkeypatch):
        db = tmp_path / 'trades.db'
        _make_db(db, [('2026-11-10', 'OLD', -500.0), ('2026-11-12', 'AAA', 80.0), ('2026-11-13', 'BEX', 900.0),
                      ('2026-11-13', 'CCC', -20.0)])
        monkeypatch.setattr(orc, 'TRADES_DB', db)
        start, _ = rs.resolve('orb', None)
        fills = orc.load_fills(start)
        # 11/13 BEX is not an accident (the accident was 11/9): only an exact (date, symbol) match is excluded
        ex = dict(orc.load_exclusions()); ex[('2026-11-13', 'BEX')] = 'test accident'
        kept, dropped = orc.split_excluded(fills, ex)
        s = orc.compute_stats(kept, 10000, None, 2, ['2026-11-12', '2026-11-13'], {})
        assert s.fills == 2 and s.pnl == 60.0
        assert 'OLD' not in {f['symbol'] for f in fills}
        assert [(f['symbol'], r) for f, r in dropped] == [('BEX', 'test accident')]
        assert s.best_fill_pnl == 80.0                                   # monster after exclusion

    def test_cli_with_no_flag_prints_zero_fills_without_error(self, capsys, monkeypatch):
        monkeypatch.setattr(sys, 'argv', ['orb_ramp_check.py'])
        assert orc.main() == 0
        out = capsys.readouterr().out
        assert 'since 2026-11-12' in out and 'amend2_parity' in out
