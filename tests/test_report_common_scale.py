"""Report-layer winner-stack coverage: a SCALED trade row through _row_pnl,
cumulative_orb_since, realized_pnl, sizing_attribution, green_verdict, and
the EoD floored-stop drift helper (review P1-5 vocabulary drill + P0-3c
defense-in-depth).
"""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

import scripts.report_common as rc

DAY = '2026-08-24'


def _scaled_row(pnl=1500.0, exit_price=10.5):
    return {
        'symbol': 'TEST', 'strategy': 'orb', 'trade_date': DAY,
        'entry_price': 10.03, 'fill_price': 10.0, 'exit_price': exit_price,
        'exit_reason': 'lock_stop', 'order_status': 'closed',
        'shares': 1000, 'filled_qty': 1000,
        'scale_qty': 400, 'scale_price': 13.0, 'scale_pnl': 1200.0,
        'scaled_at': '2026-08-24T14:00:00+00:00',
        'pnl': pnl, 'pnl_pct': 15.0,
        'real_stop_loss_price': 9.75,
        'pattern_data': json.dumps({
            'quintile': 'Q4', 'adaptive_mult': 1.0, 'pm_mult': 1.0,
            'has_news': True, 'n_articles': 2, 'pm_dollar_vol': 1e6,
            'asset_class': 'stock', 'range_low': 9.0, 'atr14': 1.0,
            'atr_floor_k': 0.25}),
    }


class TestRowPnl:
    def test_recorded_pnl_preferred(self):
        assert rc._row_pnl(_scaled_row()) == pytest.approx(1500.0)

    def test_null_pnl_recompute_scale_aware(self):
        """P0-3c: the fallback recompute uses runner qty + scale_pnl."""
        r = _scaled_row()
        r['pnl'] = None
        expected = (10.5 - 10.0) * 600 + 1200.0
        assert rc._row_pnl(r) == pytest.approx(expected)

    def test_null_pnl_unscaled_unchanged(self):
        r = _scaled_row()
        r.update({'pnl': None, 'scaled_at': None, 'scale_qty': None,
                  'scale_pnl': None})
        assert rc._row_pnl(r) == pytest.approx((10.5 - 10.0) * 1000)


class TestCumulativeSql:
    def _db(self, tmp_path, rows):
        p = tmp_path / 'trades.db'
        con = sqlite3.connect(p)
        con.execute("""CREATE TABLE trades (
            id INTEGER PRIMARY KEY, strategy TEXT, trade_date TEXT,
            entry_price REAL, fill_price REAL, exit_price REAL,
            shares INTEGER, filled_qty INTEGER, pnl REAL,
            scale_qty INTEGER, scale_pnl REAL, scaled_at TEXT)""")
        for r in rows:
            con.execute(
                "INSERT INTO trades (strategy, trade_date, entry_price, "
                "fill_price, exit_price, shares, filled_qty, pnl, scale_qty, "
                "scale_pnl, scaled_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                r)
        con.commit()
        con.close()
        return p

    def test_scaled_null_pnl_fallback(self, tmp_path, monkeypatch):
        p = self._db(tmp_path, [
            # scaled row, pnl NULL -> fallback = (10.5-10)*600 + 1200
            ('orb', DAY, 10.03, 10.0, 10.5, 1000, 1000, None, 400, 1200.0,
             '2026-08-24T14:00'),
            # unscaled closed row with recorded pnl
            ('orb', DAY, 5.0, 5.0, 5.5, 100, 100, 50.0, None, None, None),
        ])
        monkeypatch.setattr(rc, 'ROOT', tmp_path.parent)
        (tmp_path.parent / 'data').mkdir(exist_ok=True)
        import shutil
        shutil.copy(p, tmp_path.parent / 'data' / 'trades.db')
        got = rc.cumulative_orb_since('2026-01-01')
        assert got == pytest.approx((10.5 - 10.0) * 600 + 1200.0 + 50.0)


class TestSizingAttributionScale:
    def test_scaled_row_attributed(self, monkeypatch):
        monkeypatch.setattr(rc, 'load_live_rows',
                            lambda day, strategy=None: [_scaled_row()])
        monkeypatch.setattr(rc, 'eod_news_recheck', lambda syms, day: {})
        monkeypatch.setattr(rc, '_orb_sizing_cfg', lambda: {'enabled': False})
        monkeypatch.setattr(sqlite3, 'connect', _fake_conn_factory())
        attr = rc.sizing_attribution(DAY)
        t = attr['trades'][0]
        assert t['scaled'] is True
        assert t['scale_qty'] == 400
        assert t['scale_pnl'] == pytest.approx(1200.0)
        assert not attr['mult_mismatches']
        block = rc.sizing_block(attr)
        assert 'scaled 400sh' in block


def _fake_conn_factory():
    """sqlite3.connect stub for the sizing_attribution cum query."""
    class _Cur(list):
        pass

    class _Conn:
        row_factory = None

        def execute(self, *a, **kw):
            return []

        def close(self):
            pass
    return lambda *a, **kw: _Conn()


class TestGreenVerdictScaledRow:
    def test_green_with_scaled_row(self, monkeypatch):
        """P1-5 drill: day 1 of a scaled trade must not RED the streak —
        the runner's exit_reason is catalogued and nothing sticks in
        pending-verification."""
        monkeypatch.setattr(rc, 'load_live_rows',
                            lambda day, strategy=None: [_scaled_row()])
        monkeypatch.setattr(rc, 'bt_data_max_date', lambda: DAY)
        monkeypatch.setattr(rc, 'load_bt_selected',
                            lambda day: [{'symbol': 'TEST'}])
        monkeypatch.setattr(rc, 'journal_grep', lambda pat, day: [])
        monkeypatch.setattr(rc, 'read_selection_audit', lambda day: [])
        monkeypatch.setattr(rc, 'floored_stop_drift',
                            lambda day: {'available': False, 'n_checked': 0,
                                         'mismatches': []})
        v = rc.green_verdict(DAY)
        assert v['green'] is True, v['reasons']

    def test_scale_out_reason_is_catalogued(self):
        """If any reconcile path ever writes 'scale_out' into exit_reason,
        the vocabulary check must still know it (review P1-5)."""
        from trading.exit_reasons import is_attributed, is_known
        assert is_known('scale_out')
        assert is_attributed('scale_out')


class TestFlooredStopDrift:
    def _cache_db(self, tmp_path, n_bars=30):
        d = tmp_path / 'data'
        d.mkdir(exist_ok=True)
        p = d / 'cache.db'
        con = sqlite3.connect(p)
        con.execute("""CREATE TABLE daily_bars (
            symbol TEXT, bar_date TEXT, open REAL, high REAL, low REAL,
            close REAL, volume INTEGER, fetched_at TEXT,
            PRIMARY KEY (symbol, bar_date))""")
        for i in range(n_bars):
            day = f'2026-07-{i + 1:02d}'
            con.execute("INSERT INTO daily_bars VALUES "
                        "('TEST', ?, 10, 10.6, 9.6, 10.2, 1000, '')", (day,))
        con.commit()
        con.close()
        return tmp_path

    def test_skipped_when_flag_off(self, monkeypatch):
        monkeypatch.setattr(rc, '_winner_stack_cfg',
                            lambda: {'atr_stop_floor': {'enabled': False}})
        got = rc.floored_stop_drift(DAY)
        assert got['available'] is False

    def test_ok_when_recorded_matches(self, tmp_path, monkeypatch):
        root = self._cache_db(tmp_path)
        monkeypatch.setattr(rc, 'ROOT', root)
        monkeypatch.setattr(rc, '_winner_stack_cfg',
                            lambda: {'atr_stop_floor': {'enabled': True,
                                                        'k': 0.25}})
        from trading.orb_winner_stack import atr14_t1, floored_stop
        import pandas as pd
        con = sqlite3.connect(root / 'data' / 'cache.db')
        bars = pd.read_sql("SELECT bar_date, high, low, close FROM "
                           "daily_bars WHERE symbol='TEST' AND bar_date < ? "
                           "ORDER BY bar_date", con, params=(DAY,))
        con.close()
        atr = atr14_t1(bars.tail(40))
        expected, _ = floored_stop(9.0, 10.0, atr, 0.25)
        row = _scaled_row()
        row['real_stop_loss_price'] = expected
        monkeypatch.setattr(rc, 'load_live_rows',
                            lambda day, strategy=None: [row])
        got = rc.floored_stop_drift(DAY)
        assert got['available'] is True
        assert got['n_checked'] == 1
        assert not got['mismatches']

    def test_drift_flagged(self, tmp_path, monkeypatch):
        root = self._cache_db(tmp_path)
        monkeypatch.setattr(rc, 'ROOT', root)
        monkeypatch.setattr(rc, '_winner_stack_cfg',
                            lambda: {'atr_stop_floor': {'enabled': True,
                                                        'k': 0.25}})
        row = _scaled_row()
        row['real_stop_loss_price'] = 9.0    # live reverted to range_low
        monkeypatch.setattr(rc, 'load_live_rows',
                            lambda day, strategy=None: [row])
        got = rc.floored_stop_drift(DAY)
        assert got['available'] is True
        assert len(got['mismatches']) == 1

    def test_missing_recorded_stop_flagged(self, tmp_path, monkeypatch):
        root = self._cache_db(tmp_path)
        monkeypatch.setattr(rc, 'ROOT', root)
        monkeypatch.setattr(rc, '_winner_stack_cfg',
                            lambda: {'atr_stop_floor': {'enabled': True,
                                                        'k': 0.25}})
        row = _scaled_row()
        row['real_stop_loss_price'] = None
        monkeypatch.setattr(rc, 'load_live_rows',
                            lambda day, strategy=None: [row])
        got = rc.floored_stop_drift(DAY)
        assert len(got['mismatches']) == 1
