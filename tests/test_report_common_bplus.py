"""report_common B+ repoint — B+ RESTART 2026-08-15 (review P1-1, P2).

The nightly green-check must read the B+ book (config-driven path), not the
stale $100K/N4/pdr8/no-G1 orb_static_lock_trades.csv, or every night silently
compares live-B+ against the wrong ledger. Also: the news-drift soft check must
NOT fire when PM sizing is disabled (nothing gets sized by news then).
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

import scripts.report_common as rc

ROOT = Path(__file__).parent.parent


class TestBookPath:
    def test_reads_config_key(self):
        p = rc.bt_book_csv_path()
        assert p.endswith('orb_bplus_book.csv')

    def test_matches_orb_yaml(self):
        cfg = yaml.safe_load(open(ROOT / 'orb.yaml'))
        expected = cfg['backtest']['nightly_book_csv']
        assert rc.bt_book_csv_path().endswith(Path(expected).name)

    def test_latest_bt_trades_csv_uses_book_path(self, monkeypatch, tmp_path):
        fake = tmp_path / 'orb_bplus_book.csv'
        fake.write_text('date,symbol\n2026-08-17,AAA\n')
        monkeypatch.setattr(rc, 'bt_book_csv_path', lambda: str(fake))
        assert rc.latest_bt_trades_csv() == str(fake)

    def test_missing_book_returns_none(self, monkeypatch, tmp_path):
        monkeypatch.setattr(rc, 'bt_book_csv_path',
                            lambda: str(tmp_path / 'nope.csv'))
        assert rc.latest_bt_trades_csv() is None


class TestNewsDriftGatedWhenPmOff:
    def test_no_eod_recheck_when_pm_disabled(self, monkeypatch):
        """B+ pm OFF -> the news-drift EoD re-query is skipped (empty symbols),
        so no meaningless nightly noise (review P2)."""
        # _orb_sizing_cfg returns the pm mult block; force disabled.
        monkeypatch.setattr(rc, '_orb_sizing_cfg',
                            lambda: {'enabled': False, 'high_cut_usd': 5_000_000,
                                     'high_mult': 1.0, 'high_mult_news': 2.0,
                                     'news_gate': True})
        recheck = MagicMock(return_value={})
        monkeypatch.setattr(rc, 'eod_news_recheck', recheck)
        # one traded row with pm_mult recorded but has_news False
        monkeypatch.setattr(rc, 'load_live_rows', lambda day, strategy=None: [{
            'symbol': 'AAA', 'pnl': -10.0, 'fill_price': 10.0,
            'pattern_data': '{"pm_mult": 1.0, "has_news": false, '
                            '"quintile": "Q4", "pm_dollar_vol": 9000000}'}])
        # cumulative query hits the real trades.db; point it at an empty temp db
        d = tempfile.mkdtemp()
        empty = os.path.join(d, 'trades.db')
        conn = sqlite3.connect(empty)
        conn.execute("CREATE TABLE trades (symbol TEXT, pnl REAL, "
                     "pattern_data TEXT, strategy TEXT, trade_date TEXT)")
        conn.commit()
        conn.close()
        monkeypatch.setattr(rc, 'ROOT', Path(d).parent)
        # sqlite path in sizing_attribution is ROOT/'data'/'trades.db'; ensure it
        os.makedirs(Path(d) / 'data', exist_ok=True)
        sqlite3.connect(Path(d) / 'data' / 'trades.db').executescript(
            "CREATE TABLE trades (symbol TEXT, pnl REAL, pattern_data TEXT, "
            "strategy TEXT, trade_date TEXT);")
        monkeypatch.setattr(rc, 'ROOT', Path(d))
        out = rc.sizing_attribution('2026-08-17')
        # eod_news_recheck must have been called with an EMPTY symbol list
        assert recheck.call_args[0][0] == []
        assert out['news_drift'] == []
